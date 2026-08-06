import asyncio
import logging
from dataclasses import dataclass

from app.config import get_settings
from app.decision.evidence_builder import EvidenceBuilder
from app.decision.query_normalizer import QueryNormalizer
from app.decision.query_rewriter import prepare_queries, rewrite_queries
from app.decision.system_aggregator import SystemAggregator
from app.retrieval.candidate_merger import CandidateMerger
from app.retrieval.factory import (
    Retriever,
    build_keyword_retriever,
    build_vector_retriever,
    get_keyword_retriever,
    get_vector_retriever,
)
from app.retrieval.parallel_query_retriever import (
    ChannelSearchResult,
    ParallelQuerySearchResult,
    search_queries_in_parallel,
)
from app.schemas.decision import DecideResponse, SystemDecision
from app.source_registry import SourceConfig, SourceRegistry, get_source_registry
from app.utils.timing import StageTimer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SourceSearch:
    source: SourceConfig
    result: ParallelQuerySearchResult


class DecisionEngine:
    def __init__(
        self,
        normalizer: QueryNormalizer | None = None,
        keyword_retriever: Retriever | None = None,
        vector_retriever: Retriever | None = None,
        merger: CandidateMerger | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        aggregator: SystemAggregator | None = None,
        source_registry: SourceRegistry | None = None,
    ) -> None:
        self.normalizer = normalizer or QueryNormalizer()
        self.merger = merger or CandidateMerger()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.aggregator = aggregator or SystemAggregator()

        self.legacy_mode = keyword_retriever is not None or vector_retriever is not None
        self.keyword_retriever = keyword_retriever or (
            get_keyword_retriever() if self.legacy_mode else None
        )
        self.vector_retriever = vector_retriever or (
            get_vector_retriever() if self.legacy_mode else None
        )
        self.source_registry = source_registry or (
            None if self.legacy_mode else get_source_registry()
        )

    async def decide(
        self,
        task: str,
        task_id: str | None = None,
    ) -> DecideResponse:
        timer = StageTimer()
        queries = prepare_queries(task, await rewrite_queries(task))
        keyword_queries = [self.normalizer.normalize(query) for query in queries]
        timer.mark("query_rewrite")

        if self.legacy_mode:
            return await self._decide_legacy(
                task=task,
                task_id=task_id,
                queries=queries,
                keyword_queries=keyword_queries,
                timer=timer,
            )
        return await self._decide_multi_source(
            task=task,
            task_id=task_id,
            queries=queries,
            keyword_queries=keyword_queries,
            timer=timer,
        )

    async def _decide_legacy(
        self,
        *,
        task: str,
        task_id: str | None,
        queries: list[str],
        keyword_queries: list[str],
        timer: StageTimer,
    ) -> DecideResponse:
        settings = get_settings()
        assert self.keyword_retriever is not None
        assert self.vector_retriever is not None
        search_result = await search_queries_in_parallel(
            keyword_retriever=self.keyword_retriever,
            vector_retriever=self.vector_retriever,
            keyword_queries=keyword_queries,
            vector_queries=queries,
            keyword_top_k=settings.ES_TOP_K_DOCS,
            vector_top_k=settings.FAISS_TOP_K_DOCS,
        )
        timer.mark("parallel_search")
        _log_failures("legacy", "keyword", task_id, search_result.keyword)
        _log_failures("legacy", "vector", task_id, search_result.vector)
        if search_result.keyword.all_failed and search_result.vector.all_failed:
            cause = search_result.vector.failures[-1].error
            raise RuntimeError("Both ES and vector search failed") from cause

        candidates = self.merger.merge(
            search_result.keyword.hits,
            search_result.vector.hits,
        )
        timer.mark("merge")
        evidence_docs = self.evidence_builder.build(task, candidates)
        decisions = self.aggregator.aggregate(evidence_docs)
        timer.mark("aggregate")
        return _response(task_id, task, queries, decisions, timer)

    async def _decide_multi_source(
        self,
        *,
        task: str,
        task_id: str | None,
        queries: list[str],
        keyword_queries: list[str],
        timer: StageTimer,
    ) -> DecideResponse:
        assert self.source_registry is not None
        sources = self.source_registry.enabled_sources
        if not sources:
            raise RuntimeError("No enabled sources are configured")

        raw_results = await asyncio.gather(
            *(_search_source(source, keyword_queries, queries) for source in sources),
            return_exceptions=True,
        )
        timer.mark("parallel_search")

        decisions: list[SystemDecision] = []
        successful_sources = 0
        total_keyword_hits = 0
        total_vector_hits = 0
        total_candidates = 0
        last_error: Exception | None = None

        for source, raw_result in zip(sources, raw_results):
            if isinstance(raw_result, BaseException):
                last_error = (
                    raw_result
                    if isinstance(raw_result, Exception)
                    else RuntimeError(str(raw_result))
                )
                logger.error(
                    "source_search_failed task_id=%s source_id=%s",
                    task_id,
                    source.source_id,
                    exc_info=(type(raw_result), raw_result, raw_result.__traceback__),
                )
                continue

            search_result = raw_result.result
            _log_failures(source.source_id, "keyword", task_id, search_result.keyword)
            _log_failures(source.source_id, "vector", task_id, search_result.vector)
            if search_result.keyword.all_failed and search_result.vector.all_failed:
                if search_result.vector.failures:
                    last_error = search_result.vector.failures[-1].error
                elif search_result.keyword.failures:
                    last_error = search_result.keyword.failures[-1].error
                continue

            successful_sources += 1
            keyword_hits = _only_source(search_result.keyword.hits, source.source_id)
            vector_hits = _only_source(search_result.vector.hits, source.source_id)
            total_keyword_hits += len(keyword_hits)
            total_vector_hits += len(vector_hits)
            candidates = self.merger.merge(keyword_hits, vector_hits)
            total_candidates += len(candidates)
            evidence_docs = self.evidence_builder.build(task, candidates)
            source_aggregator = SystemAggregator(
                selection_threshold=source.selection_threshold,
                evidence_docs_per_system=source.evidence_docs_per_system,
                es_score_weight=source.es_score_weight,
                agreement_weight=source.agreement_weight,
                semantic_match_threshold=source.semantic_match_threshold,
                lexical_match_threshold=source.lexical_match_threshold,
            )
            decisions.extend(source_aggregator.aggregate(evidence_docs))

        timer.mark("merge")
        timer.mark("aggregate")
        if successful_sources == 0:
            raise RuntimeError("All configured source searches failed") from last_error

        decisions.sort(key=lambda item: (-item.confidence, item.system_id))
        response = _response(task_id, task, queries, decisions, timer)
        logger.info(
            "decision_completed task_id=%s source_count=%d query_count=%d keyword_hits=%d "
            "vector_hits=%d merged_candidates=%d selected_systems=%s latency_ms=%s",
            task_id,
            len(sources),
            len(queries),
            total_keyword_hits,
            total_vector_hits,
            total_candidates,
            response.selected_systems,
            response.latency_ms,
        )
        return response


async def _search_source(
    source: SourceConfig,
    keyword_queries: list[str],
    vector_queries: list[str],
) -> _SourceSearch:
    result = await search_queries_in_parallel(
        keyword_retriever=build_keyword_retriever(source),
        vector_retriever=build_vector_retriever(source),
        keyword_queries=keyword_queries,
        vector_queries=vector_queries,
        keyword_top_k=source.es_top_k,
        vector_top_k=source.faiss_top_k,
    )
    return _SourceSearch(source=source, result=result)


def _only_source(hits: list, source_id: str) -> list:
    mismatched = [hit.doc_id for hit in hits if hit.system_id != source_id]
    if mismatched:
        logger.warning(
            "source_result_mismatch source_id=%s doc_ids=%s",
            source_id,
            mismatched[:5],
        )
    return [hit for hit in hits if hit.system_id == source_id]


def _response(
    task_id: str | None,
    task: str,
    queries: list[str],
    decisions: list[SystemDecision],
    timer: StageTimer,
) -> DecideResponse:
    selected_systems = [item.system_id for item in decisions if item.selected]
    latency_ms = timer.finish()
    return DecideResponse(
        task_id=task_id,
        task=task,
        rewritten_queries=queries,
        selected_systems=selected_systems,
        decisions=decisions,
        latency_ms=latency_ms,
    )


def _log_failures(
    source_id: str,
    channel: str,
    task_id: str | None,
    result: ChannelSearchResult,
) -> None:
    for failure in result.failures:
        error = failure.error
        logger.error(
            "%s_search_failed task_id=%s source_id=%s query_index=%d",
            channel,
            task_id,
            source_id,
            failure.query_index,
            exc_info=(type(error), error, error.__traceback__),
        )
