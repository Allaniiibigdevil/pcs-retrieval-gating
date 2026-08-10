import asyncio
import logging
from collections.abc import Callable

from app.decision.evidence_builder import EvidenceBuilder
from app.decision.query_normalizer import QueryNormalizer
from app.decision.query_rewriter import prepare_queries, rewrite_queries
from app.decision.system_aggregator import SystemAggregator
from app.retrieval.candidate_merger import CandidateMerger
from app.retrieval.candidate_selector import AdaptiveCandidateSelector
from app.retrieval.factory import Retriever, build_keyword_retriever, build_vector_retriever
from app.retrieval.parallel_query_retriever import (
    ChannelSearchResult,
    ParallelQuerySearchResult,
    search_queries_in_parallel,
)
from app.schemas.decision import DecideResponse, SystemDecision
from app.source_registry import SourceConfig, SourceRegistry, get_source_registry
from app.utils.timing import StageTimer

logger = logging.getLogger(__name__)
RetrieverFactory = Callable[[SourceConfig], Retriever]


class DecisionEngine:
    def __init__(
        self,
        *,
        normalizer: QueryNormalizer | None = None,
        merger: CandidateMerger | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        source_registry: SourceRegistry | None = None,
        keyword_retriever_factory: RetrieverFactory = build_keyword_retriever,
        vector_retriever_factory: RetrieverFactory = build_vector_retriever,
    ) -> None:
        self.normalizer = normalizer or QueryNormalizer()
        self.merger = merger or CandidateMerger()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.source_registry = source_registry or get_source_registry()
        self.keyword_retriever_factory = keyword_retriever_factory
        self.vector_retriever_factory = vector_retriever_factory

    async def decide(self, task: str, task_id: str | None = None) -> DecideResponse:
        sources = self.source_registry.enabled_sources
        if not sources:
            raise RuntimeError("No enabled sources are configured")

        timer = StageTimer()
        queries = prepare_queries(task, await rewrite_queries(task))
        keyword_queries = [self.normalizer.normalize(query) for query in queries]
        timer.mark("query_rewrite")

        raw_results = await asyncio.gather(
            *(
                self._search_source(source, keyword_queries, queries)
                for source in sources
            ),
            return_exceptions=True,
        )
        timer.mark("parallel_search")

        decisions: list[SystemDecision] = []
        successful_sources = 0
        total_keyword_hits = 0
        total_vector_hits = 0
        total_vector_candidates = 0
        total_candidates = 0
        effective_thresholds: dict[str, float | None] = {}
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

            _log_failures(source.source_id, "keyword", task_id, raw_result.keyword)
            _log_failures(source.source_id, "vector", task_id, raw_result.vector)
            if raw_result.keyword.all_failed and raw_result.vector.all_failed:
                last_error = _last_failure(raw_result)
                continue

            successful_sources += 1
            keyword_hits = raw_result.keyword.hits
            raw_vector_hits = raw_result.vector.hits
            total_keyword_hits += len(keyword_hits)
            total_vector_hits += len(raw_vector_hits)

            selection = AdaptiveCandidateSelector(
                source_id=source.source_id,
                preferred_vector_threshold=source.faiss_preferred_score_threshold,
                min_vector_threshold=source.faiss_min_score_threshold,
                target_vector_hits=source.faiss_target_hits,
                merger=self.merger,
            ).select(keyword_hits, raw_vector_hits)
            effective_thresholds[source.source_id] = selection.effective_vector_threshold
            total_vector_candidates += len(selection.vector_candidates)
            candidates = selection.candidates
            total_candidates += len(candidates)

            evidence_docs = self.evidence_builder.build(task, candidates)
            decision = SystemAggregator(
                source_id=source.source_id,
                selection_threshold=source.selection_threshold,
                evidence_docs_per_system=source.evidence_docs_per_system,
                es_score_weight=source.es_score_weight,
                agreement_weight=source.agreement_weight,
                semantic_match_threshold=source.semantic_match_threshold,
                lexical_match_threshold=source.lexical_match_threshold,
            ).aggregate(evidence_docs)
            if decision is not None:
                decisions.append(decision)

        timer.mark("score_and_aggregate")
        if successful_sources == 0:
            raise RuntimeError("All configured source searches failed") from last_error

        decisions.sort(key=lambda item: (-item.confidence, item.system_id))
        response = _response(task_id, task, queries, decisions, timer)
        logger.info(
            "decision_completed task_id=%s source_count=%d query_count=%d keyword_hits=%d "
            "vector_hits=%d vector_candidates=%d effective_vector_thresholds=%s "
            "merged_candidates=%d selected_systems=%s latency_ms=%s",
            task_id,
            len(sources),
            len(queries),
            total_keyword_hits,
            total_vector_hits,
            total_vector_candidates,
            effective_thresholds,
            total_candidates,
            response.selected_systems,
            response.latency_ms,
        )
        return response

    async def _search_source(
        self,
        source: SourceConfig,
        keyword_queries: list[str],
        vector_queries: list[str],
    ) -> ParallelQuerySearchResult:
        return await search_queries_in_parallel(
            keyword_retriever=self.keyword_retriever_factory(source),
            vector_retriever=self.vector_retriever_factory(source),
            keyword_queries=keyword_queries,
            vector_queries=vector_queries,
            keyword_top_k=source.es_top_k,
            vector_top_k=source.faiss_top_k,
        )


def _last_failure(result: ParallelQuerySearchResult) -> Exception | None:
    if result.vector.failures:
        return result.vector.failures[-1].error
    if result.keyword.failures:
        return result.keyword.failures[-1].error
    return None


def _response(
    task_id: str | None,
    task: str,
    queries: list[str],
    decisions: list[SystemDecision],
    timer: StageTimer,
) -> DecideResponse:
    return DecideResponse(
        task_id=task_id,
        task=task,
        rewritten_queries=queries,
        selected_systems=[item.system_id for item in decisions if item.selected],
        decisions=decisions,
        latency_ms=timer.finish(),
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
