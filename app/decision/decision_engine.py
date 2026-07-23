import logging

from app.config import get_settings
from app.decision.evidence_builder import EvidenceBuilder
from app.decision.query_normalizer import QueryNormalizer
from app.decision.query_rewriter import prepare_queries, rewrite_queries
from app.decision.system_aggregator import SystemAggregator
from app.reranking.factory import Reranker, get_reranker
from app.retrieval.candidate_selector import AdaptiveCandidateSelector
from app.retrieval.factory import Retriever, get_keyword_retriever, get_vector_retriever
from app.retrieval.parallel_query_retriever import (
    ChannelSearchResult,
    search_queries_in_parallel,
)
from app.schemas.decision import DecideResponse
from app.utils.timing import StageTimer

logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(
        self,
        normalizer: QueryNormalizer | None = None,
        keyword_retriever: Retriever | None = None,
        vector_retriever: Retriever | None = None,
        candidate_selector: AdaptiveCandidateSelector | None = None,
        reranker: Reranker | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        aggregator: SystemAggregator | None = None,
    ) -> None:
        self.normalizer = normalizer or QueryNormalizer()
        self.keyword_retriever = keyword_retriever or get_keyword_retriever()
        self.vector_retriever = vector_retriever or get_vector_retriever()
        self.candidate_selector = candidate_selector or AdaptiveCandidateSelector()
        self.reranker = reranker or get_reranker()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.aggregator = aggregator or SystemAggregator()

    async def decide(
        self,
        task: str,
        task_id: str | None = None,
    ) -> DecideResponse:
        settings = get_settings()
        es_top_k = settings.ES_TOP_K_DOCS
        faiss_top_k = settings.FAISS_TOP_K_DOCS
        logger.info(
            "decision_started task_id=%s es_top_k=%d faiss_top_k=%d",
            task_id,
            es_top_k,
            faiss_top_k,
        )
        timer = StageTimer()
        queries = prepare_queries(task, await rewrite_queries(task))
        bm25_queries = [self.normalizer.normalize(query) for query in queries]
        timer.mark("query_rewrite")

        search_result = await search_queries_in_parallel(
            keyword_retriever=self.keyword_retriever,
            vector_retriever=self.vector_retriever,
            keyword_queries=bm25_queries,
            vector_queries=queries,
            keyword_top_k=es_top_k,
            vector_top_k=faiss_top_k,
        )
        timer.mark("parallel_search")
        _log_failures("keyword", task_id, search_result.keyword)
        _log_failures("vector", task_id, search_result.vector)

        if search_result.keyword.all_failed and search_result.vector.all_failed:
            logger.error("decision_failed task_id=%s", task_id)
            cause = search_result.vector.failures[-1].error
            raise RuntimeError("Both ES and vector search failed") from cause

        bm25_hits = search_result.keyword.hits
        vector_hits = search_result.vector.hits

        selection = self.candidate_selector.select(bm25_hits, vector_hits)
        timer.mark("candidate_select")
        evidence_docs = self.evidence_builder.build(task, selection.candidates)
        reranked_docs = await self.reranker.rerank(task, evidence_docs)
        timer.mark("rerank")
        decisions = self.aggregator.aggregate(reranked_docs)
        selected_systems = [item.system_id for item in decisions if item.selected]
        timer.mark("aggregate")
        latency_ms = timer.finish()

        logger.info(
            "decision_completed task_id=%s query_count=%d bm25_hits=%d vector_hits=%d "
            "vector_candidates=%d reranker_candidates=%d "
            "effective_vector_threshold=%s selected_systems=%s latency_ms=%s",
            task_id,
            len(queries),
            len(bm25_hits),
            len(vector_hits),
            len(selection.vector_candidates),
            len(selection.candidates),
            selection.effective_vector_threshold,
            selected_systems,
            latency_ms,
        )
        return DecideResponse(
            task_id=task_id,
            task=task,
            rewritten_queries=queries,
            selected_systems=selected_systems,
            decisions=decisions,
            latency_ms=latency_ms,
        )


def _log_failures(
    channel: str,
    task_id: str | None,
    result: ChannelSearchResult,
) -> None:
    for failure in result.failures:
        error = failure.error
        logger.error(
            "%s_search_failed task_id=%s query_index=%d",
            channel,
            task_id,
            failure.query_index,
            exc_info=(type(error), error, error.__traceback__),
        )
