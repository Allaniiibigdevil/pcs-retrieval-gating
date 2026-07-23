import logging

from app.config import get_settings
from app.decision.evidence_builder import EvidenceBuilder
from app.decision.query_normalizer import QueryNormalizer
from app.decision.system_aggregator import SystemAggregator
from app.retrieval.candidate_merger import CandidateMerger
from app.retrieval.factory import Retriever, get_keyword_retriever, get_vector_retriever
from app.retrieval.vector_candidate_selector import AdaptiveVectorCandidateSelector
from app.schemas.decision import DecideResponse
from app.schemas.search import SearchHit
from app.utils.timing import StageTimer

logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(
        self,
        normalizer: QueryNormalizer | None = None,
        keyword_retriever: Retriever | None = None,
        vector_retriever: Retriever | None = None,
        vector_selector: AdaptiveVectorCandidateSelector | None = None,
        merger: CandidateMerger | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        aggregator: SystemAggregator | None = None,
    ) -> None:
        self.normalizer = normalizer or QueryNormalizer()
        self.keyword_retriever = keyword_retriever or get_keyword_retriever()
        self.vector_retriever = vector_retriever or get_vector_retriever()
        self.vector_selector = vector_selector or AdaptiveVectorCandidateSelector()
        self.merger = merger or CandidateMerger()
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
        bm25_query = self.normalizer.normalize(task)
        vector_query = task
        timer.mark("normalize")

        bm25_hits: list[SearchHit] = []
        vector_hits: list[SearchHit] = []
        keyword_error: Exception | None = None
        vector_error: Exception | None = None

        try:
            bm25_hits = await self.keyword_retriever.search(bm25_query, es_top_k)
        except Exception as exc:
            keyword_error = exc
            logger.exception("keyword_search_failed task_id=%s", task_id)
        timer.mark("keyword_search")

        try:
            vector_hits = await self.vector_retriever.search(vector_query, faiss_top_k)
        except Exception as exc:
            vector_error = exc
            logger.exception("vector_search_failed task_id=%s", task_id)
        timer.mark("vector_search")

        if keyword_error is not None and vector_error is not None:
            logger.error("decision_failed task_id=%s", task_id)
            raise RuntimeError("Both ES and vector search failed") from vector_error

        vector_selection = self.vector_selector.select(vector_hits)
        candidates = self.merger.merge(bm25_hits, vector_selection.vector_candidates)
        timer.mark("merge")
        evidence_docs = self.evidence_builder.build(task, candidates)
        decisions = self.aggregator.aggregate(evidence_docs)
        selected_systems = [item.system_id for item in decisions if item.selected]
        timer.mark("aggregate")
        latency_ms = timer.finish()

        logger.info(
            "decision_completed task_id=%s bm25_hits=%d vector_hits=%d "
            "vector_candidates=%d effective_vector_threshold=%s "
            "merged_candidates=%d selected_systems=%s latency_ms=%s",
            task_id,
            len(bm25_hits),
            len(vector_hits),
            len(vector_selection.vector_candidates),
            vector_selection.effective_threshold,
            len(candidates),
            selected_systems,
            latency_ms,
        )
        return DecideResponse(
            task_id=task_id,
            task=task,
            selected_systems=selected_systems,
            decisions=decisions,
            latency_ms=latency_ms,
        )
