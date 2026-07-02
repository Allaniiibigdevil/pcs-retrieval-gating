import logging

from app.decision.evidence_builder import EvidenceBuilder
from app.decision.query_normalizer import QueryNormalizer
from app.decision.system_aggregator import SystemAggregator
from app.retrieval.candidate_merger import CandidateMerger
from app.retrieval.es_retriever import ESRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.decision import DecideResponse
from app.schemas.search import SearchHit
from app.utils.timing import StageTimer

logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(
        self,
        normalizer: QueryNormalizer | None = None,
        es_retriever: ESRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        merger: CandidateMerger | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        aggregator: SystemAggregator | None = None,
    ) -> None:
        self.normalizer = normalizer or QueryNormalizer()
        self.es_retriever = es_retriever or ESRetriever()
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.merger = merger or CandidateMerger()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.aggregator = aggregator or SystemAggregator()

    async def decide(
        self,
        task: str,
        task_id: str | None = None,
        top_k_docs: int = 50,
        max_systems: int = 5,
    ) -> DecideResponse:
        timer = StageTimer()
        normalized_query = self.normalizer.normalize(task)
        timer.mark("normalize")

        bm25_hits: list[SearchHit] = []
        vector_hits: list[SearchHit] = []
        es_error: Exception | None = None
        vector_error: Exception | None = None

        try:
            bm25_hits = await self.es_retriever.search(normalized_query, top_k_docs)
        except Exception as exc:
            es_error = exc
            logger.exception("es_search_failed", extra={"task_id": task_id})
        timer.mark("es_search")

        try:
            vector_hits = await self.vector_retriever.search(normalized_query, top_k_docs)
        except Exception as exc:
            vector_error = exc
            logger.exception("vector_search_failed", extra={"task_id": task_id})
        timer.mark("vector_search")

        if es_error is not None and vector_error is not None:
            logger.error("decision_failed", extra={"task_id": task_id})
            raise RuntimeError("Both ES and vector search failed") from vector_error

        candidates = self.merger.merge(bm25_hits, vector_hits)
        timer.mark("merge")
        evidence_docs = self.evidence_builder.build(normalized_query, candidates)
        decisions = self.aggregator.aggregate(evidence_docs, max_systems)
        timer.mark("aggregate")
        latency_ms = timer.finish()

        logger.info(
            "decision_completed",
            extra={
                "task_id": task_id,
                "bm25_hit_count": len(bm25_hits),
                "vector_hit_count": len(vector_hits),
                "merged_candidate_count": len(candidates),
                "selected_systems": [item.system_id for item in decisions],
                "latency_ms": latency_ms,
            },
        )
        return DecideResponse(task_id=task_id, task=task, decisions=decisions, latency_ms=latency_ms)
