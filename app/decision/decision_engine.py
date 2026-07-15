import logging
from uuid import uuid4

from app.config import get_settings
from app.decision.evidence_builder import EvidenceBuilder
from app.decision.query_normalizer import QueryNormalizer
from app.decision.gating_features import (
    append_gating_case,
    build_gating_case,
)
from app.decision.system_aggregator import SystemAggregator
from app.retrieval.candidate_merger import CandidateMerger
from app.retrieval.factory import Retriever, get_keyword_retriever, get_vector_retriever
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
        merger: CandidateMerger | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        aggregator: SystemAggregator | None = None,
    ) -> None:
        self.normalizer = normalizer or QueryNormalizer()
        self.keyword_retriever = keyword_retriever or get_keyword_retriever()
        self.vector_retriever = vector_retriever or get_vector_retriever()
        self.merger = merger or CandidateMerger()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.aggregator = aggregator or SystemAggregator()

    async def decide(
        self,
        task: str,
        task_id: str | None = None,
        top_k_docs: int = 20,
        max_systems: int = 5,
    ) -> DecideResponse:
        effective_task_id = task_id or uuid4().hex
        timer = StageTimer()
        bm25_query = self.normalizer.normalize(task)
        vector_query = task
        timer.mark("normalize")

        bm25_hits: list[SearchHit] = []
        vector_hits: list[SearchHit] = []
        keyword_error: Exception | None = None
        vector_error: Exception | None = None

        try:
            bm25_hits = await self.keyword_retriever.search(bm25_query, top_k_docs)
        except Exception as exc:
            keyword_error = exc
            logger.exception("keyword_search_failed", extra={"task_id": effective_task_id})
        timer.mark("keyword_search")

        try:
            vector_hits = await self.vector_retriever.search(vector_query, top_k_docs)
        except Exception as exc:
            vector_error = exc
            logger.exception("vector_search_failed", extra={"task_id": effective_task_id})
        timer.mark("vector_search")

        if keyword_error is not None and vector_error is not None:
            logger.error("decision_failed", extra={"task_id": effective_task_id})
            raise RuntimeError("Both ES and vector search failed") from vector_error

        candidates = self.merger.merge(bm25_hits, vector_hits)
        timer.mark("merge")
        evidence_docs = self.evidence_builder.build(task, candidates)
        settings = get_settings()
        gating_case = build_gating_case(task, evidence_docs, task_id=effective_task_id)
        if settings.GATING_CASE_LOG_ENABLED:
            append_gating_case(gating_case)

        decisions = self.aggregator.aggregate(
            evidence_docs,
            max_systems=max_systems,
            query_text=task,
            gating_case=gating_case,
        )
        selected_systems = [item.system_id for item in decisions if item.selected]
        timer.mark("aggregate")
        latency_ms = timer.finish()

        logger.info(
            "decision_completed",
            extra={
                "task_id": effective_task_id,
                "bm25_hit_count": len(bm25_hits),
                "vector_hit_count": len(vector_hits),
                "merged_candidate_count": len(candidates),
                "selected_systems": selected_systems,
                "gating_bag_count": len(gating_case.systems),
                "gating_doc_count": sum(len(system.docs) for system in gating_case.systems),
                "latency_ms": latency_ms,
            },
        )
        return DecideResponse(
            task_id=effective_task_id,
            task=task,
            selected_systems=selected_systems,
            decisions=decisions,
            latency_ms=latency_ms,
        )
