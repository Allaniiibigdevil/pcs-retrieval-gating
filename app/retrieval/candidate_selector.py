from dataclasses import dataclass

from app.retrieval.candidate_merger import CandidateMerger
from app.retrieval.vector_candidate_selector import AdaptiveVectorCandidateSelector
from app.schemas.search import SearchHit


@dataclass(frozen=True)
class CandidateSelection:
    candidates: list[SearchHit]
    vector_candidates: list[SearchHit]
    effective_vector_threshold: float | None


class AdaptiveCandidateSelector:
    """Build the complete reranker set without making a second FAISS request."""

    def __init__(
        self,
        preferred_vector_threshold: float | None = None,
        min_vector_threshold: float | None = None,
        target_vector_hits: int | None = None,
        merger: CandidateMerger | None = None,
        vector_selector: AdaptiveVectorCandidateSelector | None = None,
    ) -> None:
        self.vector_selector = vector_selector or AdaptiveVectorCandidateSelector(
            preferred_threshold=preferred_vector_threshold,
            min_threshold=min_vector_threshold,
            target_hits=target_vector_hits,
        )
        self.merger = merger or CandidateMerger()

    def select(
        self,
        bm25_hits: list[SearchHit],
        vector_hits: list[SearchHit],
    ) -> CandidateSelection:
        vector_selection = self.vector_selector.select(vector_hits)
        selected_vector_hits = vector_selection.vector_candidates
        candidates = self.merger.merge(bm25_hits, selected_vector_hits)

        return CandidateSelection(
            candidates=candidates,
            vector_candidates=selected_vector_hits,
            effective_vector_threshold=vector_selection.effective_threshold,
        )
