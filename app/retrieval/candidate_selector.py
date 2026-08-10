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
    def __init__(
        self,
        *,
        source_id: str,
        preferred_vector_threshold: float,
        min_vector_threshold: float,
        target_vector_hits: int,
        merger: CandidateMerger | None = None,
    ) -> None:
        self.source_id = source_id
        self.vector_selector = AdaptiveVectorCandidateSelector(
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
        candidates = self.merger.merge(
            self.source_id,
            bm25_hits,
            selected_vector_hits,
        )
        return CandidateSelection(
            candidates=candidates,
            vector_candidates=selected_vector_hits,
            effective_vector_threshold=vector_selection.effective_threshold,
        )
