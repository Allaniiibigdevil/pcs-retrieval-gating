from dataclasses import dataclass

from app.config import get_settings
from app.schemas.search import SearchHit


@dataclass(frozen=True)
class VectorCandidateSelection:
    vector_candidates: list[SearchHit]
    effective_threshold: float | None


class AdaptiveVectorCandidateSelector:
    """Select vector candidates from one FAISS result set using an adaptive threshold."""

    def __init__(
        self,
        preferred_threshold: float | None = None,
        min_threshold: float | None = None,
        target_hits: int | None = None,
    ) -> None:
        settings = get_settings()
        self.preferred_threshold = (
            settings.FAISS_PREFERRED_SCORE_THRESHOLD
            if preferred_threshold is None
            else preferred_threshold
        )
        self.min_threshold = (
            settings.FAISS_MIN_SCORE_THRESHOLD if min_threshold is None else min_threshold
        )
        self.target_hits = settings.FAISS_TARGET_HITS if target_hits is None else target_hits

        if self.min_threshold > self.preferred_threshold:
            raise ValueError("min_threshold must not exceed preferred_threshold")
        if self.target_hits <= 0:
            raise ValueError("target_hits must be greater than 0")

    def select(self, vector_hits: list[SearchHit]) -> VectorCandidateSelection:
        eligible = sorted(
            (
                hit
                for hit in vector_hits
                if hit.vector_score is not None and hit.vector_score >= self.min_threshold
            ),
            key=_vector_sort_key,
        )
        if not eligible:
            return VectorCandidateSelection([], None)

        preferred_count = sum(
            hit.vector_score is not None
            and hit.vector_score >= self.preferred_threshold
            for hit in eligible
        )
        if preferred_count >= self.target_hits:
            effective_threshold = self.preferred_threshold
        else:
            cutoff_index = min(self.target_hits, len(eligible)) - 1
            cutoff_score = eligible[cutoff_index].vector_score
            assert cutoff_score is not None
            effective_threshold = max(self.min_threshold, cutoff_score)

        selected = [
            hit
            for hit in eligible
            if hit.vector_score is not None and hit.vector_score >= effective_threshold
        ]
        return VectorCandidateSelection(selected, effective_threshold)


def _vector_sort_key(hit: SearchHit) -> tuple[int, float, str]:
    rank = hit.vector_rank if hit.vector_rank is not None else 2**31 - 1
    score = hit.vector_score if hit.vector_score is not None else float("-inf")
    return rank, -score, hit.doc_id
