from dataclasses import dataclass

from app.schemas.search import SearchHit


@dataclass(frozen=True)
class VectorCandidateSelection:
    vector_candidates: list[SearchHit]
    effective_threshold: float | None


class AdaptiveVectorCandidateSelector:
    def __init__(
        self,
        *,
        preferred_threshold: float,
        min_threshold: float,
        target_hits: int,
    ) -> None:
        self.preferred_threshold = preferred_threshold
        self.min_threshold = min_threshold
        self.target_hits = target_hits

        if not -1.0 <= self.min_threshold <= 1.0:
            raise ValueError("min_threshold must be between -1 and 1")
        if not -1.0 <= self.preferred_threshold <= 1.0:
            raise ValueError("preferred_threshold must be between -1 and 1")
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
