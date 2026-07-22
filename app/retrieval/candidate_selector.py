from dataclasses import dataclass

from app.config import get_settings
from app.retrieval.candidate_merger import CandidateMerger
from app.schemas.search import SearchHit


@dataclass(frozen=True)
class CandidateSelection:
    candidates: list[SearchHit]
    vector_candidates: list[SearchHit]
    effective_vector_threshold: float | None


class AdaptiveCandidateSelector:
    """Build a bounded reranker set without making a second FAISS request."""

    def __init__(
        self,
        preferred_vector_threshold: float | None = None,
        min_vector_threshold: float | None = None,
        target_vector_hits: int | None = None,
        max_candidates: int | None = None,
        merger: CandidateMerger | None = None,
    ) -> None:
        settings = get_settings()
        self.preferred_vector_threshold = (
            settings.FAISS_PREFERRED_SCORE_THRESHOLD
            if preferred_vector_threshold is None
            else preferred_vector_threshold
        )
        self.min_vector_threshold = (
            settings.FAISS_MIN_SCORE_THRESHOLD
            if min_vector_threshold is None
            else min_vector_threshold
        )
        self.target_vector_hits = (
            settings.FAISS_TARGET_HITS if target_vector_hits is None else target_vector_hits
        )
        self.max_candidates = (
            settings.RERANKER_MAX_CANDIDATES if max_candidates is None else max_candidates
        )
        self.merger = merger or CandidateMerger()

        if self.min_vector_threshold > self.preferred_vector_threshold:
            raise ValueError(
                "min_vector_threshold must not exceed preferred_vector_threshold"
            )
        if self.target_vector_hits <= 0:
            raise ValueError("target_vector_hits must be greater than 0")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be greater than 0")

    def select(
        self,
        bm25_hits: list[SearchHit],
        vector_hits: list[SearchHit],
    ) -> CandidateSelection:
        selected_vector_hits, effective_threshold = self._select_vector_hits(vector_hits)
        merged = self.merger.merge(bm25_hits, selected_vector_hits)

        if len(merged) <= self.max_candidates:
            candidates = merged
        else:
            selected_ids = _round_robin_doc_ids(
                bm25_hits,
                selected_vector_hits,
                self.max_candidates,
            )
            merged_by_id = {hit.doc_id: hit for hit in merged}
            candidates = [merged_by_id[doc_id] for doc_id in selected_ids]

        return CandidateSelection(
            candidates=candidates,
            vector_candidates=selected_vector_hits,
            effective_vector_threshold=effective_threshold,
        )

    def _select_vector_hits(
        self,
        vector_hits: list[SearchHit],
    ) -> tuple[list[SearchHit], float | None]:
        eligible = sorted(
            (
                hit
                for hit in vector_hits
                if hit.vector_score is not None
                and hit.vector_score >= self.min_vector_threshold
            ),
            key=_vector_sort_key,
        )
        if not eligible:
            return [], None

        preferred = [
            hit
            for hit in eligible
            if hit.vector_score is not None
            and hit.vector_score >= self.preferred_vector_threshold
        ]
        if len(preferred) >= self.target_vector_hits:
            effective_threshold = self.preferred_vector_threshold
        else:
            cutoff_index = min(self.target_vector_hits, len(eligible)) - 1
            cutoff_score = eligible[cutoff_index].vector_score
            assert cutoff_score is not None
            effective_threshold = max(self.min_vector_threshold, cutoff_score)

        selected = [
            hit
            for hit in eligible
            if hit.vector_score is not None and hit.vector_score >= effective_threshold
        ]
        return selected, effective_threshold


def _vector_sort_key(hit: SearchHit) -> tuple[int, float, str]:
    rank = hit.vector_rank if hit.vector_rank is not None else 2**31 - 1
    score = hit.vector_score if hit.vector_score is not None else float("-inf")
    return rank, -score, hit.doc_id


def _bm25_sort_key(hit: SearchHit) -> tuple[int, str]:
    rank = hit.bm25_rank if hit.bm25_rank is not None else 2**31 - 1
    return rank, hit.doc_id


def _round_robin_doc_ids(
    bm25_hits: list[SearchHit],
    vector_hits: list[SearchHit],
    limit: int,
) -> list[str]:
    """Keep both recall routes represented when the reranker budget is full."""

    channels = [
        sorted(bm25_hits, key=_bm25_sort_key),
        sorted(vector_hits, key=_vector_sort_key),
    ]
    positions = [0, 0]
    selected: list[str] = []
    seen: set[str] = set()

    while len(selected) < limit:
        made_progress = False
        for channel_index, hits in enumerate(channels):
            while (
                positions[channel_index] < len(hits)
                and hits[positions[channel_index]].doc_id in seen
            ):
                positions[channel_index] += 1
            if positions[channel_index] >= len(hits):
                continue

            hit = hits[positions[channel_index]]
            positions[channel_index] += 1
            seen.add(hit.doc_id)
            selected.append(hit.doc_id)
            made_progress = True
            if len(selected) >= limit:
                break

        if not made_progress:
            break

    return selected
