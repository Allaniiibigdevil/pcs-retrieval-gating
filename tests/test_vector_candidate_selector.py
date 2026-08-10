import pytest

from app.retrieval.vector_candidate_selector import AdaptiveVectorCandidateSelector
from app.schemas.search import SearchHit


def vector_hit(doc_id: str, score: float, rank: int) -> SearchHit:
    return SearchHit(doc_id=doc_id, system_id="system", vector_score=score, vector_rank=rank)


def test_selector_keeps_every_hit_above_preferred_threshold() -> None:
    selector = AdaptiveVectorCandidateSelector(
        preferred_threshold=0.70,
        min_threshold=0.30,
        target_hits=2,
    )

    result = selector.select(
        [
            vector_hit("v1", 0.90, 1),
            vector_hit("v2", 0.80, 2),
            vector_hit("v3", 0.75, 3),
            vector_hit("v4", 0.60, 4),
        ]
    )

    assert result.effective_threshold == pytest.approx(0.70)
    assert [hit.doc_id for hit in result.vector_candidates] == ["v1", "v2", "v3"]


def test_selector_lowers_threshold_to_reach_target_count() -> None:
    selector = AdaptiveVectorCandidateSelector(
        preferred_threshold=0.70,
        min_threshold=0.30,
        target_hits=3,
    )

    result = selector.select(
        [
            vector_hit("v1", 0.80, 1),
            vector_hit("v2", 0.59, 2),
            vector_hit("v3", 0.51, 3),
            vector_hit("v4", 0.20, 4),
        ]
    )

    assert result.effective_threshold == pytest.approx(0.51)
    assert [hit.doc_id for hit in result.vector_candidates] == ["v1", "v2", "v3"]


def test_selector_never_lowers_below_minimum_threshold() -> None:
    selector = AdaptiveVectorCandidateSelector(
        preferred_threshold=0.70,
        min_threshold=0.30,
        target_hits=3,
    )

    result = selector.select(
        [
            vector_hit("v1", 0.50, 1),
            vector_hit("v2", 0.40, 2),
            vector_hit("v3", 0.20, 3),
        ]
    )

    assert result.effective_threshold == pytest.approx(0.40)
    assert [hit.doc_id for hit in result.vector_candidates] == ["v1", "v2"]
