import pytest

from app.retrieval.candidate_selector import AdaptiveCandidateSelector
from app.schemas.search import SearchHit


def _selector(*, preferred: float = 0.7, minimum: float = 0.3, target: int = 3):
    return AdaptiveCandidateSelector(
        source_id="memo",
        preferred_vector_threshold=preferred,
        min_vector_threshold=minimum,
        target_vector_hits=target,
    )


def _vector(doc_id: str, score: float, rank: int) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id="memo",
        vector_score=score,
        vector_rank=rank,
    )


def _bm25(doc_id: str, rank: int) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id="memo",
        bm25_score=10.0 - rank,
        bm25_rank=rank,
    )


def test_selector_lowers_vector_threshold_to_target_hit_count() -> None:
    result = _selector().select(
        [],
        [
            _vector("v1", 0.80, 1),
            _vector("v2", 0.59, 2),
            _vector("v3", 0.51, 3),
            _vector("v4", 0.20, 4),
        ],
    )

    assert result.effective_vector_threshold == pytest.approx(0.51)
    assert [hit.doc_id for hit in result.vector_candidates] == ["v1", "v2", "v3"]


def test_selector_never_lowers_below_minimum_threshold() -> None:
    result = _selector().select(
        [],
        [_vector("v1", 0.50, 1), _vector("v2", 0.40, 2), _vector("v3", 0.20, 3)],
    )

    assert result.effective_vector_threshold == pytest.approx(0.40)
    assert [hit.doc_id for hit in result.vector_candidates] == ["v1", "v2"]


def test_selector_merges_keyword_and_vector_candidates() -> None:
    result = _selector(preferred=0.5, target=1).select(
        [_bm25("shared", 1), _bm25("keyword-only", 2)],
        [_vector("shared", 0.8, 1), _vector("vector-only", 0.7, 2)],
    )
    by_id = {hit.doc_id: hit for hit in result.candidates}

    assert set(by_id) == {"shared", "keyword-only", "vector-only"}
    assert by_id["shared"].bm25_rank == 1
    assert by_id["shared"].vector_rank == 1


def test_selector_rejects_mismatched_source_hits() -> None:
    with pytest.raises(ValueError, match="mismatched documents"):
        _selector().select(
            [],
            [SearchHit(doc_id="wrong", system_id="album", vector_score=0.8, vector_rank=1)],
        )
