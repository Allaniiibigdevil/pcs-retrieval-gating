import pytest

from app.retrieval.candidate_selector import AdaptiveCandidateSelector
from app.schemas.search import SearchHit


def vector_hit(doc_id: str, score: float, rank: int) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id="vector-system",
        vector_score=score,
        vector_rank=rank,
    )


def bm25_hit(doc_id: str, rank: int) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id="bm25-system",
        bm25_score=10.0 - rank,
        bm25_rank=rank,
    )


def test_selector_lowers_vector_threshold_to_target_hit_count() -> None:
    selector = AdaptiveCandidateSelector(
        preferred_vector_threshold=0.70,
        min_vector_threshold=0.30,
        target_vector_hits=3,
        max_candidates=10,
    )

    result = selector.select(
        [],
        [
            vector_hit("v1", 0.80, 1),
            vector_hit("v2", 0.59, 2),
            vector_hit("v3", 0.51, 3),
            vector_hit("v4", 0.20, 4),
        ],
    )

    assert result.effective_vector_threshold == pytest.approx(0.51)
    assert [hit.doc_id for hit in result.vector_candidates] == ["v1", "v2", "v3"]


def test_selector_never_lowers_below_minimum_threshold() -> None:
    selector = AdaptiveCandidateSelector(
        preferred_vector_threshold=0.70,
        min_vector_threshold=0.30,
        target_vector_hits=3,
        max_candidates=10,
    )

    result = selector.select(
        [],
        [
            vector_hit("v1", 0.50, 1),
            vector_hit("v2", 0.40, 2),
            vector_hit("v3", 0.20, 3),
        ],
    )

    assert result.effective_vector_threshold == pytest.approx(0.40)
    assert [hit.doc_id for hit in result.vector_candidates] == ["v1", "v2"]


def test_selector_keeps_both_routes_when_candidate_budget_is_full() -> None:
    selector = AdaptiveCandidateSelector(
        preferred_vector_threshold=0.50,
        min_vector_threshold=0.30,
        target_vector_hits=2,
        max_candidates=4,
    )

    result = selector.select(
        [bm25_hit(f"e{rank}", rank) for rank in range(1, 4)],
        [vector_hit(f"v{rank}", 0.9 - rank / 10, rank) for rank in range(1, 4)],
    )

    assert [hit.doc_id for hit in result.candidates] == ["e1", "v1", "e2", "v2"]


def test_selector_merges_same_document_from_both_routes() -> None:
    selector = AdaptiveCandidateSelector(
        preferred_vector_threshold=0.50,
        min_vector_threshold=0.30,
        target_vector_hits=1,
        max_candidates=10,
    )
    es_hit = bm25_hit("shared", 1)
    es_hit.system_id = "memo"
    faiss_hit = vector_hit("shared", 0.80, 2)
    faiss_hit.system_id = "memo"

    result = selector.select([es_hit], [faiss_hit])

    assert len(result.candidates) == 1
    assert result.candidates[0].bm25_rank == 1
    assert result.candidates[0].vector_rank == 2
