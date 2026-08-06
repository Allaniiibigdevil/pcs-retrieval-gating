import pytest

from app.retrieval.candidate_merger import CandidateMerger
from app.schemas.search import SearchHit


def test_candidate_merger_preserves_scores_and_metadata() -> None:
    merger = CandidateMerger()
    bm25_hits = [
        SearchHit(
            doc_id="shared",
            system_id="memo",
            bm25_score=12.5,
            bm25_score_norm=1.0,
            bm25_rank=1,
            metadata={"matched_queries": ["上海会议"]},
        ),
        SearchHit(doc_id="bm25-only", system_id="memo", bm25_score=5.0, bm25_rank=2),
    ]
    vector_hits = [
        SearchHit(
            doc_id="shared",
            system_id="memo",
            vector_score=0.86,
            vector_rank=1,
            metadata={"matched_queries": ["上海开会"]},
        ),
        SearchHit(doc_id="vector-only", system_id="memo", vector_score=0.7, vector_rank=2),
    ]

    merged = merger.merge("memo", bm25_hits, vector_hits)
    by_id = {hit.doc_id: hit for hit in merged}

    assert set(by_id) == {"shared", "bm25-only", "vector-only"}
    assert by_id["shared"].bm25_score_norm == 1.0
    assert by_id["shared"].vector_score == 0.86
    assert by_id["shared"].metadata["matched_queries"] == ["上海会议", "上海开会"]


def test_candidate_merger_rejects_cross_source_hits() -> None:
    with pytest.raises(ValueError, match="mismatched documents"):
        CandidateMerger().merge(
            "memo",
            [],
            [SearchHit(doc_id="wrong", system_id="album", vector_score=0.8)],
        )
