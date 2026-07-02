from app.retrieval.candidate_merger import CandidateMerger
from app.schemas.search import SearchHit


def test_candidate_merger_preserves_and_merges_hits() -> None:
    merger = CandidateMerger()
    bm25_hits = [
        SearchHit(
            doc_id="shared",
            system_id="memo_system",
            summary="shared summary",
            keywords=["上海", "会议"],
            bm25_score=12.5,
            bm25_rank=1,
        ),
        SearchHit(doc_id="bm25_only", system_id="memo_system", bm25_score=5.0, bm25_rank=2),
    ]
    vector_hits = [
        SearchHit(
            doc_id="shared",
            system_id="memo_system",
            keywords=["上海", "会议"],
            vector_score=0.86,
            vector_rank=3,
        ),
        SearchHit(doc_id="vector_only", system_id="album_system", vector_score=0.7, vector_rank=1),
    ]

    merged = merger.merge(bm25_hits, vector_hits)
    by_id = {hit.doc_id: hit for hit in merged}

    assert set(by_id) == {"shared", "bm25_only", "vector_only"}
    assert len(merged) == len(by_id)
    assert by_id["bm25_only"].bm25_score == 5.0
    assert by_id["bm25_only"].vector_score is None
    assert by_id["vector_only"].vector_score == 0.7
    assert by_id["vector_only"].bm25_score is None
    assert by_id["shared"].bm25_score == 12.5
    assert by_id["shared"].bm25_rank == 1
    assert by_id["shared"].vector_score == 0.86
    assert by_id["shared"].vector_rank == 3
