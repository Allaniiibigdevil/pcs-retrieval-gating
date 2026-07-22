import pytest

from app.reranking.gte_reranker import (
    attach_reranker_scores,
    build_reranker_passage,
)
from app.schemas.search import SearchHit


def test_reranker_passage_contains_only_document_content() -> None:
    hit = SearchHit(
        doc_id="doc-1",
        system_id="album",
        summary="上海出差期间拍摄的会议合照",
        keywords=["上海", "会议"],
        bm25_score=12.0,
        vector_score=0.81,
    )

    passage = build_reranker_passage(hit)

    assert passage == "关键词：上海；会议\n摘要：上海出差期间拍摄的会议合照"
    assert "album" not in passage
    assert "12.0" not in passage
    assert "0.81" not in passage


def test_attach_reranker_scores_sorts_and_assigns_global_rank() -> None:
    candidates = [
        SearchHit(doc_id="low", system_id="memo"),
        SearchHit(doc_id="high", system_id="album"),
    ]

    reranked = attach_reranker_scores(candidates, [0.2, 0.9])

    assert [hit.doc_id for hit in reranked] == ["high", "low"]
    assert [hit.reranker_rank for hit in reranked] == [1, 2]
    assert candidates[0].reranker_score is None


def test_attach_reranker_scores_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="different number"):
        attach_reranker_scores(
            [SearchHit(doc_id="doc", system_id="memo")],
            [],
        )
