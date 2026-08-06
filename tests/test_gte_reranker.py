from types import SimpleNamespace

import pytest

from app.reranking.gte_reranker import (
    GTEReranker,
    _resolve_device,
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


def test_attach_reranker_scores_rejects_invalid_output() -> None:
    with pytest.raises(ValueError, match="different number"):
        attach_reranker_scores(
            [SearchHit(doc_id="doc", system_id="memo")],
            [],
        )

    with pytest.raises(ValueError, match="invalid probability"):
        attach_reranker_scores(
            [SearchHit(doc_id="doc", system_id="memo")],
            [1.1],
        )


@pytest.mark.parametrize(
    ("batch_size", "max_length"),
    [(0, 512), (257, 512), (8, 7), (8, 8193)],
)
def test_reranker_rejects_invalid_runtime_limits(batch_size: int, max_length: int) -> None:
    with pytest.raises(ValueError):
        GTEReranker(
            "unused",
            local_files_only=False,
            batch_size=batch_size,
            max_length=max_length,
        )


def test_device_parser_rejects_malformed_cuda_device() -> None:
    torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    with pytest.raises(ValueError, match="cuda:<index>"):
        _resolve_device("cuda:abc", torch_module)
