import pytest

from app.api import rerank as rerank_api
from app.reranking.gte_reranker import GTEReranker
from app.schemas.rerank import RerankRequest


@pytest.mark.asyncio
async def test_rerank_api_returns_single_score(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeReranker:
        async def score(self, query: str, doc: str) -> float:
            calls.append((query, doc))
            return 0.73

    monkeypatch.setattr(rerank_api, "reranker", FakeReranker())

    response = await rerank_api.rerank(
        RerankRequest(query="测试 query", doc="测试 doc")
    )

    assert response.score == pytest.approx(0.73)
    assert calls == [("测试 query", "测试 doc")]


@pytest.mark.asyncio
async def test_gte_reranker_score_predicts_one_document(monkeypatch) -> None:
    reranker = GTEReranker(model_path="unused", local_files_only=False)
    calls: list[tuple[str, list[str]]] = []

    def fake_predict(query: str, passages: list[str]) -> list[float]:
        calls.append((query, passages))
        return [0.42]

    monkeypatch.setattr(reranker, "_predict", fake_predict)

    score = await reranker.score("query", "document")

    assert score == pytest.approx(0.42)
    assert calls == [("query", ["document"])]
