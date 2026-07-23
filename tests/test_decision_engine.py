from types import SimpleNamespace

import pytest

from app.decision.decision_engine import DecisionEngine
from app.decision.system_aggregator import SystemAggregator
from app.reranking.gte_reranker import attach_reranker_scores
from app.schemas.search import SearchHit


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.requests: list[tuple[str, int]] = []

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        self.requests.append((query, top_k))
        return self.hits


class FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(
        self,
        query: str,
        candidates: list[SearchHit],
    ) -> list[SearchHit]:
        self.calls.append((query, [candidate.doc_id for candidate in candidates]))
        return attach_reranker_scores(
            candidates,
            [self.scores[candidate.doc_id] for candidate in candidates],
        )


@pytest.mark.asyncio
async def test_decision_engine_reranks_candidates_and_selects_systems(monkeypatch) -> None:
    keyword_retriever = FakeRetriever([])
    vector_retriever = FakeRetriever(
        [
            SearchHit(
                doc_id="memo_1",
                system_id="notepad",
                summary="记录了用户对海鲜过敏",
                vector_score=0.82,
                vector_rank=1,
            )
        ]
    )
    reranker = FakeReranker({"memo_1": 0.91})
    engine = DecisionEngine(
        keyword_retriever=keyword_retriever,
        vector_retriever=vector_retriever,
        reranker=reranker,
        aggregator=SystemAggregator(score_threshold=0.7),
    )
    monkeypatch.setattr(
        "app.decision.decision_engine.get_settings",
        lambda: SimpleNamespace(ES_TOP_K_DOCS=7, FAISS_TOP_K_DOCS=13),
    )

    response = await engine.decide("我可以吃海鲜吗？")

    assert response.selected_systems == ["notepad"]
    assert response.rewritten_queries == ["我可以吃海鲜吗？"]
    assert response.decisions[0].system_id == "notepad"
    assert response.decisions[0].selected is True
    assert response.decisions[0].reranker_score == 0.91
    assert reranker.calls == [("我可以吃海鲜吗？", ["memo_1"])]
    assert keyword_retriever.requests == [("我可以吃海鲜吗？", 7)]
    assert vector_retriever.requests == [("我可以吃海鲜吗？", 13)]


@pytest.mark.asyncio
async def test_decision_engine_searches_all_rewritten_queries(monkeypatch) -> None:
    keyword_retriever = FakeRetriever([])
    vector_retriever = FakeRetriever([])
    reranker = FakeReranker({})
    engine = DecisionEngine(
        keyword_retriever=keyword_retriever,
        vector_retriever=vector_retriever,
        reranker=reranker,
    )

    async def fake_rewrite_queries(task: str) -> list[str]:
        assert task == "原始问题"
        return ["  改写一  ", "改写二", "改写二", ""]

    monkeypatch.setattr(
        "app.decision.decision_engine.rewrite_queries",
        fake_rewrite_queries,
    )
    monkeypatch.setattr(
        "app.decision.decision_engine.get_settings",
        lambda: SimpleNamespace(ES_TOP_K_DOCS=7, FAISS_TOP_K_DOCS=13),
    )

    response = await engine.decide("原始问题")

    assert response.task == "原始问题"
    assert response.rewritten_queries == ["改写一", "改写二"]
    assert keyword_retriever.requests == [("改写一", 7), ("改写二", 7)]
    assert vector_retriever.requests == [("改写一", 13), ("改写二", 13)]
    assert reranker.calls == [("原始问题", [])]
