import pytest

from app.decision.decision_engine import DecisionEngine
from app.decision.system_aggregator import SystemAggregator
from app.schemas.search import SearchHit


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        del query, top_k
        return self.hits


@pytest.mark.asyncio
async def test_decision_engine_returns_selected_systems() -> None:
    engine = DecisionEngine(
        keyword_retriever=FakeRetriever([]),
        vector_retriever=FakeRetriever(
            [
                SearchHit(
                    doc_id="memo_1",
                    system_id="notepad",
                    summary="记录了用户对海鲜过敏",
                    vector_score=0.82,
                    vector_rank=1,
                )
            ]
        ),
        aggregator=SystemAggregator(selection_threshold=0.45),
    )

    response = await engine.decide("我可以吃海鲜吗？")

    assert response.selected_systems == ["notepad"]
    assert response.decisions[0].system_id == "notepad"
    assert response.decisions[0].selected is True
