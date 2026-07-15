import json

import pytest

from app.config import get_settings
from app.decision.decision_engine import DecisionEngine
from app.decision.system_aggregator import SystemAggregator
from app.schemas.search import SearchHit


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    async def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        del query, top_k
        return self.hits


@pytest.mark.asyncio
async def test_decision_engine_returns_selected_systems_and_logs_one_case(
    tmp_path, monkeypatch
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    monkeypatch.setattr(get_settings(), "GATING_CASES_PATH", str(cases_path))
    monkeypatch.setattr(get_settings(), "GATING_CASE_LOG_ENABLED", True)
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

    response = await engine.decide("我可以吃海鲜吗？", task_id="task-1")

    assert response.task_id == "task-1"
    assert response.selected_systems == ["notepad"]
    assert response.decisions[0].system_id == "notepad"
    assert response.decisions[0].selected is True
    lines = cases_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    case = json.loads(lines[0])
    assert case["task_id"] == "task-1"
    assert case["query"] == "我可以吃海鲜吗？"
    assert case["systems"][0]["system_id"] == "notepad"
    assert case["systems"][0]["label"] is None
    assert [doc["doc_id"] for doc in case["systems"][0]["docs"]] == ["memo_1"]
