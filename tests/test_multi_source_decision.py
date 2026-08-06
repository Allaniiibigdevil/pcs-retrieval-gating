import pytest

from app.decision.decision_engine import DecisionEngine
from app.schemas.search import SearchHit
from app.source_registry import SourceConfig, SourceRegistry


class FakeRetriever:
    def __init__(self, source_id: str, channel: str, requests: list[tuple[str, str, int]]) -> None:
        self.source_id = source_id
        self.channel = channel
        self.requests = requests

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        self.requests.append((self.source_id, self.channel, top_k))
        if self.channel == "keyword":
            return []
        score = 0.58 if self.source_id == "photo" else 0.65
        return [
            SearchHit(
                doc_id=f"{self.source_id}-doc",
                system_id=self.source_id,
                vector_score=score,
                vector_rank=1,
            )
        ]


def _source(source_id: str, *, threshold: float, top_k: int) -> SourceConfig:
    return SourceConfig(
        source_id=source_id,
        es_index=f"pcs-{source_id}",
        enabled=True,
        es_top_k=top_k,
        faiss_top_k=top_k + 1,
        evidence_docs_per_system=2,
        selection_threshold=threshold,
        es_score_weight=0.55,
        agreement_weight=0.20,
        semantic_match_threshold=0.30,
        lexical_match_threshold=0.30,
        faiss_preferred_score_threshold=0.60,
        faiss_min_score_threshold=0.30,
        faiss_target_hits=1,
        reranker_score_threshold=0.50,
    )


@pytest.mark.asyncio
async def test_multi_source_score_fusion_uses_source_top_k_and_thresholds(monkeypatch) -> None:
    registry = SourceRegistry(
        [
            _source("photo", threshold=0.70, top_k=3),
            _source("notepad", threshold=0.60, top_k=7),
        ]
    )
    requests: list[tuple[str, str, int]] = []
    monkeypatch.setattr(
        "app.decision.decision_engine.build_keyword_retriever",
        lambda source: FakeRetriever(source.source_id, "keyword", requests),
    )
    monkeypatch.setattr(
        "app.decision.decision_engine.build_vector_retriever",
        lambda source: FakeRetriever(source.source_id, "vector", requests),
    )

    response = await DecisionEngine(source_registry=registry).decide("查找记录")

    assert response.selected_systems == ["notepad"]
    assert {item.system_id: item.selected for item in response.decisions} == {
        "notepad": True,
        "photo": False,
    }
    assert ("photo", "keyword", 3) in requests
    assert ("photo", "vector", 4) in requests
    assert ("notepad", "keyword", 7) in requests
    assert ("notepad", "vector", 8) in requests
