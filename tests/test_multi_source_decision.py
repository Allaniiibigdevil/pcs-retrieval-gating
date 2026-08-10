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


def _source(
    source_id: str,
    *,
    threshold: float,
    top_k: int,
    preferred_vector_threshold: float = 0.50,
    min_vector_threshold: float = 0.30,
    target_vector_hits: int = 1,
) -> SourceConfig:
    return SourceConfig(
        source_id=source_id,
        es_index=f"pcs-{source_id}",
        faiss_index_path=f"data/artifacts/{source_id}/faiss.index",
        faiss_doc_ids_path=f"data/artifacts/{source_id}/faiss_doc_ids.json",
        es_top_k=top_k,
        faiss_top_k=top_k + 1,
        evidence_docs_per_system=2,
        faiss_preferred_score_threshold=preferred_vector_threshold,
        faiss_min_score_threshold=min_vector_threshold,
        faiss_target_hits=target_vector_hits,
        selection_threshold=threshold,
        es_score_weight=0.55,
        agreement_weight=0.20,
        semantic_match_threshold=0.30,
        lexical_match_threshold=0.30,
    )


@pytest.mark.asyncio
async def test_multi_source_score_fusion_uses_source_top_k_and_thresholds() -> None:
    registry = SourceRegistry(
        [
            _source("photo", threshold=0.70, top_k=3),
            _source("notepad", threshold=0.60, top_k=7),
        ]
    )
    requests: list[tuple[str, str, int]] = []

    response = await DecisionEngine(
        source_registry=registry,
        keyword_retriever_factory=lambda source: FakeRetriever(
            source.source_id, "keyword", requests
        ),
        vector_retriever_factory=lambda source: FakeRetriever(
            source.source_id, "vector", requests
        ),
    ).decide("查找记录")

    assert response.selected_systems == ["notepad"]
    assert {item.system_id: item.selected for item in response.decisions} == {
        "notepad": True,
        "photo": False,
    }
    assert ("photo", "keyword", 3) in requests
    assert ("photo", "vector", 4) in requests
    assert ("notepad", "keyword", 7) in requests
    assert ("notepad", "vector", 8) in requests


@pytest.mark.asyncio
async def test_multi_source_score_fusion_applies_vector_thresholds_per_source() -> None:
    registry = SourceRegistry(
        [
            _source(
                "photo",
                threshold=0.50,
                top_k=3,
                preferred_vector_threshold=0.70,
                min_vector_threshold=0.60,
            ),
            _source(
                "notepad",
                threshold=0.60,
                top_k=7,
                preferred_vector_threshold=0.60,
                min_vector_threshold=0.30,
            ),
        ]
    )
    requests: list[tuple[str, str, int]] = []

    response = await DecisionEngine(
        source_registry=registry,
        keyword_retriever_factory=lambda source: FakeRetriever(
            source.source_id, "keyword", requests
        ),
        vector_retriever_factory=lambda source: FakeRetriever(
            source.source_id, "vector", requests
        ),
    ).decide("查找记录")

    assert response.selected_systems == ["notepad"]
    assert [item.system_id for item in response.decisions] == ["notepad"]
