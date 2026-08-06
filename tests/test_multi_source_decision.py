import pytest

from app.decision.decision_engine import DecisionEngine
from app.reranking.gte_reranker import attach_reranker_scores
from app.schemas.search import SearchHit
from app.source_registry import SourceConfig, SourceRegistry


class FakeRetriever:
    def __init__(self, source_id: str, channel: str) -> None:
        self.source_id = source_id
        self.channel = channel

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        del query, top_k
        if self.channel == "keyword":
            return []
        return [
            SearchHit(
                doc_id=f"{self.source_id}-doc",
                system_id=self.source_id,
                vector_score=0.80,
                vector_rank=1,
            )
        ]


class FakeReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, candidates: list[SearchHit]) -> list[SearchHit]:
        del query
        self.calls += 1
        return attach_reranker_scores(candidates, [0.60 for _ in candidates])


def _source(source_id: str, *, reranker_threshold: float) -> SourceConfig:
    return SourceConfig(
        source_id=source_id,
        es_index=f"pcs-{source_id}",
        es_top_k=5,
        faiss_top_k=5,
        evidence_docs_per_system=2,
        faiss_preferred_score_threshold=0.50,
        faiss_min_score_threshold=0.20,
        faiss_target_hits=1,
        reranker_score_threshold=reranker_threshold,
    )


@pytest.mark.asyncio
async def test_multi_source_reranker_batches_candidates_and_uses_source_thresholds() -> None:
    registry = SourceRegistry(
        [
            _source("photo", reranker_threshold=0.70),
            _source("notepad", reranker_threshold=0.50),
        ]
    )
    reranker = FakeReranker()

    response = await DecisionEngine(
        source_registry=registry,
        reranker=reranker,
        keyword_retriever_factory=lambda source: FakeRetriever(
            source.source_id, "keyword"
        ),
        vector_retriever_factory=lambda source: FakeRetriever(
            source.source_id, "vector"
        ),
    ).decide("查找记录")

    assert reranker.calls == 1
    assert response.selected_systems == ["notepad"]
    assert {item.system_id: item.selected for item in response.decisions} == {
        "notepad": True,
        "photo": False,
    }
