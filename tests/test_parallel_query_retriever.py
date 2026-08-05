import asyncio

import pytest

from app.retrieval.parallel_query_retriever import search_queries_in_parallel
from app.schemas.search import SearchHit


class ConcurrencyTracker:
    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.started = 0
        self.peak = 0
        self.active = 0
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.started += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.started == self.expected:
            self.all_started.set()
        await self.release.wait()
        self.active -= 1


class GatedRetriever:
    def __init__(self, tracker: ConcurrencyTracker, channel: str) -> None:
        self.tracker = tracker
        self.channel = channel

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        await self.tracker.wait()
        fields = (
            {"bm25_score": 1.0, "bm25_rank": 1}
            if self.channel == "keyword"
            else {"vector_score": 0.8, "vector_rank": 1}
        )
        return [
            SearchHit(
                doc_id=f"{self.channel}-{query}",
                system_id="system",
                **fields,
            )
        ][:top_k]


class MappingRetriever:
    def __init__(self, results: dict[str, list[SearchHit]]) -> None:
        self.results = results

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        return self.results[query][:top_k]


@pytest.mark.asyncio
async def test_all_keyword_and_vector_queries_run_concurrently() -> None:
    tracker = ConcurrencyTracker(expected=4)
    task = asyncio.create_task(
        search_queries_in_parallel(
            keyword_retriever=GatedRetriever(tracker, "keyword"),
            vector_retriever=GatedRetriever(tracker, "vector"),
            keyword_queries=["q1", "q2"],
            vector_queries=["q1", "q2"],
            keyword_top_k=10,
            vector_top_k=10,
        )
    )

    await asyncio.wait_for(tracker.all_started.wait(), timeout=1)
    tracker.release.set()
    result = await task

    assert tracker.peak == 4
    assert len(result.keyword.hits) == 2
    assert len(result.vector.hits) == 2


@pytest.mark.asyncio
async def test_keyword_scores_are_normalized_per_query_before_merge() -> None:
    keyword_retriever = MappingRetriever(
        {
            "q1": [
                SearchHit(doc_id="shared", system_id="system", bm25_score=9.0, bm25_rank=2),
                SearchHit(doc_id="q1-top", system_id="system", bm25_score=18.0, bm25_rank=1),
            ],
            "q2": [
                SearchHit(doc_id="shared", system_id="system", bm25_score=4.0, bm25_rank=1),
                SearchHit(doc_id="q2-second", system_id="system", bm25_score=2.0, bm25_rank=2),
            ],
        }
    )
    vector_retriever = MappingRetriever({"q1": [], "q2": []})

    result = await search_queries_in_parallel(
        keyword_retriever=keyword_retriever,
        vector_retriever=vector_retriever,
        keyword_queries=["q1", "q2"],
        vector_queries=["q1", "q2"],
        keyword_top_k=10,
        vector_top_k=10,
    )

    shared = next(hit for hit in result.keyword.hits if hit.doc_id == "shared")
    assert shared.bm25_score == 4.0
    assert shared.bm25_score_norm == pytest.approx(1.0)
    assert shared.metadata["matched_queries"] == ["q1", "q2"]


@pytest.mark.asyncio
async def test_vector_duplicates_keep_highest_score_and_merge_query_evidence() -> None:
    keyword_retriever = MappingRetriever({"q1": [], "q2": []})
    vector_retriever = MappingRetriever(
        {
            "q1": [
                SearchHit(
                    doc_id="shared",
                    system_id="system",
                    vector_score=0.80,
                    vector_rank=1,
                )
            ],
            "q2": [
                SearchHit(
                    doc_id="shared",
                    system_id="system",
                    vector_score=0.90,
                    vector_rank=2,
                )
            ],
        }
    )

    result = await search_queries_in_parallel(
        keyword_retriever=keyword_retriever,
        vector_retriever=vector_retriever,
        keyword_queries=["q1", "q2"],
        vector_queries=["q1", "q2"],
        keyword_top_k=10,
        vector_top_k=10,
    )

    assert len(result.vector.hits) == 1
    assert result.vector.hits[0].vector_score == 0.90
    assert result.vector.hits[0].vector_rank == 1
    assert result.vector.hits[0].metadata["matched_queries"] == ["q1", "q2"]
