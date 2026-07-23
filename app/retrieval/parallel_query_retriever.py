import asyncio
from dataclasses import dataclass
from typing import Literal

from app.retrieval.factory import Retriever
from app.schemas.search import SearchHit


Channel = Literal["keyword", "vector"]


@dataclass(frozen=True)
class SearchFailure:
    query_index: int
    error: Exception


@dataclass(frozen=True)
class ChannelSearchResult:
    hits: list[SearchHit]
    failures: list[SearchFailure]
    query_count: int

    @property
    def all_failed(self) -> bool:
        return self.query_count > 0 and len(self.failures) == self.query_count


@dataclass(frozen=True)
class ParallelQuerySearchResult:
    keyword: ChannelSearchResult
    vector: ChannelSearchResult


@dataclass(frozen=True)
class _SearchAttempt:
    hits: list[SearchHit] | None = None
    error: Exception | None = None


async def search_queries_in_parallel(
    *,
    keyword_retriever: Retriever,
    vector_retriever: Retriever,
    keyword_queries: list[str],
    vector_queries: list[str],
    keyword_top_k: int,
    vector_top_k: int,
) -> ParallelQuerySearchResult:
    """Run every keyword and vector query concurrently and merge each channel."""

    keyword_result, vector_result = await asyncio.gather(
        _search_channel(
            keyword_retriever,
            keyword_queries,
            keyword_top_k,
            channel="keyword",
        ),
        _search_channel(
            vector_retriever,
            vector_queries,
            vector_top_k,
            channel="vector",
        ),
    )
    return ParallelQuerySearchResult(keyword=keyword_result, vector=vector_result)


async def _search_channel(
    retriever: Retriever,
    queries: list[str],
    top_k: int,
    *,
    channel: Channel,
) -> ChannelSearchResult:
    attempts = await asyncio.gather(
        *(_search_one(retriever, query, top_k) for query in queries)
    )
    successful_hits: list[list[SearchHit]] = []
    failures: list[SearchFailure] = []
    for query_index, attempt in enumerate(attempts):
        if attempt.error is not None:
            failures.append(SearchFailure(query_index=query_index, error=attempt.error))
        elif attempt.hits is not None:
            successful_hits.append(attempt.hits)

    return ChannelSearchResult(
        hits=_merge_ranked_hits(successful_hits, channel),
        failures=failures,
        query_count=len(queries),
    )


async def _search_one(
    retriever: Retriever,
    query: str,
    top_k: int,
) -> _SearchAttempt:
    try:
        return _SearchAttempt(hits=await retriever.search(query, top_k))
    except Exception as exc:
        return _SearchAttempt(error=exc)


def _merge_ranked_hits(
    query_results: list[list[SearchHit]],
    channel: Channel,
) -> list[SearchHit]:
    best_by_doc_id: dict[str, SearchHit] = {}
    for hits in query_results:
        for hit in hits:
            existing = best_by_doc_id.get(hit.doc_id)
            if existing is None or _hit_sort_key(hit, channel) < _hit_sort_key(
                existing,
                channel,
            ):
                best_by_doc_id[hit.doc_id] = hit

    merged = sorted(best_by_doc_id.values(), key=lambda hit: _hit_sort_key(hit, channel))
    rank_field = "bm25_rank" if channel == "keyword" else "vector_rank"
    ranked: list[SearchHit] = []
    for rank, hit in enumerate(merged, start=1):
        copied = hit.model_copy(deep=True)
        setattr(copied, rank_field, rank)
        ranked.append(copied)
    return ranked


def _hit_sort_key(hit: SearchHit, channel: Channel) -> tuple[float, float, str]:
    if channel == "keyword":
        rank = float(hit.bm25_rank) if hit.bm25_rank is not None else float("inf")
        score = hit.bm25_score if hit.bm25_score is not None else float("-inf")
        return rank, -score, hit.doc_id

    score = hit.vector_score if hit.vector_score is not None else float("-inf")
    rank = float(hit.vector_rank) if hit.vector_rank is not None else float("inf")
    return -score, rank, hit.doc_id
