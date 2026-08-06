import asyncio
from dataclasses import dataclass
from typing import Literal

from app.retrieval.factory import Retriever
from app.retrieval.metadata import merge_metadata
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
    keyword_result, vector_result = await asyncio.gather(
        _search_channel(keyword_retriever, keyword_queries, keyword_top_k, channel="keyword"),
        _search_channel(vector_retriever, vector_queries, vector_top_k, channel="vector"),
    )
    return ParallelQuerySearchResult(keyword=keyword_result, vector=vector_result)


async def _search_channel(
    retriever: Retriever,
    queries: list[str],
    top_k: int,
    *,
    channel: Channel,
) -> ChannelSearchResult:
    attempts = await asyncio.gather(*(_search_one(retriever, query, top_k) for query in queries))
    successful_hits: list[list[SearchHit]] = []
    failures: list[SearchFailure] = []
    for query_index, (query, attempt) in enumerate(zip(queries, attempts)):
        if attempt.error is not None:
            failures.append(SearchFailure(query_index=query_index, error=attempt.error))
        elif attempt.hits is not None:
            successful_hits.append(
                _prepare_query_hits(attempt.hits, query=query, query_index=query_index)
            )

    return ChannelSearchResult(
        hits=_merge_ranked_hits(successful_hits, channel),
        failures=failures,
        query_count=len(queries),
    )


async def _search_one(retriever: Retriever, query: str, top_k: int) -> _SearchAttempt:
    try:
        return _SearchAttempt(hits=await retriever.search(query, top_k))
    except Exception as exc:
        return _SearchAttempt(error=exc)


def _prepare_query_hits(
    hits: list[SearchHit],
    *,
    query: str,
    query_index: int,
) -> list[SearchHit]:
    prepared: list[SearchHit] = []
    for hit in hits:
        copied = hit.model_copy(deep=True)
        copied.metadata = merge_metadata(
            copied.metadata,
            {
                "matched_queries": [query],
                "matched_query_indexes": [query_index],
            },
        )
        prepared.append(copied)
    return prepared


def _merge_ranked_hits(
    query_results: list[list[SearchHit]],
    channel: Channel,
) -> list[SearchHit]:
    best_by_doc_id: dict[str, tuple[SearchHit, int]] = {}
    for hits in query_results:
        for hit in hits:
            query_index = _single_query_index(hit)
            existing = best_by_doc_id.get(hit.doc_id)
            if existing is None:
                best_by_doc_id[hit.doc_id] = (hit.model_copy(deep=True), query_index)
                continue

            existing_hit, existing_query_index = existing
            if _hit_sort_key(hit, channel, query_index) < _hit_sort_key(
                existing_hit,
                channel,
                existing_query_index,
            ):
                replacement = hit.model_copy(deep=True)
                replacement.metadata = merge_metadata(
                    replacement.metadata,
                    existing_hit.metadata,
                )
                best_by_doc_id[hit.doc_id] = (replacement, query_index)
            else:
                existing_hit.metadata = merge_metadata(
                    existing_hit.metadata,
                    hit.metadata,
                )

    merged = sorted(
        best_by_doc_id.values(),
        key=lambda item: _hit_sort_key(item[0], channel, item[1]),
    )
    rank_field = "bm25_rank" if channel == "keyword" else "vector_rank"
    ranked: list[SearchHit] = []
    for rank, (hit, _) in enumerate(merged, start=1):
        copied = hit.model_copy(deep=True)
        setattr(copied, rank_field, rank)
        ranked.append(copied)
    return ranked


def _single_query_index(hit: SearchHit) -> int:
    indexes = hit.metadata.get("matched_query_indexes")
    if not isinstance(indexes, list) or len(indexes) != 1 or not isinstance(indexes[0], int):
        raise ValueError(
            f"Prepared hit {hit.doc_id!r} must contain exactly one query index"
        )
    return indexes[0]


def _hit_sort_key(
    hit: SearchHit,
    channel: Channel,
    query_index: int,
) -> tuple[float, float, float, str]:
    if channel == "keyword":
        rank = float(hit.bm25_rank) if hit.bm25_rank is not None else float("inf")
        return rank, float(query_index), 0.0, hit.doc_id

    score = hit.vector_score if hit.vector_score is not None else float("-inf")
    rank = float(hit.vector_rank) if hit.vector_rank is not None else float("inf")
    return -score, rank, float(query_index), hit.doc_id
