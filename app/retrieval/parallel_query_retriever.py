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
    for query_index, (query, attempt) in enumerate(zip(queries, attempts)):
        if attempt.error is not None:
            failures.append(SearchFailure(query_index=query_index, error=attempt.error))
        elif attempt.hits is not None:
            successful_hits.append(
                _prepare_query_hits(
                    attempt.hits,
                    query=query,
                    query_index=query_index,
                    channel=channel,
                )
            )

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


def _prepare_query_hits(
    hits: list[SearchHit],
    *,
    query: str,
    query_index: int,
    channel: Channel,
) -> list[SearchHit]:
    max_keyword_score = 0.0
    if channel == "keyword":
        max_keyword_score = max((hit.bm25_score or 0.0 for hit in hits), default=0.0)

    prepared: list[SearchHit] = []
    for hit in hits:
        copied = hit.model_copy(deep=True)
        if channel == "keyword":
            score = copied.bm25_score or 0.0
            copied.bm25_score_norm = (
                min(max(score / max_keyword_score, 0.0), 1.0)
                if score > 0.0 and max_keyword_score > 0.0
                else 0.0
            )
        copied.metadata = _merge_metadata(
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
    best_by_doc_id: dict[str, SearchHit] = {}
    for hits in query_results:
        for hit in hits:
            existing = best_by_doc_id.get(hit.doc_id)
            if existing is None:
                best_by_doc_id[hit.doc_id] = hit.model_copy(deep=True)
                continue

            if _hit_sort_key(hit, channel) < _hit_sort_key(existing, channel):
                replacement = hit.model_copy(deep=True)
                replacement.metadata = _merge_metadata(existing.metadata, replacement.metadata)
                best_by_doc_id[hit.doc_id] = replacement
            else:
                existing.metadata = _merge_metadata(existing.metadata, hit.metadata)

    merged = sorted(best_by_doc_id.values(), key=lambda hit: _hit_sort_key(hit, channel))
    rank_field = "bm25_rank" if channel == "keyword" else "vector_rank"
    ranked: list[SearchHit] = []
    for rank, hit in enumerate(merged, start=1):
        copied = hit.model_copy(deep=True)
        setattr(copied, rank_field, rank)
        ranked.append(copied)
    return ranked


def _hit_sort_key(hit: SearchHit, channel: Channel) -> tuple[float, float, float, str]:
    if channel == "keyword":
        normalized_score = (
            hit.bm25_score_norm if hit.bm25_score_norm is not None else float("-inf")
        )
        rank = float(hit.bm25_rank) if hit.bm25_rank is not None else float("inf")
        query_indexes = hit.metadata.get("matched_query_indexes")
        query_index = (
            float(query_indexes[0])
            if isinstance(query_indexes, list) and query_indexes
            else float("inf")
        )
        return -normalized_score, rank, query_index, hit.doc_id

    score = hit.vector_score if hit.vector_score is not None else float("-inf")
    rank = float(hit.vector_rank) if hit.vector_rank is not None else float("inf")
    return -score, rank, 0.0, hit.doc_id


def _merge_metadata(left: dict, right: dict) -> dict:
    merged = {**left, **right}

    for field in ("matched_queries", "matched_query_indexes", "matched_keywords"):
        values: list = []
        seen: set = set()
        for source in (left, right):
            raw_values = source.get(field)
            if not isinstance(raw_values, list):
                continue
            for value in raw_values:
                if value in seen:
                    continue
                seen.add(value)
                values.append(value)
        if values:
            merged[field] = values

    left_highlight = left.get("highlight")
    right_highlight = right.get("highlight")
    if isinstance(left_highlight, dict) or isinstance(right_highlight, dict):
        highlight: dict[str, list[str]] = {}
        for field in ("summary", "keywords"):
            values: list[str] = []
            seen_values: set[str] = set()
            for source in (left_highlight, right_highlight):
                if not isinstance(source, dict):
                    continue
                raw_values = source.get(field)
                if not isinstance(raw_values, list):
                    continue
                for value in raw_values:
                    text = str(value)
                    if text in seen_values:
                        continue
                    seen_values.add(text)
                    values.append(text)
            if values:
                highlight[field] = values
        if highlight:
            merged["highlight"] = highlight

    return merged
