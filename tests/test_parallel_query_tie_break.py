import pytest

from app.retrieval.parallel_query_retriever import search_queries_in_parallel
from app.schemas.search import SearchHit


class MappingRetriever:
    def __init__(self, results: dict[str, list[SearchHit]]) -> None:
        self.results = results

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        return self.results[query][:top_k]


@pytest.mark.asyncio
async def test_equal_keyword_evidence_prefers_earlier_query_not_larger_raw_score() -> None:
    keyword_retriever = MappingRetriever(
        {
            "original": [
                SearchHit(
                    doc_id="shared",
                    system_id="system",
                    bm25_score=2.0,
                    bm25_rank=1,
                )
            ],
            "rewrite": [
                SearchHit(
                    doc_id="shared",
                    system_id="system",
                    bm25_score=20.0,
                    bm25_rank=1,
                )
            ],
        }
    )
    vector_retriever = MappingRetriever({"original": [], "rewrite": []})

    result = await search_queries_in_parallel(
        keyword_retriever=keyword_retriever,
        vector_retriever=vector_retriever,
        keyword_queries=["original", "rewrite"],
        vector_queries=["original", "rewrite"],
        keyword_top_k=10,
        vector_top_k=10,
    )

    assert result.keyword.hits[0].bm25_score_norm == 1.0
    assert result.keyword.hits[0].bm25_score == 2.0
    assert result.keyword.hits[0].metadata["matched_queries"] == ["original", "rewrite"]
