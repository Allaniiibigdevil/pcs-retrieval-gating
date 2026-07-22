from functools import lru_cache
from typing import Protocol

from app.reranking.gte_reranker import GTEReranker
from app.schemas.search import SearchHit


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: list[SearchHit],
    ) -> list[SearchHit]:
        ...


@lru_cache
def get_reranker() -> Reranker:
    return GTEReranker()
