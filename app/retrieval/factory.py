from functools import lru_cache
from typing import Protocol

from app.retrieval.local_bm25_retriever import LocalBM25Retriever
from app.retrieval.local_faiss_retriever import LocalFaissRetriever
from app.schemas.search import SearchHit


class Retriever(Protocol):
    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        ...


@lru_cache
def get_keyword_retriever() -> Retriever:
    return LocalBM25Retriever()


@lru_cache
def get_vector_retriever() -> Retriever:
    return LocalFaissRetriever()
