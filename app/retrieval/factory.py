from functools import lru_cache
from typing import Protocol

from app.retrieval.local_es_retriever import LocalElasticsearchRetriever
from app.retrieval.local_faiss_retriever import LocalFaissRetriever
from app.schemas.search import SearchHit
from app.source_registry import SourceConfig


class Retriever(Protocol):
    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        ...


@lru_cache
def build_keyword_retriever(source: SourceConfig) -> Retriever:
    return LocalElasticsearchRetriever(
        source_id=source.source_id,
        index_name=source.es_index,
    )


@lru_cache
def build_vector_retriever(source: SourceConfig) -> Retriever:
    return LocalFaissRetriever(
        source_id=source.source_id,
        faiss_index_path=source.faiss_index_path,
        faiss_doc_ids_path=source.faiss_doc_ids_path,
    )
