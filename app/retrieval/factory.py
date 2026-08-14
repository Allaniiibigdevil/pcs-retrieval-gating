from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.config import get_settings
from app.retrieval.local_es_retriever import LocalElasticsearchRetriever
from app.retrieval.local_es_unified_retriever import LocalElasticsearchUnifiedRetriever
from app.retrieval.local_es_vector_retriever import LocalElasticsearchVectorRetriever
from app.retrieval.local_faiss_retriever import LocalFaissRetriever
from app.schemas.search import SearchHit
from app.source_registry import SourceConfig
from app.storage.local_artifact_store import LocalArtifactStore


class Retriever(Protocol):
    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        ...


@lru_cache
def build_keyword_retriever(source: SourceConfig) -> Retriever:
    settings = get_settings()
    if settings.KEYWORD_RETRIEVER_MODE == "unified":
        return LocalElasticsearchUnifiedRetriever(
            source_id=source.source_id,
            index_name=settings.LOCAL_ES_KEYWORD_INDEX,
        )
    return LocalElasticsearchRetriever(
        source_id=source.source_id,
        index_name=source.es_index,
    )


@lru_cache
def build_vector_retriever(source: SourceConfig) -> Retriever:
    settings = get_settings()
    if settings.VECTOR_RETRIEVER_BACKEND == "es":
        return LocalElasticsearchVectorRetriever(
            source_id=source.source_id,
            index_name=settings.LOCAL_ES_VECTOR_INDEX,
        )
    return LocalFaissRetriever(
        source_id=source.source_id,
        faiss_index_path=source.faiss_index_path,
        faiss_doc_ids_path=source.faiss_doc_ids_path,
        artifact_store=LocalArtifactStore(
            artifact_dir=Path(source.faiss_index_path).parent
        ),
    )
