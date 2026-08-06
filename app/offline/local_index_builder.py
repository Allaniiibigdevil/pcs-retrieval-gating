from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging

import numpy as np

from app.config import get_settings
from app.embedding.embedding_service import (
    EmbeddingService,
    build_embedding_text,
    get_embedding_service,
)
from app.offline.local_es_indexer import LocalElasticsearchIndexer
from app.schemas.doc import SourceDoc
from app.source_registry import SourceRegistry, get_source_registry
from app.storage.local_artifact_store import LocalArtifactStore


@dataclass(frozen=True)
class BuildIndexResult:
    doc_count: int
    artifact_dir: str
    embedding_model: str
    embedding_dim: int
    source_count: int
    elasticsearch_indices: dict[str, str] = field(default_factory=dict)


class LocalIndexBuilder:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        embedding_service: EmbeddingService | None = None,
        index_elasticsearch: bool = False,
        source_registry: SourceRegistry | None = None,
    ) -> None:
        self.settings = get_settings()
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service or get_embedding_service()
        self.index_elasticsearch = index_elasticsearch
        self.source_registry = source_registry or get_source_registry()

    async def build(self, docs: list[SourceDoc]) -> BuildIndexResult:
        if not docs:
            raise ValueError("Cannot build local index from an empty document set")

        doc_ids = [doc.doc_id for doc in docs]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Cannot build shared FAISS index with duplicate doc_id values")
        self.source_registry.validate_source_ids({doc.system_id for doc in docs})

        embedding_texts = [build_embedding_text(doc) for doc in docs]
        embeddings = await self.embedding_service.embed_batch(embedding_texts)
        embedding_matrix = np.asarray(embeddings, dtype="float32")
        if embedding_matrix.ndim != 2:
            raise ValueError("Embedding service must return a 2D matrix")
        if embedding_matrix.shape[0] != len(docs):
            raise ValueError("Embedding service returned a different number of vectors than docs")
        if embedding_matrix.shape[1] <= 0:
            raise ValueError("Embedding vectors must have a positive dimension")

        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        import faiss

        index = faiss.IndexFlatIP(int(embedding_matrix.shape[1]))
        index.add(embedding_matrix)

        self.artifact_store.save_docs(docs)
        self.artifact_store.save_faiss(index, doc_ids)

        docs_by_source: dict[str, list[SourceDoc]] = defaultdict(list)
        for doc in docs:
            docs_by_source[doc.system_id].append(doc)

        elasticsearch_indices: dict[str, str] = {}
        if self.index_elasticsearch:
            for source in self.source_registry.sources:
                LocalElasticsearchIndexer(index_name=source.es_index).rebuild(
                    docs_by_source.get(source.source_id, [])
                )
                elasticsearch_indices[source.source_id] = source.es_index

        self.artifact_store.save_manifest(
            {
                "version": 2,
                "built_at": datetime.now(UTC).isoformat(),
                "doc_count": len(docs),
                "source_count": len(self.source_registry.sources),
                "sources": {
                    source.source_id: {
                        "doc_count": len(docs_by_source.get(source.source_id, [])),
                        "elasticsearch_index": source.es_index,
                    }
                    for source in self.source_registry.sources
                },
                "keyword_retriever": "local_es_per_source",
                "elasticsearch_indexed": self.index_elasticsearch,
                "elasticsearch_url": self.settings.LOCAL_ES_URL,
                "elasticsearch_indices": elasticsearch_indices,
                "source_config_path": (
                    str(self.source_registry.config_path)
                    if self.source_registry.config_path is not None
                    else None
                ),
                "embedding_provider": self.settings.EMBEDDING_PROVIDER,
                "embedding_model": self.settings.EMBEDDING_MODEL_PATH,
                "embedding_dim": int(embedding_matrix.shape[1]),
                "faiss_index": "IndexFlatIP",
                "faiss_index_path": str(self.artifact_store.faiss_path),
                "faiss_doc_ids_path": str(self.artifact_store.faiss_doc_ids_path),
            }
        )
        return BuildIndexResult(
            doc_count=len(docs),
            artifact_dir=str(self.artifact_store.artifact_dir),
            embedding_model=self.settings.EMBEDDING_MODEL_PATH,
            embedding_dim=int(embedding_matrix.shape[1]),
            source_count=len(self.source_registry.sources),
            elasticsearch_indices=elasticsearch_indices,
        )
