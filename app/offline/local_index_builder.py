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
from app.source_registry import SourceConfig, SourceRegistry
from app.storage.local_artifact_store import LocalArtifactStore


@dataclass(frozen=True)
class BuildIndexResult:
    doc_count: int
    artifact_dir: str
    embedding_model: str
    embedding_dim: int
    source_count: int = 1
    faiss_indices: dict[str, str] = field(default_factory=dict)
    elasticsearch_indices: dict[str, str] = field(default_factory=dict)


class LocalIndexBuilder:
    """Build exactly one source using the original single-index build path."""

    def __init__(
        self,
        *,
        source: SourceConfig,
        artifact_store: LocalArtifactStore | None = None,
        embedding_service: EmbeddingService | None = None,
        index_elasticsearch: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.source = source
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service or get_embedding_service()
        self.index_elasticsearch = index_elasticsearch

    async def build(self, docs: list[SourceDoc]) -> BuildIndexResult:
        if not docs:
            raise ValueError("Cannot build local index from an empty document set")

        wrong_sources = sorted({doc.system_id for doc in docs if doc.system_id != self.source.source_id})
        if wrong_sources:
            raise ValueError(
                f"Single-source build for {self.source.source_id!r} received documents from: "
                + ", ".join(wrong_sources)
            )

        # Keep this path intentionally equivalent to the pre-multi-source builder:
        # build all texts -> one embed_batch call -> one matrix -> one FAISS index.
        embedding_texts = [build_embedding_text(doc) for doc in docs]
        embeddings = await self.embedding_service.embed_batch(embedding_texts)
        embedding_matrix = np.asarray(embeddings, dtype="float32")
        if len(embedding_matrix.shape) != 2:
            raise ValueError("Embedding service must return a 2D matrix")

        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        import faiss

        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        doc_ids = [doc.doc_id for doc in docs]
        self.artifact_store.save_docs(docs)
        self.artifact_store.save_faiss(
            index,
            doc_ids,
            index_path=self.source.faiss_index_path,
            doc_ids_path=self.source.faiss_doc_ids_path,
        )

        elasticsearch_indices: dict[str, str] = {}
        if self.index_elasticsearch:
            LocalElasticsearchIndexer(index_name=self.source.es_index).rebuild(docs)
            elasticsearch_indices[self.source.source_id] = self.source.es_index

        self.artifact_store.save_manifest(
            {
                "version": 3,
                "built_at": datetime.now(UTC).isoformat(),
                "doc_count": len(docs),
                "source_count": 1,
                "sources": {
                    self.source.source_id: {
                        "doc_count": len(docs),
                        "elasticsearch_index": self.source.es_index,
                        "faiss_index_path": self.source.faiss_index_path,
                        "faiss_doc_ids_path": self.source.faiss_doc_ids_path,
                    }
                },
                "keyword_retriever": "local_es_per_source",
                "vector_retriever": "local_faiss_per_source",
                "elasticsearch_indexed": self.index_elasticsearch,
                "elasticsearch_url": self.settings.LOCAL_ES_URL,
                "embedding_provider": self.settings.EMBEDDING_PROVIDER,
                "embedding_model": self.settings.EMBEDDING_MODEL_PATH,
                "embedding_dim": int(embedding_matrix.shape[1]),
                "embedding_build_mode": "legacy_single_source",
                "faiss_index_type": "IndexFlatIP",
            }
        )
        return BuildIndexResult(
            doc_count=len(docs),
            artifact_dir=str(self.artifact_store.artifact_dir),
            embedding_model=self.settings.EMBEDDING_MODEL_PATH,
            embedding_dim=int(embedding_matrix.shape[1]),
            faiss_indices={self.source.source_id: self.source.faiss_index_path},
            elasticsearch_indices=elasticsearch_indices,
        )
