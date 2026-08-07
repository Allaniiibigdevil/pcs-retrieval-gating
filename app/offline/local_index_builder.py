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
from app.utils.progress import render_progress


@dataclass(frozen=True)
class BuildIndexResult:
    doc_count: int
    artifact_dir: str
    embedding_model: str
    embedding_dim: int
    source_count: int
    faiss_indices: dict[str, str] = field(default_factory=dict)
    elasticsearch_indices: dict[str, str] = field(default_factory=dict)


class LocalIndexBuilder:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        embedding_service: EmbeddingService | None = None,
        index_elasticsearch: bool = False,
        source_registry: SourceRegistry | None = None,
        *,
        show_progress: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service or get_embedding_service()
        self.index_elasticsearch = index_elasticsearch
        self.source_registry = source_registry or get_source_registry()
        self.show_progress = show_progress

    async def build(self, docs: list[SourceDoc]) -> BuildIndexResult:
        if not docs:
            raise ValueError("Cannot build local indexes from an empty document set")

        doc_ids = [doc.doc_id for doc in docs]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Document doc_id values must be globally unique")
        self.source_registry.validate_source_ids({doc.system_id for doc in docs})

        docs_by_source: dict[str, list[SourceDoc]] = defaultdict(list)
        for doc in docs:
            docs_by_source[doc.system_id].append(doc)

        self.artifact_store.save_docs(docs)
        if self.show_progress:
            render_progress("Saving docs", 1, 1, detail=f"docs={len(docs)}")

        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        import faiss

        embedding_dim: int | None = None
        embedded_docs = 0
        faiss_done = 0
        faiss_indices: dict[str, str] = {}
        empty_sources = []

        for source in self.source_registry.sources:
            source_docs = docs_by_source.get(source.source_id, [])
            if not source_docs:
                empty_sources.append(source)
                continue

            embedding_matrix = await self._embed_source_docs(
                source_docs,
                source_id=source.source_id,
                completed_before=embedded_docs,
                total_docs=len(docs),
            )
            current_dim = int(embedding_matrix.shape[1])
            if embedding_dim is None:
                embedding_dim = current_dim
            elif current_dim != embedding_dim:
                raise ValueError(
                    "Embedding dimension changed between sources: "
                    f"expected {embedding_dim}, got {current_dim} for {source.source_id!r}"
                )

            index = faiss.IndexFlatIP(current_dim)
            index.add(embedding_matrix)
            self.artifact_store.save_faiss(
                index,
                [doc.doc_id for doc in source_docs],
                index_path=source.faiss_index_path,
                doc_ids_path=source.faiss_doc_ids_path,
            )
            faiss_indices[source.source_id] = source.faiss_index_path
            embedded_docs += len(source_docs)
            faiss_done += 1
            if self.show_progress:
                render_progress(
                    "FAISS indexes",
                    faiss_done,
                    len(self.source_registry.sources),
                    detail=source.source_id,
                )

        if embedding_dim is None:
            raise ValueError("No configured source contained documents to embed")

        for source in empty_sources:
            index = faiss.IndexFlatIP(embedding_dim)
            self.artifact_store.save_faiss(
                index,
                [],
                index_path=source.faiss_index_path,
                doc_ids_path=source.faiss_doc_ids_path,
            )
            faiss_indices[source.source_id] = source.faiss_index_path
            faiss_done += 1
            if self.show_progress:
                render_progress(
                    "FAISS indexes",
                    faiss_done,
                    len(self.source_registry.sources),
                    detail=source.source_id,
                )

        elasticsearch_indices: dict[str, str] = {}
        if self.index_elasticsearch:
            for source in self.source_registry.sources:
                indexer_kwargs = {"index_name": source.es_index}
                if self.show_progress:
                    indexer_kwargs["show_progress"] = True
                LocalElasticsearchIndexer(**indexer_kwargs).rebuild(
                    docs_by_source.get(source.source_id, [])
                )
                elasticsearch_indices[source.source_id] = source.es_index

        self.artifact_store.save_manifest(
            {
                "version": 3,
                "built_at": datetime.now(UTC).isoformat(),
                "doc_count": len(docs),
                "source_count": len(self.source_registry.sources),
                "sources": {
                    source.source_id: {
                        "doc_count": len(docs_by_source.get(source.source_id, [])),
                        "elasticsearch_index": source.es_index,
                        "faiss_index_path": source.faiss_index_path,
                        "faiss_doc_ids_path": source.faiss_doc_ids_path,
                    }
                    for source in self.source_registry.sources
                },
                "keyword_retriever": "local_es_per_source",
                "vector_retriever": "local_faiss_per_source",
                "elasticsearch_indexed": self.index_elasticsearch,
                "elasticsearch_url": self.settings.LOCAL_ES_URL,
                "source_config_path": (
                    str(self.source_registry.config_path)
                    if self.source_registry.config_path is not None
                    else None
                ),
                "embedding_provider": self.settings.EMBEDDING_PROVIDER,
                "embedding_model": self.settings.EMBEDDING_MODEL_PATH,
                "embedding_dim": embedding_dim,
                "embedding_batch_size": self.settings.EMBEDDING_BATCH_SIZE,
                "embedding_build_chunk_size": self.settings.INDEX_BUILD_EMBEDDING_CHUNK_SIZE,
                "faiss_index_type": "IndexFlatIP",
                "elasticsearch_bulk_batch_size": self.settings.LOCAL_ES_BULK_BATCH_SIZE,
            }
        )
        return BuildIndexResult(
            doc_count=len(docs),
            artifact_dir=str(self.artifact_store.artifact_dir),
            embedding_model=self.settings.EMBEDDING_MODEL_PATH,
            embedding_dim=embedding_dim,
            source_count=len(self.source_registry.sources),
            faiss_indices=faiss_indices,
            elasticsearch_indices=elasticsearch_indices,
        )

    async def _embed_source_docs(
        self,
        docs: list[SourceDoc],
        *,
        source_id: str,
        completed_before: int,
        total_docs: int,
    ) -> np.ndarray:
        chunk_size = self.settings.INDEX_BUILD_EMBEDDING_CHUNK_SIZE
        matrix: np.ndarray | None = None
        expected_dim: int | None = None

        for start in range(0, len(docs), chunk_size):
            batch_docs = docs[start : start + chunk_size]
            embedding_texts = [build_embedding_text(doc) for doc in batch_docs]
            embeddings = await self.embedding_service.embed_batch(embedding_texts)
            batch_matrix = np.asarray(embeddings, dtype="float32")
            if batch_matrix.ndim != 2:
                raise ValueError("Embedding service must return a 2D matrix")
            if batch_matrix.shape[0] != len(batch_docs):
                raise ValueError(
                    "Embedding service returned a different number of vectors than docs"
                )
            if batch_matrix.shape[1] <= 0:
                raise ValueError("Embedding vectors must have a positive dimension")

            current_dim = int(batch_matrix.shape[1])
            if matrix is None:
                expected_dim = current_dim
                matrix = np.empty((len(docs), current_dim), dtype="float32")
            elif current_dim != expected_dim:
                raise ValueError(
                    "Embedding dimension changed between batches: "
                    f"expected {expected_dim}, got {current_dim}"
                )

            end = start + len(batch_docs)
            matrix[start:end] = batch_matrix
            if self.show_progress:
                render_progress(
                    "Embedding docs",
                    completed_before + end,
                    total_docs,
                    detail=f"source={source_id}",
                )

        if matrix is None:
            raise ValueError("Cannot embed an empty source")
        return matrix
