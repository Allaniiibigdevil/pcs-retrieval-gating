from collections import defaultdict
from collections.abc import Iterable
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

        embedding_matrix = await self._embed_docs(docs, show_model_progress=self.show_progress)
        embedding_dim = int(embedding_matrix.shape[1])
        if self.show_progress:
            render_progress("Embedding docs", len(docs), len(docs), detail="single pass")

        docs_by_source: dict[str, list[SourceDoc]] = defaultdict(list)
        positions_by_source: dict[str, list[int]] = defaultdict(list)
        for position, doc in enumerate(docs):
            docs_by_source[doc.system_id].append(doc)
            positions_by_source[doc.system_id].append(position)

        self.artifact_store.save_docs(docs)
        if self.show_progress:
            render_progress("Saving docs", 1, 1, detail=f"docs={len(docs)}")

        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        import faiss

        faiss_indices: dict[str, str] = {}
        for source_number, source in enumerate(self.source_registry.sources, start=1):
            source_docs = docs_by_source.get(source.source_id, [])
            source_positions = positions_by_source.get(source.source_id, [])
            index = faiss.IndexFlatIP(embedding_dim)
            if source_positions:
                source_vectors = embedding_matrix[np.asarray(source_positions, dtype="int64")]
                index.add(source_vectors)
            self.artifact_store.save_faiss(
                index,
                [doc.doc_id for doc in source_docs],
                index_path=source.faiss_index_path,
                doc_ids_path=source.faiss_doc_ids_path,
            )
            faiss_indices[source.source_id] = source.faiss_index_path
            if self.show_progress:
                render_progress(
                    "FAISS indexes",
                    source_number,
                    len(self.source_registry.sources),
                    detail=source.source_id,
                )

        elasticsearch_indices: dict[str, str] = {}
        if self.index_elasticsearch:
            for source in self.source_registry.sources:
                LocalElasticsearchIndexer(
                    index_name=source.es_index,
                    show_progress=self.show_progress,
                ).rebuild(docs_by_source.get(source.source_id, []))
                elasticsearch_indices[source.source_id] = source.es_index

        self._save_manifest(
            docs_by_source=docs_by_source,
            embedding_dim=embedding_dim,
            elasticsearch_indices=elasticsearch_indices,
            build_mode="single_pass_then_split_by_source",
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

    async def build_streaming_source(
        self,
        docs: Iterable[SourceDoc],
        *,
        source_id: str,
        total_docs: int,
        batch_size: int | None = None,
    ) -> BuildIndexResult:
        if total_docs <= 0:
            raise ValueError(f"No documents found for source_id {source_id!r}")
        source = self.source_registry.require(source_id)
        if len(self.source_registry.sources) != 1:
            raise ValueError("Streaming source build requires a single-source registry")

        chunk_size = batch_size or self.settings.INDEX_BUILD_EMBEDDING_CHUNK_SIZE
        if chunk_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.artifact_store.ensure_dir()
        temp_docs_path = self.artifact_store.docs_path.with_suffix(".jsonl.tmp")
        doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        processed = 0
        batch: list[SourceDoc] = []
        embedding_dim: int | None = None
        index = None

        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        import faiss

        async def flush_batch() -> None:
            nonlocal processed, embedding_dim, index
            if not batch:
                return
            batch_matrix = await self._embed_docs(batch, show_model_progress=False)
            current_dim = int(batch_matrix.shape[1])
            if index is None:
                embedding_dim = current_dim
                index = faiss.IndexFlatIP(current_dim)
            elif current_dim != embedding_dim:
                raise ValueError(
                    "Embedding dimension changed between batches: "
                    f"expected {embedding_dim}, got {current_dim}"
                )
            index.add(batch_matrix)
            doc_ids.extend(doc.doc_id for doc in batch)
            processed += len(batch)
            if self.show_progress:
                render_progress(
                    "Embedding docs",
                    processed,
                    total_docs,
                    detail=f"source={source_id} batch={len(batch)}",
                )
            batch.clear()

        try:
            with temp_docs_path.open("w", encoding="utf-8", newline="\n") as docs_file:
                for doc in docs:
                    if doc.system_id != source_id:
                        raise ValueError(
                            f"Streaming source build received system_id={doc.system_id!r}; "
                            f"expected {source_id!r}"
                        )
                    if doc.doc_id in seen_doc_ids:
                        raise ValueError(f"Duplicate doc_id value for source {source_id!r}: {doc.doc_id}")
                    seen_doc_ids.add(doc.doc_id)
                    docs_file.write(doc.model_dump_json() + "\n")
                    batch.append(doc)
                    if len(batch) >= chunk_size:
                        await flush_batch()
                await flush_batch()

            if processed != total_docs:
                raise RuntimeError(
                    "Source document count changed between scan and build: "
                    f"expected {total_docs}, processed {processed}"
                )
            if index is None or embedding_dim is None:
                raise ValueError(f"No documents found for source_id {source_id!r}")

            self.artifact_store.save_faiss(
                index,
                doc_ids,
                index_path=source.faiss_index_path,
                doc_ids_path=source.faiss_doc_ids_path,
            )
            temp_docs_path.replace(self.artifact_store.docs_path)
            if self.show_progress:
                render_progress("FAISS indexes", 1, 1, detail=source_id)

            elasticsearch_indices: dict[str, str] = {}
            if self.index_elasticsearch:
                LocalElasticsearchIndexer(
                    index_name=source.es_index,
                    show_progress=self.show_progress,
                ).rebuild_iter(self.artifact_store.iter_docs(), total=processed)
                elasticsearch_indices[source_id] = source.es_index

            self._save_manifest(
                docs_by_source={source_id: []},
                embedding_dim=embedding_dim,
                elasticsearch_indices=elasticsearch_indices,
                build_mode="streaming_single_source",
                source_doc_counts={source_id: processed},
                streaming_batch_size=chunk_size,
            )
            return BuildIndexResult(
                doc_count=processed,
                artifact_dir=str(self.artifact_store.artifact_dir),
                embedding_model=self.settings.EMBEDDING_MODEL_PATH,
                embedding_dim=embedding_dim,
                source_count=1,
                faiss_indices={source_id: source.faiss_index_path},
                elasticsearch_indices=elasticsearch_indices,
            )
        finally:
            if temp_docs_path.exists():
                temp_docs_path.unlink()

    async def _embed_docs(
        self,
        docs: list[SourceDoc],
        *,
        show_model_progress: bool,
    ) -> np.ndarray:
        embedding_texts = [build_embedding_text(doc) for doc in docs]
        embeddings = await self.embedding_service.embed_batch(
            embedding_texts,
            show_progress=show_model_progress,
        )
        matrix = np.asarray(embeddings, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("Embedding service must return a 2D matrix")
        if matrix.shape[0] != len(docs):
            raise ValueError("Embedding service returned a different number of vectors than docs")
        if matrix.shape[1] <= 0:
            raise ValueError("Embedding vectors must have a positive dimension")
        return matrix

    def _save_manifest(
        self,
        *,
        docs_by_source: dict[str, list[SourceDoc]],
        embedding_dim: int,
        elasticsearch_indices: dict[str, str],
        build_mode: str,
        source_doc_counts: dict[str, int] | None = None,
        streaming_batch_size: int | None = None,
    ) -> None:
        source_doc_counts = source_doc_counts or {
            source.source_id: len(docs_by_source.get(source.source_id, []))
            for source in self.source_registry.sources
        }
        manifest = {
            "version": 3,
            "built_at": datetime.now(UTC).isoformat(),
            "doc_count": sum(source_doc_counts.values()),
            "source_count": len(self.source_registry.sources),
            "sources": {
                source.source_id: {
                    "doc_count": source_doc_counts.get(source.source_id, 0),
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
            "embedding_build_mode": build_mode,
            "faiss_index_type": "IndexFlatIP",
            "elasticsearch_bulk_batch_size": self.settings.LOCAL_ES_BULK_BATCH_SIZE,
        }
        if streaming_batch_size is not None:
            manifest["embedding_stream_batch_size"] = streaming_batch_size
        self.artifact_store.save_manifest(manifest)
