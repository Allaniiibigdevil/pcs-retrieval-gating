from dataclasses import dataclass
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
from app.storage.local_artifact_store import LocalArtifactStore


@dataclass(frozen=True)
class BuildIndexResult:
    doc_count: int
    artifact_dir: str
    embedding_model: str
    embedding_dim: int


class LocalIndexBuilder:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        embedding_service: EmbeddingService | None = None,
        index_elasticsearch: bool | None = None,
    ) -> None:
        self.settings = get_settings()
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service or get_embedding_service()
        self.index_elasticsearch = (
            self.settings.LOCAL_ES_INDEX_ON_BUILD
            if index_elasticsearch is None
            else index_elasticsearch
        )

    async def build(self, docs: list[SourceDoc]) -> BuildIndexResult:
        if not docs:
            raise ValueError("Cannot build local index from an empty document set")

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
        self.artifact_store.save_faiss(index, doc_ids)
        if self.index_elasticsearch:
            LocalElasticsearchIndexer().rebuild(docs)
        self.artifact_store.save_manifest(
            {
                "version": 1,
                "built_at": datetime.now(UTC).isoformat(),
                "doc_count": len(docs),
                "keyword_retriever": "local_es",
                "elasticsearch_indexed": self.index_elasticsearch,
                "elasticsearch_url": self.settings.LOCAL_ES_URL,
                "elasticsearch_index": self.settings.LOCAL_ES_INDEX,
                "elasticsearch_synonyms_path": self.settings.LOCAL_ES_SYNONYMS_PATH or None,
                "embedding_provider": self.settings.EMBEDDING_PROVIDER,
                "embedding_model": self.settings.EMBEDDING_MODEL_PATH,
                "embedding_dim": int(embedding_matrix.shape[1]),
                "faiss_index": "IndexFlatIP",
            }
        )
        return BuildIndexResult(
            doc_count=len(docs),
            artifact_dir=str(self.artifact_store.artifact_dir),
            embedding_model=self.settings.EMBEDDING_MODEL_PATH,
            embedding_dim=int(embedding_matrix.shape[1]),
        )
