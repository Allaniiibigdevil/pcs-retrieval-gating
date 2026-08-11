from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from app.config import get_settings
from app.embedding.embedding_service import (
    EmbeddingService,
    build_embedding_text,
    get_embedding_service,
)
from app.offline.local_es_indexer import LocalElasticsearchIndexer
from app.offline.local_es_vector_indexer import LocalElasticsearchVectorIndexer
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
    keyword_index: str
    vector_index: str
    faiss_indices: dict[str, str] = field(default_factory=dict)
    elasticsearch_indices: dict[str, str] = field(default_factory=dict)


class LocalIndexBuilder:
    """Build one unified keyword ES index and one unified vector ES index."""

    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore | None = None,
        embedding_service: EmbeddingService | None = None,
        source_registry: SourceRegistry | None = None,
    ) -> None:
        self.settings = get_settings()
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service or get_embedding_service()
        self.source_registry = source_registry or get_source_registry()

    async def build(self, docs: list[SourceDoc]) -> BuildIndexResult:
        if not docs:
            raise ValueError("Cannot build local index from an empty document set")

        source_ids = {doc.system_id for doc in docs}
        self.source_registry.validate_source_ids(source_ids)

        # Build the lightweight keyword index first so ES mapping/analyzer errors
        # fail before the expensive embedding pass starts.
        LocalElasticsearchIndexer(
            index_name=self.settings.LOCAL_ES_KEYWORD_INDEX
        ).rebuild(docs)

        # Keep the proven embedding path unchanged:
        # build all texts -> one embed_batch call -> one float32 matrix.
        embedding_texts = [build_embedding_text(doc) for doc in docs]
        embeddings = await self.embedding_service.embed_batch(embedding_texts)
        embedding_matrix = np.asarray(embeddings, dtype="float32")
        if len(embedding_matrix.shape) != 2:
            raise ValueError("Embedding service must return a 2D matrix")
        if int(embedding_matrix.shape[0]) != len(docs):
            raise ValueError("Embedding service returned the wrong number of vectors")

        LocalElasticsearchVectorIndexer().rebuild(docs, embedding_matrix)

        # Keep a mixed document artifact for build inspection.
        self.artifact_store.save_docs(docs)
        self.artifact_store.save_manifest(
            {
                "version": 5,
                "built_at": datetime.now(UTC).isoformat(),
                "doc_count": len(docs),
                "source_count": len(source_ids),
                "keyword_index": self.settings.LOCAL_ES_KEYWORD_INDEX,
                "vector_index": self.settings.LOCAL_ES_VECTOR_INDEX,
                "keyword_retriever": "local_es_unified",
                "vector_retriever": "local_es_dense_vector",
                "elasticsearch_url": self.settings.LOCAL_ES_URL,
                "embedding_provider": self.settings.EMBEDDING_PROVIDER,
                "embedding_model": self.settings.EMBEDDING_MODEL_PATH,
                "embedding_dim": int(embedding_matrix.shape[1]),
                "embedding_build_mode": "legacy_single_pass_all_sources",
            }
        )
        return BuildIndexResult(
            doc_count=len(docs),
            artifact_dir=str(self.artifact_store.artifact_dir),
            embedding_model=self.settings.EMBEDDING_MODEL_PATH,
            embedding_dim=int(embedding_matrix.shape[1]),
            source_count=len(source_ids),
            keyword_index=self.settings.LOCAL_ES_KEYWORD_INDEX,
            vector_index=self.settings.LOCAL_ES_VECTOR_INDEX,
            elasticsearch_indices={
                "keyword": self.settings.LOCAL_ES_KEYWORD_INDEX,
                "vector": self.settings.LOCAL_ES_VECTOR_INDEX,
            },
        )
