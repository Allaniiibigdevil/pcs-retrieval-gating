from dataclasses import dataclass
from datetime import UTC, datetime
import logging

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.embedding.embedding_service import (
    EmbeddingService,
    build_embedding_text,
    get_embedding_service,
)
from app.retrieval.tokenizer import build_bm25_text, tokenize
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
    ) -> None:
        self.settings = get_settings()
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service or get_embedding_service()

    async def build(self, docs: list[SourceDoc]) -> BuildIndexResult:
        if not docs:
            raise ValueError("Cannot build local index from an empty document set")

        tokenized_docs = [tokenize(build_bm25_text(doc)) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)

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
        self.artifact_store.save_bm25(bm25, tokenized_docs)
        self.artifact_store.save_faiss(index, doc_ids)
        self.artifact_store.save_manifest(
            {
                "version": 1,
                "built_at": datetime.now(UTC).isoformat(),
                "doc_count": len(docs),
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
