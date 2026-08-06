import asyncio

import numpy as np

from app.config import get_settings
from app.embedding.embedding_service import EmbeddingService, get_embedding_service
from app.schemas.doc import SourceDoc
from app.schemas.search import SearchHit
from app.storage.local_artifact_store import LocalArtifactStore


class LocalFaissRetriever:
    def __init__(
        self,
        *,
        source_id: str,
        artifact_store: LocalArtifactStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        if not source_id.strip():
            raise ValueError("source_id must not be blank")
        self.source_id = source_id
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service or get_embedding_service()
        self._docs_by_id: dict[str, SourceDoc] | None = None
        self._doc_ids: list[str] | None = None
        self._index = None
        self._source_positions: np.ndarray | None = None
        self._source_vectors: np.ndarray | None = None

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return

        docs = self.artifact_store.load_docs()
        index, doc_ids = self.artifact_store.load_faiss()
        docs_by_id = {doc.doc_id: doc for doc in docs}
        if len(docs_by_id) != len(docs):
            raise RuntimeError("Local docs artifact contains duplicate doc_id values")
        if len(set(doc_ids)) != len(doc_ids):
            raise RuntimeError("FAISS doc-id mapping contains duplicate doc_id values")
        if int(index.ntotal) != len(doc_ids):
            raise RuntimeError(
                "FAISS index and doc-id mapping are inconsistent: "
                f"index contains {int(index.ntotal)} vectors but mapping contains {len(doc_ids)} ids"
            )

        missing_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in docs_by_id]
        if missing_doc_ids:
            raise RuntimeError(
                "FAISS doc-id mapping references documents missing from docs.jsonl: "
                + ", ".join(missing_doc_ids[:5])
            )

        positions = [
            position
            for position, doc_id in enumerate(doc_ids)
            if docs_by_id[doc_id].system_id == self.source_id
        ]
        try:
            source_vectors = (
                np.vstack([index.reconstruct(int(position)) for position in positions]).astype(
                    "float32", copy=False
                )
                if positions
                else np.empty((0, int(index.d)), dtype="float32")
            )
        except Exception as exc:
            raise RuntimeError(
                "Source-filtered FAISS retrieval requires an IndexFlatIP index that supports "
                "vector reconstruction"
            ) from exc

        self._index = index
        self._doc_ids = doc_ids
        self._docs_by_id = docs_by_id
        self._source_positions = np.asarray(positions, dtype="int64")
        self._source_vectors = source_vectors

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        if not query.strip():
            return []
        self._ensure_loaded()
        assert self._index is not None
        assert self._doc_ids is not None
        assert self._docs_by_id is not None
        assert self._source_positions is not None
        assert self._source_vectors is not None

        query_embedding = await self.embedding_service.embed(query)
        vector = np.asarray(query_embedding, dtype="float32")
        if vector.ndim != 1:
            raise RuntimeError(f"Query embedding must be one-dimensional, received {vector.shape}")
        if int(vector.shape[0]) != int(self._index.d):
            raise RuntimeError(
                "FAISS index dimension mismatch: "
                f"index dimension is {int(self._index.d)}, query embedding dimension is "
                f"{int(vector.shape[0])}, configured model is "
                f"{get_settings().EMBEDDING_MODEL_PATH!r}. Rebuild the shared FAISS artifacts."
            )

        ranked_pairs = await asyncio.to_thread(
            _rank_source_vectors,
            vector,
            self._source_vectors,
            self._source_positions,
            top_k,
        )
        hits: list[SearchHit] = []
        for rank, (score, position) in enumerate(ranked_pairs, start=1):
            doc = self._docs_by_id[self._doc_ids[position]]
            hits.append(
                SearchHit(
                    doc_id=doc.doc_id,
                    system_id=self.source_id,
                    summary=doc.summary,
                    keywords=doc.keywords,
                    metadata=doc.metadata,
                    vector_score=score,
                    vector_rank=rank,
                )
            )
        return hits


def _rank_source_vectors(
    query_vector: np.ndarray,
    source_vectors: np.ndarray,
    source_positions: np.ndarray,
    top_k: int,
) -> list[tuple[float, int]]:
    if source_vectors.shape[0] == 0:
        return []
    scores = source_vectors @ query_vector
    local_order = np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]
    return [
        (float(scores[local_index]), int(source_positions[local_index]))
        for local_index in local_order
    ]
