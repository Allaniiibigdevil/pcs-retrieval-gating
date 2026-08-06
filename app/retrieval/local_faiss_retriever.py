import asyncio

import numpy as np

from app.config import get_settings
from app.embedding.embedding_service import EmbeddingService, get_embedding_service
from app.schemas.doc import SourceDoc
from app.schemas.search import SearchHit
from app.storage.local_artifact_store import LocalArtifactStore


_DEFAULT_THRESHOLD = object()


class LocalFaissRetriever:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        embedding_service: EmbeddingService | None = None,
        source_id: str | None = None,
        min_score_threshold: float | None | object = _DEFAULT_THRESHOLD,
    ) -> None:
        settings = get_settings()
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service
        self.source_id = source_id
        self.min_score_threshold = (
            getattr(settings, "FAISS_MIN_SCORE_THRESHOLD", None)
            if min_score_threshold is _DEFAULT_THRESHOLD
            else min_score_threshold
        )
        self._docs_by_id: dict[str, SourceDoc] | None = None
        self._doc_ids: list[str] | None = None
        self._index = None
        self._source_positions: np.ndarray | None = None
        self._source_vectors: np.ndarray | None = None

    def _ensure_loaded(self) -> None:
        if self._index is not None and self._docs_by_id is not None and self._doc_ids is not None:
            return

        docs = self.artifact_store.load_docs()
        index, doc_ids = self.artifact_store.load_faiss()
        docs_by_id = {doc.doc_id: doc for doc in docs}
        if len(docs_by_id) != len(docs):
            raise RuntimeError("Local docs artifact contains duplicate doc_id values")
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

        self._index = index
        self._doc_ids = doc_ids
        self._docs_by_id = docs_by_id
        if self.source_id is not None:
            positions = [
                position
                for position, doc_id in enumerate(doc_ids)
                if docs_by_id[doc_id].system_id == self.source_id
            ]
            self._source_positions = np.asarray(positions, dtype="int64")
            if positions:
                try:
                    self._source_vectors = np.vstack(
                        [index.reconstruct(int(position)) for position in positions]
                    ).astype("float32", copy=False)
                except Exception as exc:
                    raise RuntimeError(
                        "Source-filtered FAISS retrieval requires an index that supports "
                        "vector reconstruction; rebuild the shared index as IndexFlatIP"
                    ) from exc
            else:
                self._source_vectors = np.empty((0, int(index.d)), dtype="float32")

        if self.embedding_service is None:
            self.embedding_service = get_embedding_service()

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        self._ensure_loaded()
        assert self._index is not None
        assert self._doc_ids is not None
        assert self._docs_by_id is not None
        assert self.embedding_service is not None

        if top_k <= 0 or not query.strip():
            return []

        query_embedding = await self.embedding_service.embed(query)
        vector = np.asarray([query_embedding], dtype="float32")
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise RuntimeError(
                f"Query embedding must have shape (1, d), received {vector.shape}"
            )
        index_dim = int(self._index.d)
        query_dim = int(vector.shape[1])
        if query_dim != index_dim:
            model_path = get_settings().EMBEDDING_MODEL_PATH
            raise RuntimeError(
                "FAISS index dimension mismatch: "
                f"index dimension is {index_dim}, query embedding dimension is {query_dim}, "
                f"configured model is {model_path!r}. Rebuild the shared FAISS artifacts and "
                "restart the service."
            )

        if self.source_id is None:
            scores, indices = await asyncio.to_thread(
                self._index.search,
                vector,
                min(top_k, len(self._doc_ids)),
            )
            ranked_pairs = [
                (float(score), int(index))
                for score, index in zip(scores[0], indices[0])
                if int(index) >= 0
            ]
        else:
            assert self._source_positions is not None
            assert self._source_vectors is not None
            ranked_pairs = await asyncio.to_thread(
                _rank_source_vectors,
                vector[0],
                self._source_vectors,
                self._source_positions,
                top_k,
            )

        hits: list[SearchHit] = []
        output_rank = 0
        for score, index in ranked_pairs:
            if self.min_score_threshold is not None and score < self.min_score_threshold:
                continue
            output_rank += 1
            doc = self._docs_by_id[self._doc_ids[index]]
            hits.append(
                SearchHit(
                    doc_id=doc.doc_id,
                    system_id=doc.system_id,
                    summary=doc.summary,
                    keywords=doc.keywords,
                    metadata=doc.metadata,
                    vector_score=score,
                    vector_rank=output_rank,
                )
            )
        return hits


def _rank_source_vectors(
    query_vector: np.ndarray,
    source_vectors: np.ndarray,
    source_positions: np.ndarray,
    top_k: int,
) -> list[tuple[float, int]]:
    if source_vectors.shape[0] == 0 or top_k <= 0:
        return []
    scores = source_vectors @ query_vector
    local_order = np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]
    return [
        (float(scores[local_index]), int(source_positions[local_index]))
        for local_index in local_order
    ]
