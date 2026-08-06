import asyncio
from threading import Lock

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
        faiss_index_path: str,
        faiss_doc_ids_path: str,
        artifact_store: LocalArtifactStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        if not source_id.strip():
            raise ValueError("source_id must not be blank")
        self.source_id = source_id
        self.faiss_index_path = faiss_index_path
        self.faiss_doc_ids_path = faiss_doc_ids_path
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service or get_embedding_service()
        self._docs_by_id: dict[str, SourceDoc] | None = None
        self._doc_ids: list[str] | None = None
        self._index = None
        self._load_lock = Lock()

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        with self._load_lock:
            if self._index is not None:
                return

            docs = self.artifact_store.load_docs()
            index, doc_ids = self.artifact_store.load_faiss(
                index_path=self.faiss_index_path,
                doc_ids_path=self.faiss_doc_ids_path,
            )
            docs_by_id = {doc.doc_id: doc for doc in docs}
            if len(docs_by_id) != len(docs):
                raise RuntimeError("Local docs artifact contains duplicate doc_id values")

            missing_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in docs_by_id]
            if missing_doc_ids:
                raise RuntimeError(
                    "FAISS doc-id mapping references documents missing from docs.jsonl: "
                    + ", ".join(missing_doc_ids[:5])
                )

            mismatched_doc_ids = [
                doc_id
                for doc_id in doc_ids
                if docs_by_id[doc_id].system_id != self.source_id
            ]
            if mismatched_doc_ids:
                raise RuntimeError(
                    f"FAISS index for source {self.source_id!r} contains documents from "
                    "another source: "
                    + ", ".join(mismatched_doc_ids[:5])
                )

            expected_doc_ids = {
                doc.doc_id for doc in docs if doc.system_id == self.source_id
            }
            if set(doc_ids) != expected_doc_ids:
                missing_from_index = sorted(expected_doc_ids - set(doc_ids))
                unexpected_in_index = sorted(set(doc_ids) - expected_doc_ids)
                raise RuntimeError(
                    f"FAISS artifacts for source {self.source_id!r} are stale or incomplete; "
                    f"missing={missing_from_index[:5]} unexpected={unexpected_in_index[:5]}"
                )

            self._index = index
            self._doc_ids = doc_ids
            self._docs_by_id = docs_by_id

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        if top_k <= 0 or not query.strip():
            return []
        await asyncio.to_thread(self._ensure_loaded)
        assert self._index is not None
        assert self._doc_ids is not None
        assert self._docs_by_id is not None

        if not self._doc_ids:
            return []

        query_embedding = await self.embedding_service.embed(query)
        vector = np.asarray([query_embedding], dtype="float32")
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise RuntimeError(
                f"Query embedding must have shape (1, d), received {vector.shape}"
            )
        if int(vector.shape[1]) != int(self._index.d):
            raise RuntimeError(
                "FAISS index dimension mismatch: "
                f"index dimension is {int(self._index.d)}, query embedding dimension is "
                f"{int(vector.shape[1])}, configured model is "
                f"{get_settings().EMBEDDING_MODEL_PATH!r}. Rebuild this source's FAISS index."
            )

        scores, indices = await asyncio.to_thread(
            self._index.search,
            vector,
            min(top_k, len(self._doc_ids)),
        )
        hits: list[SearchHit] = []
        for rank, (score, position) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            position = int(position)
            if position < 0:
                continue
            doc = self._docs_by_id[self._doc_ids[position]]
            hits.append(
                SearchHit(
                    doc_id=doc.doc_id,
                    system_id=self.source_id,
                    summary=doc.summary,
                    keywords=doc.keywords,
                    metadata=doc.metadata,
                    vector_score=float(score),
                    vector_rank=rank,
                )
            )
        return hits
