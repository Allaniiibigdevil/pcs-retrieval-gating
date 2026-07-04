import numpy as np

from app.embedding.embedding_service import EmbeddingService, get_embedding_service
from app.schemas.doc import SourceDoc
from app.schemas.search import SearchHit
from app.storage.local_artifact_store import LocalArtifactStore


class LocalFaissRetriever:
    def __init__(
        self,
        artifact_store: LocalArtifactStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self.embedding_service = embedding_service
        self._docs_by_id: dict[str, SourceDoc] | None = None
        self._doc_ids: list[str] | None = None
        self._index = None

    def _ensure_loaded(self) -> None:
        if self._index is not None and self._docs_by_id is not None and self._doc_ids is not None:
            return
        docs = self.artifact_store.load_docs()
        self._index, self._doc_ids = self.artifact_store.load_faiss()
        self._docs_by_id = {doc.doc_id: doc for doc in docs}
        if self.embedding_service is None:
            self.embedding_service = get_embedding_service()

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        self._ensure_loaded()
        assert self._index is not None
        assert self._doc_ids is not None
        assert self._docs_by_id is not None
        assert self.embedding_service is not None

        query_embedding = await self.embedding_service.embed(query)
        vector = np.asarray([query_embedding], dtype="float32")
        scores, indices = self._index.search(vector, min(top_k, len(self._doc_ids)))

        hits: list[SearchHit] = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            if index < 0:
                continue
            doc = self._docs_by_id[self._doc_ids[int(index)]]
            hits.append(
                SearchHit(
                    doc_id=doc.doc_id,
                    system_id=doc.system_id,
                    summary=doc.summary,
                    keywords=doc.keywords,
                    metadata=doc.metadata,
                    vector_score=float(score),
                    vector_rank=rank,
                )
            )
        return hits
