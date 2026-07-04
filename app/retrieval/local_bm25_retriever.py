import numpy as np

from app.retrieval.tokenizer import tokenize
from app.schemas.doc import SourceDoc
from app.schemas.search import SearchHit
from app.storage.local_artifact_store import LocalArtifactStore


class LocalBM25Retriever:
    def __init__(self, artifact_store: LocalArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()
        self._docs_by_id: dict[str, SourceDoc] | None = None
        self._doc_ids: list[str] | None = None
        self._bm25 = None

    def _ensure_loaded(self) -> None:
        if self._bm25 is not None and self._docs_by_id is not None and self._doc_ids is not None:
            return
        docs = self.artifact_store.load_docs()
        self._bm25, _ = self.artifact_store.load_bm25()
        self._docs_by_id = {doc.doc_id: doc for doc in docs}
        self._doc_ids = [doc.doc_id for doc in docs]

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        self._ensure_loaded()
        assert self._bm25 is not None
        assert self._docs_by_id is not None
        assert self._doc_ids is not None

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        if len(scores) == 0:
            return []

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        hits: list[SearchHit] = []
        for rank, index in enumerate(ranked_indices, start=1):
            score = max(0.0, float(scores[index]))
            doc = self._docs_by_id[self._doc_ids[int(index)]]
            hits.append(
                SearchHit(
                    doc_id=doc.doc_id,
                    system_id=doc.system_id,
                    summary=doc.summary,
                    keywords=doc.keywords,
                    metadata=doc.metadata,
                    bm25_score=score,
                    bm25_rank=rank,
                )
            )
        return hits
