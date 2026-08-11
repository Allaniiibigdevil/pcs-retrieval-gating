import json
from pathlib import Path

import pytest

from app.embedding.embedding_service import MockEmbeddingService
from app.offline.local_index_builder import LocalIndexBuilder
from app.schemas.doc import SourceDoc
from app.source_registry import SourceRegistry
from app.storage.local_artifact_store import LocalArtifactStore


class CountingEmbeddingService(MockEmbeddingService):
    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.calls: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return await super().embed_batch(texts)


def _doc(doc_id: str, system_id: str) -> SourceDoc:
    return SourceDoc(
        doc_id=doc_id,
        system_id=system_id,
        summary=f"summary for {system_id}",
        keywords=[system_id],
    )


def _source(index: str, source_id: str, root: Path) -> dict:
    return {
        "es_index": index,
        "faiss_index_path": str(root / source_id / "faiss.index"),
        "faiss_doc_ids_path": str(root / source_id / "faiss_doc_ids.json"),
        "es_top_k": 10,
        "faiss_top_k": 10,
        "evidence_docs_per_system": 2,
        "faiss_preferred_score_threshold": 0.60,
        "faiss_min_score_threshold": 0.30,
        "faiss_target_hits": 10,
        "selection_threshold": 0.6,
        "es_score_weight": 0.55,
        "agreement_weight": 0.2,
        "semantic_match_threshold": 0.3,
        "lexical_match_threshold": 0.3,
    }


def _registry(tmp_path: Path) -> SourceRegistry:
    path = tmp_path / "sources.json"
    root = tmp_path / "faiss"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "memo": _source("pcs-memo", "memo", root),
                    "album": _source("pcs-album", "album", root),
                    "photo": _source("pcs-photo", "photo", root),
                }
            }
        ),
        encoding="utf-8",
    )
    return SourceRegistry.from_path(path)


@pytest.mark.asyncio
async def test_builder_builds_one_keyword_and_one_vector_index_from_all_sources_once(
    monkeypatch,
    tmp_path,
) -> None:
    keyword_calls: list[tuple[str, list[tuple[str, str]]]] = []
    vector_calls: list[tuple[list[tuple[str, str]], tuple[int, int]]] = []

    class FakeKeywordIndexer:
        def __init__(self, index_name: str) -> None:
            self.index_name = index_name

        def rebuild(self, docs: list[SourceDoc]) -> None:
            keyword_calls.append(
                (self.index_name, [(doc.system_id, doc.doc_id) for doc in docs])
            )

    class FakeVectorIndexer:
        def rebuild(self, docs, embeddings) -> None:
            vector_calls.append(
                (
                    [(doc.system_id, doc.doc_id) for doc in docs],
                    tuple(embeddings.shape),
                )
            )

    monkeypatch.setattr(
        "app.offline.local_index_builder.LocalElasticsearchIndexer",
        FakeKeywordIndexer,
    )
    monkeypatch.setattr(
        "app.offline.local_index_builder.LocalElasticsearchVectorIndexer",
        FakeVectorIndexer,
    )

    docs = [
        _doc("memo-1", "memo"),
        _doc("album-1", "album"),
        _doc("photo-1", "photo"),
    ]
    embedding = CountingEmbeddingService(dim=16)
    store = LocalArtifactStore(artifact_dir=tmp_path / "artifacts")

    result = await LocalIndexBuilder(
        artifact_store=store,
        embedding_service=embedding,
        source_registry=_registry(tmp_path),
    ).build(docs)

    expected_docs = [
        ("memo", "memo-1"),
        ("album", "album-1"),
        ("photo", "photo-1"),
    ]
    assert keyword_calls == [("pcs_retrieval_keywords", expected_docs)]
    assert vector_calls == [(expected_docs, (3, 16))]
    assert len(embedding.calls) == 1
    assert len(embedding.calls[0]) == 3
    assert result.doc_count == 3
    assert result.source_count == 3
    assert result.embedding_dim == 16
    assert result.keyword_index == "pcs_retrieval_keywords"
    assert result.vector_index == "pcs_retrieval_vectors"
    assert store.docs_path.exists()
    assert store.manifest_path.exists()
