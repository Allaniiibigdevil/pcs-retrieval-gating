import json
import math
from pathlib import Path

import pytest

from app.embedding.embedding_service import MockEmbeddingService
from app.offline.local_index_builder import LocalIndexBuilder
from app.retrieval.local_faiss_retriever import LocalFaissRetriever
from app.schemas.doc import SourceDoc
from app.source_registry import SourceRegistry
from app.storage.local_artifact_store import LocalArtifactStore


class CountingEmbeddingService(MockEmbeddingService):
    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.calls: list[list[str]] = []

    async def embed_batch(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return await super().embed_batch(texts, show_progress=show_progress)


def _memo_doc() -> SourceDoc:
    return SourceDoc(
        doc_id="memo_doc_001",
        system_id="memo",
        summary="The memo records a shanghai business trip meeting plan.",
        keywords=["memo", "shanghai", "meeting"],
    )


def _source(index: str, source_id: str, root) -> dict:
    return {
        "es_index": index,
        "faiss_index_path": str(root / source_id / "faiss.index"),
        "faiss_doc_ids_path": str(root / source_id / "faiss_doc_ids.json"),
        "es_top_k": 10,
        "faiss_top_k": 10,
        "evidence_docs_per_system": 2,
        "selection_threshold": 0.6,
        "es_score_weight": 0.55,
        "agreement_weight": 0.2,
        "semantic_match_threshold": 0.3,
        "lexical_match_threshold": 0.3,
    }


def _registry(tmp_path) -> SourceRegistry:
    path = tmp_path / "sources.json"
    faiss_root = tmp_path / "faiss"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "memo": _source("pcs-memo", "memo", faiss_root),
                    "album": _source("pcs-album", "album", faiss_root),
                }
            }
        ),
        encoding="utf-8",
    )
    return SourceRegistry.from_path(path)


@pytest.mark.asyncio
async def test_builder_uses_one_legacy_embedding_call_for_one_source(tmp_path) -> None:
    registry = _registry(tmp_path)
    source = registry.require("memo")
    store = LocalArtifactStore(artifact_dir=Path(source.faiss_index_path).parent)
    embedding = CountingEmbeddingService(dim=16)

    result = await LocalIndexBuilder(
        source=source,
        artifact_store=store,
        embedding_service=embedding,
    ).build([_memo_doc()])

    assert result.doc_count == 1
    assert result.source_count == 1
    assert result.embedding_dim == 16
    assert len(embedding.calls) == 1
    assert len(embedding.calls[0]) == 1
    assert store.docs_path.exists()
    assert store.manifest_path.exists()

    index, doc_ids = store.load_faiss(
        index_path=source.faiss_index_path,
        doc_ids_path=source.faiss_doc_ids_path,
    )
    assert doc_ids == ["memo_doc_001"]
    assert int(index.ntotal) == 1

    hits = await LocalFaissRetriever(
        source_id="memo",
        faiss_index_path=source.faiss_index_path,
        faiss_doc_ids_path=source.faiss_doc_ids_path,
        artifact_store=store,
        embedding_service=embedding,
    ).search("shanghai trip meeting", top_k=5)
    assert [hit.doc_id for hit in hits] == ["memo_doc_001"]
    assert hits[0].vector_score is not None and math.isfinite(hits[0].vector_score)


@pytest.mark.asyncio
async def test_builder_rebuilds_only_selected_source_es_index(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, list[str]]] = []

    class FakeIndexer:
        def __init__(self, index_name: str) -> None:
            self.index_name = index_name

        def rebuild(self, docs: list[SourceDoc]) -> None:
            calls.append((self.index_name, [doc.doc_id for doc in docs]))

    monkeypatch.setattr("app.offline.local_index_builder.LocalElasticsearchIndexer", FakeIndexer)
    registry = _registry(tmp_path)
    source = registry.require("memo")

    await LocalIndexBuilder(
        source=source,
        artifact_store=LocalArtifactStore(artifact_dir=Path(source.faiss_index_path).parent),
        embedding_service=MockEmbeddingService(dim=8),
        index_elasticsearch=True,
    ).build([_memo_doc()])

    assert calls == [("pcs-memo", ["memo_doc_001"])]
