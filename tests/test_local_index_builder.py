import json
import math

import pytest

from app.embedding.embedding_service import MockEmbeddingService
from app.offline.local_index_builder import LocalIndexBuilder
from app.retrieval.local_faiss_retriever import LocalFaissRetriever
from app.schemas.doc import SourceDoc
from app.source_registry import SourceRegistry
from app.storage.local_artifact_store import LocalArtifactStore


def _docs() -> list[SourceDoc]:
    return [
        SourceDoc(
            doc_id="memo_doc_001",
            system_id="memo",
            summary="The memo records a shanghai business trip meeting plan.",
            keywords=["memo", "shanghai", "meeting"],
        ),
        SourceDoc(
            doc_id="album_doc_001",
            system_id="album",
            summary="The album contains meeting photos from the shanghai trip.",
            keywords=["album", "photo", "meeting"],
        ),
    ]


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
                    "empty": _source("pcs-empty", "empty", faiss_root),
                }
            }
        ),
        encoding="utf-8",
    )
    return SourceRegistry.from_path(path)


def _store(tmp_path) -> LocalArtifactStore:
    return LocalArtifactStore(artifact_dir=tmp_path / "shared")


@pytest.mark.asyncio
async def test_builder_outputs_one_faiss_index_per_source(tmp_path) -> None:
    store = _store(tmp_path)
    registry = _registry(tmp_path)
    embedding = MockEmbeddingService(dim=16)
    result = await LocalIndexBuilder(
        artifact_store=store,
        embedding_service=embedding,
        source_registry=registry,
    ).build(_docs())

    assert result.doc_count == 2
    assert result.source_count == 3
    assert result.embedding_dim == 16
    assert store.docs_path.exists()
    assert store.manifest_path.exists()

    memo = registry.require("memo")
    album = registry.require("album")
    empty = registry.require("empty")
    memo_index, memo_ids = store.load_faiss(
        index_path=memo.faiss_index_path,
        doc_ids_path=memo.faiss_doc_ids_path,
    )
    album_index, album_ids = store.load_faiss(
        index_path=album.faiss_index_path,
        doc_ids_path=album.faiss_doc_ids_path,
    )
    empty_index, empty_ids = store.load_faiss(
        index_path=empty.faiss_index_path,
        doc_ids_path=empty.faiss_doc_ids_path,
    )

    assert memo.faiss_index_path != album.faiss_index_path
    assert memo_ids == ["memo_doc_001"]
    assert album_ids == ["album_doc_001"]
    assert empty_ids == []
    assert int(memo_index.ntotal) == 1
    assert int(album_index.ntotal) == 1
    assert int(empty_index.ntotal) == 0

    hits = await LocalFaissRetriever(
        source_id="memo",
        faiss_index_path=memo.faiss_index_path,
        faiss_doc_ids_path=memo.faiss_doc_ids_path,
        artifact_store=store,
        embedding_service=embedding,
    ).search("shanghai trip meeting", top_k=5)
    assert [hit.doc_id for hit in hits] == ["memo_doc_001"]
    assert hits[0].vector_score is not None and math.isfinite(hits[0].vector_score)


@pytest.mark.asyncio
async def test_builder_rebuilds_every_es_index_including_empty_sources(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, list[str]]] = []

    class FakeIndexer:
        def __init__(self, index_name: str) -> None:
            self.index_name = index_name

        def rebuild(self, docs: list[SourceDoc]) -> None:
            calls.append((self.index_name, [doc.doc_id for doc in docs]))

    monkeypatch.setattr("app.offline.local_index_builder.LocalElasticsearchIndexer", FakeIndexer)
    await LocalIndexBuilder(
        artifact_store=_store(tmp_path),
        embedding_service=MockEmbeddingService(dim=8),
        index_elasticsearch=True,
        source_registry=_registry(tmp_path),
    ).build(_docs())

    assert calls == [
        ("pcs-memo", ["memo_doc_001"]),
        ("pcs-album", ["album_doc_001"]),
        ("pcs-empty", []),
    ]
