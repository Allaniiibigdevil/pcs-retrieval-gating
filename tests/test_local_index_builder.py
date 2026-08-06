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


def _source(index: str) -> dict:
    return {
        "es_index": index,
        "es_top_k": 10,
        "faiss_top_k": 10,
        "evidence_docs_per_system": 2,
        "faiss_preferred_score_threshold": 0.6,
        "faiss_min_score_threshold": 0.3,
        "faiss_target_hits": 2,
        "reranker_score_threshold": 0.5,
    }


def _registry(tmp_path) -> SourceRegistry:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "memo": _source("pcs-memo"),
                    "album": _source("pcs-album"),
                    "empty": _source("pcs-empty"),
                }
            }
        ),
        encoding="utf-8",
    )
    return SourceRegistry.from_path(path)


def _store(tmp_path) -> LocalArtifactStore:
    return LocalArtifactStore(
        artifact_dir=tmp_path,
        faiss_index_path=tmp_path / "faiss.index",
        faiss_doc_ids_path=tmp_path / "faiss_doc_ids.json",
    )


@pytest.mark.asyncio
async def test_builder_outputs_shared_source_searchable_artifacts(tmp_path) -> None:
    store = _store(tmp_path)
    embedding = MockEmbeddingService(dim=16)
    result = await LocalIndexBuilder(
        artifact_store=store,
        embedding_service=embedding,
        source_registry=_registry(tmp_path),
    ).build(_docs())

    assert result.doc_count == 2
    assert result.source_count == 3
    assert result.embedding_dim == 16
    assert store.docs_path.exists()
    assert store.faiss_path.exists()
    assert store.faiss_doc_ids_path.exists()
    assert store.manifest_path.exists()

    hits = await LocalFaissRetriever(
        source_id="memo",
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
        artifact_store=_store(tmp_path / "artifacts"),
        embedding_service=MockEmbeddingService(dim=8),
        index_elasticsearch=True,
        source_registry=_registry(tmp_path),
    ).build(_docs())

    assert calls == [
        ("pcs-memo", ["memo_doc_001"]),
        ("pcs-album", ["album_doc_001"]),
        ("pcs-empty", []),
    ]
