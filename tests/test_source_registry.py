import json

import pytest
from pydantic import ValidationError

from app.source_registry import SourceRegistry


def _source(index: str, slug: str) -> dict:
    return {
        "es_index": index,
        "faiss_index_path": f"data/artifacts/{slug}/faiss.index",
        "faiss_doc_ids_path": f"data/artifacts/{slug}/faiss_doc_ids.json",
        "es_top_k": 50,
        "faiss_top_k": 40,
        "evidence_docs_per_system": 3,
        "faiss_preferred_score_threshold": 0.60,
        "faiss_min_score_threshold": 0.30,
        "faiss_target_hits": 10,
        "reranker_score_threshold": 0.50,
    }


def test_source_registry_loads_self_contained_source_settings(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "photo": {**_source("pcs-photo", "photo"), "faiss_top_k": 75},
                    "notepad": _source("pcs-notepad", "notepad"),
                }
            }
        ),
        encoding="utf-8",
    )

    registry = SourceRegistry.from_path(path)

    assert registry.config_path == path
    assert [source.source_id for source in registry.enabled_sources] == ["photo", "notepad"]
    assert registry.require("photo").faiss_top_k == 75
    assert registry.require("notepad").reranker_score_threshold == 0.50
    assert registry.require("photo").faiss_index_path.endswith("photo/faiss.index")


def test_source_registry_rejects_missing_strategy_settings(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps({"sources": {"photo": {"es_index": "pcs-photo"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        SourceRegistry.from_path(path)


def test_source_registry_rejects_invalid_vector_thresholds(tmp_path) -> None:
    path = tmp_path / "sources.json"
    invalid = _source("pcs-photo", "photo")
    invalid["faiss_min_score_threshold"] = 0.8
    path.write_text(json.dumps({"sources": {"photo": invalid}}), encoding="utf-8")

    with pytest.raises(ValidationError, match="must not exceed"):
        SourceRegistry.from_path(path)


def test_source_registry_rejects_duplicate_elasticsearch_indices(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "photo": _source("pcs-shared", "photo"),
                    "notepad": _source("pcs-shared", "notepad"),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct Elasticsearch index"):
        SourceRegistry.from_path(path)


def test_source_registry_rejects_shared_faiss_artifact_paths(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "photo": _source("pcs-photo", "shared"),
                    "notepad": _source("pcs-notepad", "shared"),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct FAISS artifact paths"):
        SourceRegistry.from_path(path)
