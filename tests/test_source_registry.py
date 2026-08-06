import json

import pytest
from pydantic import ValidationError

from app.source_registry import SourceRegistry


def _source(index: str) -> dict:
    return {
        "es_index": index,
        "es_top_k": 50,
        "faiss_top_k": 40,
        "evidence_docs_per_system": 3,
        "selection_threshold": 0.60,
        "es_score_weight": 0.55,
        "agreement_weight": 0.20,
        "semantic_match_threshold": 0.30,
        "lexical_match_threshold": 0.30,
    }


def test_source_registry_loads_self_contained_source_settings(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "photo": {**_source("pcs-photo"), "faiss_top_k": 75},
                    "notepad": _source("pcs-notepad"),
                }
            }
        ),
        encoding="utf-8",
    )

    registry = SourceRegistry.from_path(path)

    assert registry.config_path == path
    assert [source.source_id for source in registry.enabled_sources] == ["photo", "notepad"]
    assert registry.require("photo").faiss_top_k == 75
    assert registry.require("notepad").selection_threshold == 0.60


def test_source_registry_rejects_missing_strategy_settings(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps({"sources": {"photo": {"es_index": "pcs-photo"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        SourceRegistry.from_path(path)


def test_source_registry_rejects_duplicate_elasticsearch_indices(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "photo": _source("pcs-shared"),
                    "notepad": _source("pcs-shared"),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct Elasticsearch index"):
        SourceRegistry.from_path(path)
