import json
from types import SimpleNamespace

import pytest

from app.source_registry import SourceRegistry


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        LOCAL_ES_INDEX="pcs",
        ES_TOP_K_DOCS=50,
        FAISS_TOP_K_DOCS=40,
        EVIDENCE_DOCS_PER_SYSTEM=3,
        SYSTEM_SELECTION_THRESHOLD=0.60,
        ES_SCORE_WEIGHT=0.55,
        AGREEMENT_WEIGHT=0.20,
        SEMANTIC_MATCH_THRESHOLD=0.30,
        LEXICAL_MATCH_THRESHOLD=0.30,
        FAISS_PREFERRED_SCORE_THRESHOLD=0.60,
        FAISS_MIN_SCORE_THRESHOLD=0.30,
        FAISS_TARGET_HITS=10,
        RERANKER_SCORE_THRESHOLD=0.50,
    )


def test_source_registry_resolves_source_specific_and_default_values(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "photo": {
                        "es_index": "pcs-photo",
                        "faiss_top_k": 75,
                        "selection_threshold": 0.52,
                    },
                    "notepad": {"es_index": "pcs-notepad"},
                }
            }
        ),
        encoding="utf-8",
    )

    registry = SourceRegistry.from_path(path, settings=_settings())

    assert [source.source_id for source in registry.enabled_sources] == ["photo", "notepad"]
    assert registry.require("photo").faiss_top_k == 75
    assert registry.require("photo").selection_threshold == 0.52
    assert registry.require("notepad").es_top_k == 50
    assert registry.require("notepad").reranker_score_threshold == 0.50


def test_source_registry_rejects_duplicate_elasticsearch_indices(tmp_path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "photo": {"es_index": "pcs-shared"},
                    "notepad": {"es_index": "pcs-shared"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="distinct Elasticsearch index"):
        SourceRegistry.from_path(path, settings=_settings())
