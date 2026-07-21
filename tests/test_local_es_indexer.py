from types import SimpleNamespace

from app.offline.local_es_indexer import (
    SYNONYM_FILTER_NAME,
    SYNONYM_SEARCH_ANALYZER_NAME,
    LocalElasticsearchIndexer,
)


def _indexer_with_settings(**overrides) -> LocalElasticsearchIndexer:
    settings = {
        "LOCAL_ES_ANALYZER": "ik_smart",
        "LOCAL_ES_SEARCH_ANALYZER": "ik_smart",
        "LOCAL_ES_SYNONYMS_PATH": "",
        "LOCAL_ES_SYNONYM_TOKENIZER": "",
        "LOCAL_ES_SHARDS": 1,
        "LOCAL_ES_REPLICAS": 0,
    }
    settings.update(overrides)
    indexer = LocalElasticsearchIndexer(base_url="http://localhost:9200", index_name="test")
    indexer.settings = SimpleNamespace(**settings)
    return indexer


def test_mapping_uses_configured_search_analyzer_without_synonyms() -> None:
    mapping = _indexer_with_settings()._mapping()

    assert "analysis" not in mapping["settings"]
    assert mapping["mappings"]["properties"]["summary"]["search_analyzer"] == "ik_smart"


def test_mapping_registers_query_time_synonym_graph() -> None:
    mapping = _indexer_with_settings(
        LOCAL_ES_SYNONYMS_PATH="analysis/pcs_synonyms.txt"
    )._mapping()

    analysis = mapping["settings"]["analysis"]
    assert analysis["filter"][SYNONYM_FILTER_NAME] == {
        "type": "synonym_graph",
        "synonyms_path": "analysis/pcs_synonyms.txt",
        "updateable": True,
        "lenient": False,
    }
    assert analysis["analyzer"][SYNONYM_SEARCH_ANALYZER_NAME] == {
        "type": "custom",
        "tokenizer": "ik_smart",
        "filter": ["lowercase", SYNONYM_FILTER_NAME],
    }
    for field in ("summary", "keywords"):
        field_mapping = mapping["mappings"]["properties"][field]
        assert field_mapping["analyzer"] == "ik_smart"
        assert field_mapping["search_analyzer"] == SYNONYM_SEARCH_ANALYZER_NAME
