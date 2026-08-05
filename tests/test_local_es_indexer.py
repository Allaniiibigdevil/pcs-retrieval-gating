from io import BytesIO
from types import SimpleNamespace
from urllib import error

import pytest

from app.offline.local_es_indexer import (
    ElasticsearchRequestError,
    LocalElasticsearchIndexer,
)


def _indexer_with_settings(**overrides) -> LocalElasticsearchIndexer:
    settings = {
        "LOCAL_ES_ANALYZER": "ik_smart",
        "LOCAL_ES_SEARCH_ANALYZER": "ik_smart",
        "LOCAL_ES_SHARDS": 1,
        "LOCAL_ES_REPLICAS": 0,
    }
    settings.update(overrides)
    indexer = LocalElasticsearchIndexer(base_url="http://localhost:9200", index_name="test")
    indexer.settings = SimpleNamespace(**settings)
    return indexer


def test_mapping_uses_configured_index_and_search_analyzers() -> None:
    mapping = _indexer_with_settings()._mapping()

    assert "analysis" not in mapping["settings"]
    for field in ("summary", "keywords"):
        field_mapping = mapping["mappings"]["properties"][field]
        assert field_mapping["analyzer"] == "ik_smart"
        assert field_mapping["search_analyzer"] == "ik_smart"


def test_request_includes_elasticsearch_error_body(monkeypatch) -> None:
    indexer = _indexer_with_settings(LOCAL_ES_TIMEOUT_SECONDS=10)
    response_body = b'{"error":{"reason":"invalid analyzer settings"},"status":400}'

    def fail_request(*args, **kwargs):
        del args, kwargs
        raise error.HTTPError(
            url="http://localhost:9200/test",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(response_body),
        )

    monkeypatch.setattr("app.offline.local_es_indexer.request.urlopen", fail_request)

    with pytest.raises(ElasticsearchRequestError) as captured:
        indexer._request("PUT", "/test", {"settings": {}})

    assert captured.value.status_code == 400
    assert captured.value.response_body == response_body.decode("utf-8")
    assert "invalid analyzer settings" in str(captured.value)
