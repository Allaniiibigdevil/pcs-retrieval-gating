from types import SimpleNamespace

import pytest

from app.retrieval import factory
from app.source_registry import SourceConfig


class RecordingRetriever:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


SOURCE = SourceConfig(
    source_id="photo",
    es_index="pcs_retrieval_photo",
    faiss_index_path="data/artifacts/photo/faiss.index",
    faiss_doc_ids_path="data/artifacts/photo/faiss_doc_ids.json",
    enabled=True,
    es_top_k=50,
    faiss_top_k=50,
    evidence_docs_per_system=3,
    faiss_preferred_score_threshold=0.55,
    faiss_min_score_threshold=0.30,
    faiss_target_hits=10,
    selection_threshold=0.55,
    es_score_weight=0.45,
    agreement_weight=0.20,
    semantic_match_threshold=0.30,
    lexical_match_threshold=0.25,
)


def _clear_factory_caches() -> None:
    factory.build_keyword_retriever.cache_clear()
    factory.build_vector_retriever.cache_clear()


@pytest.fixture(autouse=True)
def clear_factory_caches():
    _clear_factory_caches()
    yield
    _clear_factory_caches()


def test_factory_keeps_legacy_retrievers_available(monkeypatch) -> None:
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: SimpleNamespace(
            KEYWORD_RETRIEVER_MODE="per_source",
            VECTOR_RETRIEVER_BACKEND="faiss",
            LOCAL_ES_KEYWORD_INDEX="pcs_retrieval_keywords",
            LOCAL_ES_VECTOR_INDEX="pcs_retrieval_vectors",
        ),
    )
    monkeypatch.setattr(factory, "LocalElasticsearchRetriever", RecordingRetriever)
    monkeypatch.setattr(factory, "LocalFaissRetriever", RecordingRetriever)

    keyword = factory.build_keyword_retriever(SOURCE)
    vector = factory.build_vector_retriever(SOURCE)

    assert keyword.kwargs["source_id"] == "photo"
    assert keyword.kwargs["index_name"] == "pcs_retrieval_photo"
    assert vector.kwargs["source_id"] == "photo"
    assert vector.kwargs["faiss_index_path"] == "data/artifacts/photo/faiss.index"


def test_factory_can_switch_to_unified_elasticsearch_retrievers(monkeypatch) -> None:
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: SimpleNamespace(
            KEYWORD_RETRIEVER_MODE="unified",
            VECTOR_RETRIEVER_BACKEND="es",
            LOCAL_ES_KEYWORD_INDEX="pcs_retrieval_keywords",
            LOCAL_ES_VECTOR_INDEX="pcs_retrieval_vectors",
        ),
    )
    monkeypatch.setattr(
        factory,
        "LocalElasticsearchUnifiedRetriever",
        RecordingRetriever,
    )
    monkeypatch.setattr(
        factory,
        "LocalElasticsearchVectorRetriever",
        RecordingRetriever,
    )

    keyword = factory.build_keyword_retriever(SOURCE)
    vector = factory.build_vector_retriever(SOURCE)

    assert keyword.kwargs == {
        "source_id": "photo",
        "index_name": "pcs_retrieval_keywords",
    }
    assert vector.kwargs == {
        "source_id": "photo",
        "index_name": "pcs_retrieval_vectors",
    }
