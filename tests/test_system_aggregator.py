import pytest

from app.decision.system_aggregator import SystemAggregator
from app.schemas.search import SearchHit


def _aggregator() -> SystemAggregator:
    return SystemAggregator(
        source_thresholds={"memo": 0.7, "album": 0.5},
        evidence_docs_per_source={"memo": 2, "album": 1},
    )


def _hit(
    doc_id: str,
    source_id: str,
    score: float,
    rank: int,
) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id=source_id,
        summary=f"{doc_id} summary",
        reranker_score=score,
        reranker_rank=rank,
    )


def test_source_thresholds_and_evidence_limits_are_applied() -> None:
    decisions = _aggregator().aggregate(
        [
            _hit("memo-1", "memo", 0.80, 1),
            _hit("memo-2", "memo", 0.60, 3),
            _hit("memo-3", "memo", 0.50, 4),
            _hit("album-1", "album", 0.55, 2),
        ]
    )

    by_source = {decision.system_id: decision for decision in decisions}
    assert by_source["memo"].selected is True
    assert [doc.doc_id for doc in by_source["memo"].evidence_docs] == ["memo-1", "memo-2"]
    assert by_source["album"].selected is True
    assert [doc.doc_id for doc in by_source["album"].evidence_docs] == ["album-1"]


def test_aggregator_preserves_retrieval_evidence() -> None:
    hit = _hit("allergy", "memo", 0.93, 1)
    hit.metadata = {
        "matched_keywords": ["海鲜过敏"],
        "matched_queries": ["海鲜能不能吃", "海鲜过敏"],
        "highlight": {"keywords": ["<em>海鲜过敏</em>"]},
    }
    evidence = _aggregator().aggregate([hit])[0].evidence_docs[0]

    assert evidence.matched_keywords == ["海鲜过敏"]
    assert evidence.matched_queries == ["海鲜能不能吃", "海鲜过敏"]
    assert evidence.highlight == {"keywords": ["<em>海鲜过敏</em>"]}


def test_aggregator_rejects_unknown_sources_or_unscored_documents() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        _aggregator().aggregate([_hit("x", "unknown", 0.8, 1)])

    with pytest.raises(ValueError, match="missing score or rank"):
        _aggregator().aggregate([SearchHit(doc_id="x", system_id="memo")])
