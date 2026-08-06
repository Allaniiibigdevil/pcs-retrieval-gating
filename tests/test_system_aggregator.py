from app.decision.system_aggregator import SystemAggregator
from app.schemas.search import SearchHit


def _aggregator(threshold: float = 0.6, evidence: int = 3) -> SystemAggregator:
    return SystemAggregator(
        source_id="memo",
        selection_threshold=threshold,
        evidence_docs_per_system=evidence,
        es_score_weight=0.55,
        agreement_weight=0.20,
        semantic_match_threshold=0.30,
        lexical_match_threshold=0.30,
    )


def _hit(
    doc_id: str,
    *,
    vector_score: float | None = None,
    bm25_score: float | None = None,
    bm25_score_norm: float | None = None,
) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id="memo",
        summary=f"{doc_id} summary",
        vector_score=vector_score,
        bm25_score=bm25_score,
        bm25_score_norm=bm25_score_norm,
    )


def test_vector_score_controls_source_selection() -> None:
    decision = _aggregator(threshold=0.7).aggregate([_hit("strong", vector_score=0.82)])

    assert decision is not None
    assert decision.confidence == 0.82
    assert decision.selected is True


def test_normalized_es_score_is_used_directly() -> None:
    decision = _aggregator(threshold=0.4).aggregate(
        [_hit("bm25", bm25_score=10.0, bm25_score_norm=1.0)]
    )

    assert decision is not None
    assert decision.confidence == 0.55
    assert decision.selected is True


def test_agreement_boost_requires_both_signals() -> None:
    semantic_only = _aggregator().aggregate([_hit("semantic", vector_score=0.4)])
    combined = _aggregator().aggregate(
        [
            _hit(
                "combined",
                vector_score=0.4,
                bm25_score=10.0,
                bm25_score_norm=1.0,
            )
        ]
    )

    assert semantic_only is not None and semantic_only.confidence == 0.4
    assert combined is not None and combined.confidence == 0.6765
    assert combined.selected is True


def test_aggregator_limits_and_exposes_evidence() -> None:
    source = _hit("allergy", bm25_score=3.0, bm25_score_norm=1.0)
    source.keywords = ["海鲜过敏"]
    source.metadata = {
        "matched_keywords": ["海鲜过敏"],
        "matched_queries": ["海鲜能不能吃", "海鲜过敏"],
        "highlight": {
            "summary": ["记录了用户对<em>海鲜</em>过敏"],
            "keywords": ["<em>海鲜过敏</em>"],
        },
    }
    decision = _aggregator(threshold=0.1, evidence=1).aggregate(
        [source, _hit("other", vector_score=0.2)]
    )

    assert decision is not None
    assert len(decision.evidence_docs) == 1
    evidence = decision.evidence_docs[0]
    assert evidence.doc_id == "allergy"
    assert evidence.matched_queries == ["海鲜能不能吃", "海鲜过敏"]
    assert evidence.highlight["keywords"] == ["<em>海鲜过敏</em>"]
