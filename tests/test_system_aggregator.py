import numpy as np

from app.decision.gating_features import FEATURE_NAMES
from app.decision.system_aggregator import SystemAggregator
from app.ml.mil_mlp import MilMlpGatingModel
from app.schemas.search import SearchHit


def hit(
    doc_id: str,
    system_id: str,
    vector_score: float | None = None,
    bm25_rank: int | None = None,
    bm25_score: float | None = None,
    matched_keywords: list[str] | None = None,
) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id=system_id,
        summary=f"{doc_id} summary",
        keywords=matched_keywords or [],
        vector_score=vector_score,
        bm25_score=bm25_score,
        bm25_rank=bm25_rank,
        metadata={"matched_keywords": matched_keywords or []},
    )


def test_system_aggregator_selection_thresholds() -> None:
    aggregator = SystemAggregator(selection_threshold=0.7)
    decisions = aggregator.aggregate(
        [
            hit("strong", "memo_system", vector_score=0.82),
            hit("medium", "album_system", vector_score=0.6),
            hit("weak", "todo_system", vector_score=0.3),
        ],
        max_systems=5,
    )

    by_system = {decision.system_id: decision for decision in decisions}
    assert by_system["memo_system"].confidence == 0.82
    assert by_system["memo_system"].selected is True
    assert by_system["album_system"].confidence == 0.6
    assert by_system["album_system"].selected is False
    assert by_system["todo_system"].selected is False


def test_system_aggregator_limits_evidence_sorts_and_applies_max_systems() -> None:
    aggregator = SystemAggregator(selection_threshold=0.8)
    docs = [
        hit("a1", "a", vector_score=0.9),
        hit("a2", "a", vector_score=0.8),
        hit("a3", "a", vector_score=0.7),
        hit("a4", "a", vector_score=0.6),
        hit("b1", "b", vector_score=0.95),
        hit("c1", "c", vector_score=0.5),
    ]

    decisions = aggregator.aggregate(docs, max_systems=2)

    assert [decision.system_id for decision in decisions] == ["b", "a"]
    assert len(decisions) == 2
    a_decision = next(decision for decision in decisions if decision.system_id == "a")
    assert [doc.doc_id for doc in a_decision.evidence_docs] == ["a1", "a2", "a3"]


def test_es_score_is_normalized_within_the_query() -> None:
    aggregator = SystemAggregator(selection_threshold=0.4)

    decisions = aggregator.aggregate(
        [
            hit("bm25_top", "memo_system", bm25_score=10.0, bm25_rank=1),
            hit("bm25_half", "album_system", bm25_score=5.0, bm25_rank=2),
        ],
        max_systems=5,
    )

    by_system = {decision.system_id: decision for decision in decisions}
    assert by_system["memo_system"].confidence == 0.55
    assert by_system["memo_system"].selected is True
    assert by_system["album_system"].confidence == 0.275
    assert by_system["album_system"].selected is False


def test_agreement_boost_requires_semantic_and_lexical_signals() -> None:
    aggregator = SystemAggregator(selection_threshold=0.6)

    semantic_only = aggregator.aggregate(
        [hit("semantic_only", "memo_system", vector_score=0.4)],
        max_systems=5,
    )
    semantic_and_lexical = aggregator.aggregate(
        [
            hit(
                "bm25_with_semantic",
                "memo_system",
                vector_score=0.4,
                bm25_score=10.0,
                bm25_rank=1,
            )
        ],
        max_systems=5,
    )

    assert semantic_only[0].confidence == 0.4
    assert semantic_only[0].selected is False
    assert semantic_and_lexical[0].confidence == 0.6765
    assert semantic_and_lexical[0].selected is True


def test_agreement_boost_requires_both_minimum_thresholds() -> None:
    aggregator = SystemAggregator(selection_threshold=0.6)

    decisions = aggregator.aggregate(
        [
            hit(
                "strong_es_weak_vector",
                "memo_system",
                vector_score=0.2,
                bm25_score=10.0,
                bm25_rank=1,
            )
        ],
        max_systems=5,
    )

    assert decisions[0].confidence == 0.55
    assert decisions[0].selected is False


def test_system_aggregator_exposes_doc_keywords_and_es_highlight() -> None:
    aggregator = SystemAggregator(selection_threshold=0.1)
    source = hit(
        "allergy",
        "notepad",
        bm25_score=3.0,
        bm25_rank=1,
        matched_keywords=["海鲜过敏"],
    )
    source.metadata["highlight"] = {
        "summary": ["记录了用户对<em>海鲜</em>过敏"],
        "keywords": ["<em>海鲜过敏</em>"],
    }

    decision = aggregator.aggregate([source], max_systems=1)[0]

    assert decision.evidence_docs[0].keywords == ["海鲜过敏"]
    assert decision.evidence_docs[0].matched_keywords == ["海鲜过敏"]
    assert decision.evidence_docs[0].highlight == {
        "summary": ["记录了用户对<em>海鲜</em>过敏"],
        "keywords": ["<em>海鲜过敏</em>"],
    }


def test_max_systems_does_not_drop_sources_above_threshold() -> None:
    aggregator = SystemAggregator(selection_threshold=0.5)
    decisions = aggregator.aggregate(
        [
            hit("a", "source_a", vector_score=0.9),
            hit("b", "source_b", vector_score=0.8),
            hit("c", "source_c", vector_score=0.7),
        ],
        max_systems=1,
    )
    assert len(decisions) == 3
    assert all(decision.selected for decision in decisions)


def test_mil_mlp_uses_the_strongest_representative_document() -> None:
    input_weights = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
    input_weights[0, FEATURE_NAMES.index("vector_score_norm")] = 1.0
    model = MilMlpGatingModel(
        input_weights=input_weights,
        input_bias=np.zeros(1, dtype=float),
        output_weights=np.array([10.0]),
        output_bias=-5.0,
        feature_names=list(FEATURE_NAMES),
    )
    aggregator = SystemAggregator(
        scorer="mil_mlp",
        model=model,
        selection_threshold=0.8,
        require_calibration=False,
    )

    decisions = aggregator.aggregate(
        [
            hit("memo-weak", "memo", vector_score=0.2),
            hit("memo-strong", "memo", vector_score=0.9),
            hit("album-medium", "album", vector_score=0.4),
        ],
        query_text="query",
    )

    by_system = {decision.system_id: decision for decision in decisions}
    assert by_system["memo"].selected is True
    assert by_system["memo"].trigger_doc_id == "memo-strong"
    assert by_system["memo"].confidence > 0.98
    assert by_system["album"].selected is False
