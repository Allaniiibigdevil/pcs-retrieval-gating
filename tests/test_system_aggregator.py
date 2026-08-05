from app.decision.system_aggregator import SystemAggregator
from app.schemas.search import SearchHit


def hit(
    doc_id: str,
    system_id: str,
    vector_score: float | None = None,
    bm25_rank: int | None = None,
    bm25_score: float | None = None,
    bm25_score_norm: float | None = None,
    matched_keywords: list[str] | None = None,
) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id=system_id,
        summary=f"{doc_id} summary",
        keywords=matched_keywords or [],
        vector_score=vector_score,
        bm25_score=bm25_score,
        bm25_score_norm=bm25_score_norm,
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
        ]
    )

    by_system = {decision.system_id: decision for decision in decisions}
    assert by_system["memo_system"].confidence == 0.82
    assert by_system["memo_system"].selected is True
    assert by_system["album_system"].confidence == 0.6
    assert by_system["album_system"].selected is False
    assert by_system["todo_system"].selected is False


def test_system_aggregator_returns_all_systems_and_configurable_evidence() -> None:
    aggregator = SystemAggregator(selection_threshold=0.8, evidence_docs_per_system=2)
    docs = [
        hit("a1", "a", vector_score=0.9),
        hit("a2", "a", vector_score=0.8),
        hit("a3", "a", vector_score=0.7),
        hit("b1", "b", vector_score=0.95),
        hit("c1", "c", vector_score=0.5),
    ]

    decisions = aggregator.aggregate(docs)

    assert [decision.system_id for decision in decisions] == ["b", "a", "c"]
    a_decision = next(decision for decision in decisions if decision.system_id == "a")
    assert [doc.doc_id for doc in a_decision.evidence_docs] == ["a1", "a2"]


def test_normalized_es_score_is_used_directly() -> None:
    aggregator = SystemAggregator(selection_threshold=0.4)

    decisions = aggregator.aggregate(
        [
            hit("bm25_top", "memo_system", bm25_score=10.0, bm25_score_norm=1.0),
            hit("bm25_half", "album_system", bm25_score=5.0, bm25_score_norm=0.5),
        ]
    )

    by_system = {decision.system_id: decision for decision in decisions}
    assert by_system["memo_system"].confidence == 0.55
    assert by_system["memo_system"].selected is True
    assert by_system["album_system"].confidence == 0.275
    assert by_system["album_system"].selected is False


def test_agreement_boost_requires_semantic_and_lexical_signals() -> None:
    aggregator = SystemAggregator(selection_threshold=0.6)

    semantic_only = aggregator.aggregate(
        [hit("semantic_only", "memo_system", vector_score=0.4)]
    )
    semantic_and_lexical = aggregator.aggregate(
        [
            hit(
                "bm25_with_semantic",
                "memo_system",
                vector_score=0.4,
                bm25_score=10.0,
                bm25_score_norm=1.0,
            )
        ]
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
                bm25_score_norm=1.0,
            )
        ]
    )

    assert decisions[0].confidence == 0.55
    assert decisions[0].selected is False


def test_system_aggregator_exposes_query_and_es_evidence() -> None:
    aggregator = SystemAggregator(selection_threshold=0.1)
    source = hit(
        "allergy",
        "notepad",
        bm25_score=3.0,
        bm25_score_norm=1.0,
        bm25_rank=1,
        matched_keywords=["海鲜过敏"],
    )
    source.metadata.update(
        {
            "matched_queries": ["海鲜能不能吃", "海鲜过敏"],
            "highlight": {
                "summary": ["记录了用户对<em>海鲜</em>过敏"],
                "keywords": ["<em>海鲜过敏</em>"],
            },
        }
    )

    decision = aggregator.aggregate([source])[0]

    evidence = decision.evidence_docs[0]
    assert evidence.keywords == ["海鲜过敏"]
    assert evidence.matched_keywords == ["海鲜过敏"]
    assert evidence.matched_queries == ["海鲜能不能吃", "海鲜过敏"]
    assert evidence.bm25_score_norm == 1.0
    assert evidence.highlight == {
        "summary": ["记录了用户对<em>海鲜</em>过敏"],
        "keywords": ["<em>海鲜过敏</em>"],
    }
