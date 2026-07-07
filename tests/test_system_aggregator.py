from app.decision.system_aggregator import SystemAggregator
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
    aggregator = SystemAggregator(selection_threshold=0.5)
    decisions = aggregator.aggregate(
        [
            hit("strong", "memo_system", vector_score=0.82),
            hit("medium", "album_system", vector_score=0.6),
            hit("weak", "todo_system", vector_score=0.3),
        ],
        max_systems=5,
    )

    by_system = {decision.system_id: decision for decision in decisions}
    assert by_system["memo_system"].selected is True
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


def test_bm25_score_uses_query_level_normalization() -> None:
    aggregator = SystemAggregator(selection_threshold=0.3)

    decisions = aggregator.aggregate(
        [
            hit("bm25_top", "memo_system", bm25_score=10.0, bm25_rank=1),
            hit("bm25_half", "album_system", bm25_score=5.0, bm25_rank=2),
        ],
        max_systems=5,
    )

    by_system = {decision.system_id: decision for decision in decisions}
    assert by_system["memo_system"].confidence == 0.35
    assert by_system["memo_system"].selected is True
    assert by_system["album_system"].confidence == 0.175
    assert by_system["album_system"].selected is False


def test_keyword_boost_can_cross_selection_threshold() -> None:
    aggregator = SystemAggregator(selection_threshold=0.4)

    without_keywords = aggregator.aggregate(
        [hit("bm25_only", "memo_system", bm25_score=1.0, bm25_rank=1)],
        max_systems=5,
    )
    with_keywords = aggregator.aggregate(
        [
            hit(
                "bm25_with_keywords",
                "memo_system",
                bm25_score=1.0,
                bm25_rank=1,
                matched_keywords=["shanghai", "trip", "meeting"],
            )
        ],
        max_systems=5,
    )

    assert without_keywords[0].confidence == 0.35
    assert without_keywords[0].selected is False
    assert with_keywords[0].confidence == 0.41
    assert with_keywords[0].selected is True
