import pytest

from app.decision.system_aggregator import SystemAggregator
from app.schemas.search import SearchHit


def hit(
    doc_id: str,
    system_id: str,
    *,
    reranker_score: float,
    reranker_rank: int,
    bm25_score: float | None = None,
    vector_score: float | None = None,
    matched_keywords: list[str] | None = None,
) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id=system_id,
        summary=f"{doc_id} summary",
        keywords=matched_keywords or [],
        bm25_score=bm25_score,
        vector_score=vector_score,
        reranker_score=reranker_score,
        reranker_rank=reranker_rank,
        metadata={"matched_keywords": matched_keywords or []},
    )


def test_one_relevant_document_selects_its_system() -> None:
    aggregator = SystemAggregator(score_threshold=0.7)
    decisions = aggregator.aggregate(
        [
            hit("relevant", "memo", reranker_score=0.91, reranker_rank=1),
            hit("irrelevant", "memo", reranker_score=0.12, reranker_rank=3),
            hit("album", "album", reranker_score=0.69, reranker_rank=2),
        ]
    )

    by_system = {decision.system_id: decision for decision in decisions}
    assert by_system["memo"].selected is True
    assert by_system["memo"].reranker_score == 0.91
    assert by_system["album"].selected is False
    assert by_system["album"].reranker_score == 0.69


def test_system_score_is_best_document_not_sum() -> None:
    aggregator = SystemAggregator(score_threshold=0.8)
    decisions = aggregator.aggregate(
        [
            hit("first", "memo", reranker_score=0.61, reranker_rank=1),
            hit("second", "memo", reranker_score=0.60, reranker_rank=2),
        ]
    )

    assert decisions[0].selected is False
    assert decisions[0].reranker_score == 0.61


def test_system_aggregator_limits_evidence_documents() -> None:
    aggregator = SystemAggregator(
        score_threshold=0.5,
        evidence_docs_per_system=3,
    )
    decisions = aggregator.aggregate(
        [
            hit(
                f"doc-{rank}",
                "memo",
                reranker_score=1.0 - rank / 10,
                reranker_rank=rank,
            )
            for rank in range(1, 5)
        ]
    )

    assert [doc.doc_id for doc in decisions[0].evidence_docs] == [
        "doc-1",
        "doc-2",
        "doc-3",
    ]


def test_system_aggregator_exposes_retrieval_and_reranker_signals() -> None:
    aggregator = SystemAggregator(score_threshold=0.5)
    source = hit(
        "allergy",
        "notepad",
        reranker_score=0.93,
        reranker_rank=1,
        bm25_score=3.0,
        vector_score=0.8,
        matched_keywords=["海鲜过敏"],
    )
    source.metadata["highlight"] = {
        "summary": ["记录了用户对<em>海鲜</em>过敏"],
        "keywords": ["<em>海鲜过敏</em>"],
    }

    evidence = aggregator.aggregate([source])[0].evidence_docs[0]

    assert evidence.keywords == ["海鲜过敏"]
    assert evidence.matched_keywords == ["海鲜过敏"]
    assert evidence.bm25_score == 3.0
    assert evidence.vector_score == 0.8
    assert evidence.reranker_score == 0.93
    assert evidence.reranker_rank == 1
    assert evidence.highlight == {
        "summary": ["记录了用户对<em>海鲜</em>过敏"],
        "keywords": ["<em>海鲜过敏</em>"],
    }


@pytest.mark.parametrize(
    ("score_threshold", "evidence_docs_per_system"),
    [
        (-0.1, 3),
        (1.1, 3),
        (0.5, 0),
    ],
)
def test_system_aggregator_rejects_invalid_configuration(
    score_threshold: float,
    evidence_docs_per_system: int,
) -> None:
    with pytest.raises(ValueError):
        SystemAggregator(
            score_threshold=score_threshold,
            evidence_docs_per_system=evidence_docs_per_system,
        )
