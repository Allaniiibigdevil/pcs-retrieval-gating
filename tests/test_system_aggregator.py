import pytest

from app.decision.system_aggregator import (
    SystemAggregator,
    reciprocal_rank_fusion_score,
)
from app.schemas.search import SearchHit


def hit(
    doc_id: str,
    system_id: str,
    *,
    bm25_rank: int | None = None,
    vector_rank: int | None = None,
    bm25_score: float | None = None,
    vector_score: float | None = None,
    matched_keywords: list[str] | None = None,
) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        system_id=system_id,
        summary=f"{doc_id} summary",
        keywords=matched_keywords or [],
        bm25_rank=bm25_rank,
        vector_rank=vector_rank,
        bm25_score=bm25_score,
        vector_score=vector_score,
        metadata={"matched_keywords": matched_keywords or []},
    )


def test_rrf_score_uses_available_es_and_vector_ranks() -> None:
    shared = hit("shared", "memo", bm25_rank=1, vector_rank=2)
    es_only = hit("es", "memo", bm25_rank=3)
    no_rank = hit("none", "memo")

    assert reciprocal_rank_fusion_score(shared, 20) == pytest.approx(1 / 21 + 1 / 22)
    assert reciprocal_rank_fusion_score(es_only, 20) == pytest.approx(1 / 23)
    assert reciprocal_rank_fusion_score(no_rank, 20) == 0.0


def test_global_rrf_top_docs_select_their_systems() -> None:
    aggregator = SystemAggregator(rrf_k=20, top_n_docs=2)
    decisions = aggregator.aggregate(
        [
            hit("shared", "memo", bm25_rank=1, vector_rank=2),
            hit("vector_first", "album", vector_rank=1),
            hit("es_third", "todo", bm25_rank=3),
        ]
    )

    by_system = {decision.system_id: decision for decision in decisions}
    assert by_system["memo"].rrf_score == pytest.approx(1 / 21 + 1 / 22)
    assert by_system["memo"].selected is True
    assert by_system["album"].rrf_score == pytest.approx(1 / 21)
    assert by_system["album"].selected is True
    assert by_system["todo"].rrf_score == pytest.approx(1 / 23)
    assert by_system["todo"].selected is False


def test_system_score_is_best_document_not_sum() -> None:
    aggregator = SystemAggregator(rrf_k=20, top_n_docs=1)
    decisions = aggregator.aggregate(
        [
            hit("es_first", "memo", bm25_rank=1),
            hit("vector_second", "memo", vector_rank=2),
        ]
    )

    assert len(decisions) == 1
    assert decisions[0].selected is True
    assert decisions[0].rrf_score == pytest.approx(1 / 21, abs=1e-6)
    assert [doc.doc_id for doc in decisions[0].evidence_docs] == [
        "es_first",
        "vector_second",
    ]


def test_system_aggregator_limits_evidence_documents() -> None:
    aggregator = SystemAggregator(
        rrf_k=20,
        top_n_docs=10,
        evidence_docs_per_system=2,
    )
    decisions = aggregator.aggregate(
        [hit(f"doc-{rank}", "memo", bm25_rank=rank) for rank in range(1, 5)]
    )

    assert [doc.doc_id for doc in decisions[0].evidence_docs] == [
        "doc-1",
        "doc-2",
    ]


def test_system_aggregator_exposes_doc_scores_keywords_and_highlight() -> None:
    aggregator = SystemAggregator(rrf_k=20, top_n_docs=10)
    source = hit(
        "allergy",
        "notepad",
        bm25_score=3.0,
        bm25_rank=1,
        vector_score=0.8,
        vector_rank=2,
        matched_keywords=["海鲜过敏"],
    )
    source.metadata["highlight"] = {
        "summary": ["记录了用户对<em>海鲜</em>过敏"],
        "keywords": ["<em>海鲜过敏</em>"],
    }

    evidence = aggregator.aggregate([source])[0].evidence_docs[0]

    assert evidence.keywords == ["海鲜过敏"]
    assert evidence.matched_keywords == ["海鲜过敏"]
    assert evidence.rrf_score == pytest.approx(1 / 21 + 1 / 22, abs=1e-6)
    assert evidence.rrf_rank == 1
    assert evidence.highlight == {
        "summary": ["记录了用户对<em>海鲜</em>过敏"],
        "keywords": ["<em>海鲜过敏</em>"],
    }


def test_system_aggregator_exposes_global_rrf_rank() -> None:
    aggregator = SystemAggregator(rrf_k=20, top_n_docs=10)
    decisions = aggregator.aggregate(
        [
            hit("shared", "memo", bm25_rank=1, vector_rank=2),
            hit("vector_first", "album", vector_rank=1),
            hit("es_third", "todo", bm25_rank=3),
        ]
    )

    ranks = {
        doc.doc_id: doc.rrf_rank
        for decision in decisions
        for doc in decision.evidence_docs
    }
    assert ranks == {"shared": 1, "vector_first": 2, "es_third": 3}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rrf_k": 0},
        {"top_n_docs": 0},
        {"evidence_docs_per_system": 0},
    ],
)
def test_system_aggregator_rejects_invalid_configuration(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SystemAggregator(**kwargs)
