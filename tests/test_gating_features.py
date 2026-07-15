import json

import pytest

from app.decision.gating_features import (
    FEATURE_NAMES,
    append_gating_feature_rows,
    build_gating_feature_rows,
    rows_to_mil_bags,
    select_representative_docs,
)
from app.schemas.search import SearchHit


def _nine_representative_candidates() -> list[SearchHit]:
    docs = [
        SearchHit(doc_id=f"e{i}", system_id="memo", bm25_score=11 - i, bm25_rank=i)
        for i in range(1, 5)
    ]
    docs.extend(
        SearchHit(doc_id=f"v{i}", system_id="memo", vector_score=1 - i / 10, vector_rank=i)
        for i in range(1, 5)
    )
    docs.extend(
        SearchHit(
            doc_id=f"j{i}",
            system_id="memo",
            bm25_score=7 - i,
            bm25_rank=i + 3,
            vector_score=0.7 - i / 10,
            vector_rank=i + 3,
        )
        for i in range(1, 5)
    )
    return docs


def test_selects_es_vector_and_rrf_top3_then_deduplicates() -> None:
    representatives = select_representative_docs(_nine_representative_candidates())

    assert {doc.doc_id for doc in representatives} == {
        "e1",
        "e2",
        "e3",
        "v1",
        "v2",
        "v3",
        "j1",
        "j2",
        "j3",
    }


def test_feature_rows_form_one_nine_element_mil_bag(tmp_path) -> None:
    rows = build_gating_feature_rows(
        "query",
        _nine_representative_candidates(),
        task_id="t1",
        labels_by_system={"memo": 1},
    )
    bags = rows_to_mil_bags(rows)

    assert len(rows) == 9
    assert bags[0].features.shape == (9, len(FEATURE_NAMES))
    assert bags[0].label == 1.0
    forbidden = ("source", "system", "capability", "cost", "latency", "count")
    assert all(
        not any(token in feature_name for token in forbidden) for feature_name in FEATURE_NAMES
    )

    output = tmp_path / "samples.jsonl"
    append_gating_feature_rows(rows, output)
    payload = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert payload["doc_id"]
    assert payload["label"] == 1


def test_mil_bag_rejects_duplicate_documents() -> None:
    rows = build_gating_feature_rows(
        "query",
        _nine_representative_candidates(),
        task_id="t1",
        labels_by_system={"memo": 1},
    )

    with pytest.raises(ValueError, match="duplicate doc_id"):
        rows_to_mil_bags([*rows, rows[0]])
