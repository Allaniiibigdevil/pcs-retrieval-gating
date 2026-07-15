import json

import pytest
from pydantic import ValidationError

from app.decision.gating_features import (
    FEATURE_NAMES,
    GatingCase,
    append_gating_case,
    build_gating_case,
    cases_to_mil_bags,
    select_representative_docs,
)
from app.offline.train_gating_model import load_cases
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


def test_one_jsonl_line_contains_a_case_with_system_bags(tmp_path) -> None:
    candidates = [
        *_nine_representative_candidates(),
        SearchHit(
            doc_id="album-1",
            system_id="album",
            summary="京都红色寺庙照片",
            keywords=["京都", "寺庙"],
            vector_score=0.6,
            vector_rank=10,
        ),
    ]
    case = build_gating_case(
        "去年京都的红色寺庙",
        candidates,
        task_id="t1",
        labels_by_system={"memo": 1, "album": 0},
    )
    bags = cases_to_mil_bags([case])

    systems = {system.system_id: system for system in case.systems}
    assert case.task_id == "t1"
    assert systems["memo"].label == 1
    assert len(systems["memo"].docs) == 9
    assert systems["album"].label == 0
    assert len(systems["album"].docs) == 1
    assert len(bags) == 2
    assert next(bag for bag in bags if bag.system_id == "memo").features.shape == (
        9,
        len(FEATURE_NAMES),
    )

    output = tmp_path / "cases.jsonl"
    append_gating_case(case, output)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert "schema_version" not in payload
    assert payload["task_id"] == "t1"
    assert payload["systems"][0]["label"] in (0, 1)
    assert "label" not in payload["systems"][0]["docs"][0]
    assert load_cases(output) == [case]

    assert FEATURE_NAMES == [
        "vector_score_norm",
        "vector_rank_score",
        "es_score_query_norm",
        "es_rank_score",
        "rrf_score",
    ]
    assert "same_doc_hit_by_both" not in payload["systems"][0]["docs"][0]
    assert "matched_keyword_ratio" not in payload["systems"][0]["docs"][0]

    forbidden = ("source", "system", "capability", "cost", "latency", "count")
    assert all(
        not any(token in feature_name for token in forbidden) for feature_name in FEATURE_NAMES
    )


def test_gating_case_rejects_duplicate_documents() -> None:
    case = build_gating_case(
        "query",
        _nine_representative_candidates(),
        task_id="t1",
        labels_by_system={"memo": 1},
    )
    payload = case.model_dump()
    docs = payload["systems"][0]["docs"]
    payload["systems"][0]["docs"] = [docs[0], docs[1], docs[0]]

    with pytest.raises(ValidationError, match="duplicate doc_id"):
        GatingCase.model_validate(payload)


def test_unlabeled_system_bags_are_skipped() -> None:
    case = build_gating_case(
        "query",
        _nine_representative_candidates(),
        task_id="t1",
    )

    with pytest.raises(ValueError, match="no labeled MIL bags"):
        cases_to_mil_bags([case])
