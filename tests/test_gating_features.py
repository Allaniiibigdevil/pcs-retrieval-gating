import json

import numpy as np

from app.decision.gating_features import append_gating_feature_rows, build_gating_feature_rows
from app.ml.logistic_regression import train_logistic_regression
from app.schemas.search import SearchHit


def test_build_gating_feature_rows_extracts_system_level_features(tmp_path) -> None:
    docs = [
        SearchHit(
            doc_id="shared",
            system_id="memo",
            vector_score=0.8,
            vector_rank=2,
            bm25_score=10,
            bm25_rank=1,
        ),
        SearchHit(doc_id="v", system_id="memo", vector_score=0.6, vector_rank=3),
        SearchHit(doc_id="e", system_id="album", bm25_score=5, bm25_rank=2),
    ]

    rows = build_gating_feature_rows(
        "query", docs, task_id="t1", all_system_ids={"memo", "album", "calendar"}
    )
    by_system = {row.system_id: row for row in rows}

    assert by_system["memo"].vector_top1 == 0.8
    assert by_system["memo"].vector_top3_mean == 0.7
    assert by_system["memo"].vector_best_rank_score == 0.5
    assert by_system["memo"].vector_hit_count == 2.0
    assert by_system["memo"].es_top1_norm == 1.0
    assert by_system["memo"].es_best_rank_score == 1.0
    assert by_system["memo"].es_hit_count == 1.0
    assert by_system["memo"].same_doc_hit_by_both == 1.0
    assert by_system["memo"].same_system_hit_by_both == 1.0
    assert by_system["album"].es_top1_norm == 0.5
    assert by_system["calendar"].vector_top1 == 0.0
    assert by_system["calendar"].vector_hit_count == 0.0
    assert by_system["calendar"].es_top1_norm == 0.0
    assert by_system["calendar"].es_hit_count == 0.0

    output = tmp_path / "samples.jsonl"
    append_gating_feature_rows(rows, output)
    payload = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert payload["query"] == "query"
    assert payload["label"] is None


def test_logistic_regression_training_learns_simple_boundary() -> None:
    features = np.array(
        [
            [1, 1, 1, 3, 1, 1, 1, 2, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.9, 0.8, 1, 2, 0.8, 0.7, 1, 1, 1, 1],
            [0.1, 0.1, 0, 1, 0.1, 0.1, 0, 1, 0, 0],
        ],
        dtype=float,
    )
    labels = np.array([1, 0, 1, 0], dtype=float)

    model = train_logistic_regression(features, labels, learning_rate=0.5, epochs=200)
    probs = model.predict_proba(features)

    assert probs[0] > 0.8
    assert probs[1] < 0.3
