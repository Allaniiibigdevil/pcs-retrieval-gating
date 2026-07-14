from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import json
import logging
from typing import Iterable

from app.config import get_settings
from app.schemas.search import SearchHit
from app.storage.local_artifact_store import LocalArtifactStore

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "vector_top1",
    "vector_top3_mean",
    "vector_best_rank_score",
    "vector_hit_count",
    "es_top1_norm",
    "es_top3_mean",
    "es_best_rank_score",
    "es_hit_count",
    "same_doc_hit_by_both",
    "same_system_hit_by_both",
]


@dataclass(frozen=True)
class GatingFeatureRow:
    query: str
    system_id: str
    task_id: str | None
    label: int | None
    vector_top1: float
    vector_top3_mean: float
    vector_best_rank_score: float
    vector_hit_count: float
    es_top1_norm: float
    es_top3_mean: float
    es_best_rank_score: float
    es_hit_count: float
    same_doc_hit_by_both: float
    same_system_hit_by_both: float

    def feature_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _vector_norm(hit: SearchHit) -> float:
    if hit.vector_score is not None:
        return _clamp(float(hit.vector_score))
    return 0.0


def _rank_score(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def _mean_top3(values: Iterable[float]) -> float:
    top3 = sorted((float(value) for value in values), reverse=True)[:3]
    if not top3:
        return 0.0
    return float(sum(top3) / len(top3))


def build_gating_feature_rows(
    query: str,
    evidence_docs: list[SearchHit],
    *,
    task_id: str | None = None,
    label: int | None = None,
    all_system_ids: set[str] | None = None,
) -> list[GatingFeatureRow]:
    """Build one feature row per candidate system for offline labeling/training."""
    max_es_score = max((doc.bm25_score or 0.0 for doc in evidence_docs), default=0.0)
    rows: list[GatingFeatureRow] = []
    recalled_system_ids = {doc.system_id for doc in evidence_docs}
    system_ids = sorted(recalled_system_ids | (all_system_ids or set()))

    for system_id in system_ids:
        docs = [doc for doc in evidence_docs if doc.system_id == system_id]
        vector_scores = [_vector_norm(doc) for doc in docs if doc.vector_score is not None]
        es_scores = [
            _clamp(float(doc.bm25_score) / max_es_score)
            for doc in docs
            if doc.bm25_score is not None and doc.bm25_score > 0 and max_es_score > 0
        ]
        vector_ranks = [doc.vector_rank for doc in docs if doc.vector_rank is not None]
        es_ranks = [doc.bm25_rank for doc in docs if doc.bm25_rank is not None]
        vector_hit_count = sum(
            1 for doc in docs if doc.vector_score is not None or doc.vector_rank is not None
        )
        es_hit_count = sum(
            1 for doc in docs if doc.bm25_score is not None or doc.bm25_rank is not None
        )
        same_doc = any(doc.vector_score is not None and doc.bm25_score is not None for doc in docs)
        same_system = bool(vector_scores) and bool(es_scores)

        rows.append(
            GatingFeatureRow(
                query=query,
                system_id=system_id,
                task_id=task_id,
                label=label,
                vector_top1=max(vector_scores, default=0.0),
                vector_top3_mean=_mean_top3(vector_scores),
                vector_best_rank_score=max(
                    (_rank_score(rank) for rank in vector_ranks), default=0.0
                ),
                vector_hit_count=float(vector_hit_count),
                es_top1_norm=max(es_scores, default=0.0),
                es_top3_mean=_mean_top3(es_scores),
                es_best_rank_score=max((_rank_score(rank) for rank in es_ranks), default=0.0),
                es_hit_count=float(es_hit_count),
                same_doc_hit_by_both=1.0 if same_doc else 0.0,
                same_system_hit_by_both=1.0 if same_system else 0.0,
            )
        )

    return rows


@lru_cache
def load_local_system_ids() -> frozenset[str]:
    """Load the full local system universe so unrecalled systems can be logged too."""
    try:
        docs = LocalArtifactStore().load_docs()
    except FileNotFoundError:
        logger.warning("gating_system_universe_missing")
        return frozenset()
    return frozenset(doc.system_id for doc in docs)


def append_gating_feature_rows(
    rows: list[GatingFeatureRow], path: str | Path | None = None
) -> None:
    if not rows:
        return
    output_path = Path(path or get_settings().GATING_TRAINING_DATA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(row.to_json() + "\n")
    logger.info(
        "gating_feature_rows_appended path=%s row_count=%d",
        output_path,
        len(rows),
    )


def rows_to_numpy(rows: list[GatingFeatureRow]):
    import numpy as np

    labeled = [row for row in rows if row.label is not None]
    if not labeled:
        raise ValueError("no labeled rows found")
    features = np.array([row.feature_vector() for row in labeled], dtype=np.float64)
    labels = np.array([int(row.label) for row in labeled], dtype=np.float64)
    return features, labels
