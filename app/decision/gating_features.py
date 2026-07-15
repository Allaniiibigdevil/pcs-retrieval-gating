from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from app.config import get_settings
from app.schemas.search import SearchHit


RRF_K = 60

FEATURE_NAMES = [
    "vector_score_norm",
    "vector_rank_score",
    "es_score_query_norm",
    "es_rank_score",
    "rrf_score",
    "same_doc_hit_by_both",
    "matched_keyword_ratio",
]


@dataclass(frozen=True)
class GatingFeatureRow:
    query: str
    system_id: str
    doc_id: str
    task_id: str | None
    label: int | None
    vector_score_norm: float
    vector_rank_score: float
    es_score_query_norm: float
    es_rank_score: float
    rrf_score: float
    same_doc_hit_by_both: float
    matched_keyword_ratio: float

    def feature_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class MilBag:
    query_key: str
    system_id: str
    features: np.ndarray
    label: float


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _rank_score(rank: int | None) -> float:
    return 1.0 / float(rank) if rank is not None and rank > 0 else 0.0


def _rrf_raw(hit: SearchHit) -> float:
    score = 0.0
    if hit.bm25_rank is not None and hit.bm25_rank > 0:
        score += 1.0 / (RRF_K + hit.bm25_rank)
    if hit.vector_rank is not None and hit.vector_rank > 0:
        score += 1.0 / (RRF_K + hit.vector_rank)
    return score


def _rrf_norm(hit: SearchHit) -> float:
    return _clamp(_rrf_raw(hit) / (2.0 / (RRF_K + 1)))


def _top_by_rank(
    docs: list[SearchHit], rank_field: str, score_field: str, limit: int = 3
) -> list[SearchHit]:
    ranked = [doc for doc in docs if getattr(doc, rank_field) is not None]
    if ranked:
        return sorted(ranked, key=lambda doc: int(getattr(doc, rank_field)))[:limit]
    scored = [doc for doc in docs if getattr(doc, score_field) is not None]
    return sorted(scored, key=lambda doc: float(getattr(doc, score_field)), reverse=True)[:limit]


def select_representative_docs(evidence_docs: list[SearchHit]) -> list[SearchHit]:
    """Select BM25 Top-3, Vector Top-3 and RRF Top-3 within each system."""

    grouped: dict[str, list[SearchHit]] = defaultdict(list)
    for doc in evidence_docs:
        grouped[doc.system_id].append(doc)

    representatives: list[SearchHit] = []
    for system_id in sorted(grouped):
        docs = grouped[system_id]
        selected = [
            *_top_by_rank(docs, "bm25_rank", "bm25_score"),
            *_top_by_rank(docs, "vector_rank", "vector_score"),
            *sorted(docs, key=_rrf_raw, reverse=True)[:3],
        ]
        seen_doc_ids: set[str] = set()
        for doc in selected:
            if doc.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc.doc_id)
            representatives.append(doc)
    return representatives


def _matched_keyword_ratio(hit: SearchHit) -> float:
    matched = {
        str(value).strip().lower()
        for value in hit.metadata.get("matched_keywords", [])
        if str(value).strip()
    }
    if not matched:
        return 0.0
    keywords = {value.strip().lower() for value in hit.keywords if value.strip()}
    return _clamp(len(matched & keywords) / len(keywords)) if keywords else 1.0


def build_gating_feature_rows(
    query: str,
    evidence_docs: list[SearchHit],
    *,
    task_id: str | None = None,
    label: int | None = None,
    labels_by_system: Mapping[str, int] | None = None,
) -> list[GatingFeatureRow]:
    max_es_score = max((doc.bm25_score or 0.0 for doc in evidence_docs), default=0.0)
    rows: list[GatingFeatureRow] = []
    for doc in select_representative_docs(evidence_docs):
        es_norm = (
            _clamp(float(doc.bm25_score) / max_es_score)
            if doc.bm25_score is not None and doc.bm25_score > 0 and max_es_score > 0
            else 0.0
        )
        row_label = (
            labels_by_system.get(doc.system_id, label) if labels_by_system is not None else label
        )
        rows.append(
            GatingFeatureRow(
                query=query,
                system_id=doc.system_id,
                doc_id=doc.doc_id,
                task_id=task_id,
                label=row_label,
                vector_score_norm=(
                    _clamp(float(doc.vector_score)) if doc.vector_score is not None else 0.0
                ),
                vector_rank_score=_rank_score(doc.vector_rank),
                es_score_query_norm=es_norm,
                es_rank_score=_rank_score(doc.bm25_rank),
                rrf_score=_rrf_norm(doc),
                same_doc_hit_by_both=(
                    1.0 if doc.bm25_rank is not None and doc.vector_rank is not None else 0.0
                ),
                matched_keyword_ratio=_matched_keyword_ratio(doc),
            )
        )
    return rows


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


def rows_to_numpy(rows: list[GatingFeatureRow]) -> tuple[np.ndarray, np.ndarray]:
    labeled = [row for row in rows if row.label is not None]
    if not labeled:
        raise ValueError("no labeled rows found")
    return (
        np.asarray([row.feature_vector() for row in labeled], dtype=np.float64),
        np.asarray([int(row.label) for row in labeled], dtype=np.float64),
    )


def rows_to_mil_bags(rows: list[GatingFeatureRow]) -> list[MilBag]:
    grouped: dict[tuple[str, str], list[GatingFeatureRow]] = defaultdict(list)
    for row in rows:
        if row.label is not None:
            grouped[(row.task_id or row.query, row.system_id)].append(row)

    bags: list[MilBag] = []
    for (query_key, system_id), bag_rows in grouped.items():
        labels = {int(row.label) for row in bag_rows if row.label is not None}
        if len(labels) != 1:
            raise ValueError(
                f"inconsistent labels in MIL bag query={query_key!r} system={system_id!r}"
            )
        bags.append(
            MilBag(
                query_key=query_key,
                system_id=system_id,
                features=np.asarray([row.feature_vector() for row in bag_rows], dtype=np.float64),
                label=float(labels.pop()),
            )
        )
    if not bags:
        raise ValueError("no labeled MIL bags found")
    return bags
