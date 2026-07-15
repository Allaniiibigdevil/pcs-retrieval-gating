from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import get_settings
from app.schemas.search import SearchHit


RRF_K = 60
TOP_DOCS_PER_RANKER = 3
MAX_REPRESENTATIVE_DOCS = TOP_DOCS_PER_RANKER * 3

FEATURE_NAMES = [
    "vector_score_norm",
    "vector_rank_score",
    "es_score_query_norm",
    "es_rank_score",
    "rrf_score",
]


class _GatingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GatingDoc(_GatingRecord):
    doc_id: str = Field(min_length=1)
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    vector_score_norm: float = Field(ge=0.0, le=1.0)
    vector_rank_score: float = Field(ge=0.0, le=1.0)
    es_score_query_norm: float = Field(ge=0.0, le=1.0)
    es_rank_score: float = Field(ge=0.0, le=1.0)
    rrf_score: float = Field(ge=0.0, le=1.0)

    def feature_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]


class GatingSystemBag(_GatingRecord):
    system_id: str = Field(min_length=1)
    label: Literal[0, 1] | None = None
    docs: list[GatingDoc] = Field(min_length=1, max_length=MAX_REPRESENTATIVE_DOCS)

    @model_validator(mode="after")
    def validate_unique_doc_ids(self) -> Self:
        doc_ids = [doc.doc_id for doc in self.docs]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("duplicate doc_id in MIL bag")
        return self


class GatingCase(_GatingRecord):
    task_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    systems: list[GatingSystemBag] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_system_ids(self) -> Self:
        system_ids = [system.system_id for system in self.systems]
        if len(set(system_ids)) != len(system_ids):
            raise ValueError("duplicate system_id in gating case")
        return self


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
    docs: list[SearchHit], rank_field: str, score_field: str, limit: int = TOP_DOCS_PER_RANKER
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
            *sorted(docs, key=_rrf_raw, reverse=True)[:TOP_DOCS_PER_RANKER],
        ]
        seen_doc_ids: set[str] = set()
        for doc in selected:
            if doc.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc.doc_id)
            representatives.append(doc)
    return representatives


def _build_gating_doc(doc: SearchHit, max_es_score: float) -> GatingDoc:
    es_norm = (
        _clamp(float(doc.bm25_score) / max_es_score)
        if doc.bm25_score is not None and doc.bm25_score > 0 and max_es_score > 0
        else 0.0
    )
    return GatingDoc(
        doc_id=doc.doc_id,
        summary=doc.summary,
        keywords=list(doc.keywords),
        vector_score_norm=(
            _clamp(float(doc.vector_score)) if doc.vector_score is not None else 0.0
        ),
        vector_rank_score=_rank_score(doc.vector_rank),
        es_score_query_norm=es_norm,
        es_rank_score=_rank_score(doc.bm25_rank),
        rrf_score=_rrf_norm(doc),
    )


def build_gating_case(
    query: str,
    evidence_docs: list[SearchHit],
    *,
    task_id: str,
    labels_by_system: Mapping[str, Literal[0, 1]] | None = None,
) -> GatingCase:
    max_es_score = max((doc.bm25_score or 0.0 for doc in evidence_docs), default=0.0)
    docs_by_system: dict[str, list[GatingDoc]] = defaultdict(list)
    for doc in select_representative_docs(evidence_docs):
        docs_by_system[doc.system_id].append(_build_gating_doc(doc, max_es_score))

    systems = [
        GatingSystemBag(
            system_id=system_id,
            label=labels_by_system.get(system_id) if labels_by_system is not None else None,
            docs=docs_by_system[system_id],
        )
        for system_id in sorted(docs_by_system)
    ]
    return GatingCase(task_id=task_id, query=query, systems=systems)


def append_gating_case(case: GatingCase, path: str | Path | None = None) -> None:
    output_path = Path(path or get_settings().GATING_CASES_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(case.model_dump_json() + "\n")


def cases_to_mil_bags(cases: list[GatingCase]) -> list[MilBag]:
    task_ids = [case.task_id for case in cases]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task_id in gating dataset")

    bags: list[MilBag] = []
    for case in cases:
        for system in case.systems:
            if system.label is None:
                continue
            bags.append(
                MilBag(
                    query_key=case.task_id,
                    system_id=system.system_id,
                    features=np.asarray(
                        [doc.feature_vector() for doc in system.docs], dtype=np.float32
                    ),
                    label=float(system.label),
                )
            )
    if not bags:
        raise ValueError("no labeled MIL bags found")
    return bags
