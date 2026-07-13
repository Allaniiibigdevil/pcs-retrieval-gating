import math
from collections import defaultdict

from app.config import get_settings
from app.schemas.decision import EvidenceDoc, SystemDecision
from app.schemas.search import SearchHit


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _vector_score_norm(hit: SearchHit) -> float:
    if hit.vector_score is not None:
        return _clamp(hit.vector_score)
    if hit.vector_rank is not None:
        return max(0.0, 1.0 - (hit.vector_rank - 1) / 50)
    return 0.0


def _es_score_norm(hit: SearchHit, max_es_score: float) -> float:
    """Normalize ES scores within one query's candidate set."""
    if hit.bm25_score is None or hit.bm25_score <= 0 or max_es_score <= 0:
        return 0.0
    return _clamp(hit.bm25_score / max_es_score)


def simple_doc_strength(hit: SearchHit, max_es_score: float = 0.0) -> float:
    settings = get_settings()
    semantic_score = _vector_score_norm(hit)
    lexical_score = _es_score_norm(hit, max_es_score)
    agreement_boost = (
        settings.AGREEMENT_WEIGHT * math.sqrt(semantic_score * lexical_score)
        if semantic_score >= settings.SEMANTIC_MATCH_THRESHOLD
        and lexical_score >= settings.LEXICAL_MATCH_THRESHOLD
        else 0.0
    )

    return _clamp(
        max(semantic_score, settings.ES_SCORE_WEIGHT * lexical_score) + agreement_boost
    )


def _highlight_from_metadata(doc: SearchHit) -> dict[str, list[str]]:
    highlight = doc.metadata.get("highlight")
    if not isinstance(highlight, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for field in ("summary", "keywords"):
        values = highlight.get(field)
        if isinstance(values, list):
            normalized[field] = [str(value) for value in values]
    return normalized


def _highlight_from_metadata(doc: SearchHit) -> dict[str, list[str]]:
    highlight = doc.metadata.get("highlight")
    if not isinstance(highlight, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for field in ("summary", "keywords"):
        values = highlight.get(field)
        if isinstance(values, list):
            normalized[field] = [str(value) for value in values]
    return normalized


class SystemAggregator:
    def __init__(self, selection_threshold: float | None = None) -> None:
        settings = get_settings()
        self.selection_threshold = (
            settings.SYSTEM_SELECTION_THRESHOLD
            if selection_threshold is None
            else selection_threshold
        )

    def aggregate(self, evidence_docs: list[SearchHit], max_systems: int = 5) -> list[SystemDecision]:
        grouped: dict[str, list[SearchHit]] = defaultdict(list)
        for doc in evidence_docs:
            grouped[doc.system_id].append(doc)

        max_es_score = max(
            (doc.bm25_score or 0.0 for doc in evidence_docs),
            default=0.0,
        )
        decisions: list[SystemDecision] = []
        for system_id, docs in grouped.items():
            sorted_docs = sorted(
                docs,
                key=lambda doc: simple_doc_strength(doc, max_es_score),
                reverse=True,
            )
            top_docs = sorted_docs[:3]
            confidence = simple_doc_strength(top_docs[0], max_es_score) if top_docs else 0.0
            selected = confidence >= self.selection_threshold

            decisions.append(
                SystemDecision(
                    system_id=system_id,
                    selected=selected,
                    confidence=round(confidence, 4),
                    evidence_docs=[
                        EvidenceDoc(
                            doc_id=doc.doc_id,
                            summary=doc.summary,
                            keywords=list(doc.keywords),
                            matched_keywords=list(doc.metadata.get("matched_keywords", [])),
                            highlight=_highlight_from_metadata(doc),
                            bm25_score=doc.bm25_score,
                            vector_score=doc.vector_score,
                            bm25_rank=doc.bm25_rank,
                            vector_rank=doc.vector_rank,
                        )
                        for doc in top_docs
                    ],
                )
            )

        return sorted(decisions, key=lambda item: item.confidence, reverse=True)[:max_systems]
