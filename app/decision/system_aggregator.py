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


def _es_score_norm(hit: SearchHit) -> float:
    if hit.bm25_score_norm is None:
        return 0.0
    return _clamp(hit.bm25_score_norm)


def simple_doc_strength(hit: SearchHit) -> float:
    settings = get_settings()
    semantic_score = _vector_score_norm(hit)
    lexical_score = _es_score_norm(hit)
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
    return {
        field: [str(value) for value in highlight[field]]
        for field in ("summary", "keywords")
        if isinstance(highlight.get(field), list)
    }


def _matched_queries_from_metadata(doc: SearchHit) -> list[str]:
    matched_queries = doc.metadata.get("matched_queries")
    if not isinstance(matched_queries, list):
        return []
    return [str(value) for value in matched_queries]


class SystemAggregator:
    def __init__(
        self,
        selection_threshold: float | None = None,
        evidence_docs_per_system: int | None = None,
    ) -> None:
        settings = get_settings()
        self.selection_threshold = (
            settings.SYSTEM_SELECTION_THRESHOLD
            if selection_threshold is None
            else selection_threshold
        )
        self.evidence_docs_per_system = (
            settings.EVIDENCE_DOCS_PER_SYSTEM
            if evidence_docs_per_system is None
            else evidence_docs_per_system
        )
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("selection_threshold must be between 0 and 1")
        if self.evidence_docs_per_system <= 0:
            raise ValueError("evidence_docs_per_system must be greater than 0")

    def aggregate(self, evidence_docs: list[SearchHit]) -> list[SystemDecision]:
        grouped: dict[str, list[SearchHit]] = defaultdict(list)
        for doc in evidence_docs:
            grouped[doc.system_id].append(doc)

        decisions: list[SystemDecision] = []
        for system_id, docs in grouped.items():
            sorted_docs = sorted(
                docs,
                key=lambda doc: (-simple_doc_strength(doc), doc.doc_id),
            )
            top_docs = sorted_docs[: self.evidence_docs_per_system]
            confidence = simple_doc_strength(top_docs[0]) if top_docs else 0.0
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
                            matched_queries=_matched_queries_from_metadata(doc),
                            highlight=_highlight_from_metadata(doc),
                            bm25_score=doc.bm25_score,
                            bm25_score_norm=doc.bm25_score_norm,
                            vector_score=doc.vector_score,
                            bm25_rank=doc.bm25_rank,
                            vector_rank=doc.vector_rank,
                        )
                        for doc in top_docs
                    ],
                )
            )

        return sorted(
            decisions,
            key=lambda item: (-item.confidence, item.system_id),
        )
