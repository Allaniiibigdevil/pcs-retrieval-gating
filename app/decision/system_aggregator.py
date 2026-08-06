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


def _calculate_doc_strength(
    hit: SearchHit,
    *,
    es_score_weight: float,
    agreement_weight: float,
    semantic_match_threshold: float,
    lexical_match_threshold: float,
) -> float:
    semantic_score = _vector_score_norm(hit)
    lexical_score = _es_score_norm(hit)
    agreement_boost = (
        agreement_weight * math.sqrt(semantic_score * lexical_score)
        if semantic_score >= semantic_match_threshold
        and lexical_score >= lexical_match_threshold
        else 0.0
    )
    return _clamp(max(semantic_score, es_score_weight * lexical_score) + agreement_boost)


def simple_doc_strength(hit: SearchHit) -> float:
    settings = get_settings()
    return _calculate_doc_strength(
        hit,
        es_score_weight=settings.ES_SCORE_WEIGHT,
        agreement_weight=settings.AGREEMENT_WEIGHT,
        semantic_match_threshold=settings.SEMANTIC_MATCH_THRESHOLD,
        lexical_match_threshold=settings.LEXICAL_MATCH_THRESHOLD,
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
        es_score_weight: float | None = None,
        agreement_weight: float | None = None,
        semantic_match_threshold: float | None = None,
        lexical_match_threshold: float | None = None,
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
        self.es_score_weight = (
            settings.ES_SCORE_WEIGHT if es_score_weight is None else es_score_weight
        )
        self.agreement_weight = (
            settings.AGREEMENT_WEIGHT if agreement_weight is None else agreement_weight
        )
        self.semantic_match_threshold = (
            settings.SEMANTIC_MATCH_THRESHOLD
            if semantic_match_threshold is None
            else semantic_match_threshold
        )
        self.lexical_match_threshold = (
            settings.LEXICAL_MATCH_THRESHOLD
            if lexical_match_threshold is None
            else lexical_match_threshold
        )

        for name in (
            "selection_threshold",
            "es_score_weight",
            "agreement_weight",
            "semantic_match_threshold",
            "lexical_match_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.evidence_docs_per_system <= 0:
            raise ValueError("evidence_docs_per_system must be greater than 0")

    def doc_strength(self, hit: SearchHit) -> float:
        return _calculate_doc_strength(
            hit,
            es_score_weight=self.es_score_weight,
            agreement_weight=self.agreement_weight,
            semantic_match_threshold=self.semantic_match_threshold,
            lexical_match_threshold=self.lexical_match_threshold,
        )

    def aggregate(self, evidence_docs: list[SearchHit]) -> list[SystemDecision]:
        grouped: dict[str, list[SearchHit]] = defaultdict(list)
        for doc in evidence_docs:
            grouped[doc.system_id].append(doc)

        decisions: list[SystemDecision] = []
        for system_id, docs in grouped.items():
            sorted_docs = sorted(
                docs,
                key=lambda doc: (-self.doc_strength(doc), doc.doc_id),
            )
            top_docs = sorted_docs[: self.evidence_docs_per_system]
            confidence = self.doc_strength(top_docs[0]) if top_docs else 0.0
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

        return sorted(decisions, key=lambda item: (-item.confidence, item.system_id))
