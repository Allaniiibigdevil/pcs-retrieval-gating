import math

from app.schemas.decision import EvidenceDoc, SystemDecision
from app.schemas.search import SearchHit


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


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
        *,
        source_id: str,
        selection_threshold: float,
        evidence_docs_per_system: int,
        es_score_weight: float,
        agreement_weight: float,
        semantic_match_threshold: float,
        lexical_match_threshold: float,
    ) -> None:
        self.source_id = source_id
        self.selection_threshold = selection_threshold
        self.evidence_docs_per_system = evidence_docs_per_system
        self.es_score_weight = es_score_weight
        self.agreement_weight = agreement_weight
        self.semantic_match_threshold = semantic_match_threshold
        self.lexical_match_threshold = lexical_match_threshold

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
        semantic_score = _clamp(hit.vector_score) if hit.vector_score is not None else 0.0
        lexical_score = _clamp(hit.bm25_score_norm or 0.0)
        agreement_boost = (
            self.agreement_weight * math.sqrt(semantic_score * lexical_score)
            if semantic_score >= self.semantic_match_threshold
            and lexical_score >= self.lexical_match_threshold
            else 0.0
        )
        return _clamp(
            max(semantic_score, self.es_score_weight * lexical_score) + agreement_boost
        )

    def aggregate(self, evidence_docs: list[SearchHit]) -> SystemDecision | None:
        mismatched = [doc.doc_id for doc in evidence_docs if doc.system_id != self.source_id]
        if mismatched:
            raise ValueError(
                f"Evidence for source {self.source_id!r} contains mismatched documents: "
                + ", ".join(mismatched[:5])
            )
        if not evidence_docs:
            return None

        sorted_docs = sorted(
            evidence_docs,
            key=lambda doc: (-self.doc_strength(doc), doc.doc_id),
        )
        top_docs = sorted_docs[: self.evidence_docs_per_system]
        confidence = self.doc_strength(top_docs[0])
        return SystemDecision(
            system_id=self.source_id,
            selected=confidence >= self.selection_threshold,
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
