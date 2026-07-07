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


def _bm25_score_norm(hit: SearchHit, max_bm25_score: float) -> float:
    if hit.bm25_score is None or hit.bm25_score <= 0 or max_bm25_score <= 0:
        return 0.0
    return _clamp(hit.bm25_score / max_bm25_score)


def _keyword_match_score(hit: SearchHit) -> float:
    settings = get_settings()
    matched_keywords = hit.metadata.get("matched_keywords", [])
    return min(
        settings.KEYWORD_MATCH_MAX,
        settings.KEYWORD_MATCH_PER_HIT * len(matched_keywords),
    )


def simple_doc_strength(hit: SearchHit, max_bm25_score: float = 0.0) -> float:
    settings = get_settings()
    semantic_score = _vector_score_norm(hit)
    lexical_score = max(_bm25_score_norm(hit, max_bm25_score), _keyword_match_score(hit))
    agreement_boost = (
        settings.AGREEMENT_BOOST
        if semantic_score >= settings.SEMANTIC_MATCH_THRESHOLD
        and lexical_score >= settings.LEXICAL_MATCH_THRESHOLD
        else 0.0
    )

    return _clamp(
        settings.VECTOR_SCORE_WEIGHT * semantic_score
        + settings.BM25_SCORE_WEIGHT * lexical_score
        + agreement_boost
    )


def build_reason(system_id: str, evidence_docs: list[SearchHit]) -> str:
    del system_id
    seen: set[str] = set()
    keywords: list[str] = []
    for doc in evidence_docs:
        for keyword in doc.metadata.get("matched_keywords", []):
            if keyword not in seen:
                seen.add(keyword)
                keywords.append(keyword)

    if keywords:
        return f"命中相关关键词：{', '.join(keywords)}"
    return "存在相关摘要证据。"


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

        max_bm25_score = max(
            (doc.bm25_score or 0.0 for doc in evidence_docs if (doc.bm25_score or 0.0) > 0),
            default=0.0,
        )
        decisions: list[SystemDecision] = []
        for system_id, docs in grouped.items():
            sorted_docs = sorted(
                docs,
                key=lambda doc: simple_doc_strength(doc, max_bm25_score),
                reverse=True,
            )
            top_docs = sorted_docs[:3]
            confidence = simple_doc_strength(top_docs[0], max_bm25_score) if top_docs else 0.0
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
                            matched_keywords=list(doc.metadata.get("matched_keywords", [])),
                            bm25_score=doc.bm25_score,
                            vector_score=doc.vector_score,
                            bm25_rank=doc.bm25_rank,
                            vector_rank=doc.vector_rank,
                        )
                        for doc in top_docs
                    ],
                    reason=build_reason(system_id, top_docs),
                )
            )

        return sorted(decisions, key=lambda item: item.confidence, reverse=True)[:max_systems]
