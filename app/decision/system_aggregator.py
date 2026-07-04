from collections import defaultdict

from app.config import get_settings
from app.schemas.decision import EvidenceDoc, SystemDecision
from app.schemas.search import SearchHit


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def simple_doc_strength(hit: SearchHit) -> float:
    settings = get_settings()
    scores: list[float] = []

    if hit.vector_score is not None and 0.0 <= hit.vector_score <= 1.0:
        scores.append(hit.vector_score)

    if hit.bm25_rank is not None and (hit.bm25_score is None or hit.bm25_score > 0):
        rank_score = max(0.0, 1.0 - (hit.bm25_rank - 1) / 50)
        scores.append(settings.BM25_RANK_WEIGHT * rank_score)

    if hit.vector_score is None and hit.vector_rank is not None:
        scores.append(max(0.0, 1.0 - (hit.vector_rank - 1) / 50))

    matched_keywords = hit.metadata.get("matched_keywords", [])
    keyword_boost = min(
        settings.KEYWORD_BOOST_MAX,
        settings.KEYWORD_BOOST_PER_MATCH * len(matched_keywords),
    )
    return _clamp(max(scores, default=0.0) + keyword_boost)


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

        decisions: list[SystemDecision] = []
        for system_id, docs in grouped.items():
            sorted_docs = sorted(docs, key=simple_doc_strength, reverse=True)
            top_docs = sorted_docs[:3]
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
