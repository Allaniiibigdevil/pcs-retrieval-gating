from collections import defaultdict

from app.config import get_settings
from app.schemas.decision import EvidenceDoc, SystemDecision
from app.schemas.search import SearchHit


def _highlight_from_metadata(doc: SearchHit) -> dict[str, list[str]]:
    highlight = doc.metadata.get("highlight")
    if not isinstance(highlight, dict):
        return {}
    return {
        field: [str(value) for value in highlight[field]]
        for field in ("summary", "keywords")
        if isinstance(highlight.get(field), list)
    }


def _doc_sort_key(doc: SearchHit) -> tuple[int, float, str]:
    rank = doc.reranker_rank if doc.reranker_rank is not None else 2**31 - 1
    score = doc.reranker_score if doc.reranker_score is not None else float("-inf")
    return rank, -score, doc.doc_id


class SystemAggregator:
    def __init__(
        self,
        score_threshold: float | None = None,
        evidence_docs_per_system: int | None = None,
    ) -> None:
        settings = get_settings()
        self.score_threshold = (
            settings.RERANKER_SCORE_THRESHOLD
            if score_threshold is None
            else score_threshold
        )
        self.evidence_docs_per_system = (
            settings.RERANKER_EVIDENCE_DOCS_PER_SYSTEM
            if evidence_docs_per_system is None
            else evidence_docs_per_system
        )
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        if self.evidence_docs_per_system <= 0:
            raise ValueError("evidence_docs_per_system must be greater than 0")

    def aggregate(self, reranked_docs: list[SearchHit]) -> list[SystemDecision]:
        grouped: dict[str, list[SearchHit]] = defaultdict(list)
        for doc in reranked_docs:
            if doc.reranker_score is not None and doc.reranker_rank is not None:
                grouped[doc.system_id].append(doc)

        decisions: list[SystemDecision] = []
        for system_id, system_docs in grouped.items():
            system_docs.sort(key=_doc_sort_key)
            best_score = system_docs[0].reranker_score
            assert best_score is not None
            evidence = system_docs[: self.evidence_docs_per_system]
            decisions.append(
                SystemDecision(
                    system_id=system_id,
                    selected=best_score >= self.score_threshold,
                    reranker_score=best_score,
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
                            reranker_score=doc.reranker_score,
                            reranker_rank=doc.reranker_rank,
                        )
                        for doc in evidence
                        if doc.reranker_score is not None
                        and doc.reranker_rank is not None
                    ],
                )
            )

        return sorted(
            decisions,
            key=lambda item: (-item.reranker_score, item.system_id),
        )
