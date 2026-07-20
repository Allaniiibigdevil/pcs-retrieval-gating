from collections import defaultdict

from app.config import get_settings
from app.schemas.decision import EvidenceDoc, SystemDecision
from app.schemas.search import SearchHit


def reciprocal_rank_fusion_score(hit: SearchHit, rank_constant: int) -> float:
    """Fuse the ES and FAISS ranks for one document."""

    score = 0.0
    if hit.bm25_rank is not None and hit.bm25_rank > 0:
        score += 1.0 / (rank_constant + hit.bm25_rank)
    if hit.vector_rank is not None and hit.vector_rank > 0:
        score += 1.0 / (rank_constant + hit.vector_rank)
    return score


def _highlight_from_metadata(doc: SearchHit) -> dict[str, list[str]]:
    highlight = doc.metadata.get("highlight")
    if not isinstance(highlight, dict):
        return {}
    return {
        field: [str(value) for value in highlight[field]]
        for field in ("summary", "keywords")
        if isinstance(highlight.get(field), list)
    }


def _doc_sort_key(item: tuple[SearchHit, float]) -> tuple[float, int, int, str]:
    doc, score = item
    ranks = [
        rank
        for rank in (doc.bm25_rank, doc.vector_rank)
        if rank is not None and rank > 0
    ]
    return (-score, min(ranks), sum(ranks), doc.doc_id)


class SystemAggregator:
    def __init__(
        self,
        rrf_k: int | None = None,
        top_n_docs: int | None = None,
    ) -> None:
        settings = get_settings()
        self.rrf_k = settings.RRF_K if rrf_k is None else rrf_k
        self.top_n_docs = settings.RRF_TOP_N_DOCS if top_n_docs is None else top_n_docs
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        if self.top_n_docs <= 0:
            raise ValueError("top_n_docs must be greater than 0")

    def aggregate(self, evidence_docs: list[SearchHit]) -> list[SystemDecision]:
        scored_docs = [
            (doc, reciprocal_rank_fusion_score(doc, self.rrf_k)) for doc in evidence_docs
        ]
        scored_docs = [item for item in scored_docs if item[1] > 0.0]
        scored_docs.sort(key=_doc_sort_key)

        selected_doc_ids = {doc.doc_id for doc, _ in scored_docs[: self.top_n_docs]}
        grouped: dict[str, list[tuple[SearchHit, float]]] = defaultdict(list)
        for doc, score in scored_docs:
            grouped[doc.system_id].append((doc, score))

        decisions: list[SystemDecision] = []
        for system_id, system_docs in grouped.items():
            system_docs.sort(key=_doc_sort_key)
            evidence = system_docs[:3]
            decisions.append(
                SystemDecision(
                    system_id=system_id,
                    selected=any(doc.doc_id in selected_doc_ids for doc, _ in system_docs),
                    rrf_score=system_docs[0][1],
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
                            rrf_score=score,
                        )
                        for doc, score in evidence
                    ],
                )
            )

        return sorted(decisions, key=lambda item: (-item.rrf_score, item.system_id))
