from collections import defaultdict

from app.config import get_settings
from app.schemas.decision import EvidenceDoc, SystemDecision
from app.schemas.search import SearchHit


def reciprocal_rank_fusion_score(
    hit: SearchHit,
    rank_constant: int,
    es_only_weight: float = 0.70,
) -> float:
    """Fuse ES and FAISS ranks, downweighting documents found only by ES."""

    score = 0.0
    has_es_rank = hit.bm25_rank is not None and hit.bm25_rank > 0
    has_vector_rank = hit.vector_rank is not None and hit.vector_rank > 0
    if has_es_rank:
        weight = 1.0 if has_vector_rank else es_only_weight
        score += weight / (rank_constant + hit.bm25_rank)
    if has_vector_rank:
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
        evidence_docs_per_system: int | None = None,
        es_only_weight: float | None = None,
    ) -> None:
        settings = get_settings()
        self.rrf_k = settings.RRF_K if rrf_k is None else rrf_k
        self.top_n_docs = settings.RRF_TOP_N_DOCS if top_n_docs is None else top_n_docs
        self.evidence_docs_per_system = (
            settings.EVIDENCE_DOCS_PER_SYSTEM
            if evidence_docs_per_system is None
            else evidence_docs_per_system
        )
        self.es_only_weight = (
            settings.RRF_ES_ONLY_WEIGHT
            if es_only_weight is None
            else es_only_weight
        )
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        if self.top_n_docs <= 0:
            raise ValueError("top_n_docs must be greater than 0")
        if self.evidence_docs_per_system <= 0:
            raise ValueError("evidence_docs_per_system must be greater than 0")
        if not 0.0 < self.es_only_weight <= 1.0:
            raise ValueError("es_only_weight must be greater than 0 and at most 1")

    def aggregate(self, evidence_docs: list[SearchHit]) -> list[SystemDecision]:
        scored_docs = [
            (
                doc,
                reciprocal_rank_fusion_score(
                    doc,
                    self.rrf_k,
                    self.es_only_weight,
                ),
            )
            for doc in evidence_docs
        ]
        scored_docs = [item for item in scored_docs if item[1] > 0.0]
        scored_docs.sort(key=_doc_sort_key)
        rrf_rank_by_doc_id = {
            doc.doc_id: rank
            for rank, (doc, _) in enumerate(scored_docs, start=1)
        }

        selected_doc_ids = {doc.doc_id for doc, _ in scored_docs[: self.top_n_docs]}
        grouped: dict[str, list[tuple[SearchHit, float]]] = defaultdict(list)
        for doc, score in scored_docs:
            grouped[doc.system_id].append((doc, score))

        decisions: list[SystemDecision] = []
        for system_id, system_docs in grouped.items():
            system_docs.sort(key=_doc_sort_key)
            evidence = system_docs[: self.evidence_docs_per_system]
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
                            rrf_rank=rrf_rank_by_doc_id[doc.doc_id],
                        )
                        for doc, score in evidence
                    ],
                )
            )

        return sorted(decisions, key=lambda item: (-item.rrf_score, item.system_id))
