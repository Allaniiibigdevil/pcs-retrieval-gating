from collections import defaultdict

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


def _matched_queries_from_metadata(doc: SearchHit) -> list[str]:
    values = doc.metadata.get("matched_queries")
    return [str(value) for value in values] if isinstance(values, list) else []


def _doc_sort_key(doc: SearchHit) -> tuple[int, float, str]:
    rank = doc.reranker_rank if doc.reranker_rank is not None else 2**31 - 1
    score = doc.reranker_score if doc.reranker_score is not None else float("-inf")
    return rank, -score, doc.doc_id


class SystemAggregator:
    def __init__(
        self,
        *,
        source_thresholds: dict[str, float],
        evidence_docs_per_source: dict[str, int],
    ) -> None:
        if not source_thresholds:
            raise ValueError("source_thresholds must not be empty")
        if set(source_thresholds) != set(evidence_docs_per_source):
            raise ValueError(
                "source_thresholds and evidence_docs_per_source must contain the same sources"
            )
        self.source_thresholds = dict(source_thresholds)
        self.evidence_docs_per_source = dict(evidence_docs_per_source)

        for source_id, threshold in self.source_thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"source threshold for {source_id!r} must be between 0 and 1")
        for source_id, limit in self.evidence_docs_per_source.items():
            if limit <= 0:
                raise ValueError(
                    f"evidence document limit for {source_id!r} must be greater than 0"
                )

    def aggregate(self, reranked_docs: list[SearchHit]) -> list[SystemDecision]:
        grouped: dict[str, list[SearchHit]] = defaultdict(list)
        for doc in reranked_docs:
            if doc.system_id not in self.source_thresholds:
                raise ValueError(f"Reranked document has unknown source {doc.system_id!r}")
            if doc.reranker_score is None or doc.reranker_rank is None:
                raise ValueError(f"Reranked document {doc.doc_id!r} is missing score or rank")
            grouped[doc.system_id].append(doc)

        decisions: list[SystemDecision] = []
        for system_id, system_docs in grouped.items():
            system_docs.sort(key=_doc_sort_key)
            best_score = system_docs[0].reranker_score
            assert best_score is not None
            evidence = system_docs[: self.evidence_docs_per_source[system_id]]
            decisions.append(
                SystemDecision(
                    system_id=system_id,
                    selected=best_score >= self.source_thresholds[system_id],
                    reranker_score=best_score,
                    evidence_docs=[
                        EvidenceDoc(
                            doc_id=doc.doc_id,
                            summary=doc.summary,
                            keywords=list(doc.keywords),
                            matched_keywords=list(doc.metadata.get("matched_keywords", [])),
                            matched_queries=_matched_queries_from_metadata(doc),
                            highlight=_highlight_from_metadata(doc),
                            bm25_score=doc.bm25_score,
                            vector_score=doc.vector_score,
                            bm25_rank=doc.bm25_rank,
                            vector_rank=doc.vector_rank,
                            reranker_score=doc.reranker_score,
                            reranker_rank=doc.reranker_rank,
                        )
                        for doc in evidence
                    ],
                )
            )

        return sorted(decisions, key=lambda item: (-item.reranker_score, item.system_id))
