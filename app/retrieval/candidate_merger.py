from app.retrieval.metadata import merge_metadata
from app.schemas.search import SearchHit


class CandidateMerger:
    def merge(
        self,
        source_id: str,
        bm25_hits: list[SearchHit],
        vector_hits: list[SearchHit],
    ) -> list[SearchHit]:
        _validate_source(source_id, bm25_hits)
        _validate_source(source_id, vector_hits)
        merged: dict[str, SearchHit] = {}

        for hit in bm25_hits:
            merged[hit.doc_id] = hit.model_copy(deep=True)

        for hit in vector_hits:
            existing = merged.get(hit.doc_id)
            if existing is None:
                merged[hit.doc_id] = hit.model_copy(deep=True)
                continue

            existing.vector_score = hit.vector_score
            existing.vector_rank = hit.vector_rank
            if existing.summary is None:
                existing.summary = hit.summary
            if not existing.keywords:
                existing.keywords = list(hit.keywords)
            existing.metadata = merge_metadata(existing.metadata, hit.metadata)

        return list(merged.values())


def _validate_source(source_id: str, hits: list[SearchHit]) -> None:
    mismatched = [hit.doc_id for hit in hits if hit.system_id != source_id]
    if mismatched:
        raise ValueError(
            f"Candidate hits for source {source_id!r} contain mismatched documents: "
            + ", ".join(mismatched[:5])
        )
