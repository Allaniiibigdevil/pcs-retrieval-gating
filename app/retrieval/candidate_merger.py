from app.schemas.search import SearchHit


class CandidateMerger:
    def merge(self, bm25_hits: list[SearchHit], vector_hits: list[SearchHit]) -> list[SearchHit]:
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
            existing.metadata = {**hit.metadata, **existing.metadata}

        return list(merged.values())
