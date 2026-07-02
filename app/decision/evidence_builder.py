from app.schemas.search import SearchHit


class EvidenceBuilder:
    def build(self, normalized_query: str, candidates: list[SearchHit]) -> list[SearchHit]:
        enriched: list[SearchHit] = []
        for hit in candidates:
            copy = hit.model_copy(deep=True)
            matched_keywords = [kw for kw in copy.keywords if kw and kw in normalized_query]
            copy.metadata = {**copy.metadata, "matched_keywords": matched_keywords}
            enriched.append(copy)
        return enriched
