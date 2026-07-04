from app.retrieval.tokenizer import tokenize
from app.schemas.search import SearchHit


class EvidenceBuilder:
    def build(self, query_text: str, candidates: list[SearchHit]) -> list[SearchHit]:
        query_tokens = set(tokenize(query_text))
        enriched: list[SearchHit] = []
        for hit in candidates:
            copy = hit.model_copy(deep=True)
            matched_keywords = [
                kw for kw in copy.keywords if kw and _keyword_matches(kw, query_text, query_tokens)
            ]
            copy.metadata = {**copy.metadata, "matched_keywords": matched_keywords}
            enriched.append(copy)
        return enriched


def _keyword_matches(keyword: str, query_text: str, query_tokens: set[str]) -> bool:
    if keyword in query_text:
        return True

    keyword_tokens = set(tokenize(keyword))
    if keyword_tokens & query_tokens:
        return True

    for token in query_tokens:
        if len(token) >= 2 and token in keyword:
            return True

    for token in keyword_tokens:
        if len(token) >= 2 and token in query_text:
            return True

    return False
