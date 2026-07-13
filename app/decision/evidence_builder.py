import re

from app.schemas.search import SearchHit


class EvidenceBuilder:
    def build(self, query_text: str, candidates: list[SearchHit]) -> list[SearchHit]:
        enriched: list[SearchHit] = []
        for hit in candidates:
            copy = hit.model_copy(deep=True)
            matched_keywords = list(copy.metadata.get("matched_keywords") or [])
            if not matched_keywords:
                matched_keywords = [
                    kw for kw in copy.keywords if kw and _keyword_matches(kw, query_text)
                ]
            copy.metadata = {**copy.metadata, "matched_keywords": matched_keywords}
            enriched.append(copy)
        return enriched


def _keyword_matches(keyword: str, query_text: str) -> bool:
    keyword = keyword.strip().lower()
    query_text = query_text.strip().lower()
    if not keyword or not query_text:
        return False
    if keyword in query_text or query_text in keyword:
        return True

    return bool(_text_bigrams(keyword) & _text_bigrams(query_text))


def _text_bigrams(text: str) -> set[str]:
    grams: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text):
        normalized = run.lower()
        if len(normalized) < 2:
            continue
        grams.update(normalized[index : index + 2] for index in range(len(normalized) - 1))
    return grams
