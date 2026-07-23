import asyncio
import json
from urllib import request

from app.config import get_settings
from app.schemas.search import SearchHit


class LocalElasticsearchRetriever:
    def __init__(self, base_url: str | None = None, index_name: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.LOCAL_ES_URL).rstrip("/")
        self.index_name = index_name or settings.LOCAL_ES_INDEX
        self.fields = [
            f"summary^{settings.LOCAL_ES_SUMMARY_BOOST}",
            f"keywords^{settings.LOCAL_ES_KEYWORDS_BOOST}",
        ]

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        if not query.strip():
            return []

        response = await asyncio.to_thread(
            self._request,
            "POST",
            f"/{self.index_name}/_search",
            {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": self.fields,
                        "type": "cross_fields",
                        "operator": "or",
                        "minimum_should_match": "1<2",
                    }
                },
                "size": top_k,
                "highlight": {
                    "fields": {
                        "keywords": {"number_of_fragments": 0},
                        "summary": {"number_of_fragments": 2},
                    }
                },
            },
        )
        hits: list[SearchHit] = []
        for rank, item in enumerate(response.get("hits", {}).get("hits", []), start=1):
            source = item.get("_source", {})
            highlight = item.get("highlight", {})
            metadata = dict(source.get("metadata", {}))
            if highlight:
                metadata["highlight"] = highlight
            matched_keywords = _matched_keywords_from_highlight(
                list(source.get("keywords", [])),
                highlight.get("keywords", []),
            )
            if matched_keywords:
                metadata["matched_keywords"] = matched_keywords
            hits.append(
                SearchHit(
                    doc_id=source.get("doc_id") or item.get("_id"),
                    system_id=source["system_id"],
                    summary=source.get("summary"),
                    keywords=list(source.get("keywords", [])),
                    metadata=metadata,
                    bm25_score=float(item.get("_score") or 0.0),
                    bm25_rank=rank,
                )
            )
        return hits

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=get_settings().LOCAL_ES_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload) if payload else {}


def _strip_highlight_tags(text: str) -> str:
    return text.replace("<em>", "").replace("</em>", "")


def _matched_keywords_from_highlight(
    keywords: list[str],
    highlighted_keywords: list[str],
) -> list[str]:
    if not highlighted_keywords:
        return []

    highlighted = {_strip_highlight_tags(item) for item in highlighted_keywords}
    matches: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if keyword in highlighted and keyword not in seen:
            seen.add(keyword)
            matches.append(keyword)
    return matches
