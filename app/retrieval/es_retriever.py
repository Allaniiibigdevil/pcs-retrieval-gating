from elasticsearch import AsyncElasticsearch

from app.config import get_settings
from app.schemas.search import SearchHit


class ESRetriever:
    def __init__(self, client: AsyncElasticsearch | None = None) -> None:
        settings = get_settings()
        self.index_name = settings.ES_INDEX_NAME
        self.client = client or AsyncElasticsearch(settings.ES_URL)

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        response = await self.client.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["summary^2", "keywords_text^3"],
                }
            },
            size=top_k,
        )
        hits: list[SearchHit] = []
        for index, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
            source = hit.get("_source", {})
            hits.append(
                SearchHit(
                    doc_id=source["doc_id"],
                    system_id=source["system_id"],
                    summary=source.get("summary"),
                    keywords=source.get("keywords") or [],
                    metadata=source.get("metadata") or {},
                    bm25_score=hit.get("_score"),
                    bm25_rank=index,
                )
            )
        return hits
