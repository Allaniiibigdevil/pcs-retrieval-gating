import asyncio
import json
from urllib import request

from app.config import get_settings
from app.embedding.embedding_service import EmbeddingService, get_embedding_service
from app.schemas.search import SearchHit


class LocalElasticsearchVectorRetriever:
    def __init__(
        self,
        *,
        source_id: str,
        index_name: str,
        embedding_service: EmbeddingService | None = None,
        base_url: str | None = None,
    ) -> None:
        if not source_id.strip() or not index_name.strip():
            raise ValueError("source_id and index_name must not be blank")
        settings = get_settings()
        self.source_id = source_id
        self.index_name = index_name
        self.embedding_service = embedding_service or get_embedding_service()
        self.base_url = (base_url or settings.LOCAL_ES_URL).rstrip("/")
        self.timeout_seconds = settings.LOCAL_ES_TIMEOUT_SECONDS

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        if top_k <= 0 or not query.strip():
            return []

        query_vector = await self.embedding_service.embed(query)
        response = await asyncio.to_thread(
            self._request,
            "POST",
            f"/{self.index_name}/_search",
            {
                "size": top_k,
                "_source": ["doc_id", "system_id", "summary"],
                "query": {
                    "script_score": {
                        "query": {
                            "term": {"system_id": self.source_id},
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding')",
                            "params": {
                                "query_vector": query_vector,
                            },
                        },
                    }
                },
            },
        )

        hits: list[SearchHit] = []
        for rank, item in enumerate(response.get("hits", {}).get("hits", []), start=1):
            source = item.get("_source", {})
            stored_source_id = source.get("system_id")
            if stored_source_id != self.source_id:
                raise RuntimeError(
                    f"Elasticsearch index {self.index_name!r} returned source "
                    f"{stored_source_id!r}; expected {self.source_id!r}"
                )
            hits.append(
                SearchHit(
                    doc_id=source.get("doc_id") or item.get("_id"),
                    system_id=self.source_id,
                    summary=source.get("summary"),
                    vector_score=float(item.get("_score") or 0.0),
                    vector_rank=rank,
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
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload) if payload else {}
