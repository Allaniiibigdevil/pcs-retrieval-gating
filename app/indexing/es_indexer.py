from elasticsearch import AsyncElasticsearch

from app.config import get_settings
from app.schemas.doc import SourceDoc


class ESIndexer:
    def __init__(self, client: AsyncElasticsearch | None = None) -> None:
        settings = get_settings()
        self.index_name = settings.ES_INDEX_NAME
        self.client = client or AsyncElasticsearch(settings.ES_URL)

    async def ensure_index(self) -> None:
        exists = await self.client.indices.exists(index=self.index_name)
        if exists:
            return

        # TODO: Support Chinese analyzers, stop words, and synonym dictionaries.
        await self.client.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "system_id": {"type": "keyword"},
                    "summary": {"type": "text"},
                    "keywords": {"type": "keyword"},
                    "keywords_text": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True},
                    "updated_at": {"type": "date"},
                }
            },
        )

    async def upsert(self, doc: SourceDoc) -> None:
        await self.ensure_index()
        payload = {
            "doc_id": doc.doc_id,
            "system_id": doc.system_id,
            "summary": doc.summary,
            "keywords": doc.keywords,
            "keywords_text": " ".join(doc.keywords),
            "metadata": doc.metadata,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        }
        await self.client.index(index=self.index_name, id=doc.doc_id, document=payload)
