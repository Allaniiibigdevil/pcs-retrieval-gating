import logging

from app.embedding.embedding_service import (
    EmbeddingService,
    build_embedding_text,
    get_embedding_service,
)
from app.indexing.es_indexer import ESIndexer
from app.indexing.vector_indexer import VectorIndexer
from app.schemas.doc import SourceDoc

logger = logging.getLogger(__name__)


class IndexService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        es_indexer: ESIndexer | None = None,
        vector_indexer: VectorIndexer | None = None,
    ) -> None:
        self.embedding_service = embedding_service or get_embedding_service()
        self.es_indexer = es_indexer or ESIndexer()
        self.vector_indexer = vector_indexer or VectorIndexer()

    async def upsert_doc(self, doc: SourceDoc) -> None:
        embedding = await self.embedding_service.embed(build_embedding_text(doc))
        try:
            await self.es_indexer.upsert(doc)
        except Exception:
            logger.exception("es_index_failed", extra={"doc_id": doc.doc_id})
            raise

        try:
            await self.vector_indexer.upsert(doc, embedding)
        except Exception:
            logger.exception("vector_index_failed", extra={"doc_id": doc.doc_id})
            raise
