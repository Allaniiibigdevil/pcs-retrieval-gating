import asyncpg

from app.config import get_settings
from app.embedding.embedding_service import EmbeddingService, get_embedding_service
from app.schemas.search import SearchHit


class VectorRetriever:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        pool: asyncpg.Pool | None = None,
    ) -> None:
        settings = get_settings()
        self.dsn = settings.GAUSSDB_DSN
        self.table = settings.GAUSSDB_VECTOR_TABLE
        self.embedding_service = embedding_service or get_embedding_service()
        self.pool = pool

    async def _get_pool(self) -> asyncpg.Pool:
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.dsn)
        return self.pool

    async def search(self, query: str, top_k: int = 50) -> list[SearchHit]:
        query_embedding = await self.embedding_service.embed(query)
        embedding_literal = "[" + ",".join(str(value) for value in query_embedding) + "]"
        # TODO: Confirm GaussDB distance operator. This assumes pgvector-compatible cosine distance.
        sql = f"""
            SELECT
                doc_id,
                system_id,
                summary,
                keywords,
                metadata,
                GREATEST(0.0, LEAST(1.0, 1.0 - (embedding <=> $1))) AS vector_score
            FROM {self.table}
            ORDER BY embedding <=> $1
            LIMIT $2
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, embedding_literal, top_k)

        return [
            SearchHit(
                doc_id=row["doc_id"],
                system_id=row["system_id"],
                summary=row["summary"],
                keywords=list(row["keywords"] or []),
                metadata=dict(row["metadata"] or {}),
                vector_score=float(row["vector_score"]),
                vector_rank=index,
            )
            for index, row in enumerate(rows, start=1)
        ]
