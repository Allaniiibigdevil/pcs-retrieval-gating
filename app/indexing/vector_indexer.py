import json

import asyncpg

from app.config import get_settings
from app.schemas.doc import SourceDoc


class VectorIndexer:
    def __init__(self, pool: asyncpg.Pool | None = None) -> None:
        settings = get_settings()
        self.dsn = settings.GAUSSDB_DSN
        self.table = settings.GAUSSDB_VECTOR_TABLE
        self.embedding_model_version = settings.EMBEDDING_PROVIDER
        self.pool = pool

    async def _get_pool(self) -> asyncpg.Pool:
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.dsn)
        return self.pool

    async def upsert(self, doc: SourceDoc, embedding: list[float]) -> None:
        pool = await self._get_pool()
        embedding_literal = "[" + ",".join(str(value) for value in embedding) + "]"
        # TODO: Confirm GaussDB vector type/operator syntax for the deployed version.
        query = f"""
            INSERT INTO {self.table} (
                doc_id, system_id, summary, keywords, metadata, embedding,
                updated_at, embedding_model_version
            )
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7, $8)
            ON CONFLICT (doc_id) DO UPDATE SET
                system_id = EXCLUDED.system_id,
                summary = EXCLUDED.summary,
                keywords = EXCLUDED.keywords,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = EXCLUDED.updated_at,
                embedding_model_version = EXCLUDED.embedding_model_version
        """
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                doc.doc_id,
                doc.system_id,
                doc.summary,
                json.dumps(doc.keywords, ensure_ascii=False),
                json.dumps(doc.metadata, ensure_ascii=False),
                embedding_literal,
                doc.updated_at,
                self.embedding_model_version,
            )
