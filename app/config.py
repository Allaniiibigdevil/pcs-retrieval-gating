from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "retrieval-gating-service"
    APP_ENV: str = "local"

    ES_URL: str = "http://localhost:9200"
    ES_INDEX_NAME: str = "source_docs"

    GAUSSDB_DSN: str = "postgresql://user:password@localhost:5432/retrieval"
    GAUSSDB_VECTOR_TABLE: str = "source_doc_vectors"

    EMBEDDING_PROVIDER: str = "mock"
    EMBEDDING_DIM: int = 384

    DEFAULT_TOP_K_DOCS: int = 50
    DEFAULT_MAX_SYSTEMS: int = 5

    RETRIEVE_THRESHOLD: float = 0.80
    MAYBE_RETRIEVE_THRESHOLD: float = 0.55


@lru_cache
def get_settings() -> Settings:
    return Settings()
