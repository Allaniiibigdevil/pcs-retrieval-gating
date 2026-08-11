from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "retrieval-gating-service"

    LOCAL_RAW_DOCS_PATH: str = "data/raw/docs.jsonl"
    LOCAL_ARTIFACT_DIR: str = "data/artifacts"
    LOCAL_SOURCE_CONFIG_PATH: str = "config/sources.json"

    LOCAL_ES_URL: str = "http://127.0.0.1:9200"
    LOCAL_ES_KEYWORD_INDEX: str = "pcs_retrieval_keywords"
    LOCAL_ES_VECTOR_INDEX: str = "pcs_retrieval_vectors"
    LOCAL_ES_ANALYZER: str = "standard"
    LOCAL_ES_SEARCH_ANALYZER: str = "standard"
    LOCAL_ES_SHARDS: int = Field(default=1, ge=1)
    LOCAL_ES_REPLICAS: int = Field(default=0, ge=0)
    LOCAL_ES_TIMEOUT_SECONDS: int = Field(default=10, ge=1)
    LOCAL_ES_SUMMARY_BOOST: float = Field(default=1.0, gt=0.0)
    LOCAL_ES_KEYWORDS_BOOST: float = Field(default=1.0, gt=0.0)
    LOCAL_ES_BULK_BATCH_SIZE: int = Field(default=1000, ge=1, le=100000)

    EMBEDDING_PROVIDER: str = "bge"
    EMBEDDING_MODEL_PATH: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIM: int = Field(default=512, ge=1)

    REWRITTEN_QUERY_ES_WEIGHT: float = Field(default=1.0, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
