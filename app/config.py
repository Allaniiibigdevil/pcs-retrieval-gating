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
    APP_ENV: str = "local"
    APP_MODE: str = "local"

    LOCAL_DATA_DIR: str = "data"
    LOCAL_RAW_DOCS_PATH: str = "data/raw/docs.jsonl"
    LOCAL_ARTIFACT_DIR: str = "data/artifacts"

    LOCAL_ES_URL: str = "http://127.0.0.1:9200"
    LOCAL_ES_INDEX: str = "pcs_retrieval_docs"
    LOCAL_ES_ANALYZER: str = "standard"
    LOCAL_ES_SEARCH_ANALYZER: str = "standard"
    LOCAL_ES_SHARDS: int = 1
    LOCAL_ES_REPLICAS: int = 0
    LOCAL_ES_TIMEOUT_SECONDS: int = 10
    LOCAL_ES_INDEX_ON_BUILD: bool = False
    LOCAL_ES_SUMMARY_BOOST: float = 1.0
    LOCAL_ES_KEYWORDS_BOOST: float = 3.0

    EMBEDDING_PROVIDER: str = "bge"
    EMBEDDING_MODEL_PATH: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIM: int = 512

    DEFAULT_TOP_K_DOCS: int = Field(default=50, ge=1, le=1000)
    FAISS_SCORE_THRESHOLD: float = Field(default=0.60, ge=-1.0, le=1.0)
    RRF_K: int = Field(default=20, ge=1)
    RRF_TOP_N_DOCS: int = Field(default=10, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
