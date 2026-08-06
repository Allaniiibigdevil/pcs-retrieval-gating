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
    LOCAL_SOURCE_CONFIG_PATH: str = "config/sources.json"
    LOCAL_FAISS_INDEX_PATH: str = "data/artifacts/faiss.index"
    LOCAL_FAISS_DOC_IDS_PATH: str = "data/artifacts/faiss_doc_ids.json"

    LOCAL_ES_URL: str = "http://127.0.0.1:9200"
    LOCAL_ES_INDEX: str = "pcs_retrieval_docs"
    LOCAL_ES_ANALYZER: str = "standard"
    LOCAL_ES_SEARCH_ANALYZER: str = "standard"
    LOCAL_ES_SHARDS: int = 1
    LOCAL_ES_REPLICAS: int = 0
    LOCAL_ES_TIMEOUT_SECONDS: int = 10
    LOCAL_ES_INDEX_ON_BUILD: bool = False
    LOCAL_ES_SUMMARY_BOOST: float = 1.0
    LOCAL_ES_KEYWORDS_BOOST: float = 1.0

    EMBEDDING_PROVIDER: str = "bge"
    EMBEDDING_MODEL_PATH: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIM: int = 512

    ES_TOP_K_DOCS: int = Field(default=50, ge=1, le=1000)
    FAISS_TOP_K_DOCS: int = Field(default=50, ge=1, le=1000)
    EVIDENCE_DOCS_PER_SYSTEM: int = Field(default=3, ge=1, le=100)
    REWRITTEN_QUERY_ES_WEIGHT: float = Field(default=1.0, ge=0.0, le=1.0)

    SYSTEM_SELECTION_THRESHOLD: float = Field(default=0.60, ge=0.0, le=1.0)
    ES_SCORE_WEIGHT: float = Field(default=0.55, ge=0.0, le=1.0)
    AGREEMENT_WEIGHT: float = Field(default=0.20, ge=0.0, le=1.0)
    SEMANTIC_MATCH_THRESHOLD: float = Field(default=0.30, ge=0.0, le=1.0)
    LEXICAL_MATCH_THRESHOLD: float = Field(default=0.30, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
