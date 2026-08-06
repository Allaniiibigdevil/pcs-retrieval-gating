from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
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
    FAISS_PREFERRED_SCORE_THRESHOLD: float = Field(default=0.60, ge=-1.0, le=1.0)
    FAISS_MIN_SCORE_THRESHOLD: float = Field(default=0.30, ge=-1.0, le=1.0)
    FAISS_TARGET_HITS: int = Field(default=10, ge=1, le=1000)
    EVIDENCE_DOCS_PER_SYSTEM: int = Field(default=3, ge=1, le=100)

    RERANKER_MODEL_PATH: str = "Alibaba-NLP/gte-multilingual-reranker-base"
    RERANKER_LOCAL_FILES_ONLY: bool = True
    RERANKER_DEVICE: str = "auto"
    RERANKER_BATCH_SIZE: int = Field(default=8, ge=1, le=256)
    RERANKER_MAX_LENGTH: int = Field(default=512, ge=8, le=8192)
    RERANKER_SCORE_THRESHOLD: float = Field(default=0.50, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_retrieval_thresholds(self) -> "Settings":
        if self.FAISS_MIN_SCORE_THRESHOLD > self.FAISS_PREFERRED_SCORE_THRESHOLD:
            raise ValueError(
                "FAISS_MIN_SCORE_THRESHOLD must not exceed "
                "FAISS_PREFERRED_SCORE_THRESHOLD"
            )
        if self.FAISS_TARGET_HITS > self.FAISS_TOP_K_DOCS:
            raise ValueError("FAISS_TARGET_HITS must not exceed FAISS_TOP_K_DOCS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
