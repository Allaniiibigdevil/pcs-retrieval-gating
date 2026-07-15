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

    DEFAULT_TOP_K_DOCS: int = 20
    DEFAULT_MAX_SYSTEMS: int = 5
    VECTOR_TOP_K: int = 20

    SYSTEM_SELECTION_THRESHOLD: float = 0.60
    SOURCE_SELECTION_THRESHOLDS: dict[str, float] = Field(default_factory=dict)
    ES_SCORE_WEIGHT: float = 0.55
    AGREEMENT_WEIGHT: float = 0.20
    SEMANTIC_MATCH_THRESHOLD: float = 0.30
    LEXICAL_MATCH_THRESHOLD: float = 0.30

    GATING_SCORER: str = "fixed"
    GATING_FEATURE_LOG_ENABLED: bool = True
    GATING_TRAINING_DATA_PATH: str = "data/gating/training_samples.jsonl"
    GATING_MODEL_PATH: str = "data/gating/nine_representative_mil_mlp.json"
    GATING_REQUIRE_CALIBRATION: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
