from functools import lru_cache
from pathlib import Path

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
    LOCAL_SYNONYMS_PATH: str = "examples/dicts/synonyms.txt"
    LOCAL_STOPWORDS_PATH: str = "examples/dicts/stopwords.txt"

    EMBEDDING_PROVIDER: str = "bge"
    EMBEDDING_MODEL_PATH: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIM: int = 512

    DEFAULT_TOP_K_DOCS: int = 50
    DEFAULT_MAX_SYSTEMS: int = 5
    BM25_TOP_K: int = 50
    VECTOR_TOP_K: int = 50

    SYSTEM_SELECTION_THRESHOLD: float = 0.55
    VECTOR_SCORE_WEIGHT: float = 0.60
    BM25_SCORE_WEIGHT: float = 0.35
    AGREEMENT_BOOST: float = 0.05
    SEMANTIC_MATCH_THRESHOLD: float = 0.35
    LEXICAL_MATCH_THRESHOLD: float = 0.40
    KEYWORD_MATCH_PER_HIT: float = 0.20
    KEYWORD_MATCH_MAX: float = 0.60


@lru_cache
def get_settings() -> Settings:
    return Settings()
