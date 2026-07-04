from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "retrieval-gating-service"
    APP_ENV: str = "local"
    APP_MODE: str = "local"

    LOCAL_DATA_DIR: str = "data"
    LOCAL_RAW_DOCS_PATH: str = "data/raw/docs.jsonl"
    LOCAL_ARTIFACT_DIR: str = "data/artifacts"

    EMBEDDING_PROVIDER: str = "bge"
    EMBEDDING_MODEL_PATH: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIM: int = 512

    DEFAULT_TOP_K_DOCS: int = 50
    DEFAULT_MAX_SYSTEMS: int = 5
    BM25_TOP_K: int = 50
    VECTOR_TOP_K: int = 50

    SYSTEM_SELECTION_THRESHOLD: float = 0.75
    BM25_RANK_WEIGHT: float = 0.75
    KEYWORD_BOOST_PER_MATCH: float = 0.02
    KEYWORD_BOOST_MAX: float = 0.10


@lru_cache
def get_settings() -> Settings:
    return Settings()
