import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import get_settings
from app.utils.paths import resolve_project_path


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    es_index: str = Field(min_length=1)
    faiss_index_path: str = Field(min_length=1)
    faiss_doc_ids_path: str = Field(min_length=1)
    enabled: bool = True
    es_top_k: int = Field(ge=1, le=1000)
    faiss_top_k: int = Field(ge=1, le=1000)
    evidence_docs_per_system: int = Field(ge=1, le=100)
    faiss_preferred_score_threshold: float = Field(ge=-1.0, le=1.0)
    faiss_min_score_threshold: float = Field(ge=-1.0, le=1.0)
    faiss_target_hits: int = Field(ge=1, le=1000)
    reranker_score_threshold: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_vector_thresholds(self) -> "SourceConfig":
        if self.faiss_min_score_threshold > self.faiss_preferred_score_threshold:
            raise ValueError(
                "faiss_min_score_threshold must not exceed "
                "faiss_preferred_score_threshold"
            )
        if self.faiss_target_hits > self.faiss_top_k:
            raise ValueError("faiss_target_hits must not exceed faiss_top_k")
        return self


class SourceRegistry:
    def __init__(self, sources: list[SourceConfig], *, config_path: str | Path | None = None) -> None:
        if not sources:
            raise ValueError("Source registry must contain at least one source")

        self.sources = tuple(sources)
        self.config_path = (
            resolve_project_path(config_path) if config_path is not None else None
        )
        self.by_id = {source.source_id: source for source in sources}
        if len(self.by_id) != len(sources):
            raise ValueError("Source registry contains duplicate source ids")

        es_indices = [source.es_index for source in sources]
        if len(set(es_indices)) != len(es_indices):
            raise ValueError("Each source must use a distinct Elasticsearch index")

        faiss_paths = [
            str(resolve_project_path(path))
            for source in sources
            for path in (source.faiss_index_path, source.faiss_doc_ids_path)
        ]
        if len(set(faiss_paths)) != len(faiss_paths):
            raise ValueError("Each source must use distinct FAISS artifact paths")

    @property
    def enabled_sources(self) -> tuple[SourceConfig, ...]:
        return tuple(source for source in self.sources if source.enabled)

    def require(self, source_id: str) -> SourceConfig:
        try:
            return self.by_id[source_id]
        except KeyError as exc:
            raise KeyError(f"Unknown source_id {source_id!r}") from exc

    def validate_source_ids(self, source_ids: list[str] | set[str]) -> None:
        unknown = sorted(set(source_ids) - set(self.by_id))
        if unknown:
            raise ValueError(
                "Documents reference source ids missing from the source registry: "
                + ", ".join(unknown)
            )

    @classmethod
    def from_path(cls, path: str | Path) -> "SourceRegistry":
        config_path = resolve_project_path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Missing source configuration: {config_path}")

        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw_sources = payload.get("sources") if isinstance(payload, dict) else None
        if not isinstance(raw_sources, dict):
            raise ValueError("Source configuration must contain an object named 'sources'")

        sources: list[SourceConfig] = []
        for source_id, raw_config in raw_sources.items():
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError("Source ids must be non-empty strings")
            if not isinstance(raw_config, dict):
                raise ValueError(f"Source {source_id!r} configuration must be an object")
            sources.append(
                SourceConfig.model_validate({"source_id": source_id, **raw_config})
            )
        return cls(sources, config_path=config_path)


@lru_cache
def get_source_registry() -> SourceRegistry:
    return SourceRegistry.from_path(get_settings().LOCAL_SOURCE_CONFIG_PATH)
