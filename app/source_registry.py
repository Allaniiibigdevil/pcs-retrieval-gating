import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    es_index: str
    enabled: bool
    es_top_k: int
    faiss_top_k: int
    evidence_docs_per_system: int

    selection_threshold: float
    es_score_weight: float
    agreement_weight: float
    semantic_match_threshold: float
    lexical_match_threshold: float

    faiss_preferred_score_threshold: float
    faiss_min_score_threshold: float
    faiss_target_hits: int
    reranker_score_threshold: float


class SourceRegistry:
    def __init__(self, sources: list[SourceConfig]) -> None:
        if not sources:
            raise ValueError("Source registry must contain at least one source")
        self.sources = sources
        self.by_id = {source.source_id: source for source in sources}
        if len(self.by_id) != len(sources):
            raise ValueError("Source registry contains duplicate source ids")

        es_indices = [source.es_index for source in sources]
        if len(set(es_indices)) != len(es_indices):
            raise ValueError("Each source must use a distinct Elasticsearch index")

    @property
    def enabled_sources(self) -> list[SourceConfig]:
        return [source for source in self.sources if source.enabled]

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
    def from_path(
        cls,
        path: str | Path,
        *,
        settings: Any | None = None,
    ) -> "SourceRegistry":
        settings = settings or get_settings()
        config_path = Path(path)
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
            sources.append(_resolve_source(source_id.strip(), raw_config, settings))
        return cls(sources)


def _resolve_source(source_id: str, raw: dict[str, Any], settings: Any) -> SourceConfig:
    source = SourceConfig(
        source_id=source_id,
        es_index=str(
            raw.get("es_index")
            or f"{getattr(settings, 'LOCAL_ES_INDEX', 'pcs_retrieval_docs')}-{_slug(source_id)}"
        ),
        enabled=bool(raw.get("enabled", True)),
        es_top_k=_int(raw, "es_top_k", getattr(settings, "ES_TOP_K_DOCS", 50)),
        faiss_top_k=_int(raw, "faiss_top_k", getattr(settings, "FAISS_TOP_K_DOCS", 50)),
        evidence_docs_per_system=_int(
            raw,
            "evidence_docs_per_system",
            getattr(settings, "EVIDENCE_DOCS_PER_SYSTEM", 3),
        ),
        selection_threshold=_float(
            raw,
            "selection_threshold",
            getattr(settings, "SYSTEM_SELECTION_THRESHOLD", 0.60),
        ),
        es_score_weight=_float(
            raw,
            "es_score_weight",
            getattr(settings, "ES_SCORE_WEIGHT", 0.55),
        ),
        agreement_weight=_float(
            raw,
            "agreement_weight",
            getattr(settings, "AGREEMENT_WEIGHT", 0.20),
        ),
        semantic_match_threshold=_float(
            raw,
            "semantic_match_threshold",
            getattr(settings, "SEMANTIC_MATCH_THRESHOLD", 0.30),
        ),
        lexical_match_threshold=_float(
            raw,
            "lexical_match_threshold",
            getattr(settings, "LEXICAL_MATCH_THRESHOLD", 0.30),
        ),
        faiss_preferred_score_threshold=_float(
            raw,
            "faiss_preferred_score_threshold",
            getattr(settings, "FAISS_PREFERRED_SCORE_THRESHOLD", 0.60),
        ),
        faiss_min_score_threshold=_float(
            raw,
            "faiss_min_score_threshold",
            getattr(settings, "FAISS_MIN_SCORE_THRESHOLD", 0.30),
        ),
        faiss_target_hits=_int(
            raw,
            "faiss_target_hits",
            getattr(settings, "FAISS_TARGET_HITS", 10),
        ),
        reranker_score_threshold=_float(
            raw,
            "reranker_score_threshold",
            getattr(settings, "RERANKER_SCORE_THRESHOLD", 0.50),
        ),
    )
    _validate_source(source)
    return source


def _validate_source(source: SourceConfig) -> None:
    if not source.es_index:
        raise ValueError(f"Source {source.source_id!r} must define an Elasticsearch index")
    for field_name in ("es_top_k", "faiss_top_k", "evidence_docs_per_system", "faiss_target_hits"):
        if getattr(source, field_name) <= 0:
            raise ValueError(f"{source.source_id}.{field_name} must be greater than 0")
    for field_name in (
        "selection_threshold",
        "es_score_weight",
        "agreement_weight",
        "semantic_match_threshold",
        "lexical_match_threshold",
        "reranker_score_threshold",
    ):
        value = getattr(source, field_name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{source.source_id}.{field_name} must be between 0 and 1")
    for field_name in ("faiss_preferred_score_threshold", "faiss_min_score_threshold"):
        value = getattr(source, field_name)
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{source.source_id}.{field_name} must be between -1 and 1")
    if source.faiss_min_score_threshold > source.faiss_preferred_score_threshold:
        raise ValueError(
            f"{source.source_id}.faiss_min_score_threshold must not exceed "
            "faiss_preferred_score_threshold"
        )
    if source.faiss_target_hits > source.faiss_top_k:
        raise ValueError(
            f"{source.source_id}.faiss_target_hits must not exceed faiss_top_k"
        )


def _int(raw: dict[str, Any], key: str, default: int) -> int:
    return int(raw.get(key, default))


def _float(raw: dict[str, Any], key: str, default: float) -> float:
    return float(raw.get(key, default))


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot derive Elasticsearch index name from source_id {value!r}")
    return slug


@lru_cache
def get_source_registry() -> SourceRegistry:
    settings = get_settings()
    return SourceRegistry.from_path(settings.LOCAL_SOURCE_CONFIG_PATH, settings=settings)
