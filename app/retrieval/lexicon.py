from functools import lru_cache
from pathlib import Path

from app.config import PROJECT_ROOT, get_settings


def _resolve_path(path: str) -> Path:
    target = Path(path)
    if target.is_absolute():
        return target
    return PROJECT_ROOT / target


@lru_cache
def load_stopwords() -> set[str]:
    path = _resolve_path(get_settings().LOCAL_STOPWORDS_PATH)
    if not path.exists():
        return set()

    words: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#"):
            words.add(word)
    return words


@lru_cache
def load_synonyms() -> dict[str, list[str]]:
    path = _resolve_path(get_settings().LOCAL_SYNONYMS_PATH)
    if not path.exists():
        return {}

    synonyms: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip().lower()
        if not raw or raw.startswith("#"):
            continue

        if "=>" in raw:
            source, targets = raw.split("=>", 1)
            terms = [source.strip(), *[item.strip() for item in targets.split(",")]]
        else:
            terms = [item.strip() for item in raw.split(",")]

        terms = [term for term in terms if term]
        for term in terms:
            bucket = synonyms.setdefault(term, set())
            bucket.update(item for item in terms if item != term)

    return {term: sorted(values) for term, values in synonyms.items()}
