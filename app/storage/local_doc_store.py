import json
from collections.abc import Iterator
from pathlib import Path

from app.config import get_settings
from app.schemas.doc import SourceDoc
from app.utils.paths import resolve_project_path
from app.utils.progress import render_progress


class LocalDocStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = resolve_project_path(path or get_settings().LOCAL_RAW_DOCS_PATH)

    def load_all(self, *, show_progress: bool = False) -> list[SourceDoc]:
        if not self.path.exists():
            return []
        docs = _load_jsonl(self.path, show_progress=show_progress)
        _validate_unique_doc_ids(docs, self.path)
        return docs

    def save_all(self, docs: list[SourceDoc]) -> None:
        _validate_unique_doc_ids(docs, self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as file:
            for doc in docs:
                file.write(doc.model_dump_json() + "\n")

    def append_many(self, docs: list[SourceDoc]) -> None:
        _validate_unique_doc_ids(docs, "incoming documents")
        existing = {doc.doc_id: doc for doc in self.load_all()}
        for doc in docs:
            existing[doc.doc_id] = doc
        self.save_all(list(existing.values()))


def iter_docs_from_jsonl(
    path: str | Path,
    *,
    source_id: str | None = None,
) -> Iterator[SourceDoc]:
    input_path = resolve_project_path(path)
    if input_path.suffix.lower() != ".jsonl":
        raise ValueError("Streaming index build requires a .jsonl input file")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing SourceDoc JSONL: {input_path}")

    with input_path.open("rb") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                doc = SourceDoc.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid SourceDoc JSONL at {input_path}:{line_no}") from exc
            if source_id is None or doc.system_id == source_id:
                yield doc


def count_docs_from_jsonl(
    path: str | Path,
    *,
    source_id: str | None = None,
) -> int:
    return sum(1 for _ in iter_docs_from_jsonl(path, source_id=source_id))


def load_docs_from_json_or_jsonl(
    path: str | Path,
    *,
    show_progress: bool = False,
) -> list[SourceDoc]:
    input_path = resolve_project_path(path)
    if input_path.suffix.lower() == ".jsonl":
        docs = _load_jsonl(input_path, show_progress=show_progress)
        _validate_unique_doc_ids(docs, input_path)
        return docs

    content = input_path.read_text(encoding="utf-8").strip()
    if not content:
        if show_progress:
            render_progress("Loading docs", 0, 0, detail="docs=0")
        return []

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        parsed = parsed.get("docs", [parsed])
    if not isinstance(parsed, list):
        raise ValueError("Input JSON must be a SourceDoc object, a list, or {'docs': [...]}")
    docs = [SourceDoc.model_validate(item) for item in parsed]
    _validate_unique_doc_ids(docs, input_path)
    if show_progress:
        size = input_path.stat().st_size
        render_progress("Loading docs", size, size, detail=f"docs={len(docs)}")
    return docs


def _load_jsonl(path: Path, *, show_progress: bool) -> list[SourceDoc]:
    total_bytes = path.stat().st_size
    processed_bytes = 0
    docs: list[SourceDoc] = []

    with path.open("rb") as file:
        for line_no, raw_line in enumerate(file, start=1):
            processed_bytes += len(raw_line)
            line = raw_line.strip()
            if line:
                try:
                    docs.append(SourceDoc.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"Invalid SourceDoc JSONL at {path}:{line_no}") from exc
            if show_progress and (
                line_no % 1000 == 0 or processed_bytes >= total_bytes
            ):
                render_progress(
                    "Loading docs",
                    processed_bytes,
                    total_bytes,
                    detail=f"docs={len(docs)}",
                )

    if show_progress and total_bytes == 0:
        render_progress("Loading docs", 0, 0, detail="docs=0")
    return docs


def _validate_unique_doc_ids(docs: list[SourceDoc], context: object) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for doc in docs:
        if doc.doc_id in seen and doc.doc_id not in duplicates:
            duplicates.append(doc.doc_id)
        seen.add(doc.doc_id)
    if duplicates:
        raise ValueError(
            f"Duplicate doc_id values in {context}: " + ", ".join(duplicates[:5])
        )
