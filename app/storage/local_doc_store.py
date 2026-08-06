import json
from pathlib import Path

from app.config import get_settings
from app.schemas.doc import SourceDoc
from app.utils.paths import resolve_project_path


class LocalDocStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = resolve_project_path(path or get_settings().LOCAL_RAW_DOCS_PATH)

    def load_all(self) -> list[SourceDoc]:
        if not self.path.exists():
            return []

        docs: list[SourceDoc] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(SourceDoc.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"Invalid SourceDoc JSONL at {self.path}:{line_no}") from exc
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


def load_docs_from_json_or_jsonl(path: str | Path) -> list[SourceDoc]:
    input_path = resolve_project_path(path)
    content = input_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    if input_path.suffix.lower() == ".jsonl":
        docs: list[SourceDoc] = []
        for line_no, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(SourceDoc.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid SourceDoc JSONL at {input_path}:{line_no}") from exc
        _validate_unique_doc_ids(docs, input_path)
        return docs

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        parsed = parsed.get("docs", [parsed])
    if not isinstance(parsed, list):
        raise ValueError("Input JSON must be a SourceDoc object, a list, or {'docs': [...]}")
    docs = [SourceDoc.model_validate(item) for item in parsed]
    _validate_unique_doc_ids(docs, input_path)
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
