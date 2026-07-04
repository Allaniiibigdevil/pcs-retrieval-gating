import json
from pathlib import Path

from app.config import get_settings
from app.schemas.doc import SourceDoc


class LocalDocStore:
    def __init__(self, path: str | Path | None = None) -> None:
        settings = get_settings()
        self.path = Path(path or settings.LOCAL_RAW_DOCS_PATH)

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
        return docs

    def save_all(self, docs: list[SourceDoc]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as file:
            for doc in docs:
                file.write(doc.model_dump_json() + "\n")

    def append_many(self, docs: list[SourceDoc]) -> None:
        existing = {doc.doc_id: doc for doc in self.load_all()}
        for doc in docs:
            existing[doc.doc_id] = doc
        self.save_all(list(existing.values()))


def load_docs_from_json_or_jsonl(path: str | Path) -> list[SourceDoc]:
    input_path = Path(path)
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
        return docs

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        parsed = parsed.get("docs", [parsed])
    if not isinstance(parsed, list):
        raise ValueError("Input JSON must be a SourceDoc object, a list, or {'docs': [...]}")
    return [SourceDoc.model_validate(item) for item in parsed]
