import json
import logging
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, get_settings
from app.schemas.doc import SourceDoc


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


class LocalArtifactStore:
    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        configured_dir = artifact_dir or get_settings().LOCAL_ARTIFACT_DIR
        self.artifact_dir = _resolve_path(configured_dir)
        self.docs_path = self.artifact_dir / "docs.jsonl"
        self.manifest_path = self.artifact_dir / "manifest.json"

    def ensure_dir(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def save_docs(self, docs: list[SourceDoc]) -> None:
        self.ensure_dir()
        with self.docs_path.open("w", encoding="utf-8", newline="\n") as file:
            for doc in docs:
                file.write(doc.model_dump_json() + "\n")

    def load_docs(self) -> list[SourceDoc]:
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Missing local docs artifact: {self.docs_path}")

        docs: list[SourceDoc] = []
        with self.docs_path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(SourceDoc.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"Invalid docs artifact at {self.docs_path}:{line_no}") from exc
        return docs

    def save_faiss(
        self,
        index: Any,
        doc_ids: list[str],
        *,
        index_path: str | Path,
        doc_ids_path: str | Path,
    ) -> None:
        if int(index.ntotal) != len(doc_ids):
            raise ValueError(
                "FAISS index and doc-id mapping must have the same number of entries"
            )
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("FAISS doc-id mapping contains duplicate doc_id values")

        resolved_index_path = _resolve_path(index_path)
        resolved_doc_ids_path = _resolve_path(doc_ids_path)
        resolved_index_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_doc_ids_path.parent.mkdir(parents=True, exist_ok=True)

        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        import faiss

        faiss.write_index(index, str(resolved_index_path))
        resolved_doc_ids_path.write_text(
            json.dumps(doc_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_faiss(
        self,
        *,
        index_path: str | Path,
        doc_ids_path: str | Path,
    ) -> tuple[Any, list[str]]:
        resolved_index_path = _resolve_path(index_path)
        resolved_doc_ids_path = _resolve_path(doc_ids_path)
        if not resolved_index_path.exists():
            raise FileNotFoundError(f"Missing FAISS artifact: {resolved_index_path}")
        if not resolved_doc_ids_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS doc ids artifact: {resolved_doc_ids_path}"
            )

        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        import faiss

        index = faiss.read_index(str(resolved_index_path))
        raw_doc_ids = json.loads(resolved_doc_ids_path.read_text(encoding="utf-8"))
        if not isinstance(raw_doc_ids, list) or not all(
            isinstance(doc_id, str) and doc_id for doc_id in raw_doc_ids
        ):
            raise ValueError(
                f"Invalid FAISS doc-id mapping: {resolved_doc_ids_path} must contain strings"
            )
        if len(set(raw_doc_ids)) != len(raw_doc_ids):
            raise ValueError(
                f"Invalid FAISS doc-id mapping: {resolved_doc_ids_path} contains duplicates"
            )
        if int(index.ntotal) != len(raw_doc_ids):
            raise RuntimeError(
                "FAISS index and doc-id mapping are inconsistent: "
                f"index contains {int(index.ntotal)} vectors but mapping contains "
                f"{len(raw_doc_ids)} ids"
            )
        return index, raw_doc_ids

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        self.ensure_dir()
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest artifact: {self.manifest_path}")
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))
