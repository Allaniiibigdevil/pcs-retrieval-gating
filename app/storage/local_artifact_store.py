import json
import logging
import pickle
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.schemas.doc import SourceDoc


class LocalArtifactStore:
    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self.artifact_dir = Path(artifact_dir or settings.LOCAL_ARTIFACT_DIR)
        self.docs_path = self.artifact_dir / "docs.jsonl"
        self.bm25_path = self.artifact_dir / "bm25.pkl"
        self.faiss_path = self.artifact_dir / "faiss.index"
        self.faiss_doc_ids_path = self.artifact_dir / "faiss_doc_ids.json"
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

    def save_bm25(self, bm25_index: Any, tokenized_docs: list[list[str]]) -> None:
        self.ensure_dir()
        with self.bm25_path.open("wb") as file:
            pickle.dump({"index": bm25_index, "tokenized_docs": tokenized_docs}, file)

    def load_bm25(self) -> tuple[Any, list[list[str]]]:
        if not self.bm25_path.exists():
            raise FileNotFoundError(f"Missing BM25 artifact: {self.bm25_path}")

        with self.bm25_path.open("rb") as file:
            payload = pickle.load(file)
        return payload["index"], payload["tokenized_docs"]

    def save_faiss(self, index: Any, doc_ids: list[str]) -> None:
        self.ensure_dir()
        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        import faiss

        faiss.write_index(index, str(self.faiss_path))
        self.faiss_doc_ids_path.write_text(
            json.dumps(doc_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_faiss(self) -> tuple[Any, list[str]]:
        if not self.faiss_path.exists():
            raise FileNotFoundError(f"Missing FAISS artifact: {self.faiss_path}")
        if not self.faiss_doc_ids_path.exists():
            raise FileNotFoundError(f"Missing FAISS doc ids artifact: {self.faiss_doc_ids_path}")

        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        import faiss

        index = faiss.read_index(str(self.faiss_path))
        doc_ids = json.loads(self.faiss_doc_ids_path.read_text(encoding="utf-8"))
        return index, doc_ids

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
