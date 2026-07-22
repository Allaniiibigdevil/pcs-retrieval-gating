import asyncio
import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import PROJECT_ROOT, get_settings
from app.schemas.search import SearchHit

logger = logging.getLogger(__name__)


def build_reranker_passage(hit: SearchHit) -> str:
    """Build content-only input; retrieval scores and source identity stay out."""

    parts: list[str] = []
    if hit.keywords:
        parts.append(f"关键词：{'；'.join(hit.keywords)}")
    if hit.summary:
        parts.append(f"摘要：{hit.summary}")
    return "\n".join(parts)


def attach_reranker_scores(
    candidates: list[SearchHit],
    scores: list[float],
) -> list[SearchHit]:
    if len(candidates) != len(scores):
        raise ValueError(
            "Reranker returned a different number of scores than input candidates"
        )

    scored: list[SearchHit] = []
    for candidate, score in zip(candidates, scores):
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"Reranker returned a non-finite score for {candidate.doc_id}")
        copy = candidate.model_copy(deep=True)
        copy.reranker_score = numeric_score
        scored.append(copy)

    scored.sort(
        key=lambda hit: (
            -(hit.reranker_score if hit.reranker_score is not None else float("-inf")),
            hit.doc_id,
        )
    )
    for rank, hit in enumerate(scored, start=1):
        hit.reranker_rank = rank
    return scored


class GTEReranker:
    """Local Alibaba GTE cross-encoder with lazy, batched inference."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        local_files_only: bool | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model_path = model_path or settings.RERANKER_MODEL_PATH
        self.local_files_only = (
            settings.RERANKER_LOCAL_FILES_ONLY
            if local_files_only is None
            else local_files_only
        )
        self.requested_device = device or settings.RERANKER_DEVICE
        self.batch_size = settings.RERANKER_BATCH_SIZE if batch_size is None else batch_size
        self.max_length = settings.RERANKER_MAX_LENGTH if max_length is None else max_length
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.max_length <= 0:
            raise ValueError("max_length must be greater than 0")

        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    async def rerank(
        self,
        query: str,
        candidates: list[SearchHit],
    ) -> list[SearchHit]:
        if not candidates:
            return []
        passages = [build_reranker_passage(hit) for hit in candidates]
        scores = await asyncio.to_thread(self._predict, query, passages)
        return attach_reranker_scores(candidates, scores)

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        scores: list[float] = []
        with self._inference_lock, self._torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = passages[start : start + self.batch_size]
                pairs = [[query, passage] for passage in batch]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                inputs = {
                    name: tensor.to(self._device) for name, tensor in inputs.items()
                }
                logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
                probabilities = self._torch.sigmoid(logits)
                scores.extend(probabilities.detach().cpu().tolist())
        return [float(score) for score in scores]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            model_source = self._resolve_model_source()
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = _resolve_device(self.requested_device, torch)
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            logger.info(
                "reranker_loading model=%s device=%s max_length=%d",
                model_source,
                device,
                self.max_length,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                local_files_only=self.local_files_only,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                torch_dtype=dtype,
            )
            model.to(device)
            model.eval()

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            logger.info("reranker_loaded model=%s device=%s", model_source, device)

    def _resolve_model_source(self) -> str:
        configured = Path(self.model_path).expanduser()
        local_path = configured if configured.is_absolute() else PROJECT_ROOT / configured
        if local_path.exists():
            return str(local_path.resolve())
        if self.local_files_only:
            raise FileNotFoundError(
                "Local reranker model was not found at "
                f"{local_path}. Set RERANKER_MODEL_PATH to the downloaded "
                "gte-multilingual-reranker-base directory."
            )
        return self.model_path


def _resolve_device(requested: str, torch_module: Any) -> str:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("RERANKER_DEVICE requests CUDA, but CUDA is unavailable")
    if normalized == "cpu" or normalized.startswith("cuda"):
        return normalized
    raise ValueError("RERANKER_DEVICE must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")
