import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.decision.gating_features import build_gating_feature_rows
from app.ml.mil_mlp import MilMlpGatingModel
from app.schemas.decision import EvidenceDoc, SystemDecision
from app.schemas.search import SearchHit


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _vector_score_norm(hit: SearchHit) -> float:
    if hit.vector_score is not None:
        return _clamp(hit.vector_score)
    return 0.0


def _es_score_norm(hit: SearchHit, max_es_score: float) -> float:
    if hit.bm25_score is None or hit.bm25_score <= 0 or max_es_score <= 0:
        return 0.0
    return _clamp(hit.bm25_score / max_es_score)


def simple_doc_strength(hit: SearchHit, max_es_score: float = 0.0) -> float:
    settings = get_settings()
    semantic_score = _vector_score_norm(hit)
    lexical_score = _es_score_norm(hit, max_es_score)
    agreement_boost = (
        settings.AGREEMENT_WEIGHT * math.sqrt(semantic_score * lexical_score)
        if semantic_score >= settings.SEMANTIC_MATCH_THRESHOLD
        and lexical_score >= settings.LEXICAL_MATCH_THRESHOLD
        else 0.0
    )
    return _clamp(max(semantic_score, settings.ES_SCORE_WEIGHT * lexical_score) + agreement_boost)


def _highlight_from_metadata(doc: SearchHit) -> dict[str, list[str]]:
    highlight = doc.metadata.get("highlight")
    if not isinstance(highlight, dict):
        return {}
    return {
        field: [str(value) for value in highlight[field]]
        for field in ("summary", "keywords")
        if isinstance(highlight.get(field), list)
    }


class SystemAggregator:
    def __init__(
        self,
        selection_threshold: float | None = None,
        *,
        system_thresholds: dict[str, float] | None = None,
        scorer: str | None = None,
        model_path: str | Path | None = None,
        require_calibration: bool | None = None,
        model: MilMlpGatingModel | None = None,
    ) -> None:
        settings = get_settings()
        self.selection_threshold = (
            settings.SYSTEM_SELECTION_THRESHOLD
            if selection_threshold is None
            else selection_threshold
        )
        self.system_thresholds = dict(
            settings.SYSTEM_SELECTION_THRESHOLDS if system_thresholds is None else system_thresholds
        )
        self.scorer = scorer or settings.GATING_SCORER
        if self.scorer not in {"fixed", "mil_mlp"}:
            raise ValueError(f"unsupported gating scorer: {self.scorer}")
        self.model_path = Path(model_path or settings.GATING_MODEL_PATH)
        self.require_calibration = (
            settings.GATING_REQUIRE_CALIBRATION
            if require_calibration is None
            else require_calibration
        )
        self._model = model

    def _mil_doc_scores(
        self, query_text: str, evidence_docs: list[SearchHit]
    ) -> tuple[dict[str, float], str]:
        rows = build_gating_feature_rows(query_text, evidence_docs)
        if not rows:
            return {}, "calibrated_probability"
        features = np.asarray([row.feature_vector() for row in rows], dtype=np.float32)
        if self._model is None:
            self._model = MilMlpGatingModel.load(self.model_path)
        if self.require_calibration and not self._model.calibrated:
            raise RuntimeError("MIL model is not calibrated")
        probabilities = self._model.predict_proba(features)
        kind = "calibrated_probability" if self._model.calibrated else "uncalibrated_probability"
        return (
            {
                row.doc_id: float(probability)
                for row, probability in zip(rows, probabilities, strict=True)
            },
            kind,
        )

    def aggregate(
        self,
        evidence_docs: list[SearchHit],
        max_systems: int = 5,
        *,
        query_text: str = "",
    ) -> list[SystemDecision]:
        grouped: dict[str, list[SearchHit]] = defaultdict(list)
        for doc in evidence_docs:
            grouped[doc.system_id].append(doc)

        max_es_score = max((doc.bm25_score or 0.0 for doc in evidence_docs), default=0.0)
        if self.scorer == "fixed":
            doc_scores = {
                doc.doc_id: simple_doc_strength(doc, max_es_score) for doc in evidence_docs
            }
            score_kind = "heuristic"
        else:
            doc_scores, score_kind = self._mil_doc_scores(query_text, evidence_docs)

        decisions: list[SystemDecision] = []
        for system_id, docs in grouped.items():
            scored_docs = [doc for doc in docs if doc.doc_id in doc_scores]
            scored_docs.sort(key=lambda doc: doc_scores[doc.doc_id], reverse=True)
            top_docs = scored_docs[:3]
            confidence = doc_scores[top_docs[0].doc_id] if top_docs else 0.0
            threshold = self.system_thresholds.get(system_id, self.selection_threshold)
            decisions.append(
                SystemDecision(
                    system_id=system_id,
                    selected=confidence >= threshold,
                    confidence=round(confidence, 4),
                    confidence_kind=score_kind,
                    threshold=threshold,
                    trigger_doc_id=top_docs[0].doc_id if top_docs else None,
                    evidence_docs=[
                        EvidenceDoc(
                            doc_id=doc.doc_id,
                            summary=doc.summary,
                            keywords=list(doc.keywords),
                            matched_keywords=list(doc.metadata.get("matched_keywords", [])),
                            highlight=_highlight_from_metadata(doc),
                            bm25_score=doc.bm25_score,
                            vector_score=doc.vector_score,
                            bm25_rank=doc.bm25_rank,
                            vector_rank=doc.vector_rank,
                            gating_score=round(doc_scores[doc.doc_id], 4),
                        )
                        for doc in top_docs
                    ],
                )
            )

        ranked = sorted(decisions, key=lambda item: item.confidence, reverse=True)
        selected = [item for item in ranked if item.selected]
        if max_systems <= 0:
            return selected
        unselected = [item for item in ranked if not item.selected]
        visible = selected + unselected[: max(max_systems - len(selected), 0)]
        return sorted(visible, key=lambda item: item.confidence, reverse=True)
