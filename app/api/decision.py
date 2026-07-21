import logging

from fastapi import APIRouter, HTTPException

from app.decision.decision_engine import DecisionEngine
from app.schemas.decision import DecideRequest, DecideResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["decision"])
decision_engine = DecisionEngine()


@router.post("/decide", response_model=DecideResponse)
async def decide(request: DecideRequest) -> DecideResponse:
    try:
        return await decision_engine.decide(
            task=request.task,
            task_id=request.task_id,
        )
    except Exception as exc:
        logger.exception("decision_failed", extra={"task_id": request.task_id})
        raise HTTPException(status_code=500, detail="decision failed") from exc
