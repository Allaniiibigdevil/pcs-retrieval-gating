import logging

from fastapi import APIRouter, HTTPException

from app.reranking.factory import get_reranker
from app.schemas.rerank import RerankRequest, RerankResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/rerank", tags=["rerank"])
reranker = get_reranker()


@router.post("", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    try:
        score = await reranker.score(request.query, request.doc)
    except Exception as exc:
        logger.exception("rerank_failed")
        raise HTTPException(status_code=500, detail="rerank failed") from exc
    return RerankResponse(score=score)
