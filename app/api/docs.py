import logging

from fastapi import APIRouter, HTTPException

from app.indexing.index_service import IndexService
from app.schemas.doc import UpsertDocRequest, UpsertDocResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/docs", tags=["docs"])
index_service = IndexService()


@router.post("/upsert", response_model=UpsertDocResponse)
async def upsert_doc(request: UpsertDocRequest) -> UpsertDocResponse:
    try:
        await index_service.upsert_doc(request)
    except Exception as exc:
        logger.exception("doc_upsert_failed", extra={"doc_id": request.doc_id})
        raise HTTPException(status_code=500, detail="doc upsert failed") from exc
    return UpsertDocResponse(doc_id=request.doc_id, status="indexed")
