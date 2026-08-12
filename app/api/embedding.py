import logging

from fastapi import APIRouter

from app.embedding.embedding_service import get_embedding_service
from app.schemas.embedding import Content, Request, Response, Result

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/embedding", tags=["embedding"])
embedding_service = get_embedding_service()


@router.post("", response_model=Response)
async def embedding(request: Request) -> Response:
    if not request.texts.strip():
        return Response(
            result=Result(
                code="2",
                des="texts must not be empty",
                content=None,
            )
        )

    try:
        embeddings = await embedding_service.embed_batch([request.texts])
    except Exception:
        logger.exception("embedding_failed")
        return Response(
            result=Result(
                code="2",
                des="embedding failed",
                content=None,
            )
        )

    return Response(
        result=Result(
            code="0",
            des="success",
            content=[Content(embeddings=embeddings)],
        )
    )
