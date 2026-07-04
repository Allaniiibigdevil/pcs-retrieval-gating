import logging

from fastapi import APIRouter, HTTPException

from app.decision.query_normalizer import QueryNormalizer
from app.retrieval.factory import get_keyword_retriever, get_vector_retriever
from app.schemas.search import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/search", tags=["search"])
normalizer = QueryNormalizer()


@router.post("/bm25", response_model=SearchResponse)
async def search_bm25(request: SearchRequest) -> SearchResponse:
    query = normalizer.normalize(request.query)
    try:
        hits = await get_keyword_retriever().search(query, request.top_k)
    except Exception as exc:
        logger.exception("keyword_search_failed")
        raise HTTPException(status_code=500, detail="keyword search failed") from exc
    return SearchResponse(query=query, hits=hits)


@router.post("/es", response_model=SearchResponse)
async def search_es_alias(request: SearchRequest) -> SearchResponse:
    return await search_bm25(request)


@router.post("/vector", response_model=SearchResponse)
async def search_vector(request: SearchRequest) -> SearchResponse:
    try:
        hits = await get_vector_retriever().search(request.query, request.top_k)
    except Exception as exc:
        logger.exception("vector_search_failed")
        raise HTTPException(status_code=500, detail="vector search failed") from exc
    return SearchResponse(query=request.query, hits=hits)
