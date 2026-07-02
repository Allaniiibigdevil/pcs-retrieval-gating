import logging

from fastapi import APIRouter, HTTPException

from app.decision.query_normalizer import QueryNormalizer
from app.retrieval.es_retriever import ESRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.search import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/search", tags=["search"])
normalizer = QueryNormalizer()
es_retriever = ESRetriever()
vector_retriever = VectorRetriever()


@router.post("/es", response_model=SearchResponse)
async def search_es(request: SearchRequest) -> SearchResponse:
    query = normalizer.normalize(request.query)
    try:
        hits = await es_retriever.search(query, request.top_k)
    except Exception as exc:
        logger.exception("es_search_failed")
        raise HTTPException(status_code=500, detail="es search failed") from exc
    return SearchResponse(query=query, hits=hits)


@router.post("/vector", response_model=SearchResponse)
async def search_vector(request: SearchRequest) -> SearchResponse:
    query = normalizer.normalize(request.query)
    try:
        hits = await vector_retriever.search(query, request.top_k)
    except Exception as exc:
        logger.exception("vector_search_failed")
        raise HTTPException(status_code=500, detail="vector search failed") from exc
    return SearchResponse(query=query, hits=hits)
