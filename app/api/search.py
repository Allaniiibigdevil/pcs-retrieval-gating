import logging

from fastapi import APIRouter, HTTPException

from app.decision.query_normalizer import QueryNormalizer
from app.retrieval.factory import build_keyword_retriever, build_vector_retriever
from app.schemas.search import SearchRequest, SearchResponse
from app.source_registry import SourceConfig, get_source_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/search", tags=["search"])
normalizer = QueryNormalizer()


@router.post("/bm25", response_model=SearchResponse)
async def search_bm25(request: SearchRequest) -> SearchResponse:
    source = _resolve_source(request.source_id)
    query = normalizer.normalize(request.query)
    try:
        hits = await build_keyword_retriever(source).search(query, request.top_k)
    except Exception as exc:
        logger.exception("keyword_search_failed", extra={"source_id": source.source_id})
        raise HTTPException(status_code=500, detail="keyword search failed") from exc
    return SearchResponse(query=query, source_id=source.source_id, hits=hits)


@router.post("/es", response_model=SearchResponse)
async def search_es_alias(request: SearchRequest) -> SearchResponse:
    return await search_bm25(request)


@router.post("/vector", response_model=SearchResponse)
async def search_vector(request: SearchRequest) -> SearchResponse:
    source = _resolve_source(request.source_id)
    try:
        hits = await build_vector_retriever(source).search(request.query, request.top_k)
    except Exception as exc:
        logger.exception("vector_search_failed", extra={"source_id": source.source_id})
        raise HTTPException(status_code=500, detail="vector search failed") from exc
    return SearchResponse(query=request.query, source_id=source.source_id, hits=hits)


def _resolve_source(source_id: str | None) -> SourceConfig:
    registry = get_source_registry()
    if source_id:
        try:
            source = registry.require(source_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown source_id: {source_id}") from exc
        if not source.enabled:
            raise HTTPException(status_code=409, detail=f"source is disabled: {source_id}")
        return source

    enabled_sources = registry.enabled_sources
    if len(enabled_sources) == 1:
        return enabled_sources[0]
    raise HTTPException(
        status_code=400,
        detail="source_id is required when multiple sources are enabled",
    )
