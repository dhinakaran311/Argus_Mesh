"""
backend/app/api/rag.py
=======================
AbuseRing Sentinel — RAG Similarity Search Endpoint
#8: Protected by X-API-Key + rate limit (10 req/min) since it triggers paid embedding calls.
"""

import logging

from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..models.investigation import RAGRequest
from .investigate import _require_api_key

log = logging.getLogger(__name__)
router = APIRouter(tags=["RAG"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/rag/similar", dependencies=[Depends(_require_api_key)])
@limiter.limit("10/minute")
async def find_similar_cases(body: RAGRequest, request: Request) -> dict:
    """
    Finds the top-k most similar historical investigation cases
    stored in Qdrant, given a free-text query.
    Falls back to empty list if Qdrant is unavailable.
    """
    try:
        results = request.app.state.vectors.find_similar_cases(
            query_text=body.query,
            top_k=body.top_k,
        )
    except Exception as e:
        log.warning(f"Qdrant similarity search failed ({e}), returning empty results")
        results = []
    return {"query": body.query, "results": results}


@router.get("/rag/cases", dependencies=[Depends(_require_api_key)])
@limiter.limit("10/minute")
async def list_cases(request: Request) -> dict:
    """Lists all investigation cases stored in Qdrant."""
    try:
        cases = request.app.state.vectors.get_all_cases()
    except Exception as e:
        log.warning(f"Qdrant list cases failed ({e}), returning empty list")
        cases = []
    return {"count": len(cases), "cases": cases}
