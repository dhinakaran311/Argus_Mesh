"""
backend/app/api/rag.py
=======================
AbuseRing Sentinel — RAG Similarity Search Endpoint
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from ..models.investigation import RAGRequest

router = APIRouter(tags=["RAG"])


@router.post("/rag/similar")
async def find_similar_cases(body: RAGRequest, request: Request) -> dict:
    """
    Finds the top-k most similar historical investigation cases
    stored in Qdrant, given a free-text query.
    """
    results = request.app.state.vectors.find_similar_cases(
        query_text=body.query,
        top_k=body.top_k,
    )
    return {"query": body.query, "results": results}


@router.get("/rag/cases")
async def list_cases(request: Request) -> dict:
    """Lists all investigation cases stored in Qdrant."""
    cases = request.app.state.vectors.get_all_cases()
    return {"count": len(cases), "cases": cases}
