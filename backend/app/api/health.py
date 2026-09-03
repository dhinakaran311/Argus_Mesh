"""
backend/app/api/health.py
==========================
AbuseRing Sentinel — Health Check Endpoint
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(request: Request) -> dict:
    app = request.app
    state = app.state

    neo4j_ok   = state.neo4j.health_check()   if hasattr(state, "neo4j")   else False
    qdrant_ok  = state.qdrant.health_check()  if hasattr(state, "qdrant")  else False
    ml_ok      = state.ml.is_loaded()         if hasattr(state, "ml")      else False
    data_ok    = state.data.health_check()    if hasattr(state, "data")    else False

    overall = "ok" if (neo4j_ok and qdrant_ok and ml_ok and data_ok) else "degraded"

    return {
        "status":    overall,
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "neo4j":    "ok" if neo4j_ok  else "error",
            "qdrant":   "ok" if qdrant_ok else "error",
            "ml_model": "ok" if ml_ok     else "error",
            "data":     "ok" if data_ok   else "error",
        },
    }
