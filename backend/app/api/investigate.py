"""
backend/app/api/investigate.py
================================
AbuseRing Sentinel — AI Investigation SSE Endpoint

POST /api/investigate
Body: { "cluster_id": "string" }
Response: text/event-stream

Streams 7 events in real time:
  starting → facts → graph → ml → rag → reasoning → complete

#8: Protected by X-API-Key header check + per-IP rate limit (5 req/min).
    Set SECRET_KEY in .env; pass it as X-API-Key header.
"""
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from ..agents.orchestrator import investigation_graph
from ..config import get_settings
from ..models.investigation import InvestigationRequest

router = APIRouter(tags=["Investigation"])
log = logging.getLogger(__name__)

# Thread pool for running synchronous LangGraph in async context
_executor = ThreadPoolExecutor(max_workers=4)

# #8: Rate limiter — 5 investigations per minute per IP
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _require_api_key(x_api_key: str = Header(default="")) -> None:
    """#8: Reject requests without a valid X-API-Key header in production."""
    settings = get_settings()
    # Bypassed in local development mode or if secret_key is placeholder
    if settings.environment.lower() == "development" or settings.secret_key in ("", "changeme"):
        return
    if x_api_key != settings.secret_key:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key")


def _run_graph(cluster_id: str, services: dict) -> list[dict]:
    """
    Run the LangGraph pipeline synchronously (in executor thread).
    Returns the accumulated SSE events list.
    """
    initial_state = {
        "cluster_id":    cluster_id,
        "facts":         None,
        "graph_data":    None,
        "ml_data":       None,
        "similar_cases": None,
        "report":        None,
        "events":        [],
        "error":         None,
        "_services":     services,   # injected for tools
    }
    final_state = investigation_graph.invoke(initial_state)
    return final_state.get("events", [])


@router.post("/investigate", dependencies=[Depends(_require_api_key)])
@limiter.limit("5/minute")
async def investigate(body: InvestigationRequest, request: Request):
    """
    Start an AI investigation of a ring cluster.
    Streams structured JSON events via Server-Sent Events.

    Requires: X-API-Key header matching SECRET_KEY (when set in .env).
    Rate limited: 5 requests per minute per IP.
    """
    cluster_id = body.cluster_id
    state      = request.app.state

    # Assemble services dict to inject into LangGraph nodes
    # Use getattr with None defaults — app.state attributes are set by lifespan
    # but may be None if startup failed gracefully.
    settings    = getattr(state, "settings", get_settings())
    groq_key    = settings.groq_api_key or ""

    # Also try os.environ as fallback (ChatGroq validates key is non-empty)
    import os as _os
    if not groq_key:
        groq_key = _os.environ.get("GROQ_API_KEY", "")

    services = {
        "data":          getattr(state, "data",    None),
        "neo4j":         getattr(state, "neo4j",   None),
        "ml":            getattr(state, "ml",      None),
        "vectors":       getattr(state, "vectors", None),
        "groq_api_key":  groq_key,
        "groq_model":    settings.groq_model,
    }

    # Diagnostic: log service availability to catch None injection issues
    log.info(
        f"[investigate] services for {cluster_id}: "
        f"data={type(services['data']).__name__}, "
        f"neo4j={type(services['neo4j']).__name__}, "
        f"ml={type(services['ml']).__name__}, "
        f"vectors={type(services['vectors']).__name__}, "
        f"model={services['groq_model']}, "
        f"key={'SET' if services['groq_api_key'] else 'MISSING'}"
    )

    async def event_generator():
        # Emit "starting" immediately
        yield {
            "data": json.dumps({
                "step":    "starting",
                "message": f"Initialising investigation for cluster {cluster_id}...",
                "data":    None,
            })
        }

        try:
            # Run LangGraph in thread pool (it's synchronous)
            loop = asyncio.get_event_loop()
            events = await loop.run_in_executor(
                _executor,
                _run_graph,
                cluster_id,
                services,
            )
            # Stream each event as it was collected
            for event in events:
                yield {"data": json.dumps(event, default=str)}
                await asyncio.sleep(0)  # yield control to event loop

        except Exception as e:
            log.error(f"Investigation error for {cluster_id}: {e}")
            yield {
                "data": json.dumps({
                    "step":    "error",
                    "message": f"Investigation failed: {str(e)}",
                    "data":    None,
                })
            }

    return EventSourceResponse(event_generator())
