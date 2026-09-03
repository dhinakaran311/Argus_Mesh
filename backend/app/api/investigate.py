"""
backend/app/api/investigate.py
================================
AbuseRing Sentinel — AI Investigation SSE Endpoint

POST /api/investigate
Body: { "cluster_id": "string" }
Response: text/event-stream

Streams 7 events in real time:
  starting → facts → graph → ml → rag → reasoning → complete
"""
from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from ..agents.orchestrator import investigation_graph
from ..models.investigation import InvestigationRequest

router = APIRouter(tags=["Investigation"])
log = logging.getLogger(__name__)

# Thread pool for running synchronous LangGraph in async context
_executor = ThreadPoolExecutor(max_workers=4)


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


@router.post("/investigate")
async def investigate(body: InvestigationRequest, request: Request):
    """
    Start an AI investigation of a ring cluster.
    Streams structured JSON events via Server-Sent Events.
    """
    cluster_id = body.cluster_id
    state      = request.app.state

    # Assemble services dict to inject into LangGraph nodes
    settings = state.settings
    services = {
        "data":          state.data,
        "neo4j":         state.neo4j,
        "ml":            state.ml,
        "vectors":       state.vectors,
        "groq_api_key":  settings.groq_api_key,
        "groq_model":    settings.groq_model,
    }

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
