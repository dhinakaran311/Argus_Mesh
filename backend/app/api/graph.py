"""
backend/app/api/graph.py
=========================
AbuseRing Sentinel — React Flow Graph Endpoint
"""
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

from ..models.cluster import ReactFlowGraph

router = APIRouter(tags=["Graph"])


@router.get("/graph/{cluster_id}", response_model=ReactFlowGraph)
async def get_graph(cluster_id: str, request: Request) -> ReactFlowGraph:
    """
    Returns a React Flow-compatible graph for the given cluster_id.
    The cluster_id is the anchor device_id returned by /api/clusters.
    """
    graph = request.app.state.graph.get_react_flow_graph(cluster_id)
    if not graph.nodes:
        raise HTTPException(
            status_code=404,
            detail=f"No graph data found for cluster '{cluster_id}'"
        )
    return graph
