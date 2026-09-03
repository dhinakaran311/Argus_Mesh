"""
backend/app/api/clusters.py
=============================
AbuseRing Sentinel — Ring Cluster Endpoints
"""
from __future__ import annotations

from fastapi import APIRouter, Request, Query, HTTPException

from ..models.cluster import ClusterSummary, ClusterDetail

router = APIRouter(tags=["Clusters"])


@router.get("/clusters", response_model=list[ClusterSummary])
async def list_clusters(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
) -> list[ClusterSummary]:
    """
    Returns top abuse ring clusters sorted by combined risk score.
    Optionally filter by minimum risk score.
    """
    clusters = request.app.state.graph.get_clusters(limit=limit)
    if min_risk > 0:
        clusters = [c for c in clusters if c.combined_risk_score >= min_risk]
    return clusters


@router.get("/clusters/{cluster_id}", response_model=ClusterDetail)
async def get_cluster_detail(cluster_id: str, request: Request) -> ClusterDetail:
    """
    Returns full cluster detail including member list and React Flow graph data.
    cluster_id is the device_id that anchors the cluster.
    """
    detail = request.app.state.graph.get_cluster_detail(cluster_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found")
    return detail
