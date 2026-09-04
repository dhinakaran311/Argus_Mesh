"""
backend/app/api/clusters.py
=============================
AbuseRing Sentinel — Ring Cluster Endpoints

Falls back to DataStore (CSV) when Neo4j graph has no edges seeded yet.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

from fastapi import APIRouter, Request, Query, HTTPException, Path

from ..models.cluster import ClusterSummary, ClusterDetail, ReactFlowGraph, ReactFlowNode, ReactFlowEdge
from ..models.customer import CustomerSummary
from ..services.risk_engine import RiskEngine

log = logging.getLogger(__name__)
router = APIRouter(tags=["Clusters"])
_re = RiskEngine()


# ── DataStore fallback ──────────────────────────────────────────────────────

def _clusters_from_datastore(ds, limit: int) -> list[ClusterSummary]:
    """Build cluster summaries directly from abuse_labels.csv + customers.csv."""
    if ds.labels.empty:
        return []

    # group labels by cluster_id
    grp = ds.labels.groupby("cluster_id")
    results = []
    for cluster_id, group in grp:
        member_ids = group["customer_id"].tolist()
        ring_type  = str(group["ring_type"].iloc[0]) if "ring_type" in group.columns else "unknown"
        size       = len(member_ids)

        # pull risk scores from customer index
        scores = []
        for cid in member_ids[:50]:  # sample up to 50
            c = ds._customer_idx.get(str(cid))
            if c and c.get("risk_score") is not None:
                scores.append(float(c["risk_score"]))

        avg_ml   = sum(scores) / len(scores) if scores else 0.5
        # graph score proxy: use ring size
        gs       = min(1.0, size / 30) * 0.6 + avg_ml * 0.4
        combined = round(0.5 * avg_ml + 0.5 * gs, 4)
        abuse_ct = int(group["is_abuse"].astype(bool).sum()) if "is_abuse" in group.columns else size

        results.append(ClusterSummary(
            cluster_id          = str(cluster_id),
            cluster_size        = size,
            risk_level          = _re.get_risk_level(combined),
            combined_risk_score = combined,
            avg_ml_score        = round(avg_ml, 4),
            graph_score         = round(gs, 4),
            cluster_return_rate = 0.0,
            total_transactions  = 0,
            total_returns       = 0,
            member_ids          = [str(m) for m in member_ids],
            abuse_count         = abuse_ct,
            ring_type           = ring_type,
        ))

    results.sort(key=lambda c: c.combined_risk_score, reverse=True)
    return results[:limit]


def _graph_from_datastore(ds, cluster_id: str) -> Optional[ReactFlowGraph]:
    """Build a React Flow graph from DataStore for a cluster."""
    if ds.labels.empty:
        return None

    members_df = ds.labels[ds.labels["cluster_id"] == cluster_id]
    if members_df.empty:
        return None

    member_ids = members_df["customer_id"].tolist()
    nodes: list[ReactFlowNode] = []
    edges: list[ReactFlowEdge] = []

    n = len(member_ids)
    for i, cid in enumerate(member_ids[:30]):  # cap at 30 nodes for display
        cust = ds._customer_idx.get(str(cid), {})
        risk  = float(cust.get("risk_score", 0.5))
        angle = (2 * math.pi * i) / max(n, 1)
        radius = 200

        nodes.append(ReactFlowNode(
            id   = f"c:{cid}",
            type = "customer",
            data = {
                "id":         cid[:12] + "…",
                "risk_score": risk,
                "risk_level": _re.get_risk_level(risk),
                "is_abuse":   bool(cust.get("is_abuse", True)),
                "ring_type":  str(cust.get("ring_type", "")),
                "return_rate":float(cust.get("return_rate", 0)),
            },
            position = {
                "x": round(radius * math.cos(angle) + 300),
                "y": round(radius * math.sin(angle) + 250),
            },
        ))

    # Add shared-device proxy edges (connect sequential members to simulate ring)
    for i in range(min(len(member_ids), 30) - 1):
        a = member_ids[i]
        b = member_ids[i + 1]
        edges.append(ReactFlowEdge(
            id     = f"e:{i}",
            source = f"c:{a}",
            target = f"c:{b}",
            label  = "shared device",
            data   = {"type": "device"},
        ))
    # Close the ring
    if len(member_ids) >= 3:
        edges.append(ReactFlowEdge(
            id     = f"e:close",
            source = f"c:{member_ids[min(29, len(member_ids)-1)]}",
            target = f"c:{member_ids[0]}",
            label  = "shared device",
            data   = {"type": "device"},
        ))

    return ReactFlowGraph(nodes=nodes, edges=edges)


def _detail_from_datastore(ds, cluster_id: str) -> Optional[ClusterDetail]:
    clusters = _clusters_from_datastore(ds, 1000)
    summary  = next((c for c in clusters if c.cluster_id == cluster_id), None)
    if not summary:
        return None

    members = []
    for cid in summary.member_ids[:100]:
        c = ds._customer_idx.get(str(cid), {})
        rs = float(c.get("risk_score", 0.5))
        members.append(CustomerSummary(
            customer_id       = str(cid),
            is_abuse          = bool(c.get("is_abuse", True)),
            risk_score        = rs,
            risk_level        = _re.get_risk_level(rs),
            return_rate       = float(c.get("return_rate", 0)),
            num_transactions  = int(c.get("num_transactions", 0)),
            num_orders        = int(c.get("num_orders", 0)),
            num_returns       = int(c.get("num_returns", 0)),
            cluster_id        = str(cid),
            ring_type         = str(c.get("ring_type", "")),
            location_city     = str(c.get("location_city", "")),
            email_domain      = str(c.get("email_domain", "")),
            account_created_at= str(c.get("account_created_at", "")),
        ))

    rf = _graph_from_datastore(ds, cluster_id) or ReactFlowGraph(nodes=[], edges=[])
    return ClusterDetail(**summary.model_dump(), members=members, react_flow_graph=rf)


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/clusters", response_model=list[ClusterSummary])
async def list_clusters(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
) -> list[ClusterSummary]:
    """Rings ranked by combined risk. Falls back to DataStore if Neo4j has no edges."""
    try:
        clusters = request.app.state.graph.get_clusters(limit=limit)
    except Exception as e:
        log.warning(f"Neo4j cluster query failed ({e}), using DataStore fallback")
        clusters = []

    if not clusters:
        log.info("Neo4j returned no clusters — using DataStore fallback")
        clusters = _clusters_from_datastore(request.app.state.data, limit)

    if min_risk > 0:
        clusters = [c for c in clusters if c.combined_risk_score >= min_risk]
    return clusters


@router.get("/clusters/{cluster_id}", response_model=ClusterDetail)
async def get_cluster_detail(
    cluster_id: str = Path(..., min_length=1, max_length=64),
    request: Request = None,
) -> ClusterDetail:
    """Full cluster detail. Falls back to DataStore if Neo4j has no edges."""
    detail = None
    try:
        detail = request.app.state.graph.get_cluster_detail(cluster_id)
    except Exception as e:
        log.warning(f"Neo4j detail query failed ({e}), using DataStore fallback")

    if not detail:
        detail = _detail_from_datastore(request.app.state.data, cluster_id)

    if not detail:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found")
    return detail


@router.get("/graph/{cluster_id}")
async def get_cluster_graph(
    cluster_id: str = Path(..., min_length=1, max_length=64),
    request: Request = None,
) -> dict:
    """React Flow graph data for a cluster. Falls back to DataStore layout."""
    rf = None

    # Primary: ask GraphService to build the React Flow graph directly from Neo4j
    try:
        rf = request.app.state.graph.get_react_flow_graph(cluster_id)
    except Exception as e:
        log.warning(f"Neo4j graph query failed ({e}), using DataStore fallback")

    # DataStore fallback: works for RING-XXX label IDs from abuse_labels.csv
    if not rf or not rf.nodes:
        rf = _graph_from_datastore(request.app.state.data, cluster_id)

    # Last resort: return an empty graph rather than 404 so the UI shows
    # "Graph data unavailable" instead of crashing with a network error
    if not rf or not rf.nodes:
        return {"nodes": [], "edges": []}

    return {
        "nodes": [n.model_dump() for n in rf.nodes],
        "edges": [e.model_dump() for e in rf.edges],
    }
