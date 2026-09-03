"""
backend/app/api/dashboard.py
==============================
AbuseRing Sentinel — Dashboard Overview Endpoint
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from ..models.risk import DashboardStats
from ..services.risk_engine import RiskEngine

router = APIRouter(tags=["Dashboard"])
_engine = RiskEngine()


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard(request: Request) -> DashboardStats:
    """
    Returns aggregate statistics for the dashboard overview panel.
    Sourced from Neo4j graph + in-memory data store.
    """
    state = request.app.state

    # Primary stats from Neo4j
    neo4j_stats = state.graph.get_dashboard_summary()

    # Fallback to in-memory data store if Neo4j is unavailable
    if not neo4j_stats:
        summary = state.data.summary_stats()
        return DashboardStats(
            total_customers=summary.get("customers", 0),
            abuse_customers=0,
            abuse_rate_pct=0.0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            total_rings=0,
            avg_risk_score=0.0,
            total_devices=summary.get("devices", 0),
            high_share_devices=0,
            total_ips=summary.get("ips", 0),
        )

    return DashboardStats(
        total_customers=int(neo4j_stats.get("total_customers", 0)),
        abuse_customers=int(neo4j_stats.get("abuse_customers", 0)),
        abuse_rate_pct=float(neo4j_stats.get("abuse_rate_pct", 0.0)),
        critical_count=int(neo4j_stats.get("critical_count", 0)),
        high_count=int(neo4j_stats.get("high_count", 0)),
        medium_count=int(neo4j_stats.get("medium_count", 0)),
        low_count=int(neo4j_stats.get("low_count", 0)),
        total_rings=int(neo4j_stats.get("total_rings", 0)),
        avg_risk_score=float(neo4j_stats.get("avg_risk_score", 0.0)),
        total_devices=int(neo4j_stats.get("total_devices", 0)),
        high_share_devices=int(neo4j_stats.get("high_share_devices", 0)),
        total_ips=int(neo4j_stats.get("total_ips", 0)),
    )
