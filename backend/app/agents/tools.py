"""
backend/app/agents/tools.py
=============================
AbuseRing Sentinel — LangGraph Tool Definitions

Five tools used by the investigation nodes:
  1. get_cluster_facts      — DataStore + feature index
  2. get_graph_topology     — Neo4j entity traversal
  3. get_ml_explanation     — XGBoost inference + SHAP
  4. search_similar_cases   — Qdrant semantic search
  5. store_investigation    — Qdrant upsert (case memory)

These are NOT LangChain @tool decorated (to avoid overhead).
They are plain functions called directly by the nodes,
with the services injected at call time.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..db.supabase import DataStore
    from ..db.neo4j import Neo4jClient
    from ..services.ml_service import MLService
    from ..services.vector_service import VectorService
    from ..services.risk_engine import RiskEngine

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1: Cluster Facts
# ---------------------------------------------------------------------------

def get_cluster_facts(cluster_id: str, data: "DataStore", ml: "MLService") -> dict:
    """
    Aggregate cluster member stats from the in-memory DataStore + feature index.
    Returns: size, member ids, avg risk, return rates, account ages, etc.
    """
    members = data.get_customers_in_cluster(cluster_id)
    if not members:
        return {"cluster_id": cluster_id, "error": "No members found in DataStore"}

    risk_scores = [ml.predict(str(m.get("customer_id", ""))) for m in members]
    return_rates = [float(m.get("return_rate", 0.0) or 0.0) for m in members]
    num_txns = [int(m.get("num_transactions", 0) or 0) for m in members]
    ring_type = members[0].get("ring_type", "UNKNOWN") if members else "UNKNOWN"
    abuse_count = sum(1 for m in members if m.get("is_abuse"))

    return {
        "cluster_id":         cluster_id,
        "ring_type":          ring_type,
        "cluster_size":       len(members),
        "abuse_count":        abuse_count,
        "avg_risk_score":     round(sum(risk_scores) / max(len(risk_scores), 1), 4),
        "max_risk_score":     round(max(risk_scores, default=0.0), 4),
        "avg_return_rate":    round(sum(return_rates) / max(len(return_rates), 1), 4),
        "max_return_rate":    round(max(return_rates, default=0.0), 4),
        "total_transactions": sum(num_txns),
        "member_sample":      [str(m.get("customer_id", "")) for m in members[:5]],
        "locations":          list({m.get("location_city", "Unknown") for m in members}),
    }


# ---------------------------------------------------------------------------
# Tool 2: Graph Topology
# ---------------------------------------------------------------------------

def get_graph_topology(cluster_id: str, neo4j: "Neo4jClient") -> dict:
    """
    Traverses Neo4j entity graph for a cluster and returns topology summary.
    """
    rows = neo4j.get_entity_graph(cluster_id)
    if not rows:
        return {"cluster_id": cluster_id, "error": "No graph data found"}

    customer_ids = list({str(r["customer_id"]) for r in rows if r.get("customer_id")})
    device_ids   = list({str(r["device_id"])   for r in rows if r.get("device_id")})
    ip_ids       = list({str(r["ip_id"])        for r in rows if r.get("ip_id")})
    merchant_ids = list({str(r["merchant_id"]) for r in rows if r.get("merchant_id")})
    cities       = list({str(r["ip_city"])      for r in rows if r.get("ip_city")})

    avg_risk = sum(float(r.get("risk_score", 0.0) or 0.0) for r in rows) / max(len(rows), 1)
    abuse_count = sum(1 for r in rows if r.get("is_abuse"))

    return {
        "cluster_id":       cluster_id,
        "total_customers":  len(customer_ids),
        "shared_devices":   len(device_ids),
        "shared_ips":       len(ip_ids),
        "merchants_hit":    len(merchant_ids),
        "ip_cities":        cities[:5],
        "avg_risk_score":   round(avg_risk, 4),
        "abuse_members":    abuse_count,
        "device_ids":       device_ids[:3],
        "ip_ids":           ip_ids[:3],
    }


# ---------------------------------------------------------------------------
# Tool 3: ML Explanation
# ---------------------------------------------------------------------------

def get_ml_explanation(cluster_id: str, data: "DataStore", ml: "MLService") -> dict:
    """
    Runs XGBoost inference + SHAP for all cluster members.
    Returns top 5 globally most important features across the cluster.
    """
    members = data.get_customers_in_cluster(cluster_id)
    if not members:
        return {"cluster_id": cluster_id, "error": "No members"}

    member_ids = [str(m.get("customer_id", "")) for m in members]
    scores = ml.predict_batch(member_ids)

    # Aggregate SHAP across all members
    feature_totals: dict[str, float] = {}
    for cid in member_ids:
        for feat in ml.explain(cid, top_n=10):
            name = feat["feature"]
            feature_totals[name] = feature_totals.get(name, 0.0) + abs(feat["importance"])

    top_features = sorted(
        [{"feature": k, "importance": round(v / max(len(member_ids), 1), 4)}
         for k, v in feature_totals.items()],
        key=lambda x: x["importance"],
        reverse=True,
    )[:5]

    return {
        "cluster_id":       cluster_id,
        "scored_members":   len(scores),
        "avg_ml_score":     round(sum(scores.values()) / max(len(scores), 1), 4),
        "max_ml_score":     round(max(scores.values(), default=0.0), 4),
        "high_risk_count":  sum(1 for s in scores.values() if s >= 0.6),
        "top_shap_features": top_features,
        "member_scores":    [{"customer_id": k, "score": v} for k, v in list(scores.items())[:10]],
    }


# ---------------------------------------------------------------------------
# Tool 4: Search Similar Cases
# ---------------------------------------------------------------------------

def search_similar_cases(query_text: str, vectors: "VectorService", top_k: int = 3) -> list[dict]:
    """Embed query and search Qdrant for similar historical cases."""
    return vectors.find_similar_cases(query_text, top_k=top_k)


# ---------------------------------------------------------------------------
# Tool 5: Store Investigation
# ---------------------------------------------------------------------------

def store_investigation(
    case_id: str,
    summary: str,
    metadata: dict,
    vectors: "VectorService",
) -> bool:
    """Persist a completed investigation into Qdrant for future RAG."""
    return vectors.store_investigation(case_id, summary, metadata)
