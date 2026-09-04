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
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..db.datastore import DataStore
    from ..db.neo4j import Neo4jClient
    from ..services.ml_service import MLService
    from ..services.vector_service import VectorService

log = logging.getLogger(__name__)

# UUID pattern — device node IDs returned by Neo4j (e.g. 4e7ccaff-419d-44dc-8f07-97fd8cfe9fd2)
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


def _is_uuid(value: str) -> bool:
    """Return True if value looks like a UUID4 device-node ID from Neo4j."""
    return bool(_UUID_RE.match(value.strip()))


# ---------------------------------------------------------------------------
# Tool 1: Cluster Facts
# ---------------------------------------------------------------------------

def get_cluster_facts(
    cluster_id: str,
    data: "DataStore",
    ml: "MLService",
    neo4j: Optional["Neo4jClient"] = None,
) -> dict:
    """
    Aggregate cluster member stats from DataStore + feature index.

    #2: Two-path resolution for the cluster_id identity mismatch:
      - Primary:   DataStore.get_customers_in_cluster(cluster_id) — works when
                   the ID is a label like "RING-001" / "FAM-042" (abuse_labels.csv).
      - Fallback:  Neo4j get_cluster_members(device_id) — works when the ID is a
                   UUID device-node ID (e.g. "4e7ccaff-419d-...") returned by Neo4j.
    UUID IDs skip the DataStore primary path to avoid the wasted lookup.
    """
    # Only try DataStore when the ID is NOT a UUID device ID
    members = [] if _is_uuid(cluster_id) else data.get_customers_in_cluster(cluster_id)

    # Fallback: resolve via Neo4j then enrich via DataStore.
    if not members and neo4j is not None:
        try:
            neo4j_rows = neo4j.get_cluster_members(cluster_id)
            if neo4j_rows:
                customer_ids = [str(r.get("customer_id", "")) for r in neo4j_rows if r.get("customer_id")]
                # Enrich: try DataStore first, fall back to raw Neo4j row
                ds_members = data.get_customers_by_ids(customer_ids)
                ds_by_id = {str(m["customer_id"]): m for m in ds_members}
                members = [ds_by_id.get(cid, {
                    "customer_id":      cid,
                    "is_abuse":         neo4j_row.get("is_abuse"),
                    "risk_score":       neo4j_row.get("risk_score", 0.0),
                    "return_rate":      neo4j_row.get("return_rate", 0.0),
                    "num_transactions": neo4j_row.get("num_transactions", 0),
                    "ring_type":        neo4j_row.get("ring_type", ""),
                    "location_city":    neo4j_row.get("location_city", ""),
                }) for cid, neo4j_row in zip(customer_ids, neo4j_rows)]
        except Exception as exc:
            log.warning(f"[tools] Neo4j member fallback failed for {cluster_id}: {exc}")

    if not members:
        return {"cluster_id": cluster_id, "error": "No members found in DataStore or Neo4j"}

    risk_scores  = [ml.predict(str(m.get("customer_id", ""))) for m in members]
    return_rates = [float(m.get("return_rate", 0.0) or 0.0) for m in members]
    num_txns     = [int(m.get("num_transactions", 0) or 0) for m in members]
    ring_type    = members[0].get("ring_type", "UNKNOWN") if members else "UNKNOWN"
    abuse_count  = sum(1 for m in members if m.get("is_abuse"))

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

def get_graph_topology(cluster_id: str, neo4j: Optional["Neo4jClient"]) -> dict:
    """
    Traverses Neo4j entity graph for a cluster and returns topology summary.
    """
    if neo4j is None:
        log.warning(f"[tools] get_graph_topology: neo4j client is None for {cluster_id}")
        return {"cluster_id": cluster_id, "error": "Neo4j client not available"}
    rows = neo4j.get_entity_graph(cluster_id)
    if not rows:
        return {"cluster_id": cluster_id, "error": "No graph data found"}

    customer_ids = list({str(r["customer_id"]) for r in rows if r.get("customer_id")})
    device_ids   = list({str(r["device_id"])   for r in rows if r.get("device_id")})
    ip_ids       = list({str(r["ip_id"])        for r in rows if r.get("ip_id")})
    merchant_ids = list({str(r["merchant_id"]) for r in rows if r.get("merchant_id")})
    cities       = list({str(r["ip_city"])      for r in rows if r.get("ip_city")})

    avg_risk    = sum(float(r.get("risk_score", 0.0) or 0.0) for r in rows) / max(len(rows), 1)
    abuse_count = sum(1 for r in rows if r.get("is_abuse"))

    return {
        "cluster_id":      cluster_id,
        "total_customers": len(customer_ids),
        "shared_devices":  len(device_ids),
        "shared_ips":      len(ip_ids),
        "merchants_hit":   len(merchant_ids),
        "ip_cities":       cities[:5],
        "avg_risk_score":  round(avg_risk, 4),
        "abuse_members":   abuse_count,
        "device_ids":      device_ids[:3],
        "ip_ids":          ip_ids[:3],
    }


# ---------------------------------------------------------------------------
# Tool 3: ML Explanation
# ---------------------------------------------------------------------------

def get_ml_explanation(
    cluster_id: str,
    data: "DataStore",
    ml: "MLService",
    neo4j: Optional["Neo4jClient"] = None,
) -> dict:
    """
    Runs XGBoost inference + SHAP for all cluster members.
    Returns top 5 globally most important features across the cluster.

    #2: Same two-path resolution as get_cluster_facts for device vs label IDs.
    UUID IDs skip the DataStore primary path.
    """
    # Only try DataStore when the ID is NOT a UUID device ID
    members = [] if _is_uuid(cluster_id) else data.get_customers_in_cluster(cluster_id)

    # Fallback: resolve via Neo4j if DataStore returned nothing
    if not members and neo4j is not None:
        try:
            neo4j_rows = neo4j.get_cluster_members(cluster_id)
            if neo4j_rows:
                customer_ids = [str(r.get("customer_id", "")) for r in neo4j_rows if r.get("customer_id")]
                ds_members = data.get_customers_by_ids(customer_ids)
                if ds_members:
                    members = ds_members
                else:
                    # Still nothing in DataStore — use synthetic stubs so scoring works
                    members = [{"customer_id": cid} for cid in customer_ids]
        except Exception as exc:
            log.warning(f"[tools] Neo4j ML fallback failed for {cluster_id}: {exc}")

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
        "cluster_id":        cluster_id,
        "scored_members":    len(scores),
        "avg_ml_score":      round(sum(scores.values()) / max(len(scores), 1), 4),
        "max_ml_score":      round(max(scores.values(), default=0.0), 4),
        "high_risk_count":   sum(1 for s in scores.values() if s >= 0.6),
        "top_shap_features": top_features,
        "member_scores":     [{"customer_id": k, "score": v} for k, v in list(scores.items())[:10]],
    }


# ---------------------------------------------------------------------------
# Tool 4: Search Similar Cases
# ---------------------------------------------------------------------------

def search_similar_cases(query_text: str, vectors: Optional["VectorService"], top_k: int = 3) -> list[dict]:
    """Embed query and search Qdrant for similar historical cases."""
    if vectors is None:
        log.warning("[tools] search_similar_cases: VectorService is None")
        return []
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
