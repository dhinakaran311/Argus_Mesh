"""
backend/app/agents/retrieval.py
==================================
AbuseRing Sentinel — Retrieval Node

Builds a summary string from the evidence gathered by the Investigator,
then searches Qdrant for the top-3 most similar historical cases.

Appends one SSE event: "rag"
"""
from __future__ import annotations

import logging

from .state import InvestigationState
from .tools import search_similar_cases

log = logging.getLogger(__name__)


def _build_query_text(state: InvestigationState) -> str:
    """Construct a human-readable cluster summary for Qdrant embedding."""
    facts = state.get("facts") or {}
    graph = state.get("graph_data") or {}
    ml    = state.get("ml_data") or {}

    size        = facts.get("cluster_size", "?")
    ring_type   = facts.get("ring_type", "UNKNOWN")
    avg_rr      = facts.get("avg_return_rate", 0.0)
    total_txns  = facts.get("total_transactions", 0)
    shared_devs = graph.get("shared_devices", 0)
    shared_ips  = graph.get("shared_ips", 0)
    avg_ml      = ml.get("avg_ml_score", 0.0)
    top_feats   = ml.get("top_shap_features", [])[:3]
    feat_names  = ", ".join(f["feature"] for f in top_feats)

    return (
        f"{size} accounts in a {ring_type} cluster. "
        f"Average return rate {avg_rr:.1%}. "
        f"Sharing {shared_devs} device(s) and {shared_ips} IP(s). "
        f"Total transactions: {total_txns}. "
        f"Average ML risk score: {avg_ml:.2f}. "
        f"Key risk signals: {feat_names}."
    )


def retrieval_node(state: InvestigationState) -> InvestigationState:
    """LangGraph node: search Qdrant for similar historical cases."""
    events = list(state.get("events", []))
    _services = state.get("_services", {})  # type: ignore
    vectors = _services.get("vectors")

    query_text = _build_query_text(state)
    log.info(f"  [retrieval] query: {query_text[:100]}...")

    try:
        similar = search_similar_cases(query_text, vectors, top_k=3)
        events.append({
            "step":    "rag",
            "message": f"Found {len(similar)} similar historical case(s)",
            "data":    {"query": query_text, "cases": similar},
        })
        log.info(f"  [retrieval] found {len(similar)} similar cases")
    except Exception as e:
        log.error(f"  [retrieval] error: {e}")
        similar = []
        events.append({"step": "rag", "message": "Case retrieval failed", "data": {"cases": []}})

    return {
        **state,
        "similar_cases": similar,
        "events":        events,
    }
