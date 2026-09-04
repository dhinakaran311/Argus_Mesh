"""
backend/app/agents/investigator.py
=====================================
AbuseRing Sentinel — Investigator Node

Collects raw evidence about a cluster:
  1. Cluster facts (from DataStore + ML feature index)
  2. Graph topology (from Neo4j)
  3. ML risk scores + SHAP explanations (from XGBoost)

Appends three SSE events: "facts", "graph", "ml"
"""
from __future__ import annotations

import logging

from .state import InvestigationState
from .tools import get_cluster_facts, get_graph_topology, get_ml_explanation

log = logging.getLogger(__name__)


def investigator_node(state: InvestigationState) -> InvestigationState:
    """LangGraph node: gather cluster evidence."""
    cluster_id = state["cluster_id"]
    events = list(state.get("events", []))

    # Retrieve services from the global app state
    # (injected via the orchestrator before graph.invoke())
    _services = state.get("_services", {})  # type: ignore
    data    = _services.get("data")
    neo4j   = _services.get("neo4j")
    ml      = _services.get("ml")

    # -- Facts ---------------------------------------------------------------
    try:
        facts = get_cluster_facts(cluster_id, data, ml, neo4j=neo4j)
        events.append({
            "step":    "facts",
            "message": f"Retrieved facts for cluster {cluster_id}",
            "data":    facts,
        })
        log.info(f"  [investigator] facts done: {facts.get('cluster_size')} members")
    except Exception as e:
        log.error(f"  [investigator] facts error: {e}")
        facts = {"cluster_id": cluster_id, "error": str(e)}
        events.append({"step": "facts", "message": "Could not retrieve cluster facts", "data": facts})

    # -- Graph topology ------------------------------------------------------
    try:
        graph_data = get_graph_topology(cluster_id, neo4j)
        events.append({
            "step":    "graph",
            "message": f"Graph traversal complete — {graph_data.get('shared_devices')} shared device(s)",
            "data":    graph_data,
        })
        log.info(f"  [investigator] graph done: {graph_data.get('shared_devices')} devices")
    except Exception as e:
        log.error(f"  [investigator] graph error: {e}")
        graph_data = {"cluster_id": cluster_id, "error": str(e)}
        events.append({"step": "graph", "message": "Graph traversal failed", "data": graph_data})

    # -- ML explanation ------------------------------------------------------
    try:
        ml_data = get_ml_explanation(cluster_id, data, ml, neo4j=neo4j)
        events.append({
            "step":    "ml",
            "message": f"ML scoring complete — avg risk {ml_data.get('avg_ml_score', 0.0):.2f}",
            "data":    ml_data,
        })
        log.info(f"  [investigator] ml done: avg={ml_data.get('avg_ml_score')}")
    except Exception as e:
        log.error(f"  [investigator] ml error: {e}")
        ml_data = {"cluster_id": cluster_id, "error": str(e)}
        events.append({"step": "ml", "message": "ML explanation failed", "data": ml_data})

    return {
        **state,
        "facts":      facts,
        "graph_data": graph_data,
        "ml_data":    ml_data,
        "events":     events,
    }
