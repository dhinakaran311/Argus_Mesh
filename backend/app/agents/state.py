"""
backend/app/agents/state.py
=============================
AbuseRing Sentinel — LangGraph Investigation State
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class InvestigationState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    Each node reads and writes specific keys.

    NOTE: _services MUST be declared here so LangGraph does not strip it
    when passing state between nodes. LangGraph only forwards keys that are
    declared in the TypedDict schema.
    """
    cluster_id:    str
    facts:         Optional[dict]    # Investigator node output
    graph_data:    Optional[dict]    # Investigator node output
    ml_data:       Optional[dict]    # Investigator node output
    similar_cases: Optional[list]   # Retrieval node output
    report:        Optional[dict]    # Analyst node output
    events:        list[dict]        # Accumulated SSE events (all nodes append here)
    error:         Optional[str]     # Set on any failure
    _services:     dict              # Injected services (data, neo4j, ml, vectors, groq key/model)
