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
    """
    cluster_id:    str
    facts:         Optional[dict]    # Investigator node output
    graph_data:    Optional[dict]    # Investigator node output
    ml_data:       Optional[dict]    # Investigator node output
    similar_cases: Optional[list]   # Retrieval node output
    report:        Optional[dict]    # Analyst node output
    events:        list[dict]        # Accumulated SSE events (all nodes append here)
    error:         Optional[str]     # Set on any failure
