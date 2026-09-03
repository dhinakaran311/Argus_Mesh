"""
backend/app/agents/orchestrator.py
=====================================
AbuseRing Sentinel — LangGraph Investigation Orchestrator

Builds and compiles the 3-node investigation graph:
  investigator → retrieval → analyst → END

Services are injected into the state under the "_services" key
before graph.invoke() is called from the SSE endpoint.
"""
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from .state import InvestigationState
from .investigator import investigator_node
from .retrieval import retrieval_node
from .analyst import analyst_node

log = logging.getLogger(__name__)


def build_investigation_graph():
    """Build and compile the LangGraph investigation pipeline."""
    graph = StateGraph(InvestigationState)

    graph.add_node("investigator", investigator_node)
    graph.add_node("retrieval",    retrieval_node)
    graph.add_node("analyst",      analyst_node)

    graph.set_entry_point("investigator")
    graph.add_edge("investigator", "retrieval")
    graph.add_edge("retrieval",    "analyst")
    graph.add_edge("analyst",      END)

    compiled = graph.compile()
    log.info("  ✅ LangGraph investigation graph compiled")
    return compiled


# Singleton — compiled once at import time
investigation_graph = build_investigation_graph()
