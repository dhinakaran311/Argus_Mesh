"""
backend/app/models/cluster.py
==============================
AbuseRing Sentinel — Cluster Pydantic Schemas
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .customer import CustomerSummary


class ReactFlowNode(BaseModel):
    id: str
    type: str          # customer | device | ip | merchant
    data: dict
    position: dict = Field(default_factory=lambda: {"x": 0, "y": 0})


class ReactFlowEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str = ""
    animated: bool = False
    style: dict = Field(default_factory=dict)


class ReactFlowGraph(BaseModel):
    nodes: list[ReactFlowNode]
    edges: list[ReactFlowEdge]


class ClusterSummary(BaseModel):
    cluster_id: str
    cluster_size: int
    risk_level: str
    combined_risk_score: float = Field(ge=0.0, le=1.0)
    avg_ml_score: float = Field(ge=0.0, le=1.0)
    graph_score: float = Field(ge=0.0, le=1.0)
    cluster_return_rate: float = Field(ge=0.0, le=1.0)
    total_transactions: int
    total_returns: int
    member_ids: list[str]
    abuse_count: int
    ring_type: Optional[str] = None  # populated from member data or labels CSV


class ClusterDetail(ClusterSummary):
    members: list[CustomerSummary] = []
    react_flow_graph: Optional[ReactFlowGraph] = None
