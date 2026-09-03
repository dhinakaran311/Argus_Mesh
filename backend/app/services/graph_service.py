"""
backend/app/services/graph_service.py
========================================
AbuseRing Sentinel — Graph Service

Wraps Neo4jClient to provide:
  1. Ranked ring cluster list for the dashboard
  2. Cluster detail with member list
  3. React Flow graph data for the frontend visualizer
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from ..db.neo4j import Neo4jClient
from ..models.cluster import (
    ClusterSummary,
    ClusterDetail,
    ReactFlowGraph,
    ReactFlowNode,
    ReactFlowEdge,
)
from ..models.customer import CustomerSummary
from ..services.risk_engine import RiskEngine

log = logging.getLogger(__name__)

_risk_engine = RiskEngine()


def _risk_level(score: float) -> str:
    return _risk_engine.get_risk_level(score)


class GraphService:
    def __init__(self, neo4j: Neo4jClient):
        self._neo4j = neo4j

    # -----------------------------------------------------------------------
    # Cluster list
    # -----------------------------------------------------------------------

    def get_clusters(self, limit: int = 50) -> list[ClusterSummary]:
        rows = self._neo4j.get_top_clusters(limit=limit)
        result = []
        for r in rows:
            score = float(r.get("combined_risk_score", 0.0))
            result.append(ClusterSummary(
                cluster_id=str(r["cluster_id"]),
                cluster_size=int(r.get("cluster_size", 0)),
                risk_level=r.get("risk_level") or _risk_level(score),
                combined_risk_score=score,
                avg_ml_score=float(r.get("avg_ml_score", 0.0)),
                graph_score=float(r.get("graph_score", 0.0)),
                cluster_return_rate=float(r.get("cluster_return_rate", 0.0)),
                total_transactions=int(r.get("total_transactions", 0)),
                total_returns=int(r.get("total_returns", 0)),
                member_ids=[str(m) for m in (r.get("member_ids") or [])],
                abuse_count=int(r.get("abuse_count", 0)),
            ))
        return result

    # -----------------------------------------------------------------------
    # Cluster detail
    # -----------------------------------------------------------------------

    def get_cluster_detail(self, cluster_id: str) -> Optional[ClusterDetail]:
        # Fetch the cluster summary row
        clusters = self.get_clusters(limit=200)
        summary = next((c for c in clusters if c.cluster_id == cluster_id), None)
        if not summary:
            # Build a minimal summary from member query
            members_raw = self._neo4j.get_cluster_members(cluster_id)
            if not members_raw:
                return None
            summary = ClusterSummary(
                cluster_id=cluster_id,
                cluster_size=len(members_raw),
                risk_level="MEDIUM",
                combined_risk_score=0.5,
                avg_ml_score=0.0,
                graph_score=0.0,
                cluster_return_rate=0.0,
                total_transactions=0,
                total_returns=0,
                member_ids=[str(m["customer_id"]) for m in members_raw],
                abuse_count=sum(1 for m in members_raw if m.get("is_abuse")),
            )
        else:
            members_raw = self._neo4j.get_cluster_members(cluster_id)

        # Build CustomerSummary list
        members = []
        for m in members_raw:
            score = float(m.get("risk_score", 0.0))
            members.append(CustomerSummary(
                customer_id=str(m["customer_id"]),
                is_abuse=bool(m.get("is_abuse", False)),
                risk_score=score,
                risk_level=_risk_level(score),
                return_rate=float(m.get("return_rate", 0.0)),
                num_transactions=int(m.get("num_transactions", 0)),
                num_orders=int(m.get("num_orders", 0)),
                num_returns=int(m.get("num_returns", 0)),
                cluster_id=str(m.get("cluster_id") or ""),
                ring_type=str(m.get("ring_type") or ""),
                location_city=str(m.get("location_city") or ""),
                email_domain=str(m.get("email_domain") or ""),
                account_created_at=str(m.get("account_created_at") or ""),
            ))

        # Build React Flow graph
        rf_graph = self.get_react_flow_graph(cluster_id)

        return ClusterDetail(
            **summary.model_dump(),
            members=members,
            react_flow_graph=rf_graph,
        )

    # -----------------------------------------------------------------------
    # React Flow graph builder
    # -----------------------------------------------------------------------

    def get_react_flow_graph(self, cluster_id: str) -> ReactFlowGraph:
        """
        Query Neo4j entity graph and convert to React Flow format.
        Positions nodes in concentric circles: customers → inner, devices/IPs → outer.
        """
        rows = self._neo4j.get_entity_graph(cluster_id)
        if not rows:
            return ReactFlowGraph(nodes=[], edges=[])

        nodes: dict[str, ReactFlowNode] = {}
        edges: dict[str, ReactFlowEdge] = {}

        customer_count = 0
        for row in rows:
            cid = str(row["customer_id"])
            did = str(row["device_id"]) if row.get("device_id") else None
            iid = str(row["ip_id"]) if row.get("ip_id") else None
            mid = str(row["merchant_id"]) if row.get("merchant_id") else None

            # Customer node
            if cid not in nodes:
                angle = (2 * math.pi * customer_count) / max(len(rows), 1)
                nodes[f"c:{cid}"] = ReactFlowNode(
                    id=f"c:{cid}",
                    type="customer",
                    data={
                        "label":       f"C-{cid[:6]}",
                        "risk_score":  float(row.get("risk_score", 0.0)),
                        "is_abuse":    bool(row.get("is_abuse", False)),
                        "return_rate": float(row.get("return_rate", 0.0)),
                    },
                    position={
                        "x": round(150 * math.cos(angle)),
                        "y": round(150 * math.sin(angle)),
                    },
                )
                customer_count += 1

            # Device node
            if did and f"d:{did}" not in nodes:
                nodes[f"d:{did}"] = ReactFlowNode(
                    id=f"d:{did}",
                    type="device",
                    data={
                        "label":    f"Device {row.get('device_type', '')}",
                        "accounts": int(row.get("device_accounts", 1)),
                    },
                    position={"x": 400, "y": 0},
                )
            if did:
                eid = f"e:c{cid}-d{did}"
                if eid not in edges:
                    edges[eid] = ReactFlowEdge(
                        id=eid,
                        source=f"c:{cid}",
                        target=f"d:{did}",
                        label="USES",
                        animated=bool(row.get("is_abuse", False)),
                    )

            # IP node
            if iid and f"i:{iid}" not in nodes:
                nodes[f"i:{iid}"] = ReactFlowNode(
                    id=f"i:{iid}",
                    type="ip",
                    data={
                        "label": f"IP {row.get('ip_city', '')}",
                        "isp":   str(row.get("ip_isp", "")),
                    },
                    position={"x": 400, "y": 200},
                )
            if iid:
                eid = f"e:c{cid}-i{iid}"
                if eid not in edges:
                    edges[eid] = ReactFlowEdge(
                        id=eid,
                        source=f"c:{cid}",
                        target=f"i:{iid}",
                        label="CONNECTS",
                    )

            # Merchant node
            if mid and f"m:{mid}" not in nodes:
                nodes[f"m:{mid}"] = ReactFlowNode(
                    id=f"m:{mid}",
                    type="merchant",
                    data={"label": str(row.get("merchant_name", mid[:8]))},
                    position={"x": 0, "y": -200},
                )

        return ReactFlowGraph(
            nodes=list(nodes.values()),
            edges=list(edges.values()),
        )

    # -----------------------------------------------------------------------
    # Dashboard summary
    # -----------------------------------------------------------------------

    def get_dashboard_summary(self) -> dict:
        return self._neo4j.get_dashboard_summary()
