"""
backend/app/db/neo4j.py
=======================
AbuseRing Sentinel — Neo4j AuraDB Client

Wraps the neo4j Python driver and exposes high-level query methods
tied to the pre-built Cypher files in graph/queries/*.cypher.

All queries are cached as module-level string constants so they are
read from disk once at import time, not on every request.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase, Driver

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load Cypher query files from graph/queries/
# ---------------------------------------------------------------------------
_QUERIES_DIR = Path(__file__).parent.parent.parent.parent / "graph" / "queries"


def _load_cypher(filename: str) -> str:
    path = _QUERIES_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    log.warning(f"Cypher file not found: {path}")
    return ""


# Pre-load all query files once
_RING_DETECTION_CYPHER = _load_cypher("abuse_ring_detection.cypher")
_FIND_CLUSTERS_CYPHER = _load_cypher("find_clusters.cypher")
_CLUSTER_DETAIL_CYPHER = _load_cypher("get_cluster_detail.cypher")
_ENTITY_CONNECTIONS_CYPHER = _load_cypher("get_entity_connections.cypher")

# ---------------------------------------------------------------------------
# Inline queries for specific needs (not in files)
# ---------------------------------------------------------------------------

_DASHBOARD_QUERY = """
MATCH (c:Customer)
WITH
    count(c)                                                               AS total_customers,
    sum(CASE WHEN c.is_abuse THEN 1 ELSE 0 END)                           AS abuse_customers,
    sum(CASE WHEN c.risk_score >= 0.80 THEN 1 ELSE 0 END)                 AS critical_count,
    sum(CASE WHEN c.risk_score >= 0.60 AND c.risk_score < 0.80 THEN 1 ELSE 0 END) AS high_count,
    sum(CASE WHEN c.risk_score >= 0.30 AND c.risk_score < 0.60 THEN 1 ELSE 0 END) AS medium_count,
    sum(CASE WHEN c.risk_score < 0.30 THEN 1 ELSE 0 END)                  AS low_count,
    avg(c.risk_score)                                                      AS avg_risk_score
MATCH (d:Device)
WITH total_customers, abuse_customers, critical_count, high_count, medium_count, low_count, avg_risk_score,
     count(d)                                                              AS total_devices,
     sum(CASE WHEN d.accounts_count >= 10 THEN 1 ELSE 0 END)              AS high_share_devices
MATCH (i:IP)
WITH total_customers, abuse_customers, critical_count, high_count, medium_count, low_count,
     avg_risk_score, total_devices, high_share_devices,
     count(i)                                                              AS total_ips
MATCH (c2:Customer)
WHERE c2.cluster_id IS NOT NULL
RETURN
    total_customers,
    abuse_customers,
    round(toFloat(abuse_customers) / total_customers * 10000) / 100        AS abuse_rate_pct,
    critical_count,
    high_count,
    medium_count,
    low_count,
    round(avg_risk_score * 1000) / 1000                                    AS avg_risk_score,
    total_devices,
    high_share_devices,
    total_ips,
    count(DISTINCT c2.cluster_id)                                          AS total_rings
"""

_TOP_CLUSTERS_QUERY = """
MATCH (c:Customer)-[:USES]->(d:Device)
WITH d,
     collect(DISTINCT c)        AS members,
     count(DISTINCT c)          AS cluster_size,
     avg(c.return_rate)         AS cluster_return_rate,
     avg(c.risk_score)          AS avg_ml_score,
     sum(c.num_transactions)    AS total_transactions,
     sum(c.num_returns)         AS total_returns,
     min(c.account_created_at)  AS earliest_created,
     max(c.account_created_at)  AS latest_created
WHERE cluster_size >= 3
WITH d, members, cluster_size, cluster_return_rate, avg_ml_score,
     total_transactions, total_returns, earliest_created, latest_created,
     CASE WHEN cluster_size >= 30 THEN 1.0 ELSE toFloat(cluster_size) / 30 END AS size_signal,
     cluster_return_rate AS return_signal,
     CASE
         WHEN duration.between(earliest_created, latest_created).hours <= 24 THEN 1.0
         WHEN duration.between(earliest_created, latest_created).hours <= 72 THEN 0.7
         WHEN duration.between(earliest_created, latest_created).hours <= 168 THEN 0.4
         ELSE 0.1
     END AS burst_signal
WITH d, members, cluster_size, cluster_return_rate, avg_ml_score,
     total_transactions, total_returns,
     (0.35 * size_signal + 0.40 * return_signal + 0.25 * burst_signal) AS graph_score,
     avg_ml_score AS ml_score
WHERE (0.35 * size_signal + 0.40 * cluster_return_rate + 0.25 * burst_signal) > 0.1
RETURN
    d.id                                            AS cluster_id,
    cluster_size,
    round(cluster_return_rate * 1000) / 1000        AS cluster_return_rate,
    round(avg_ml_score * 1000) / 1000               AS avg_ml_score,
    total_transactions,
    total_returns,
    round(graph_score * 1000) / 1000                AS graph_score,
    round(
        (0.40 * avg_ml_score + 0.30 * graph_score + 0.30 * cluster_return_rate) * 1000
    ) / 1000                                        AS combined_risk_score,
    CASE
        WHEN (0.40 * avg_ml_score + 0.30 * graph_score + 0.30 * cluster_return_rate) >= 0.80 THEN 'CRITICAL'
        WHEN (0.40 * avg_ml_score + 0.30 * graph_score + 0.30 * cluster_return_rate) >= 0.60 THEN 'HIGH'
        WHEN (0.40 * avg_ml_score + 0.30 * graph_score + 0.30 * cluster_return_rate) >= 0.30 THEN 'MEDIUM'
        ELSE 'LOW'
    END                                             AS risk_level,
    [m IN members | m.id]                           AS member_ids,
    size([m IN members WHERE m.is_abuse])           AS abuse_count
ORDER BY combined_risk_score DESC
LIMIT $limit
"""

_CLUSTER_MEMBERS_QUERY = """
MATCH (c:Customer)-[:USES]->(d:Device {id: $device_id})
RETURN
    c.id                  AS customer_id,
    c.is_abuse            AS is_abuse,
    c.risk_score          AS risk_score,
    c.return_rate         AS return_rate,
    c.num_transactions    AS num_transactions,
    c.num_orders          AS num_orders,
    c.num_returns         AS num_returns,
    c.cluster_id          AS cluster_id,
    c.ring_type           AS ring_type,
    c.location_city       AS location_city,
    c.email_domain        AS email_domain,
    c.account_created_at  AS account_created_at
ORDER BY c.risk_score DESC
"""

_ENTITY_GRAPH_QUERY = """
MATCH (c:Customer)-[:USES]->(d:Device {id: $device_id})
WITH collect(c) AS cust_list, d
UNWIND cust_list AS c
OPTIONAL MATCH (c)-[r2:CONNECTS_FROM]->(i:IP)
OPTIONAL MATCH (c)-[r3:TRANSACTED_WITH]->(m:Merchant)
RETURN
    c.id              AS customer_id,
    c.risk_score      AS risk_score,
    c.is_abuse        AS is_abuse,
    c.return_rate     AS return_rate,
    d.id              AS device_id,
    d.device_type     AS device_type,
    d.accounts_count  AS device_accounts,
    i.id              AS ip_id,
    i.city            AS ip_city,
    i.isp             AS ip_isp,
    r2.count          AS ip_txn_count,
    m.id              AS merchant_id,
    m.name            AS merchant_name
"""


class Neo4jClient:
    """Thin wrapper around the Neo4j Python driver for AbuseRing Sentinel."""

    def __init__(self, uri: str, username: str, password: str):
        self._uri = uri
        self._username = username
        self._password = password
        self._driver: Driver | None = None

    def connect(self) -> None:
        log.info(f"Connecting to Neo4j: {self._uri}")
        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._username, self._password)
        )
        self._driver.verify_connectivity()
        log.info("  ✅ Neo4j connected")

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            log.info("Neo4j driver closed")

    def run_query(self, cypher: str, params: dict | None = None) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as a list of dicts."""
        if not self._driver:
            raise RuntimeError("Neo4j driver not connected — call connect() first")
        params = params or {}
        with self._driver.session() as session:
            result = session.run(cypher, params)
            return [dict(record) for record in result]

    # -----------------------------------------------------------------------
    # High-level query methods
    # -----------------------------------------------------------------------

    def get_dashboard_summary(self) -> dict:
        rows = self.run_query(_DASHBOARD_QUERY)
        return rows[0] if rows else {}

    def get_top_clusters(self, limit: int = 50) -> list[dict]:
        return self.run_query(_TOP_CLUSTERS_QUERY, {"limit": limit})

    def get_cluster_members(self, device_id: str) -> list[dict]:
        return self.run_query(_CLUSTER_MEMBERS_QUERY, {"device_id": device_id})

    def get_entity_graph(self, device_id: str) -> list[dict]:
        """Returns raw rows for building a React Flow graph."""
        return self.run_query(_ENTITY_GRAPH_QUERY, {"device_id": device_id})

    def health_check(self) -> bool:
        try:
            self.run_query("RETURN 1 AS ok")
            return True
        except Exception as e:
            log.error(f"Neo4j health check failed: {e}")
            return False
