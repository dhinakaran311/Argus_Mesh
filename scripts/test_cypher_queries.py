"""
scripts/test_cypher_queries.py
Runs Day 4 Cypher queries against Neo4j Aura to confirm ring detection works.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Load .env
env_path = ROOT_DIR / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

from neo4j import GraphDatabase

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

def test_queries():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            print("\n--- Test 1: Top Shared Device Clusters (cluster size >= 3) ---")
            q1 = """
            MATCH (c:Customer)-[:USES]->(d:Device)
            WITH d, collect(c) AS members, count(c) AS sz, avg(c.return_rate) AS avg_rr, sum(c.num_transactions) AS txns
            WHERE sz >= 3
            RETURN d.id AS device_id, sz AS cluster_size, round(avg_rr * 100) / 100 AS avg_return_rate, txns,
                   size([m IN members WHERE m.is_abuse]) AS abuse_count
            ORDER BY sz DESC LIMIT 5
            """
            res1 = session.run(q1).data()
            for r in res1:
                print(f"Device: {r['device_id'][:15]}... | Size: {r['cluster_size']} accounts | Avg Return Rate: {r['avg_return_rate']*100:.1f}% | Abuse Accounts: {r['abuse_count']}")

            print("\n--- Test 2: Known Abuse Ring Clusters in Neo4j ---")
            q2 = """
            MATCH (c:Customer)
            WHERE c.is_abuse = true AND c.cluster_id IS NOT NULL
            RETURN count(DISTINCT c.cluster_id) AS total_rings, count(c) AS total_abuse_members
            """
            res2 = session.run(q2).single()
            print(f"Total Rings Injected in Graph: {res2['total_rings']} | Total Abuse Accounts: {res2['total_abuse_members']}")

            print("\n--- Test 3: Top Abuse Ring Cluster Details ---")
            q3 = """
            MATCH (c:Customer)
            WHERE c.cluster_id IS NOT NULL
            WITH c.cluster_id AS ring_id, count(c) AS size, avg(c.return_rate) AS avg_rr, avg(c.risk_score) AS avg_rs
            RETURN ring_id, size, round(avg_rr * 100) / 100 AS avg_return_rate, round(avg_rs * 100) / 100 AS avg_risk_score
            ORDER BY size DESC LIMIT 5
            """
            res3 = session.run(q3).data()
            for r in res3:
                print(f"Ring ID: {r['ring_id']} | Members: {r['size']} | Avg Return Rate: {r['avg_return_rate']*100:.1f}% | Avg Risk Score: {r['avg_risk_score']*100:.1f}%")

    finally:
        driver.close()

if __name__ == "__main__":
    test_queries()
