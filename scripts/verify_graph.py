"""
scripts/verify_graph.py
Quick script to verify existing Neo4j graph nodes and relationships.
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

def verify():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        queries = {
            "Customer Nodes":        "MATCH (n:Customer) RETURN count(n) AS c",
            "Device Nodes":          "MATCH (n:Device)   RETURN count(n) AS c",
            "IP Nodes":              "MATCH (n:IP)        RETURN count(n) AS c",
            "Merchant Nodes":        "MATCH (n:Merchant)  RETURN count(n) AS c",
            "USES Rels":            "MATCH ()-[r:USES]->()              RETURN count(r) AS c",
            "CONNECTS_FROM Rels":   "MATCH ()-[r:CONNECTS_FROM]->()    RETURN count(r) AS c",
            "TRANSACTED_WITH Rels": "MATCH ()-[r:TRANSACTED_WITH]->()  RETURN count(r) AS c",
            "Abuse Ring Members":    "MATCH (c:Customer {is_abuse: true}) RETURN count(c) AS c",
        }
        print("\n" + "=" * 50)
        print("  NEO4J AURA GRAPH VERIFICATION REPORT")
        print("=" * 50)
        with driver.session() as session:
            for label, q in queries.items():
                res = session.run(q).single()
                count = res["c"] if res else 0
                print(f"  {label:<25} : {count:>10,}")
        print("=" * 50 + "\n")
    finally:
        driver.close()

if __name__ == "__main__":
    verify()
