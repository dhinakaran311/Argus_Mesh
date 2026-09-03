"""
scripts/seed_neo4j.py
======================
Seeds Neo4j with relationship edges from CSV data.
Run once to connect Customer → Device, Customer → IP, Customer → Merchant edges.

Usage:
  cd Argus_Mesh
  backend\\.venv\\Scripts\\python.exe scripts/seed_neo4j.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
NEO4J_URI  = "neo4j+s://636ccb86.databases.neo4j.io"
NEO4J_USER = "636ccb86"
NEO4J_PASS = "iM_hCaLse6TI756LNFMuaBMlYdt3YpiyP9ES7yMj4ZM"

BATCH = 500

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def seed(driver):
    log.info("Loading CSVs...")
    txns = pd.read_csv(RAW / "transactions.csv")
    labels = pd.read_csv(RAW / "abuse_labels.csv")
    customers = pd.read_csv(RAW / "customers.csv")

    # Merge cluster_id + ring_type into customers
    cust = customers.merge(
        labels[["customer_id", "cluster_id", "is_abuse", "ring_type"]],
        on="customer_id", how="left"
    )

    log.info(f"  {len(cust):,} customers | {len(txns):,} txns")

    with driver.session() as s:
        # ── 1. Set cluster_id and is_abuse on Customer nodes ─────────────
        log.info("Setting cluster_id + ring_type on Customer nodes...")
        abuse_rows = cust[cust["is_abuse"] == True][["customer_id", "cluster_id", "ring_type"]].dropna()
        for batch in batched(abuse_rows.to_dict("records"), BATCH):
            s.run("""
                UNWIND $rows AS row
                MATCH (c:Customer {id: row.customer_id})
                SET c.cluster_id = row.cluster_id,
                    c.ring_type  = row.ring_type,
                    c.is_abuse   = true
            """, rows=batch)
        log.info(f"  Tagged {len(abuse_rows):,} abuse customers")

        # ── 2. USES edges: Customer → Device ──────────────────────────────
        log.info("Creating USES (Customer → Device) edges...")
        rels = txns[["customer_id", "device_id"]].dropna().drop_duplicates()
        total = 0
        for batch in batched(rels.to_dict("records"), BATCH):
            s.run("""
                UNWIND $rows AS row
                MATCH (c:Customer {id: row.customer_id})
                MATCH (d:Device   {id: row.device_id})
                MERGE (c)-[:USES]->(d)
            """, rows=batch)
            total += len(batch)
            if total % 10000 == 0:
                log.info(f"  {total:,} USES edges written...")
        log.info(f"  ✅ {total:,} USES edges total")

        # ── 3. CONNECTS_FROM: Customer → IP ───────────────────────────────
        log.info("Creating CONNECTS_FROM (Customer → IP) edges...")
        rels = txns[["customer_id", "ip_id"]].dropna().drop_duplicates()
        total = 0
        for batch in batched(rels.to_dict("records"), BATCH):
            s.run("""
                UNWIND $rows AS row
                MATCH (c:Customer {id: row.customer_id})
                MATCH (i:IP       {id: row.ip_id})
                MERGE (c)-[:CONNECTS_FROM]->(i)
            """, rows=batch)
            total += len(batch)
            if total % 10000 == 0:
                log.info(f"  {total:,} CONNECTS_FROM edges...")
        log.info(f"  ✅ {total:,} CONNECTS_FROM edges total")

        # ── 4. TRANSACTED_WITH: Customer → Merchant ────────────────────────
        log.info("Creating TRANSACTED_WITH (Customer → Merchant) edges...")
        rels = txns[["customer_id", "merchant_id"]].dropna().drop_duplicates()
        total = 0
        for batch in batched(rels.to_dict("records"), BATCH):
            s.run("""
                UNWIND $rows AS row
                MATCH (c:Customer {id: row.customer_id})
                MATCH (m:Merchant {id: row.merchant_id})
                MERGE (c)-[:TRANSACTED_WITH]->(m)
            """, rows=batch)
            total += len(batch)
        log.info(f"  ✅ {total:,} TRANSACTED_WITH edges total")

        # ── 5. Verify ─────────────────────────────────────────────────────
        result = s.run("MATCH ()-[r]->() RETURN type(r) as t, count(r) as cnt")
        log.info("Edge counts after seeding:")
        for row in result:
            log.info(f"  {row['t']:<25} {row['cnt']:>10,}")

    log.info("✅ Seeding complete!")


if __name__ == "__main__":
    log.info(f"Connecting to {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        seed(driver)
    finally:
        driver.close()
