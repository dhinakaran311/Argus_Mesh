"""
graph/graph_builder.py
======================
AbuseRing Sentinel — Neo4j Graph Builder

Reads all raw CSVs from data/raw/ and populates the Neo4j Aura instance
(argus-mesh-graph) with the full customer-device-IP-transaction graph.

Node types created:
    Customer, Device, IP, Merchant

Relationship types created:
    (:Customer)-[:USES]->(:Device)
    (:Customer)-[:CONNECTS_FROM]->(:IP)
    (:Customer)-[:MADE]->(:Transaction)
    (:Transaction)-[:AT]->(:Merchant)

Usage:
    python graph/graph_builder.py
    python scripts/seed_neo4j.py       ← recommended entry point

Environment:
    Reads NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD from .env
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env before anything else
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

def load_env():
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

load_env()

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR       = ROOT_DIR / "data" / "raw"
MODELS_DIR    = ROOT_DIR / "ml" / "models"
BATCH_SIZE    = 500    # records per Neo4j transaction (safe for Aura Free)
TXNS_SAMPLE   = None   # set to e.g. 50_000 to only load a subset of transactions

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Driver
# ===========================================================================

def get_driver():
    log.info(f"Connecting to Neo4j: {NEO4J_URI}")
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    log.info("  ✅ Connected")
    return driver


# ===========================================================================
# Schema setup
# ===========================================================================

CONSTRAINTS_AND_INDEXES = [
    "CREATE CONSTRAINT customer_id_unique IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT device_id_unique   IF NOT EXISTS FOR (d:Device)   REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT ip_id_unique       IF NOT EXISTS FOR (i:IP)       REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT merchant_id_unique IF NOT EXISTS FOR (m:Merchant) REQUIRE m.id IS UNIQUE",
    "CREATE INDEX customer_cluster_idx IF NOT EXISTS FOR (c:Customer) ON (c.cluster_id)",
    "CREATE INDEX customer_abuse_idx   IF NOT EXISTS FOR (c:Customer) ON (c.is_abuse)",
    "CREATE INDEX customer_risk_idx    IF NOT EXISTS FOR (c:Customer) ON (c.risk_score)",
    "CREATE INDEX device_accounts_idx  IF NOT EXISTS FOR (d:Device)   ON (d.accounts_count)",
    "CREATE INDEX ip_accounts_idx      IF NOT EXISTS FOR (i:IP)       ON (i.accounts_count)",
]

def setup_schema(driver):
    log.info("Setting up schema constraints and indexes...")
    with driver.session() as session:
        for stmt in CONSTRAINTS_AND_INDEXES:
            try:
                session.run(stmt)
            except Exception as e:
                log.warning(f"  Schema stmt skipped: {e}")
    log.info("  ✅ Schema ready")


# ===========================================================================
# Loaders
# ===========================================================================

def load_raw_data() -> dict:
    log.info("Loading raw CSVs...")
    dfs = {}

    files = {
        "customers":    "customers.csv",
        "transactions": "transactions.csv",
        "orders":       "orders.csv",
        "returns":      "returns.csv",
        "devices":      "devices.csv",
        "ips":          "ips.csv",
        "labels":       "abuse_labels.csv",
        "merchants":    "merchants.csv",
    }

    for key, fname in files.items():
        path = RAW_DIR / fname
        if not path.exists():
            log.warning(f"  Missing: {path}")
            continue
        dfs[key] = pd.read_csv(path)
        log.info(f"  {fname:<25} {len(dfs[key]):>10,} rows")

    return dfs


def load_ml_risk_scores() -> dict:
    """Load XGBoost risk scores from saved model metadata if available."""
    meta_path = MODELS_DIR / "model_meta.json"
    if not meta_path.exists():
        log.warning("No model_meta.json found — risk_score will default to 0.0")
        return {}
    # Risk scores per customer are in features.parquet
    feat_path = ROOT_DIR / "data" / "processed" / "features.parquet"
    if not feat_path.exists():
        log.warning("features.parquet not found — risk_score will default to 0.0")
        return {}
    feat = pd.read_parquet(feat_path, columns=["customer_id", "is_abuse"])
    log.info(f"  Loaded feature risk proxies for {len(feat):,} customers")
    return dict(zip(feat["customer_id"], feat["is_abuse"].astype(float)))


# ===========================================================================
# Node creators
# ===========================================================================

def batch_run(driver, query: str, records: list, desc: str = ""):
    """Run a parameterised Cypher query in batches."""
    total = len(records)
    with tqdm(total=total, desc=desc, unit="records", ncols=80) as pbar:
        for start in range(0, total, BATCH_SIZE):
            batch = records[start:start + BATCH_SIZE]
            with driver.session() as session:
                session.run(query, {"records": batch})
            pbar.update(len(batch))


def create_merchant_nodes(driver, dfs: dict):
    log.info("Creating Merchant nodes...")
    if "merchants" not in dfs:
        # Derive from transactions
        merchants = dfs["transactions"][["merchant_id"]].drop_duplicates()
        merchants["name"] = merchants["merchant_id"].str[:8]
        merchants["baseline_refund_rate"] = 0.08
    else:
        merchants = dfs["merchants"]

    records = merchants.rename(columns={"merchant_id": "id"}).to_dict("records")
    query = """
    UNWIND $records AS row
    MERGE (m:Merchant {id: row.id})
    SET m.name                 = coalesce(row.name, row.id),
        m.baseline_refund_rate = toFloat(coalesce(row.baseline_refund_rate, 0.08))
    """
    batch_run(driver, query, records, desc="Merchants")
    log.info(f"  ✅ {len(records):,} Merchant nodes")


def create_device_nodes(driver, dfs: dict):
    log.info("Creating Device nodes...")
    devices = dfs["devices"].copy() if "devices" in dfs else pd.DataFrame()

    if devices.empty or "device_id" not in devices.columns:
        # Derive from transactions
        dev_counts = dfs["transactions"].groupby("device_id")["customer_id"].nunique().rename("accounts_count")
        devices = dev_counts.reset_index().rename(columns={"device_id": "id"})
        devices["device_type"] = "MOBILE"
        devices["browser"] = "Unknown"
        devices["os"] = "Unknown"
        devices["first_seen"] = None
    else:
        devices = devices.rename(columns={"device_id": "id"})
        devices["accounts_count"] = devices.get("total_accounts", 1)

    records = devices[["id", "device_type", "browser", "os", "accounts_count"]].to_dict("records")
    query = """
    UNWIND $records AS row
    MERGE (d:Device {id: row.id})
    SET d.device_type     = row.device_type,
        d.browser         = row.browser,
        d.os              = row.os,
        d.accounts_count  = toInteger(coalesce(row.accounts_count, 1))
    """
    batch_run(driver, query, records, desc="Devices  ")
    log.info(f"  ✅ {len(records):,} Device nodes")


def create_ip_nodes(driver, dfs: dict):
    log.info("Creating IP nodes...")
    ips = dfs.get("ips", pd.DataFrame())

    if ips.empty or "ip_id" not in ips.columns:
        ip_counts = dfs["transactions"].groupby("ip_id")["customer_id"].nunique().rename("accounts_count")
        ips = ip_counts.reset_index().rename(columns={"ip_id": "id"})
        ips["country"] = "IN"
        ips["region"]  = "Unknown"
        ips["city"]    = "Unknown"
        ips["isp"]     = "Unknown"
    else:
        ips = ips.rename(columns={"ip_id": "id"})
        ips["accounts_count"] = ips.get("total_accounts", 1)

    records = ips[["id", "country", "region", "city", "isp", "accounts_count"]].to_dict("records")
    query = """
    UNWIND $records AS row
    MERGE (i:IP {id: row.id})
    SET i.country        = row.country,
        i.region         = row.region,
        i.city           = row.city,
        i.isp            = row.isp,
        i.accounts_count = toInteger(coalesce(row.accounts_count, 1))
    """
    batch_run(driver, query, records, desc="IPs      ")
    log.info(f"  ✅ {len(records):,} IP nodes")


def create_customer_nodes(driver, dfs: dict, risk_scores: dict):
    log.info("Creating Customer nodes...")

    customers = dfs["customers"].copy()
    labels    = dfs.get("labels", pd.DataFrame())

    if not labels.empty:
        customers = customers.merge(
            labels[["customer_id", "is_abuse", "cluster_id", "ring_type"]],
            on="customer_id", how="left"
        )
    else:
        customers["is_abuse"]   = False
        customers["cluster_id"] = None
        customers["ring_type"]  = None

    customers["is_abuse"] = customers["is_abuse"].fillna(False).astype(bool)

    # Per-customer stats
    order_counts  = dfs["orders"].groupby("customer_id").size().rename("num_orders")
    return_counts = dfs["returns"].groupby("customer_id").size().rename("num_returns")
    txn_counts    = dfs["transactions"].groupby("customer_id").size().rename("num_transactions")

    customers = (customers
                 .merge(txn_counts, on="customer_id", how="left")
                 .merge(order_counts, on="customer_id", how="left")
                 .merge(return_counts, on="customer_id", how="left"))

    customers["num_transactions"] = customers["num_transactions"].fillna(0).astype(int)
    customers["num_orders"]       = customers["num_orders"].fillna(0).astype(int)
    customers["num_returns"]      = customers["num_returns"].fillna(0).astype(int)
    customers["return_rate"]      = (
        customers["num_returns"] / customers["num_orders"].clip(lower=1)
    )
    customers["risk_score"]       = customers["customer_id"].map(risk_scores).fillna(0.0)

    records = []
    for _, row in customers.iterrows():
        records.append({
            "id":                 str(row["customer_id"]),
            "account_created_at": str(row.get("account_created_at", "")),
            "customer_age_days":  int(row.get("customer_age_days", 0)),
            "location_city":      str(row.get("location_city", "")),
            "location_state":     str(row.get("location_state", "")),
            "email_domain":       str(row.get("email_domain", "")),
            "is_verified":        bool(row.get("is_verified", False)),
            "is_abuse":           bool(row["is_abuse"]),
            "cluster_id":         str(row["cluster_id"]) if pd.notna(row.get("cluster_id")) else None,
            "ring_type":          str(row["ring_type"]) if pd.notna(row.get("ring_type")) else None,
            "risk_score":         float(row["risk_score"]),
            "return_rate":        float(row["return_rate"]),
            "num_transactions":   int(row["num_transactions"]),
            "num_orders":         int(row["num_orders"]),
            "num_returns":        int(row["num_returns"]),
        })

    query = """
    UNWIND $records AS row
    MERGE (c:Customer {id: row.id})
    SET c.account_created_at = row.account_created_at,
        c.customer_age_days  = row.customer_age_days,
        c.location_city      = row.location_city,
        c.location_state     = row.location_state,
        c.email_domain       = row.email_domain,
        c.is_verified        = row.is_verified,
        c.is_abuse           = row.is_abuse,
        c.cluster_id         = row.cluster_id,
        c.ring_type          = row.ring_type,
        c.risk_score         = toFloat(row.risk_score),
        c.return_rate        = toFloat(row.return_rate),
        c.num_transactions   = toInteger(row.num_transactions),
        c.num_orders         = toInteger(row.num_orders),
        c.num_returns        = toInteger(row.num_returns)
    """
    batch_run(driver, query, records, desc="Customers")
    log.info(f"  ✅ {len(records):,} Customer nodes")


# ===========================================================================
# Relationship creators
# ===========================================================================

def create_uses_relationships(driver, dfs: dict):
    """(:Customer)-[:USES]->(:Device)"""
    log.info("Creating USES relationships (Customer → Device)...")

    cust_dev = (
        dfs["transactions"][["customer_id", "device_id"]]
        .drop_duplicates()
        .dropna()
    )
    records = [
        {"customer_id": str(r["customer_id"]), "device_id": str(r["device_id"])}
        for _, r in cust_dev.iterrows()
    ]
    query = """
    UNWIND $records AS row
    MATCH (c:Customer {id: row.customer_id})
    MATCH (d:Device   {id: row.device_id})
    MERGE (c)-[:USES]->(d)
    """
    batch_run(driver, query, records, desc="USES     ")
    log.info(f"  ✅ {len(records):,} USES relationships")


def create_connects_from_relationships(driver, dfs: dict):
    """(:Customer)-[:CONNECTS_FROM]->(:IP)"""
    log.info("Creating CONNECTS_FROM relationships (Customer → IP)...")

    cust_ip = (
        dfs["transactions"].groupby(["customer_id", "ip_id"])
        .size()
        .reset_index(name="count")
        .dropna()
    )
    records = [
        {
            "customer_id": str(r["customer_id"]),
            "ip_id":       str(r["ip_id"]),
            "count":       int(r["count"]),
        }
        for _, r in cust_ip.iterrows()
    ]
    query = """
    UNWIND $records AS row
    MATCH (c:Customer {id: row.customer_id})
    MATCH (i:IP       {id: row.ip_id})
    MERGE (c)-[r:CONNECTS_FROM]->(i)
    SET r.count = row.count
    """
    batch_run(driver, query, records, desc="CONNECTS ")
    log.info(f"  ✅ {len(records):,} CONNECTS_FROM relationships")


def create_made_and_at_relationships(driver, dfs: dict):
    """
    (:Customer)-[:MADE]->(:Transaction)-[:AT]->(:Merchant)
    Note: We skip Transaction nodes by default to avoid 200K nodes on free tier.
    Instead we create a direct TRANSACTED_WITH relationship for efficiency.
    """
    log.info("Creating TRANSACTED_WITH relationships (Customer → Merchant)...")

    txns = dfs["transactions"].copy()
    if TXNS_SAMPLE:
        txns = txns.sample(min(TXNS_SAMPLE, len(txns)), random_state=42)

    cust_merch = (
        txns.groupby(["customer_id", "merchant_id"])
        .agg(
            txn_count=("transaction_id", "count"),
            total_amount=("amount", "sum"),
        )
        .reset_index()
        .dropna()
    )

    records = [
        {
            "customer_id": str(r["customer_id"]),
            "merchant_id": str(r["merchant_id"]),
            "txn_count":   int(r["txn_count"]),
            "total_amount": float(r["total_amount"]),
        }
        for _, r in cust_merch.iterrows()
    ]

    query = """
    UNWIND $records AS row
    MATCH (c:Customer {id: row.customer_id})
    MATCH (m:Merchant {id: row.merchant_id})
    MERGE (c)-[r:TRANSACTED_WITH]->(m)
    SET r.txn_count   = row.txn_count,
        r.total_amount = row.total_amount
    """
    batch_run(driver, query, records, desc="TRANSACT ")
    log.info(f"  ✅ {len(records):,} TRANSACTED_WITH relationships")


# ===========================================================================
# Verification
# ===========================================================================

def verify_graph(driver) -> dict:
    log.info("Verifying graph contents...")
    counts = {}
    queries = {
        "Customer":        "MATCH (n:Customer) RETURN count(n) AS c",
        "Device":          "MATCH (n:Device)   RETURN count(n) AS c",
        "IP":              "MATCH (n:IP)        RETURN count(n) AS c",
        "Merchant":        "MATCH (n:Merchant)  RETURN count(n) AS c",
        "USES":            "MATCH ()-[r:USES]->()              RETURN count(r) AS c",
        "CONNECTS_FROM":   "MATCH ()-[r:CONNECTS_FROM]->()    RETURN count(r) AS c",
        "TRANSACTED_WITH": "MATCH ()-[r:TRANSACTED_WITH]->()  RETURN count(r) AS c",
        "Abuse Customers": "MATCH (c:Customer {is_abuse: true}) RETURN count(c) AS c",
    }
    with driver.session() as session:
        for label, q in queries.items():
            result = session.run(q).single()
            counts[label] = result["c"] if result else 0
            log.info(f"  {label:<20} {counts[label]:>10,}")
    return counts


# ===========================================================================
# Main
# ===========================================================================

def build_graph(skip_relationships: bool = False):
    log.info("=" * 60)
    log.info("AbuseRing Sentinel — Neo4j Graph Builder")
    log.info("=" * 60)
    log.info(f"  URI      : {NEO4J_URI}")
    log.info(f"  Username : {NEO4J_USERNAME}")
    log.info(f"  Batch sz : {BATCH_SIZE}")
    log.info("")

    driver = get_driver()

    try:
        # Schema
        setup_schema(driver)

        # Load data
        dfs = load_raw_data()
        risk_scores = load_ml_risk_scores()

        # Nodes
        log.info("\n--- Creating Nodes ---")
        create_merchant_nodes(driver, dfs)
        create_device_nodes(driver, dfs)
        create_ip_nodes(driver, dfs)
        create_customer_nodes(driver, dfs, risk_scores)

        if not skip_relationships:
            # Relationships
            log.info("\n--- Creating Relationships ---")
            create_uses_relationships(driver, dfs)
            create_connects_from_relationships(driver, dfs)
            create_made_and_at_relationships(driver, dfs)

        # Verify
        log.info("\n--- Verification ---")
        counts = verify_graph(driver)

        log.info("\n" + "=" * 60)
        log.info("✅ Graph build complete!")
        log.info("=" * 60)
        return counts

    finally:
        driver.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Neo4j graph for AbuseRing Sentinel")
    parser.add_argument("--nodes-only", action="store_true",
                        help="Only create nodes, skip relationships (for testing)")
    args = parser.parse_args()
    build_graph(skip_relationships=args.nodes_only)
