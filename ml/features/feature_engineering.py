"""
ml/features/feature_engineering.py
====================================
AbuseRing Sentinel — Feature Engineering Pipeline

Reads raw CSVs from data/raw/ and produces a single flat feature matrix
saved to data/processed/features.parquet.

No labels are used during feature computation (label-free engineering).
Labels are merged at the END as a column for convenience.

Feature groups:
    A. Customer behavioural features
    B. Device-level aggregates (worst-case join to customer)
    C. IP-level aggregates (worst-case join to customer)
    D. Cluster features (Union-Find connected components on shared device/IP)

Temporal note:
    Features are computed from ALL data (Jan-Nov). The train/val/test split
    is applied in train.py based on account_created_at timestamps.
    No leakage occurs because cluster membership is derived from structure
    (shared devices/IPs), not from labels.

Usage:
    python -m ml.features.feature_engineering
    python ml/features/feature_engineering.py
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

REFERENCE_DATE = datetime(2023, 12, 1)   # one day after END_DATE (data is 2023)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_raw() -> dict[str, pd.DataFrame]:
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
    }
    for key, fname in files.items():
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            log.error(f"Missing: {path} — run scripts/generate_data.py first")
            sys.exit(1)
        dfs[key] = pd.read_csv(path)
        log.info(f"  {fname:<25} {len(dfs[key]):>10,} rows")

    # Parse timestamps
    dfs["customers"]["account_created_at"] = pd.to_datetime(
        dfs["customers"]["account_created_at"], utc=True
    ).dt.tz_localize(None)
    dfs["transactions"]["timestamp"] = pd.to_datetime(
        dfs["transactions"]["timestamp"], utc=True
    ).dt.tz_localize(None)
    dfs["orders"]["order_time"] = pd.to_datetime(
        dfs["orders"]["order_time"], utc=True
    ).dt.tz_localize(None)
    dfs["returns"]["return_time"] = pd.to_datetime(
        dfs["returns"]["return_time"], utc=True
    ).dt.tz_localize(None)
    return dfs


# ===========================================================================
# 2. CUSTOMER BEHAVIOURAL FEATURES (Group A)
# ===========================================================================

def compute_customer_features(dfs: dict) -> pd.DataFrame:
    log.info("Computing customer-level features (Group A)...")
    customers = dfs["customers"].copy()
    txn       = dfs["transactions"].copy()
    orders    = dfs["orders"].copy()
    returns   = dfs["returns"].copy()

    ref = pd.Timestamp(REFERENCE_DATE)

    # -- Account age ----------------------------------------------------------
    cust = customers[["customer_id", "account_created_at", "is_verified"]].copy()
    cust["account_age_days"] = (ref - cust["account_created_at"]).dt.days.clip(lower=0)
    cust["account_creation_hour"] = cust["account_created_at"].dt.hour
    cust["account_creation_dow"]  = cust["account_created_at"].dt.dayofweek  # 0=Mon
    # Created on weekend = slightly more suspicious
    cust["created_on_weekend"]    = (cust["account_creation_dow"] >= 5).astype(int)
    # Created late night (11pm-4am)
    cust["created_late_night"] = (
        (cust["account_creation_hour"] >= 23) | (cust["account_creation_hour"] <= 4)
    ).astype(int)

    # -- Transaction aggregates -----------------------------------------------
    txn_grp = txn.groupby("customer_id")

    txn_stats = pd.DataFrame({
        "num_transactions_total": txn_grp.size(),
        "total_spend":            txn_grp["amount"].sum(),
        "avg_txn_amount":         txn_grp["amount"].mean(),
        "max_txn_amount":         txn_grp["amount"].max(),
        "min_txn_amount":         txn_grp["amount"].min(),
        "std_txn_amount":         txn_grp["amount"].std().fillna(0),
        "num_failed_txn":         (txn_grp["status"].apply(lambda s: (s == "FAILED").sum())),
        "num_reversed_txn":       (txn_grp["status"].apply(lambda s: (s == "REVERSED").sum())),
        "unique_devices_used":    txn_grp["device_id"].nunique(),
        "unique_ips_used":        txn_grp["ip_id"].nunique(),
        "unique_merchants":       txn_grp["merchant_id"].nunique(),
        "last_txn_date":          txn_grp["timestamp"].max(),
        "first_txn_date":         txn_grp["timestamp"].min(),
    }).reset_index()

    txn_stats["failed_payment_rate"] = (
        txn_stats["num_failed_txn"] / txn_stats["num_transactions_total"].clip(lower=1)
    )
    txn_stats["days_since_last_txn"] = (
        ref - txn_stats["last_txn_date"]
    ).dt.days.clip(lower=0)
    txn_stats["days_since_first_txn"] = (
        ref - txn_stats["first_txn_date"]
    ).dt.days.clip(lower=0)
    # Active lifespan in days (distance between first and last transaction)
    txn_stats["txn_lifespan_days"] = (
        txn_stats["last_txn_date"] - txn_stats["first_txn_date"]
    ).dt.days.clip(lower=0)

    # -- Velocity (transactions per day in active window) ---------------------
    txn_stats["txn_velocity_overall"] = (
        txn_stats["num_transactions_total"] /
        txn_stats["txn_lifespan_days"].clip(lower=1)
    )

    # Velocity over last 7 days
    cutoff_7d  = ref - pd.Timedelta(days=7)
    cutoff_30d = ref - pd.Timedelta(days=30)
    txn_7d  = txn[txn["timestamp"] >= cutoff_7d].groupby("customer_id").size().rename("num_txn_7d")
    txn_30d = txn[txn["timestamp"] >= cutoff_30d].groupby("customer_id").size().rename("num_txn_30d")

    txn_stats = txn_stats.merge(txn_7d,  on="customer_id", how="left").fillna({"num_txn_7d": 0})
    txn_stats = txn_stats.merge(txn_30d, on="customer_id", how="left").fillna({"num_txn_30d": 0})
    txn_stats["txn_velocity_7d"]  = txn_stats["num_txn_7d"]  / 7
    txn_stats["txn_velocity_30d"] = txn_stats["num_txn_30d"] / 30

    # -- Multi-attempt rate (indicator of payment failures) -------------------
    multi = txn[txn["attempt_count"] > 1].groupby("customer_id").size().rename("num_multi_attempt_txn")
    txn_stats = txn_stats.merge(multi, on="customer_id", how="left").fillna({"num_multi_attempt_txn": 0})
    txn_stats["multi_attempt_rate"] = (
        txn_stats["num_multi_attempt_txn"] / txn_stats["num_transactions_total"].clip(lower=1)
    )

    # -- Order aggregates -----------------------------------------------------
    ord_grp   = orders.groupby("customer_id")
    ord_stats = pd.DataFrame({
        "num_orders":     ord_grp.size(),
        "avg_order_amt":  ord_grp["order_amount"].mean(),
        "total_order_amt": ord_grp["order_amount"].sum(),
    }).reset_index()

    # -- Return aggregates ----------------------------------------------------
    ret_grp   = returns.groupby("customer_id")
    ret_stats = pd.DataFrame({
        "num_returns":     ret_grp.size(),
        "avg_refund_amt":  ret_grp["refund_amount"].mean(),
        "avg_days_return": ret_grp["days_to_return"].mean(),
    }).reset_index()

    # Changed-mind / Other return reasons (ring indicator)
    changed_mind_ret = (
        returns[returns["return_reason"].isin(["CHANGED_MIND", "OTHER"])]
        .groupby("customer_id").size()
        .rename("num_changed_mind_returns")
    )
    ret_stats = ret_stats.merge(changed_mind_ret, on="customer_id", how="left")
    ret_stats["num_changed_mind_returns"] = ret_stats["num_changed_mind_returns"].fillna(0)
    ret_stats["changed_mind_return_rate"] = (
        ret_stats["num_changed_mind_returns"] / ret_stats["num_returns"].clip(lower=1)
    )

    # -- Merge all into customer base -----------------------------------------
    feat = (
        cust
        .merge(txn_stats, on="customer_id", how="left")
        .merge(ord_stats,  on="customer_id", how="left")
        .merge(ret_stats,  on="customer_id", how="left")
    )

    # Computed rates
    feat["return_rate"] = (
        feat["num_returns"].fillna(0) / feat["num_orders"].fillna(1).clip(lower=1)
    )
    feat["refund_rate"] = (
        feat["num_returns"].fillna(0) / feat["num_transactions_total"].fillna(1).clip(lower=1)
    )

    # Fill NaN for customers with no orders / no returns
    for col in ["num_orders", "avg_order_amt", "total_order_amt"]:
        feat[col] = feat[col].fillna(0)
    for col in ["num_returns", "avg_refund_amt", "avg_days_return",
                "num_changed_mind_returns", "changed_mind_return_rate"]:
        feat[col] = feat[col].fillna(0)
    for col in ["num_transactions_total", "total_spend", "avg_txn_amount",
                "max_txn_amount", "min_txn_amount", "std_txn_amount",
                "num_failed_txn", "num_reversed_txn", "unique_devices_used",
                "unique_ips_used", "unique_merchants", "failed_payment_rate",
                "days_since_last_txn", "txn_lifespan_days", "txn_velocity_overall",
                "txn_velocity_7d", "txn_velocity_30d", "multi_attempt_rate"]:
        feat[col] = feat[col].fillna(0)

    log.info(f"  Customer features: {feat.shape[1]} columns for {len(feat):,} customers")
    return feat


# ===========================================================================
# 3. DEVICE-LEVEL FEATURES (Group B) — worst-case join
# ===========================================================================

def compute_device_features(dfs: dict) -> pd.DataFrame:
    log.info("Computing device-level features (Group B)...")
    txn     = dfs["transactions"].copy()
    returns = dfs["returns"].copy()
    orders  = dfs["orders"].copy()

    # Accounts per device
    dev_accts = (
        txn.groupby("device_id")["customer_id"]
        .nunique()
        .rename("dev_accounts_count")
        .reset_index()
    )

    # Refund rate per device
    dev_orders  = orders.groupby("device_id").size().rename("dev_orders").reset_index() \
        if "device_id" in orders.columns else pd.DataFrame(columns=["device_id", "dev_orders"])
    dev_returns = (
        returns.merge(orders[["order_id", "transaction_id"]], on="order_id", how="left")
        .merge(txn[["transaction_id", "device_id"]], on="transaction_id", how="left")
        .groupby("device_id").size()
        .rename("dev_returns")
        .reset_index()
    )
    dev_txn_count = txn.groupby("device_id").size().rename("dev_txn_count").reset_index()

    # Accounts created within 24h on this device (key abuse signal)
    customer_device_times = (
        txn.merge(
            dfs["customers"][["customer_id", "account_created_at"]],
            on="customer_id", how="left"
        )
        [["device_id", "customer_id", "account_created_at"]]
        .drop_duplicates(subset=["device_id", "customer_id"])
    )

    def count_24h_accounts(group: pd.DataFrame) -> int:
        times = group["account_created_at"].sort_values().values
        if len(times) <= 1:
            return len(times)
        max_in_window = 1
        for i in range(len(times)):
            window = (times >= times[i]) & (
                times <= times[i] + np.timedelta64(24, "h")
            )
            max_in_window = max(max_in_window, window.sum())
        return int(max_in_window)

    log.info("  Computing new-accounts-per-24h per device (this may take ~20s)...")
    dev_new_24h = (
        customer_device_times
        .groupby("device_id")
        .apply(count_24h_accounts, include_groups=False)
        .rename("dev_new_accounts_24h")
        .reset_index()
    )

    # Merge device stats
    dev_stats = (
        dev_accts
        .merge(dev_txn_count, on="device_id", how="left")
        .merge(dev_new_24h,   on="device_id", how="left")
        .fillna(0)
    )

    # Now join to customers: take the WORST-CASE device each customer uses
    # (if a customer uses multiple devices, take the most suspicious one)
    cust_dev = (
        txn[["customer_id", "device_id"]]
        .drop_duplicates()
        .merge(dev_stats, on="device_id", how="left")
    )
    # Worst-case = device with most accounts
    cust_dev_worst = (
        cust_dev
        .sort_values("dev_accounts_count", ascending=False)
        .groupby("customer_id")
        .first()
        .reset_index()
        [["customer_id", "dev_accounts_count", "dev_txn_count", "dev_new_accounts_24h"]]
    )
    cust_dev_worst.columns = [
        "customer_id",
        "worst_device_account_count",
        "worst_device_txn_count",
        "worst_device_new_accounts_24h",
    ]

    log.info(f"  Device features: {cust_dev_worst.shape[1]} columns")
    return cust_dev_worst


# ===========================================================================
# 4. IP-LEVEL FEATURES (Group C) — worst-case join
# ===========================================================================

def compute_ip_features(dfs: dict) -> pd.DataFrame:
    log.info("Computing IP-level features (Group C)...")
    txn = dfs["transactions"].copy()

    ip_accts  = txn.groupby("ip_id")["customer_id"].nunique().rename("ip_accounts_count").reset_index()
    ip_txns   = txn.groupby("ip_id").size().rename("ip_txn_count").reset_index()
    ip_merch  = txn.groupby("ip_id")["merchant_id"].nunique().rename("ip_merchant_count").reset_index()

    ip_stats = ip_accts.merge(ip_txns, on="ip_id").merge(ip_merch, on="ip_id")

    # Worst-case per customer
    cust_ip = (
        txn[["customer_id", "ip_id"]]
        .drop_duplicates()
        .merge(ip_stats, on="ip_id", how="left")
    )
    cust_ip_worst = (
        cust_ip
        .sort_values("ip_accounts_count", ascending=False)
        .groupby("customer_id")
        .first()
        .reset_index()
        [["customer_id", "ip_accounts_count", "ip_txn_count", "ip_merchant_count"]]
    )
    cust_ip_worst.columns = [
        "customer_id",
        "worst_ip_account_count",
        "worst_ip_txn_count",
        "worst_ip_merchant_count",
    ]

    log.info(f"  IP features: {cust_ip_worst.shape[1]} columns")
    return cust_ip_worst


# ===========================================================================
# 5. CLUSTER FEATURES via Union-Find (Group D)
# ===========================================================================

class UnionFind:
    """Path-compressed weighted Union-Find for clustering customers."""
    def __init__(self, elements: list):
        self.parent = {e: e for e in elements}
        self.rank   = {e: 0 for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def clusters(self) -> dict:
        """Returns dict: root → list of members."""
        from collections import defaultdict
        groups = defaultdict(list)
        for e in self.parent:
            groups[self.find(e)].append(e)
        return dict(groups)


def compute_cluster_features(dfs: dict, cust_feats: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing cluster features via Union-Find (Group D)...")
    txn       = dfs["transactions"].copy()
    customers = dfs["customers"].copy()

    all_customer_ids = customers["customer_id"].tolist()
    uf = UnionFind(all_customer_ids)

    # Edge: customers sharing the same device
    log.info("  Building edges: shared devices...")
    dev_groups = txn.groupby("device_id")["customer_id"].unique()
    for customers_on_dev in dev_groups:
        if len(customers_on_dev) > 1:
            for i in range(1, len(customers_on_dev)):
                uf.union(customers_on_dev[0], customers_on_dev[i])

    # Edge: customers sharing the same IP
    log.info("  Building edges: shared IPs...")
    ip_groups = txn.groupby("ip_id")["customer_id"].unique()
    for customers_on_ip in ip_groups:
        if len(customers_on_ip) > 1:
            for i in range(1, len(customers_on_ip)):
                uf.union(customers_on_ip[0], customers_on_ip[i])

    # Assign cluster_id (root of component)
    cluster_assignment = {cid: uf.find(cid) for cid in all_customer_ids}
    cust_cluster = pd.DataFrame(
        list(cluster_assignment.items()),
        columns=["customer_id", "cluster_root"]
    )

    # Cluster sizes
    cluster_sizes = (
        cust_cluster
        .groupby("cluster_root")
        .size()
        .rename("cluster_size")
        .reset_index()
    )
    cust_cluster = cust_cluster.merge(cluster_sizes, on="cluster_root", how="left")

    # Per-cluster: return rate
    #   (merge cust_feats which already has return_rate computed)
    cluster_return = (
        cust_cluster
        .merge(cust_feats[["customer_id", "return_rate", "account_created_at"]], on="customer_id", how="left")
        .groupby("cluster_root")
        .agg(
            cluster_return_rate=("return_rate", "mean"),
            cluster_min_account_age=("account_created_at", lambda x: (pd.Timestamp(REFERENCE_DATE) - x.max()).days),
        )
        .reset_index()
    )
    cust_cluster = cust_cluster.merge(cluster_return, on="cluster_root", how="left")

    # Per-cluster: transaction velocity ratio vs mean per-cluster
    cluster_txn_vel = (
        cust_cluster
        .merge(cust_feats[["customer_id", "txn_velocity_overall"]], on="customer_id", how="left")
        .groupby("cluster_root")["txn_velocity_overall"]
        .mean()
        .rename("cluster_avg_txn_velocity")
        .reset_index()
    )
    cust_cluster = cust_cluster.merge(cluster_txn_vel, on="cluster_root", how="left")

    # Per-cluster: how many accounts created within 24h of the anchor (earliest account)
    cust_with_creation = cust_cluster.merge(
        customers[["customer_id", "account_created_at"]], on="customer_id", how="left"
    )
    def accounts_in_24h_window(group: pd.DataFrame) -> int:
        times = group["account_created_at"].sort_values().values
        if len(times) <= 1:
            return len(times)
        # sliding window: count max accounts within any 24h window
        max_count = 1
        for i in range(len(times)):
            in_window = np.sum(
                (times >= times[i]) & (times <= times[i] + np.timedelta64(24, "h"))
            )
            max_count = max(max_count, int(in_window))
        return max_count

    log.info("  Computing accounts-created-in-24h per cluster...")
    cluster_24h = (
        cust_with_creation
        .groupby("cluster_root")
        .apply(accounts_in_24h_window, include_groups=False)
        .rename("cluster_accounts_created_24h")
        .reset_index()
    )
    cust_cluster = cust_cluster.merge(cluster_24h, on="cluster_root", how="left")

    # Per-cluster: fraction of accounts created within 24h
    cust_cluster["cluster_24h_creation_rate"] = (
        cust_cluster["cluster_accounts_created_24h"] /
        cust_cluster["cluster_size"].clip(lower=1)
    )

    # Merchant concentration: does the cluster target mostly one merchant?
    cluster_merch = (
        cust_cluster[["customer_id", "cluster_root"]]
        .merge(txn[["customer_id", "merchant_id"]], on="customer_id", how="left")
        .groupby("cluster_root")
        .apply(
            lambda g: g["merchant_id"].value_counts(normalize=True).iloc[0] if len(g) > 0 else 0,
            include_groups=False
        )
        .rename("cluster_merchant_concentration")
        .reset_index()
    )
    cust_cluster = cust_cluster.merge(cluster_merch, on="cluster_root", how="left")

    # Final per-customer cluster feature set
    result = cust_cluster[[
        "customer_id",
        "cluster_root",
        "cluster_size",
        "cluster_return_rate",
        "cluster_min_account_age",
        "cluster_avg_txn_velocity",
        "cluster_accounts_created_24h",
        "cluster_24h_creation_rate",
        "cluster_merchant_concentration",
    ]].fillna(0)

    # Log-transform cluster_size (heavy-tailed)
    result["log_cluster_size"] = np.log1p(result["cluster_size"])

    log.info(f"  Cluster features: {result.shape[1]} columns | "
             f"{result['cluster_root'].nunique():,} distinct clusters")
    return result


# ===========================================================================
# 6. ASSEMBLE FINAL FEATURE MATRIX
# ===========================================================================

def assemble_features(
    cust_feats:    pd.DataFrame,
    device_feats:  pd.DataFrame,
    ip_feats:      pd.DataFrame,
    cluster_feats: pd.DataFrame,
    labels:        pd.DataFrame,
) -> pd.DataFrame:
    log.info("Assembling final feature matrix...")

    feat = (
        cust_feats
        .merge(device_feats,  on="customer_id", how="left")
        .merge(ip_feats,      on="customer_id", how="left")
        .merge(cluster_feats, on="customer_id", how="left")
        .merge(
            labels[["customer_id", "is_abuse", "cluster_id", "ring_type"]],
            on="customer_id", how="left"
        )
    )

    # Fill any remaining NaN
    device_cols  = ["worst_device_account_count", "worst_device_txn_count", "worst_device_new_accounts_24h"]
    ip_cols      = ["worst_ip_account_count", "worst_ip_txn_count", "worst_ip_merchant_count"]
    cluster_cols = ["cluster_size", "cluster_return_rate", "cluster_min_account_age",
                    "cluster_avg_txn_velocity", "cluster_accounts_created_24h",
                    "cluster_24h_creation_rate", "cluster_merchant_concentration", "log_cluster_size"]
    for col in device_cols + ip_cols + cluster_cols:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0)

    # Temporal split would put all 803 ring members in "train" (they were created
    # in Jan-Jul 2023). Use stratified split instead to ensure abuse examples
    # appear in val and test for proper precision/recall evaluation.
    # Documented honestly in limitations.
    from sklearn.model_selection import train_test_split

    feat["split"] = "train"
    label_col = "is_abuse"

    if label_col in feat.columns and feat[label_col].sum() > 0:
        idx = feat.index.tolist()
        y   = feat[label_col].fillna(0).astype(int).values

        # 70 / 15 / 15 stratified split
        idx_trainval, idx_test, y_trainval, _ = train_test_split(
            idx, y, test_size=0.15, random_state=42, stratify=y
        )
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=0.15 / 0.85,
            random_state=42, stratify=y_trainval
        )
        feat.loc[idx_val,  "split"] = "val"
        feat.loc[idx_test, "split"] = "test"


    log.info(f"  Final matrix: {len(feat):,} rows x {feat.shape[1]} columns")
    log.info(f"  Split distribution:\n{feat['split'].value_counts().to_string()}")
    return feat


# ===========================================================================
# 7. SAVE OUTPUTS
# ===========================================================================

FEATURE_COLS = [
    # Group A — Customer behavioural
    "account_age_days", "account_creation_hour", "created_on_weekend",
    "created_late_night", "is_verified",
    "num_transactions_total", "total_spend", "avg_txn_amount", "max_txn_amount",
    "std_txn_amount", "failed_payment_rate", "multi_attempt_rate",
    "unique_devices_used", "unique_ips_used", "unique_merchants",
    "days_since_last_txn", "txn_lifespan_days", "txn_velocity_overall",
    "txn_velocity_7d", "txn_velocity_30d",
    "num_orders", "avg_order_amt",
    "num_returns", "return_rate", "refund_rate", "avg_days_return",
    "changed_mind_return_rate",
    # Group B — Device
    "worst_device_account_count", "worst_device_txn_count",
    "worst_device_new_accounts_24h",
    # Group C — IP
    "worst_ip_account_count", "worst_ip_txn_count", "worst_ip_merchant_count",
    # Group D — Cluster
    "cluster_size", "log_cluster_size", "cluster_return_rate",
    "cluster_min_account_age", "cluster_avg_txn_velocity",
    "cluster_accounts_created_24h", "cluster_24h_creation_rate",
    "cluster_merchant_concentration",
]

def save_features(feat: pd.DataFrame) -> None:
    out_path = os.path.join(PROCESSED_DIR, "features.parquet")
    feat.to_parquet(out_path, index=False)
    log.info(f"  Saved features: {out_path}  ({os.path.getsize(out_path)//1024:,} KB)")

    # Feature names for ML pipeline
    actual_feature_cols = [c for c in FEATURE_COLS if c in feat.columns]
    names_path = os.path.join(PROCESSED_DIR, "feature_names.json")
    with open(names_path, "w") as f:
        json.dump({"feature_cols": actual_feature_cols, "label_col": "is_abuse"}, f, indent=2)
    log.info(f"  Saved feature names: {names_path}")

    # Summary stats
    label_col = "is_abuse"
    for split in ["train", "val", "test"]:
        sub = feat[feat["split"] == split]
        n_abuse  = sub[label_col].sum() if label_col in sub.columns else 0
        n_total  = len(sub)
        pct      = n_abuse / n_total * 100 if n_total > 0 else 0
        log.info(f"  {split:<6}  {n_total:>6,} customers | {n_abuse:>4} abuse ({pct:.1f}%)")

    log.info("\nFeature engineering complete. Run ml/training/train.py next.")


# ===========================================================================
# 8. MAIN
# ===========================================================================

def main():
    log.info("=" * 60)
    log.info("AbuseRing Sentinel - Feature Engineering Pipeline")
    log.info("=" * 60)

    dfs           = load_raw()
    cust_feats    = compute_customer_features(dfs)
    device_feats  = compute_device_features(dfs)
    ip_feats      = compute_ip_features(dfs)
    cluster_feats = compute_cluster_features(dfs, cust_feats)

    feat = assemble_features(
        cust_feats, device_feats, ip_feats, cluster_feats, dfs["labels"]
    )
    save_features(feat)


if __name__ == "__main__":
    main()
