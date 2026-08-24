#!/usr/bin/env python3
"""
AbuseRing Sentinel — Data Validation Script
============================================
Validates the generated synthetic dataset for quality and consistency.

Usage:
    python scripts/validate_data.py                     # default: data/raw
    python scripts/validate_data.py --input data/raw
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FILES = [
    "customers.csv",
    "transactions.csv",
    "orders.csv",
    "returns.csv",
    "devices.csv",
    "ips.csv",
    "merchants.csv",
    "abuse_labels.csv",
    "generation_summary.json",
]

REQUIRED_COLUMNS = {
    "customers.csv":    ["customer_id", "account_created_at", "location_city",
                         "location_state", "email_domain", "phone_hash", "is_verified"],
    "transactions.csv": ["transaction_id", "customer_id", "merchant_id", "timestamp",
                         "amount", "payment_method", "device_id", "ip_id", "status",
                         "attempt_count"],
    "orders.csv":       ["order_id", "customer_id", "transaction_id", "merchant_id",
                         "product_category", "order_time", "delivery_time",
                         "order_amount", "quantity"],
    "returns.csv":      ["return_id", "order_id", "customer_id", "merchant_id",
                         "return_time", "return_reason", "refund_amount", "days_to_return"],
    "devices.csv":      ["device_id", "device_type", "browser", "os",
                         "first_seen", "total_accounts"],
    "ips.csv":          ["ip_id", "country", "region", "city", "isp",
                         "first_seen", "total_accounts"],
    "merchants.csv":    ["merchant_id", "merchant_name", "category", "city",
                         "state", "baseline_refund_rate", "is_primary_target"],
    "abuse_labels.csv": ["customer_id", "cluster_id", "is_abuse", "ring_type"],
}

EXPECTED_MIN_COUNTS = {
    "customers.csv":    900,
    "transactions.csv": 9_000,
    "orders.csv":       6_000,
    "returns.csv":      500,
    "devices.csv":      200,
    "ips.csv":          200,
    "merchants.csv":    5,
    "abuse_labels.csv": 900,
}

TRAIN_END = datetime(2024, 9, 30)
VAL_END   = datetime(2024, 10, 31)
END_DATE  = datetime(2024, 11, 30)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def check(condition: bool, msg: str, warn: bool = False) -> bool:
    status = "✅" if condition else ("⚠️ " if warn else "❌")
    print(f"  {status}  {msg}")
    return condition

def validate_files_exist(data_dir: str) -> bool:
    print("\n── [1] File existence ──────────────────────────────────────────")
    ok = True
    for fname in REQUIRED_FILES:
        path   = os.path.join(data_dir, fname)
        exists = os.path.isfile(path)
        size   = os.path.getsize(path) if exists else 0
        ok    &= check(exists, f"{fname:<30} {'exists' if exists else 'MISSING'}"
                               + (f"  ({size/1024:.0f} KB)" if exists else ""))
    return ok

def validate_columns(data_dir: str, dfs: dict) -> bool:
    print("\n── [2] Column completeness ─────────────────────────────────────")
    ok = True
    for fname, required in REQUIRED_COLUMNS.items():
        if fname not in dfs:
            continue
        df      = dfs[fname]
        missing = [c for c in required if c not in df.columns]
        ok     &= check(
            len(missing) == 0,
            f"{fname:<30} columns OK" if not missing else f"{fname}: missing {missing}"
        )
    return ok

def validate_row_counts(dfs: dict, summary: dict) -> bool:
    print("\n── [3] Row counts ──────────────────────────────────────────────")
    ok       = True
    expected = summary.get("counts", {})
    for fname, df in dfs.items():
        n            = len(df)
        exp_key      = fname.replace(".csv", "")
        exp_count    = expected.get(exp_key, EXPECTED_MIN_COUNTS.get(fname, 0))
        min_ok       = n >= EXPECTED_MIN_COUNTS.get(fname, 0)
        match_ok     = (exp_count == 0) or (n == exp_count)
        check(min_ok,   f"{fname:<30} {n:>10,} rows  (expected ≥ {EXPECTED_MIN_COUNTS.get(fname, 0):,})")
        ok &= min_ok
    return ok

def validate_null_rates(dfs: dict) -> bool:
    print("\n── [4] Missing values ──────────────────────────────────────────")
    ok = True
    for fname, df in dfs.items():
        null_cols = df.columns[df.isnull().any()].tolist()
        null_pcts = {c: round(df[c].isnull().mean() * 100, 1) for c in null_cols}
        high_null = {c: p for c, p in null_pcts.items() if p > 10.0}
        # cluster_id / ring_type can be null (legitimate customers have no ring)
        skip_null = {"cluster_id", "ring_type"}
        real_high = {c: p for c, p in high_null.items() if c not in skip_null}
        if real_high:
            check(False, f"{fname:<30} HIGH nulls: {real_high}")
            ok = False
        else:
            check(True, f"{fname:<30} nulls within acceptable range")
    return ok

def validate_foreign_keys(dfs: dict) -> bool:
    print("\n── [5] Foreign key integrity ───────────────────────────────────")
    ok = True
    if "transactions.csv" in dfs and "customers.csv" in dfs:
        cust_ids   = set(dfs["customers.csv"]["customer_id"])
        txn_cids   = set(dfs["transactions.csv"]["customer_id"])
        orphans    = txn_cids - cust_ids
        ok        &= check(len(orphans) == 0,
                           f"transactions→customers  {len(orphans)} orphan customer_ids")

    if "transactions.csv" in dfs and "devices.csv" in dfs:
        dev_ids    = set(dfs["devices.csv"]["device_id"])
        txn_devs   = set(dfs["transactions.csv"]["device_id"])
        orphans    = txn_devs - dev_ids
        ok        &= check(len(orphans) == 0,
                           f"transactions→devices    {len(orphans)} orphan device_ids")

    if "orders.csv" in dfs and "transactions.csv" in dfs:
        txn_ids    = set(dfs["transactions.csv"]["transaction_id"])
        ord_txns   = set(dfs["orders.csv"]["transaction_id"])
        orphans    = ord_txns - txn_ids
        ok        &= check(len(orphans) == 0,
                           f"orders→transactions     {len(orphans)} orphan transaction_ids")

    if "returns.csv" in dfs and "orders.csv" in dfs:
        order_ids  = set(dfs["orders.csv"]["order_id"])
        ret_orders = set(dfs["returns.csv"]["order_id"])
        orphans    = ret_orders - order_ids
        ok        &= check(len(orphans) == 0,
                           f"returns→orders          {len(orphans)} orphan order_ids")
    return ok

def validate_abuse_labels(dfs: dict, summary: dict) -> bool:
    print("\n── [6] Abuse label distribution ────────────────────────────────")
    ok = True
    if "abuse_labels.csv" not in dfs:
        return False

    labels     = dfs["abuse_labels.csv"]
    total      = len(labels)
    n_abuse    = int(labels["is_abuse"].sum())
    pct        = n_abuse / total * 100
    n_clusters = labels["cluster_id"].nunique() - 1  # exclude NaN

    ok &= check(n_abuse > 0,         f"Abuse members:  {n_abuse:,} ({pct:.1f}% of customers)")
    ok &= check(3 <= pct <= 20,      f"Abuse %:        {pct:.1f}% (expected 3–20%)", warn=True)
    ok &= check(n_clusters >= 1,     f"Distinct rings: {n_clusters}")

    # Check all ring members have a cluster_id
    ring_rows    = labels[labels["is_abuse"] == 1]
    missing_cids = ring_rows["cluster_id"].isnull().sum()
    ok          &= check(missing_cids == 0,
                         f"Ring members with cluster_id: {n_abuse - missing_cids}/{n_abuse}")
    return ok

def validate_temporal_structure(dfs: dict) -> bool:
    print("\n── [7] Temporal structure ──────────────────────────────────────")
    ok = True
    if "transactions.csv" not in dfs:
        return False

    txn = dfs["transactions.csv"].copy()
    txn["ts"] = pd.to_datetime(txn["timestamp"], format="ISO8601")

    train_n = (txn["ts"] <= TRAIN_END).sum()
    val_n   = ((txn["ts"] > TRAIN_END) & (txn["ts"] <= VAL_END)).sum()
    test_n  = (txn["ts"] > VAL_END).sum()
    total   = len(txn)

    ok &= check(train_n > 0, f"Train transactions:  {train_n:>10,}  ({train_n/total*100:.1f}%)")
    ok &= check(val_n > 0,   f"Val transactions:    {val_n:>10,}  ({val_n/total*100:.1f}%)")
    ok &= check(test_n > 0,  f"Test transactions:   {test_n:>10,}  ({test_n/total*100:.1f}%)")
    ok &= check(40 <= train_n / total * 100 <= 80,
                "Train proportion 40–80% ✓" if 40 <= train_n / total * 100 <= 80 else
                "Train proportion outside expected range", warn=True)
    return ok

def validate_behavioral_signals(dfs: dict) -> bool:
    print("\n── [8] Behavioral signal validation ────────────────────────────")
    ok = True
    if "returns.csv" not in dfs or "abuse_labels.csv" not in dfs or "orders.csv" not in dfs:
        return True  # skip if data missing

    labels  = dfs["abuse_labels.csv"].set_index("customer_id")["is_abuse"]
    orders  = dfs["orders.csv"]
    returns = dfs["returns.csv"]

    order_counts  = orders.groupby("customer_id").size().rename("order_count")
    return_counts = returns.groupby("customer_id").size().rename("return_count")

    merged = pd.concat([order_counts, return_counts, labels], axis=1).fillna(0)
    merged["return_rate"] = merged["return_count"] / merged["order_count"].clip(lower=1)

    ring_rr   = merged[merged["is_abuse"] == 1]["return_rate"].mean()
    normal_rr = merged[merged["is_abuse"] == 0]["return_rate"].mean()

    ok &= check(ring_rr > normal_rr,
                f"Ring return rate ({ring_rr:.2%}) > Normal ({normal_rr:.2%})")
    ok &= check(ring_rr > 0.4,
                f"Ring return rate {ring_rr:.2%} (expected > 40%)", warn=ring_rr < 0.4)
    ok &= check(normal_rr < 0.25,
                f"Normal return rate {normal_rr:.2%} (expected < 25%)", warn=normal_rr > 0.25)

    # Device sharing
    if "transactions.csv" in dfs:
        txn = dfs["transactions.csv"]
        dev_accts = txn.groupby("device_id")["customer_id"].nunique()
        max_dev   = dev_accts.max()
        ok &= check(max_dev >= 5,
                    f"Max accounts per device: {max_dev} (ring device sharing present)")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AbuseRing Sentinel — Data Validator")
    parser.add_argument("--input", default="data/raw", help="Input data directory")
    args = parser.parse_args()

    data_dir = args.input
    print(f"\n🔍  Validating dataset in: {data_dir}\n")

    # Load summary
    summary_path = os.path.join(data_dir, "generation_summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"  📋 Summary: generated at {summary.get('generated_at', 'unknown')}")
        print(f"     Seed: {summary.get('seed', '?')} | "
              f"Rings: {summary.get('abuse_stats', {}).get('ring_count', '?')}")

    # Check files exist
    if not validate_files_exist(data_dir):
        print("\n❌  Some required files are missing. Run generate_data.py first.\n")
        sys.exit(1)

    # Load all CSVs
    dfs = {}
    for fname in REQUIRED_COLUMNS:
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            dfs[fname] = pd.read_csv(path)

    # Run all validations
    results = [
        validate_columns(data_dir, dfs),
        validate_row_counts(dfs, summary),
        validate_null_rates(dfs),
        validate_foreign_keys(dfs),
        validate_abuse_labels(dfs, summary),
        validate_temporal_structure(dfs),
        validate_behavioral_signals(dfs),
    ]

    all_ok = all(results)
    print(f"""
{'═'*55}
  Validation: {'✅  ALL CHECKS PASSED' if all_ok else '❌  SOME CHECKS FAILED'}
{'═'*55}
""")

    if not all_ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
