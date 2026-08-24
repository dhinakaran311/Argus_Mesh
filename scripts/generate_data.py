#!/usr/bin/env python3
"""
AbuseRing Sentinel — Synthetic Dataset Generator
=================================================
Generates a realistic payment-ecosystem dataset with injected
coordinated abuse rings for ML training and evaluation.

Design principle — realistic overlaps:
    Legitimate families ALSO share devices (2-4 accounts, normal behaviour).
    The model must combine MULTIPLE signals to separate abuse from normal.
    A single feature (e.g. shared device) is NOT sufficient.

Dataset (full scale):
    customers.csv       10,000 customers
    transactions.csv   200,000 transactions  (Jan–Nov 2024)
    orders.csv         150,000 orders
    returns.csv         ~22,000 returns
    devices.csv          3,500 devices
    ips.csv              4,200 IP addresses
    abuse_labels.csv    10,000 rows (one per customer, ground truth)

Abuse rings:
    15 rings | 10–100 members each | ~750 total ring members (~7.5%)
    Ring members share devices/IPs, were created in short windows,
    and exhibit abnormally high refund + transaction velocity.

Temporal split (for train/val/test):
    Jan 01 – Sep 30  →  Training   (70%)
    Oct 01 – Oct 31  →  Validation (15%)
    Nov 01 – Nov 30  →  Held-out test (15%)

Usage:
    pip install pandas numpy
    python scripts/generate_data.py              # full scale
    python scripts/generate_data.py --scale sample  # 1,000 customers (quick)
    python scripts/generate_data.py --output data/raw
"""

import os
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 11, 30)
TOTAL_DAYS = (END_DATE - START_DATE).days  # 334 days

TRAIN_END  = datetime(2024, 9, 30)
VAL_END    = datetime(2024, 10, 31)
# Nov 1–30 = held-out test

SCALE_CONFIG = {
    "full": {
        "n_customers":                10_000,
        "n_devices":                   3_500,
        "n_ips":                       4_200,
        "n_merchants":                    51,   # 1 target + 50 others
        "n_rings":                        15,
        "n_families":                    200,   # legitimate device sharing noise
        "avg_txn_per_normal_customer":    18,
        "avg_txn_per_ring_member":        28,
    },
    "sample": {
        "n_customers":                 1_000,
        "n_devices":                     350,
        "n_ips":                         420,
        "n_merchants":                    11,
        "n_rings":                         3,
        "n_families":                     20,
        "avg_txn_per_normal_customer":    18,
        "avg_txn_per_ring_member":        28,
    },
}

# Ring sizes: 5 small, 5 medium, 5 large
RING_SIZE_RANGES = {
    "full":   {"small": (10, 25),  "medium": (30, 60),  "large":  (70, 100)},
    "sample": {"small": (5,  10),  "medium": (15, 25),  "large":  (30,  45)},
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────

INDIAN_CITIES = {
    "Mumbai": "Maharashtra",    "Delhi": "Delhi",
    "Bangalore": "Karnataka",   "Chennai": "Tamil Nadu",
    "Hyderabad": "Telangana",   "Kolkata": "West Bengal",
    "Pune": "Maharashtra",      "Ahmedabad": "Gujarat",
    "Jaipur": "Rajasthan",      "Surat": "Gujarat",
    "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh",
    "Nagpur": "Maharashtra",    "Indore": "Madhya Pradesh",
    "Bhopal": "Madhya Pradesh", "Visakhapatnam": "Andhra Pradesh",
    "Patna": "Bihar",           "Vadodara": "Gujarat",
    "Coimbatore": "Tamil Nadu", "Kochi": "Kerala",
}

CITY_LIST  = list(INDIAN_CITIES.keys())
STATE_MAP  = INDIAN_CITIES

PAYMENT_METHODS = ["UPI", "CARD", "NETBANKING", "WALLET"]
PAYMENT_WEIGHTS = [0.60, 0.25, 0.10, 0.05]   # UPI dominant in India

PRODUCT_CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books",
    "Sports", "Beauty", "Grocery", "Toys", "Automotive",
]

RETURN_REASONS = [
    "DEFECTIVE", "WRONG_ITEM", "NOT_DELIVERED", "CHANGED_MIND", "OTHER"
]
# Ring members use CHANGED_MIND/OTHER disproportionately
NORMAL_RETURN_REASON_WEIGHTS = [0.35, 0.25, 0.20, 0.12, 0.08]
RING_RETURN_REASON_WEIGHTS   = [0.20, 0.15, 0.10, 0.35, 0.20]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com",
    "hotmail.com", "rediffmail.com", "ymail.com",
]

DEVICE_TYPES  = ["MOBILE", "DESKTOP", "TABLET"]
DEVICE_WEIGHTS = [0.70, 0.22, 0.08]

BROWSERS = ["Chrome", "Firefox", "Safari", "Edge", "Samsung Internet"]
OS_TYPES = ["Android", "Windows", "iOS", "macOS", "Linux"]

RING_TYPES = ["REFUND_RING", "PROMO_ABUSE", "RETURN_FRAUD"]

ISP_NAMES = [
    "Jio", "Airtel", "BSNL", "Vodafone", "Reliance",
    "ACT Fibernet", "Hathway", "BSNL Broadband",
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

_counters: dict = {}

def fmt_id(prefix: str, n: int, width: int = 6) -> str:
    return f"{prefix}{str(n).zfill(width)}"

def next_id(prefix: str, width: int = 6) -> str:
    _counters[prefix] = _counters.get(prefix, 0) + 1
    return fmt_id(prefix, _counters[prefix], width)

def rand_datetime(start: datetime, end: datetime) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, delta))

def rand_datetime_near(base: datetime, hours_window: float) -> datetime:
    """Return a datetime within ±hours_window of base."""
    offset = random.uniform(0, hours_window * 3600)
    return base + timedelta(seconds=offset)

def lognormal_amount(mean: float, sigma: float = 0.5) -> float:
    """Return a positive INR amount from a lognormal distribution."""
    val = np.random.lognormal(math.log(mean), sigma)
    return round(max(50.0, min(val, 50_000.0)), 2)

def phone_hash(n: int) -> str:
    """Simulate a hashed phone (last 4 digits visible)."""
    return f"XXXX-XXXX-{n % 10000:04d}"

def split_label(ts: datetime) -> str:
    if ts <= TRAIN_END:
        return "train"
    elif ts <= VAL_END:
        return "val"
    else:
        return "test"

# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERATE DEVICES
# ─────────────────────────────────────────────────────────────────────────────

def generate_devices(n: int) -> pd.DataFrame:
    print(f"  Generating {n:,} devices...")
    rows = []
    for _ in range(n):
        did = next_id("D", 4)
        rows.append({
            "device_id":    did,
            "device_type":  np.random.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS),
            "browser":      random.choice(BROWSERS),
            "os":           random.choice(OS_TYPES),
            "first_seen":   rand_datetime(START_DATE, END_DATE).isoformat(),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERATE IP ADDRESSES
# ─────────────────────────────────────────────────────────────────────────────

def generate_ips(n: int) -> pd.DataFrame:
    print(f"  Generating {n:,} IP addresses...")
    rows = []
    for _ in range(n):
        city = random.choice(CITY_LIST)
        rows.append({
            "ip_id":      next_id("I", 4),
            "country":    "India",
            "region":     STATE_MAP[city],
            "city":       city,
            "isp":        random.choice(ISP_NAMES),
            "first_seen": rand_datetime(START_DATE, END_DATE).isoformat(),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERATE MERCHANTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_merchants(n: int) -> pd.DataFrame:
    print(f"  Generating {n:,} merchants (M0001 = primary target)...")
    rows = []
    for i in range(n):
        mid = fmt_id("M", i + 1, 4)
        city = random.choice(CITY_LIST)
        rows.append({
            "merchant_id":            mid,
            "merchant_name":          f"Merchant_{mid}",
            "category":               random.choice(PRODUCT_CATEGORIES),
            "city":                   city,
            "state":                  STATE_MAP[city],
            "baseline_refund_rate":   round(random.uniform(0.06, 0.14), 3),
            "is_primary_target":      (i == 0),   # M0001 is the abused merchant
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 7. GENERATE ABUSE RINGS
# ─────────────────────────────────────────────────────────────────────────────

def generate_abuse_rings(cfg: dict, device_ids: list, ip_ids: list) -> tuple:
    """
    Returns:
        ring_customers: list of customer dicts (ring members)
        ring_meta:      list of ring metadata dicts
        ring_device_map: dict customer_id → [device_ids]
        ring_ip_map:    dict customer_id → [ip_ids]
        ring_member_map: dict customer_id → ring_id
    """
    n_rings    = cfg["n_rings"]
    scale_name = "full" if cfg["n_customers"] >= 5000 else "sample"
    size_cfg   = RING_SIZE_RANGES[scale_name]

    # Distribute rings across time periods:
    #   60% train, 20% val, 20% test
    ring_period_assignments = (
        ["train"] * max(1, int(n_rings * 0.60)) +
        ["val"]   * max(1, int(n_rings * 0.20)) +
        ["test"]  * max(1, int(n_rings * 0.20))
    )[:n_rings]
    random.shuffle(ring_period_assignments)

    # 5 small, 5 medium, 5 large (scaled to n_rings)
    sizes_per_tier = n_rings // 3
    ring_sizes = (
        [random.randint(*size_cfg["small"])  for _ in range(sizes_per_tier)] +
        [random.randint(*size_cfg["medium"]) for _ in range(sizes_per_tier)] +
        [random.randint(*size_cfg["large"])  for _ in range(n_rings - 2 * sizes_per_tier)]
    )
    random.shuffle(ring_sizes)

    ring_customers  = []
    ring_meta       = []
    ring_device_map = {}   # customer_id → [device_ids used by this customer]
    ring_ip_map     = {}   # customer_id → [ip_ids used by this customer]
    ring_member_map = {}   # customer_id → ring_id

    for ring_idx in range(n_rings):
        ring_id   = f"RING-{ring_idx + 1:03d}"
        ring_size = ring_sizes[ring_idx]
        ring_type = random.choice(RING_TYPES)
        period    = ring_period_assignments[ring_idx]

        # Pick creation start time within the period
        if period == "train":
            period_start, period_end = START_DATE, TRAIN_END
        elif period == "val":
            period_start, period_end = TRAIN_END + timedelta(days=1), VAL_END
        else:
            period_start, period_end = VAL_END + timedelta(days=1), END_DATE

        # Leave at least 30 days for ring to operate after creation
        creation_pool_end  = period_end - timedelta(days=30)
        if creation_pool_end <= period_start:
            creation_pool_end = period_start + timedelta(days=5)

        ring_anchor_time   = rand_datetime(period_start, creation_pool_end)
        creation_window_h  = random.uniform(1.0, 24.0)   # accounts created in this window
        refund_rate        = random.uniform(0.70, 0.95)
        shared_device_ids  = random.sample(device_ids, k=random.randint(1, 3))
        shared_ip_ids      = random.sample(ip_ids,     k=random.randint(1, 2))

        ring_meta.append({
            "ring_id":          ring_id,
            "ring_type":        ring_type,
            "ring_size":        ring_size,
            "period":           period,
            "anchor_time":      ring_anchor_time.isoformat(),
            "creation_window_h": round(creation_window_h, 2),
            "refund_rate":      round(refund_rate, 3),
            "shared_devices":   shared_device_ids,
            "shared_ips":       shared_ip_ids,
        })

        for member_idx in range(ring_size):
            cid           = next_id("C")
            city          = random.choice(CITY_LIST)
            created_at    = rand_datetime_near(ring_anchor_time, creation_window_h)
            # Each member uses 1-2 of the shared devices and 1 of the shared IPs
            member_devices = random.sample(shared_device_ids, k=min(2, len(shared_device_ids)))
            member_ips     = random.sample(shared_ip_ids,     k=1)

            ring_customers.append({
                "customer_id":       cid,
                "account_created_at": created_at.isoformat(),
                "location_city":     city,
                "location_state":    STATE_MAP[city],
                "email_domain":      random.choice(EMAIL_DOMAINS),
                "phone_hash":        phone_hash(hash(cid) % 10000),
                "is_verified":       random.random() > 0.6,   # less often verified
                "_ring_id":          ring_id,
                "_ring_type":        ring_type,
                "_is_ring":          True,
                "_refund_rate":      refund_rate,
                "_primary_devices":  member_devices,
                "_primary_ips":      member_ips,
                "_period":           period,
                "_ring_anchor":      ring_anchor_time,
            })
            ring_device_map[cid] = member_devices
            ring_ip_map[cid]     = member_ips
            ring_member_map[cid] = ring_id

    print(f"  Injected {n_rings} abuse rings | "
          f"{len(ring_customers):,} total ring members")
    return ring_customers, ring_meta, ring_device_map, ring_ip_map, ring_member_map

# ─────────────────────────────────────────────────────────────────────────────
# 8. GENERATE LEGITIMATE CUSTOMERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_normal_customers(n: int, device_ids: list, ip_ids: list) -> tuple:
    """
    Returns:
        normal_customers: list of customer dicts
        normal_device_map: dict customer_id → [device_ids]
        normal_ip_map:    dict customer_id → [ip_ids]
    """
    print(f"  Generating {n:,} legitimate customers...")
    normal_customers  = []
    normal_device_map = {}
    normal_ip_map     = {}

    for _ in range(n):
        cid        = next_id("C")
        city       = random.choice(CITY_LIST)
        created_at = rand_datetime(START_DATE, END_DATE - timedelta(days=7))
        # Normal customers use 1–4 devices spread over time, 1–6 IPs
        n_devices  = np.random.choice([1, 2, 3, 4], p=[0.65, 0.22, 0.10, 0.03])
        n_ips      = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.45, 0.28, 0.15, 0.07, 0.03, 0.02])
        cust_devices = random.sample(device_ids, k=min(n_devices, len(device_ids)))
        cust_ips     = random.sample(ip_ids,     k=min(n_ips,     len(ip_ids)))

        normal_customers.append({
            "customer_id":       cid,
            "account_created_at": created_at.isoformat(),
            "location_city":     city,
            "location_state":    STATE_MAP[city],
            "email_domain":      random.choice(EMAIL_DOMAINS),
            "phone_hash":        phone_hash(hash(cid) % 10000),
            "is_verified":       random.random() > 0.25,
            "_ring_id":          None,
            "_ring_type":        None,
            "_is_ring":          False,
            "_refund_rate":      random.uniform(0.05, 0.15),
            "_primary_devices":  cust_devices,
            "_primary_ips":      cust_ips,
            "_period":           None,
            "_ring_anchor":      None,
        })
        normal_device_map[cid] = cust_devices
        normal_ip_map[cid]     = cust_ips

    return normal_customers, normal_device_map, normal_ip_map

# ─────────────────────────────────────────────────────────────────────────────
# 9. INJECT FAMILY SHARING (Legitimate noise — makes detection harder)
# ─────────────────────────────────────────────────────────────────────────────

def inject_family_sharing(
    normal_customers: list,
    device_ids: list,
    n_families: int,
) -> None:
    """
    Pick random legitimate customers and make them share a device —
    simulating a household. This ensures device_account_count alone
    is NOT sufficient to detect abuse rings.
    Modifies normal_customers in-place.
    """
    print(f"  Injecting {n_families} legitimate family device-sharing groups...")
    # Shuffle a copy of indices
    eligible  = [i for i, c in enumerate(normal_customers)]
    random.shuffle(eligible)

    cursor = 0
    for _ in range(n_families):
        family_size    = random.randint(2, 4)
        shared_device  = random.choice(device_ids)
        member_indices = eligible[cursor: cursor + family_size]
        cursor        += family_size
        if cursor >= len(eligible):
            break
        for idx in member_indices:
            # Insert the shared device at the start of their device list
            existing = normal_customers[idx]["_primary_devices"]
            if shared_device not in existing:
                normal_customers[idx]["_primary_devices"] = [shared_device] + existing

# ─────────────────────────────────────────────────────────────────────────────
# 10. GENERATE TRANSACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_transactions(
    all_customers: list,
    merchant_ids: list,
    cfg: dict,
) -> pd.DataFrame:
    """
    For ring members:
        - High velocity (burst within days)
        - Concentrate on primary merchant (M0001)
        - Amounts clustered around promotion-typical values

    For normal customers:
        - Spread over their account lifetime
        - Random merchant distribution
        - Varied amounts
    """
    print("  Generating transactions...")
    primary_merchant = merchant_ids[0]   # M0001 — the abused merchant
    rows = []

    for cust in all_customers:
        cid        = cust["customer_id"]
        is_ring    = cust["_is_ring"]
        created_at = datetime.fromisoformat(cust["account_created_at"])
        devices    = cust["_primary_devices"]
        ips        = cust["_primary_ips"]
        refund_r   = cust["_refund_rate"]

        if is_ring:
            n_txn          = int(np.random.normal(cfg["avg_txn_per_ring_member"], 8))
            n_txn          = max(5, n_txn)
            ring_anchor    = cust["_ring_anchor"]
            # Transactions happen in a burst: within 0–45 days after account creation
            txn_window_end = min(ring_anchor + timedelta(days=45), END_DATE)
            # 70% of ring transactions go to the primary (abused) merchant
            merchant_pool  = ([primary_merchant] * 7 +
                              random.sample(merchant_ids[1:], k=min(3, len(merchant_ids)-1)))
            mean_amount    = random.uniform(800, 4000)   # promotion-like amounts
        else:
            n_txn          = int(np.random.normal(cfg["avg_txn_per_normal_customer"], 12))
            n_txn          = max(1, n_txn)
            txn_window_end = END_DATE
            merchant_pool  = merchant_ids
            mean_amount    = random.uniform(300, 8000)

        for _ in range(n_txn):
            if is_ring:
                ts = rand_datetime(created_at, txn_window_end)
            else:
                ts = rand_datetime(
                    created_at + timedelta(days=random.randint(0, 30)),
                    txn_window_end
                )
            if ts > END_DATE:
                ts = END_DATE - timedelta(hours=random.randint(1, 24))

            # Occasionally simulate failed payment attempts
            attempt_count = np.random.choice([1, 2, 3], p=[0.82, 0.13, 0.05])
            status        = np.random.choice(
                ["SUCCESS", "FAILED", "REVERSED"],
                p=[0.88, 0.08, 0.04] if not is_ring else [0.92, 0.04, 0.04]
            )

            rows.append({
                "transaction_id":  next_id("T"),
                "customer_id":     cid,
                "merchant_id":     random.choice(merchant_pool),
                "timestamp":       ts.isoformat(),
                "amount":          lognormal_amount(mean_amount),
                "payment_method":  np.random.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS),
                "device_id":       random.choice(devices),
                "ip_id":           random.choice(ips),
                "status":          status,
                "attempt_count":   attempt_count,
                "_is_ring":        is_ring,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"    → {len(df):,} transactions generated")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 11. GENERATE ORDERS (from successful transactions)
# ─────────────────────────────────────────────────────────────────────────────

def generate_orders(txn_df: pd.DataFrame) -> pd.DataFrame:
    print("  Generating orders from successful transactions...")
    # ~75% of SUCCESS transactions result in an order
    success_txns = txn_df[txn_df["status"] == "SUCCESS"].copy()
    order_mask   = np.random.random(len(success_txns)) < 0.75
    order_txns   = success_txns[order_mask].copy()

    rows = []
    for _, txn in order_txns.iterrows():
        txn_time     = datetime.fromisoformat(txn["timestamp"])
        # Delivery 2–15 days after order
        delivery_days = random.randint(2, 15)
        rows.append({
            "order_id":         next_id("O"),
            "customer_id":      txn["customer_id"],
            "transaction_id":   txn["transaction_id"],
            "merchant_id":      txn["merchant_id"],
            "product_category": random.choice(PRODUCT_CATEGORIES),
            "order_time":       txn["timestamp"],
            "delivery_time":    (txn_time + timedelta(days=delivery_days)).isoformat(),
            "order_amount":     txn["amount"],
            "quantity":         np.random.choice([1, 2, 3, 4], p=[0.65, 0.22, 0.09, 0.04]),
            "_is_ring":         txn["_is_ring"],
        })

    df = pd.DataFrame(rows)
    print(f"    → {len(df):,} orders generated")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 12. GENERATE RETURNS
# ─────────────────────────────────────────────────────────────────────────────

def generate_returns(
    order_df: pd.DataFrame,
    customer_refund_rates: dict,
) -> pd.DataFrame:
    print("  Generating returns...")
    rows = []

    for _, order in order_df.iterrows():
        cid       = order["customer_id"]
        is_ring   = order["_is_ring"]
        ref_rate  = customer_refund_rates.get(cid, 0.10)

        if random.random() > ref_rate:
            continue   # no return for this order

        order_time  = datetime.fromisoformat(order["order_time"])
        # Ring members return within 1–5 days; normal within 3–30 days
        if is_ring:
            days_to_return = random.randint(1, 5)
        else:
            days_to_return = random.randint(3, 30)

        return_time = order_time + timedelta(days=days_to_return)
        if return_time > END_DATE:
            return_time = END_DATE - timedelta(hours=1)

        reason = np.random.choice(
            RETURN_REASONS,
            p=RING_RETURN_REASON_WEIGHTS if is_ring else NORMAL_RETURN_REASON_WEIGHTS,
        )
        # Refund amount: 70–100% of order value (some items partially restocked)
        refund_pct = random.uniform(0.70, 1.00) if is_ring else random.uniform(0.85, 1.00)

        rows.append({
            "return_id":     next_id("R", 5),
            "order_id":      order["order_id"],
            "customer_id":   cid,
            "merchant_id":   order["merchant_id"],
            "return_time":   return_time.isoformat(),
            "return_reason": reason,
            "refund_amount": round(order["order_amount"] * refund_pct, 2),
            "days_to_return": days_to_return,
        })

    df = pd.DataFrame(rows)
    print(f"    → {len(df):,} returns generated")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 13. BUILD FINAL CUSTOMER / DEVICE / IP DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────────

def finalise_customers(all_customers: list) -> pd.DataFrame:
    rows = []
    for c in all_customers:
        rows.append({
            "customer_id":       c["customer_id"],
            "account_created_at": c["account_created_at"],
            "location_city":     c["location_city"],
            "location_state":    c["location_state"],
            "email_domain":      c["email_domain"],
            "phone_hash":        c["phone_hash"],
            "is_verified":       c["is_verified"],
        })
    return pd.DataFrame(rows)

def build_abuse_labels(all_customers: list, ring_member_map: dict) -> pd.DataFrame:
    rows = []
    for c in all_customers:
        cid     = c["customer_id"]
        is_ring = cid in ring_member_map
        rows.append({
            "customer_id": cid,
            "cluster_id":  ring_member_map.get(cid, None),
            "is_abuse":    int(is_ring),
            "ring_type":   c.get("_ring_type"),
        })
    return pd.DataFrame(rows)

def enrich_devices(device_df: pd.DataFrame, txn_df: pd.DataFrame) -> pd.DataFrame:
    """Add total_accounts per device from transaction data."""
    accts_per_dev = (
        txn_df.groupby("device_id")["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "total_accounts"})
    )
    return device_df.merge(accts_per_dev, on="device_id", how="left").fillna({"total_accounts": 0})

def enrich_ips(ip_df: pd.DataFrame, txn_df: pd.DataFrame) -> pd.DataFrame:
    """Add total_accounts per IP from transaction data."""
    accts_per_ip = (
        txn_df.groupby("ip_id")["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "total_accounts"})
    )
    return ip_df.merge(accts_per_ip, on="ip_id", how="left").fillna({"total_accounts": 0})

# ─────────────────────────────────────────────────────────────────────────────
# 14. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    output_dir: str,
    customers_df: pd.DataFrame,
    txn_df: pd.DataFrame,
    order_df: pd.DataFrame,
    return_df: pd.DataFrame,
    device_df: pd.DataFrame,
    ip_df: pd.DataFrame,
    merchant_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ring_meta: list,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Drop internal helper columns before saving
    internal_cols = [c for c in txn_df.columns   if c.startswith("_")]
    order_cols    = [c for c in order_df.columns  if c.startswith("_")]

    files = {
        "customers.csv":     customers_df,
        "transactions.csv":  txn_df.drop(columns=internal_cols),
        "orders.csv":        order_df.drop(columns=order_cols),
        "returns.csv":       return_df,
        "devices.csv":       device_df,
        "ips.csv":           ip_df,
        "merchants.csv":     merchant_df,
        "abuse_labels.csv":  labels_df,
    }

    for fname, df in files.items():
        path = os.path.join(output_dir, fname)
        df.to_csv(path, index=False)
        print(f"  ✓  {fname:<22} {len(df):>10,} rows  →  {path}")

    # Generation summary (for validate_data.py)
    abuse_count  = int(labels_df["is_abuse"].sum())
    summary = {
        "seed":                SEED,
        "generated_at":        datetime.now().isoformat(),
        "date_range":          {"start": START_DATE.isoformat(), "end": END_DATE.isoformat()},
        "counts": {
            "customers":       len(customers_df),
            "transactions":    len(txn_df),
            "orders":          len(order_df),
            "returns":         len(return_df),
            "devices":         len(device_df),
            "ips":             len(ip_df),
            "merchants":       len(merchant_df),
            "abuse_labels":    len(labels_df),
        },
        "abuse_stats": {
            "ring_count":      len(ring_meta),
            "ring_member_count": abuse_count,
            "abuse_pct":       round(abuse_count / len(customers_df) * 100, 2),
            "ring_meta":       ring_meta,
        },
        "split_dist": {
            "train_end": TRAIN_END.isoformat(),
            "val_end":   VAL_END.isoformat(),
            "test_end":  END_DATE.isoformat(),
        }
    }

    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved → {summary_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AbuseRing Sentinel — Dataset Generator")
    parser.add_argument("--scale",  choices=["full", "sample"], default="full",
                        help="Dataset scale (default: full)")
    parser.add_argument("--output", default="data/raw",
                        help="Output directory (default: data/raw)")
    args = parser.parse_args()

    cfg = SCALE_CONFIG[args.scale]
    print(f"\n🔧  Scale       : {args.scale.upper()}")
    print(f"📁  Output dir  : {args.output}")
    print(f"🎲  Random seed : {SEED}")
    print(f"📅  Date range  : {START_DATE.date()} → {END_DATE.date()}\n")

    # ── Step 1: Base entities ─────────────────────────────────────────────
    print("── [1/7] Generating base entities ─────────────────────────────")
    device_df   = generate_devices(cfg["n_devices"])
    ip_df       = generate_ips(cfg["n_ips"])
    merchant_df = generate_merchants(cfg["n_merchants"])

    device_ids   = device_df["device_id"].tolist()
    ip_ids       = ip_df["ip_id"].tolist()
    merchant_ids = merchant_df["merchant_id"].tolist()

    # ── Step 2: Abuse rings ───────────────────────────────────────────────
    print("\n── [2/7] Injecting abuse rings ─────────────────────────────────")
    ring_customers, ring_meta, ring_device_map, ring_ip_map, ring_member_map = (
        generate_abuse_rings(cfg, device_ids, ip_ids)
    )
    n_ring_members = len(ring_customers)

    # ── Step 3: Legitimate customers ──────────────────────────────────────
    print("\n── [3/7] Generating legitimate customers ───────────────────────")
    n_normal     = cfg["n_customers"] - n_ring_members
    normal_customers, normal_device_map, normal_ip_map = (
        generate_normal_customers(n_normal, device_ids, ip_ids)
    )
    inject_family_sharing(normal_customers, device_ids, cfg["n_families"])

    all_customers = ring_customers + normal_customers
    random.shuffle(all_customers)

    # ── Step 4: Transactions ──────────────────────────────────────────────
    print("\n── [4/7] Generating transactions ───────────────────────────────")
    txn_df = generate_transactions(all_customers, merchant_ids, cfg)

    # ── Step 5: Orders ────────────────────────────────────────────────────
    print("\n── [5/7] Generating orders ─────────────────────────────────────")
    order_df = generate_orders(txn_df)

    # ── Step 6: Returns ───────────────────────────────────────────────────
    print("\n── [6/7] Generating returns ────────────────────────────────────")
    refund_rate_map = {c["customer_id"]: c["_refund_rate"] for c in all_customers}
    return_df       = generate_returns(order_df, refund_rate_map)

    # ── Step 7: Finalise and save ─────────────────────────────────────────
    print("\n── [7/7] Finalising and saving ─────────────────────────────────")
    customers_df = finalise_customers(all_customers)
    device_df    = enrich_devices(device_df, txn_df)
    ip_df        = enrich_ips(ip_df, txn_df)
    labels_df    = build_abuse_labels(all_customers, ring_member_map)

    save_outputs(
        args.output,
        customers_df, txn_df, order_df, return_df,
        device_df, ip_df, merchant_df, labels_df,
        ring_meta,
    )

    # ── Summary stats ─────────────────────────────────────────────────────
    ring_count    = int(labels_df["is_abuse"].sum())
    ring_pct      = ring_count / len(customers_df) * 100
    ring_txns     = int(txn_df["_is_ring"].sum())
    normal_refund = return_df.shape[0] / max(order_df.shape[0], 1) * 100

    print(f"""
╔══════════════════════════════════════════════════════╗
║           DATASET GENERATION COMPLETE                ║
╠══════════════════════════════════════════════════════╣
║  Customers         {len(customers_df):>10,}                     ║
║    → Ring members  {ring_count:>10,}  ({ring_pct:.1f}%)               ║
║    → Legitimate    {len(customers_df)-ring_count:>10,}                     ║
║  Transactions      {len(txn_df):>10,}                     ║
║    → Ring txns     {ring_txns:>10,}                     ║
║  Orders            {len(order_df):>10,}                     ║
║  Returns           {len(return_df):>10,}  ({normal_refund:.1f}% return rate)  ║
║  Devices           {len(device_df):>10,}                     ║
║  IPs               {len(ip_df):>10,}                     ║
║  Abuse rings       {len(ring_meta):>10,}                     ║
╠══════════════════════════════════════════════════════╣
║  Temporal split:                                     ║
║    Train  Jan–Sep  {TRAIN_END.strftime('%Y-%m-%d')}                   ║
║    Val    Oct      {VAL_END.strftime('%Y-%m-%d')}                   ║
║    Test   Nov      {END_DATE.strftime('%Y-%m-%d')}                   ║
╚══════════════════════════════════════════════════════╝

  Next step: python scripts/validate_data.py --input {args.output}
""")

if __name__ == "__main__":
    main()
