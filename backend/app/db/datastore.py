"""
backend/app/db/datastore.py
============================
AbuseRing Sentinel — Data Access Layer (In-Memory CSV Store)

Loads all raw CSVs into pandas DataFrames at startup.
Customer + Transaction indexes are built immediately (fast).
Orders + Returns are indexed lazily on first access.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


class DataStore:
    """In-memory data access layer backed by raw CSV files."""

    def __init__(self, raw_dir: Path):
        self._raw_dir = raw_dir
        # DataFrames (always in memory)
        self.customers: pd.DataFrame = pd.DataFrame()
        self.transactions: pd.DataFrame = pd.DataFrame()
        self.orders: pd.DataFrame = pd.DataFrame()
        self.returns: pd.DataFrame = pd.DataFrame()
        self.devices: pd.DataFrame = pd.DataFrame()
        self.ips: pd.DataFrame = pd.DataFrame()
        self.labels: pd.DataFrame = pd.DataFrame()
        self.merchants: pd.DataFrame = pd.DataFrame()
        # Indexes (dict for O(1) lookup)
        self._customer_idx: dict[str, dict] = {}
        self._txn_by_customer: dict[str, list[dict]] = {}
        self._orders_by_customer: dict[str, list[dict]] = {}
        self._returns_by_customer: dict[str, list[dict]] = {}
        self._orders_indexed = False
        self._returns_indexed = False

    # -----------------------------------------------------------------------
    # Load & Index
    # -----------------------------------------------------------------------

    def load(self) -> None:
        """Load all CSVs then build fast startup indexes."""
        log.info(f"Loading raw CSV data from: {self._raw_dir}")

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

        for attr, fname in files.items():
            path = self._raw_dir / fname
            if path.exists():
                df = pd.read_csv(path)
                setattr(self, attr, df)
                log.info(f"  {fname:<25} {len(df):>10,} rows")
            else:
                log.warning(f"  Missing: {path} — skipping")

        # #14: Sort transactions by timestamp DESC once at load time so
        # get_recent_transactions() returns the actual most-recent rows.
        if not self.transactions.empty and "timestamp" in self.transactions.columns:
            self.transactions = (
                self.transactions
                .sort_values("timestamp", ascending=False)
                .reset_index(drop=True)
            )

        self._build_startup_indexes()
        log.info("  ✅ DataStore ready")

    def _fast_group(self, df: pd.DataFrame) -> dict[str, list[dict]]:
        """Group a DataFrame by customer_id in a single linear pass."""
        if df.empty:
            return {}
        records = df.where(df.notna(), other=None).to_dict("records")
        res: dict[str, list[dict]] = {}
        for r in records:
            cid = str(r.get("customer_id", ""))
            res.setdefault(cid, []).append(r)
        return res

    def _build_startup_indexes(self) -> None:
        """Build customer + transaction indexes at startup (both fast)."""
        log.info("  Building indexes...")

        # Merge abuse labels into customers
        if not self.labels.empty and not self.customers.empty:
            merged = self.customers.merge(
                self.labels[["customer_id", "is_abuse", "cluster_id", "ring_type"]],
                on="customer_id",
                how="left",
            )
        else:
            merged = self.customers.copy()
            for col in ["is_abuse", "cluster_id", "ring_type"]:
                if col not in merged.columns:
                    merged[col] = None

        merged = merged.where(merged.notna(), other=None)

        # Customer dict (10K rows — sub-second)
        self._customer_idx = {
            str(row["customer_id"]): row
            for row in merged.to_dict("records")
        }
        log.info(f"  Indexed {len(self._customer_idx):,} customers")

        # Transaction dict (200K rows — ~2s)
        if not self.transactions.empty:
            self._txn_by_customer = self._fast_group(self.transactions)
            log.info(f"  Indexed {len(self._txn_by_customer):,} transaction groups")

        log.info("  Orders/returns will be indexed on first access (lazy)")

    def _ensure_orders(self) -> None:
        if not self._orders_indexed and not self.orders.empty:
            log.info("  Lazily indexing orders...")
            self._orders_by_customer = self._fast_group(self.orders)
            self._orders_indexed = True

    def _ensure_returns(self) -> None:
        if not self._returns_indexed and not self.returns.empty:
            log.info("  Lazily indexing returns...")
            self._returns_by_customer = self._fast_group(self.returns)
            self._returns_indexed = True

    # -----------------------------------------------------------------------
    # Customer queries
    # -----------------------------------------------------------------------

    def get_customer(self, customer_id: str) -> Optional[dict]:
        return self._customer_idx.get(str(customer_id))

    def get_customers_by_ids(self, customer_ids: list[str]) -> list[dict]:
        return [self._customer_idx[cid] for cid in customer_ids if cid in self._customer_idx]

    def get_all_customers(self, limit: int = 1000, offset: int = 0) -> list[dict]:
        items = list(self._customer_idx.values())
        return items[offset: offset + limit]

    def get_customers_in_cluster(self, cluster_id: str) -> list[dict]:
        return [c for c in self._customer_idx.values() if c.get("cluster_id") == cluster_id]

    # -----------------------------------------------------------------------
    # Transaction queries
    # -----------------------------------------------------------------------

    def get_transactions_for_customer(self, customer_id: str, limit: int = 50) -> list[dict]:
        return self._txn_by_customer.get(str(customer_id), [])[:limit]

    def get_recent_transactions(self, limit: int = 100) -> list[dict]:
        if self.transactions.empty:
            return []
        # #14: transactions are sorted DESC at load, so head() returns most-recent
        df = self.transactions.head(limit).where(self.transactions.notna(), other=None)
        return df.to_dict("records")

    # -----------------------------------------------------------------------
    # Orders / Returns (lazy)
    # -----------------------------------------------------------------------

    def get_orders_for_customer(self, customer_id: str) -> list[dict]:
        self._ensure_orders()
        return self._orders_by_customer.get(str(customer_id), [])

    def get_returns_for_customer(self, customer_id: str) -> list[dict]:
        self._ensure_returns()
        return self._returns_by_customer.get(str(customer_id), [])

    # -----------------------------------------------------------------------
    # Stats helpers
    # -----------------------------------------------------------------------

    def summary_stats(self) -> dict:
        return {
            "customers":    len(self.customers),
            "transactions": len(self.transactions),
            "orders":       len(self.orders),
            "returns":      len(self.returns),
            "devices":      len(self.devices),
            "ips":          len(self.ips),
        }

    def health_check(self) -> bool:
        return len(self._customer_idx) > 0
