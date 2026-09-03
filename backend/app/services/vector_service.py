"""
backend/app/services/vector_service.py
=========================================
AbuseRing Sentinel — Qdrant Vector Service

Manages investigation case memory:
  - Seed with 6 synthetic historical cases on first startup
  - Store new investigation summaries after each Groq analysis
  - Search for similar cases during investigation
"""
from __future__ import annotations

import logging
from typing import Optional

from ..db.qdrant import QdrantDB
from .embed_service import EmbedService

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed cases: synthetic historical investigations derived from known rings
# ---------------------------------------------------------------------------
_SEED_CASES = [
    {
        "case_id":    "CASE-001",
        "ring_type":  "REFUND_RING",
        "summary":    (
            "22 accounts sharing 2 devices created within 3 hours. "
            "Cluster return rate 91%, targeting electronics category. "
            "Average account age 2 days. All transactions targeted one merchant."
        ),
        "outcome":    "BLOCKED — all 22 accounts suspended, ₹1.1L exposure prevented",
        "risk_score": 0.96,
        "recommended_action": "Block all accounts, flag device fingerprints",
    },
    {
        "case_id":    "CASE-002",
        "ring_type":  "PROMO_ABUSE",
        "summary":    (
            "15 accounts sharing 3 IPs, all created within 6 hours of a promotion launch. "
            "Zero return rate but 100% promo code redemption. "
            "Merchant concentration: 98% transactions to one merchant."
        ),
        "outcome":    "BLOCKED — promo flagged, 15 accounts deactivated",
        "risk_score": 0.89,
        "recommended_action": "Revoke promo credits, block accounts, alert merchant",
    },
    {
        "case_id":    "CASE-003",
        "ring_type":  "RETURN_FRAUD",
        "summary":    (
            "8 accounts with 100% DEFECTIVE return reason claims. "
            "All accounts share 1 device, created over 48 hours. "
            "Average transaction ₹8,200. All returns within 1 day of delivery."
        ),
        "outcome":    "ESCALATED — forensic review, ₹65,600 recovered",
        "risk_score": 0.91,
        "recommended_action": "Require photo evidence for returns, escalate to fraud team",
    },
    {
        "case_id":    "CASE-004",
        "ring_type":  "REFUND_RING",
        "summary":    (
            "30 accounts (largest ring detected), 2 shared devices, 1 shared IP. "
            "All created within 2 hours on a weekend. "
            "Cluster return rate 88%, total exposure ₹4.5L."
        ),
        "outcome":    "CRITICAL — immediate block, law enforcement notification",
        "risk_score": 0.99,
        "recommended_action": "Immediate block, freeze refunds, notify law enforcement",
    },
    {
        "case_id":    "CASE-005",
        "ring_type":  "PROMO_ABUSE",
        "summary":    (
            "19 accounts, 3 IPs, extremely high merchant concentration (99.7%). "
            "Transaction velocity 6× merchant baseline. "
            "All accounts unverified, phone numbers sequential."
        ),
        "outcome":    "BLOCKED — promotional system hardened",
        "risk_score": 0.87,
        "recommended_action": "Block accounts, implement velocity limits on promotions",
    },
    {
        "case_id":    "CASE-006",
        "ring_type":  "RETURN_FRAUD",
        "summary":    (
            "12 accounts, 95% changed_mind return rate. "
            "All orders in fashion category, returned within 12 hours. "
            "Suspected try-and-return scheme."
        ),
        "outcome":    "BLOCKED — return policy tightened for this device cluster",
        "risk_score": 0.84,
        "recommended_action": "Restrict returns to 48h with photo evidence required",
    },
]


class VectorService:
    """Manages Qdrant case memory for semantic investigation search."""

    def __init__(self, qdrant: QdrantDB, embedder: EmbedService):
        self._qdrant = qdrant
        self._embedder = embedder

    def setup(self) -> None:
        """Ensure collection exists and seed if empty."""
        self._qdrant.ensure_collection()
        count = self._qdrant.count()
        log.info(f"  Qdrant collection '{self._qdrant._collection}': {count} vectors")
        if count == 0:
            log.info("  Seeding Qdrant with historical cases...")
            self._seed_cases()

    def _seed_cases(self) -> None:
        for case in _SEED_CASES:
            vec = self._embedder.embed(case["summary"])
            self._qdrant.upsert(
                point_id=case["case_id"],
                vector=vec,
                payload=case,
            )
        log.info(f"  ✅ Seeded {len(_SEED_CASES)} historical cases into Qdrant")

    def store_investigation(
        self,
        case_id: str,
        summary: str,
        metadata: dict,
    ) -> bool:
        """Embed and store a new investigation summary."""
        try:
            vec = self._embedder.embed(summary)
            payload = {"case_id": case_id, "summary": summary, **metadata}
            self._qdrant.upsert(point_id=case_id, vector=vec, payload=payload)
            log.info(f"  Stored investigation: {case_id}")
            return True
        except Exception as e:
            log.error(f"  Failed to store investigation {case_id}: {e}")
            return False

    def find_similar_cases(self, query_text: str, top_k: int = 3) -> list[dict]:
        """Embed query and search for most similar historical cases."""
        vec = self._embedder.embed(query_text)
        results = self._qdrant.search(vec, top_k=top_k, score_threshold=0.0)
        return results

    def get_all_cases(self) -> list[dict]:
        return self._qdrant.get_all(limit=100)
