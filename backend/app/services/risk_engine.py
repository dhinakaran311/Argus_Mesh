"""
backend/app/services/risk_engine.py
=====================================
AbuseRing Sentinel — Multi-Modal Risk Engine

Combines four risk signals into a final cluster risk score:
  40% XGBoost ML score
  30% Graph score (from Neo4j topology analysis)
  20% Behaviour score (return rate signal)
  10% Velocity score (transaction velocity signal)

Mirrors the Cypher formula in graph/queries/abuse_ring_detection.cypher.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
_W_ML         = 0.40
_W_GRAPH      = 0.30
_W_BEHAVIOUR  = 0.20
_W_VELOCITY   = 0.10

# Merchant baseline refund rate (from merchants.csv / domain knowledge)
_BASELINE_REFUND_RATE = 0.08
_MAX_VELOCITY_RATIO   = 8.0  # ring velocity can be up to 8× baseline


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


class RiskEngine:
    """Compute multi-modal risk scores for customers and clusters."""

    # -----------------------------------------------------------------------
    # Score components
    # -----------------------------------------------------------------------

    @staticmethod
    def behaviour_score(return_rate: float) -> float:
        """
        Behaviour signal based on return rate relative to merchant baseline.
        Normal: ~9%, Abuse: ~85-95%
        """
        excess = max(0.0, return_rate - _BASELINE_REFUND_RATE)
        # Normalise: excess of 0.9 maps to 1.0
        return _clamp(excess / 0.9)

    @staticmethod
    def velocity_score(txn_velocity: float, baseline_velocity: float = 1.0) -> float:
        """
        Velocity signal: how many times faster than baseline.
        """
        ratio = txn_velocity / max(baseline_velocity, 0.01)
        return _clamp(ratio / _MAX_VELOCITY_RATIO)

    @staticmethod
    def graph_score(
        cluster_size: int,
        cluster_return_rate: float,
        burst_hours: Optional[float] = None,
    ) -> float:
        """
        Graph topology score combining cluster size + return rate + burst signal.
        Mirrors the Cypher formula in abuse_ring_detection.cypher.
        """
        size_signal   = _clamp(cluster_size / 30.0)
        return_signal = _clamp(cluster_return_rate)
        if burst_hours is None:
            burst_signal = 0.5  # unknown — use neutral
        elif burst_hours <= 24:
            burst_signal = 1.0
        elif burst_hours <= 72:
            burst_signal = 0.7
        elif burst_hours <= 168:
            burst_signal = 0.4
        else:
            burst_signal = 0.1
        return _clamp(0.35 * size_signal + 0.40 * return_signal + 0.25 * burst_signal)

    @staticmethod
    def compute_final_risk(
        ml_score: float,
        graph_score: float,
        behaviour_score: float,
        velocity_score: float,
    ) -> float:
        """Weighted combination of the four sub-scores."""
        return _clamp(
            _W_ML        * ml_score        +
            _W_GRAPH     * graph_score     +
            _W_BEHAVIOUR * behaviour_score +
            _W_VELOCITY  * velocity_score
        )

    @staticmethod
    def get_risk_level(score: float) -> str:
        if score >= 0.80:
            return "CRITICAL"
        if score >= 0.60:
            return "HIGH"
        if score >= 0.30:
            return "MEDIUM"
        return "LOW"

    # -----------------------------------------------------------------------
    # Cluster-level scoring (from Neo4j query result dict)
    # -----------------------------------------------------------------------

    def score_cluster(self, cluster_row: dict) -> dict:
        """
        Take a raw Neo4j cluster result dict and return enriched risk dict.
        cluster_row keys: cluster_id, cluster_size, cluster_return_rate,
                          avg_ml_score, total_transactions, total_returns
        """
        ml    = float(cluster_row.get("avg_ml_score", 0.0))
        cr    = float(cluster_row.get("cluster_return_rate", 0.0))
        size  = int(cluster_row.get("cluster_size", 1))
        txns  = int(cluster_row.get("total_transactions", 0))

        g_score  = self.graph_score(size, cr)
        b_score  = self.behaviour_score(cr)
        vel      = txns / max(size, 1) / 20.0   # crude velocity estimate
        v_score  = self.velocity_score(vel)
        final    = self.compute_final_risk(ml, g_score, b_score, v_score)

        return {
            **cluster_row,
            "graph_score":      round(g_score, 4),
            "behaviour_score":  round(b_score, 4),
            "velocity_score":   round(v_score, 4),
            "combined_risk_score": round(final, 4),
            "risk_level":       self.get_risk_level(final),
        }

    def score_clusters(self, cluster_rows: list[dict]) -> list[dict]:
        scored = [self.score_cluster(r) for r in cluster_rows]
        scored.sort(key=lambda r: r["combined_risk_score"], reverse=True)
        return scored
