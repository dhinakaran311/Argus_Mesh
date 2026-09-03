"""
backend/app/models/risk.py
===========================
AbuseRing Sentinel — Risk Score Pydantic Schemas
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RiskScore(BaseModel):
    customer_id: str
    ml_score: float = Field(ge=0.0, le=1.0)
    graph_score: float = Field(ge=0.0, le=1.0)
    behaviour_score: float = Field(ge=0.0, le=1.0)
    velocity_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    risk_level: str


class DashboardStats(BaseModel):
    total_customers: int
    abuse_customers: int
    abuse_rate_pct: float
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    total_rings: int
    avg_risk_score: float
    total_devices: int
    high_share_devices: int
    total_ips: int = 0


class ModelMetrics(BaseModel):
    model_config = {"protected_namespaces": ()}

    evaluation_date: str
    model_version: str
    n_test_samples: int
    n_abuse_true: int
    abuse_rate_test: float
    primary_metrics: dict
    threshold_sweep: list[dict]
    baseline_comparison: dict
    limitations: list[str]
    score_distribution: dict
    risk_level_distribution: dict
