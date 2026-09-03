"""
backend/app/models/customer.py
===============================
AbuseRing Sentinel — Customer Pydantic Schemas
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ShapFeature(BaseModel):
    feature: str
    value: float
    importance: float
    direction: str  # "increases_risk" | "decreases_risk"


class CustomerSummary(BaseModel):
    customer_id: str
    is_abuse: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str   # LOW | MEDIUM | HIGH | CRITICAL
    return_rate: float = Field(ge=0.0, le=1.0)
    num_transactions: int
    num_orders: int
    num_returns: int
    cluster_id: Optional[str] = None
    ring_type: Optional[str] = None
    location_city: Optional[str] = None
    email_domain: Optional[str] = None
    account_created_at: Optional[str] = None


class CustomerDetail(CustomerSummary):
    top_shap_features: list[ShapFeature] = []
    transactions: list[dict] = []
