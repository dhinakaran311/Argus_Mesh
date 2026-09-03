"""
backend/app/models/investigation.py
=====================================
AbuseRing Sentinel — Investigation Request/Response Pydantic Schemas
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class InvestigationRequest(BaseModel):
    cluster_id: str


class InvestigationStep(BaseModel):
    step: str   # starting | facts | graph | ml | rag | reasoning | complete | error
    message: str
    data: Optional[Any] = None


class SimilarCase(BaseModel):
    case_id: str
    ring_type: str
    similarity: float
    summary: str
    outcome: str
    risk_score: float


class InvestigationReport(BaseModel):
    cluster_id: str
    risk_level: str
    final_risk_score: float
    summary: str
    key_evidence: list[str]
    similar_cases: list[SimilarCase] = []
    recommended_action: str
    confidence: str   # HIGH | MEDIUM | LOW
    generated_at: datetime = datetime.utcnow()


class RAGRequest(BaseModel):
    query: str
    top_k: int = 3
