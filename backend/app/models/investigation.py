"""
backend/app/models/investigation.py
=====================================
AbuseRing Sentinel — Investigation Request/Response Pydantic Schemas
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):
    cluster_id: str = Field(..., min_length=1, max_length=64)  # #15 input validation


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
    """
    #10: Used as the validated structure for the 'complete' SSE event data.
    #11: generated_at uses default_factory to evaluate per-instance, not at import time.
    """
    cluster_id: str
    risk_level: str
    final_risk_score: float
    summary: str
    key_evidence: list[str]
    similar_cases: list[SimilarCase] = []
    recommended_action: str
    confidence: str   # HIGH | MEDIUM | LOW
    generated_at: datetime = Field(default_factory=datetime.utcnow)  # #11 frozen-default fix


class RAGRequest(BaseModel):
    query: str
    top_k: int = Field(3, ge=1, le=20)  # #15 bound top_k
