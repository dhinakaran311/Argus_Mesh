"""
backend/app/api/model.py
=========================
AbuseRing Sentinel — Model Metrics & Threshold Analysis Endpoints
"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["Model"])


@router.get("/model/metrics")
async def get_model_metrics(request: Request) -> dict:
    """Returns full evaluation results from ml/evaluation/results.json."""
    return request.app.state.ml.get_model_metrics()


@router.get("/model/thresholds")
async def get_threshold_sweep(request: Request) -> list:
    """Returns the threshold cost-sweep array for the cost analysis chart."""
    return request.app.state.ml.get_threshold_sweep()


@router.get("/model/features")
async def get_feature_importance(request: Request) -> dict:
    """Returns SHAP feature importances (global) for the importance chart."""
    raw = request.app.state.ml.get_shap_importance()
    # Sort by absolute importance
    sorted_features = sorted(
        [{"feature": k, "importance": v} for k, v in raw.items()],
        key=lambda x: abs(x["importance"]),
        reverse=True,
    )
    return {
        "features": sorted_features,
        "model_meta": request.app.state.ml.get_meta(),
    }
