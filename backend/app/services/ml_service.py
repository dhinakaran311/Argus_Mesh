"""
backend/app/services/ml_service.py
====================================
AbuseRing Sentinel — XGBoost ML Service

Loads the trained XGBoost model + SHAP importance from disk.
Provides per-customer risk scoring and feature explanation.

Data strategy:
  - features.parquet is loaded once at startup into a customer_id → row dict
  - Prediction = dict lookup + model.predict_proba()  (microseconds)
  - SHAP = top-5 feature contributions per prediction
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class MLService:
    """XGBoost inference + SHAP explanation service."""

    def __init__(self, models_dir: Path, processed_dir: Path, evaluation_dir: Path):
        self._models_dir = models_dir
        self._processed_dir = processed_dir
        self._evaluation_dir = evaluation_dir

        self._model = None
        self._feature_cols: list[str] = []
        self._threshold: float = 0.1
        self._meta: dict = {}
        self._shap_importance: dict = {}
        self._eval_results: dict = {}

        # customer_id → feature row (numpy array)
        self._feature_index: dict[str, np.ndarray] = {}
        self._feature_df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Load model, metadata, SHAP importance, and feature index."""
        log.info("Loading ML artifacts...")

        # --- Model -----------------------------------------------------------
        model_file = self._find_model_file("xgboost_*.pkl")
        if model_file:
            with open(model_file, "rb") as f:
                self._model = pickle.load(f)
            log.info(f"  ✅ XGBoost model loaded: {model_file.name}")
        else:
            log.error("  ❌ No XGBoost model file found")

        # --- Metadata --------------------------------------------------------
        meta_path = self._models_dir / "model_meta.json"
        if meta_path.exists():
            self._meta = json.loads(meta_path.read_text())
            self._feature_cols = self._meta.get("feature_cols", [])
            self._threshold = self._meta.get("optimal_threshold", 0.1)
            log.info(f"  ✅ Model metadata: {len(self._feature_cols)} features, threshold={self._threshold}")

        # --- SHAP importance -------------------------------------------------
        shap_path = self._models_dir / "shap_importance.json"
        if shap_path.exists():
            self._shap_importance = json.loads(shap_path.read_text())
            log.info(f"  ✅ SHAP importance loaded: {len(self._shap_importance)} features")

        # --- Evaluation results ----------------------------------------------
        eval_path = self._evaluation_dir / "results.json"
        if eval_path.exists():
            self._eval_results = json.loads(eval_path.read_text())
            log.info("  ✅ Evaluation results loaded")

        # --- Feature index ---------------------------------------------------
        self._load_feature_index()

    def _find_model_file(self, pattern: str) -> Optional[Path]:
        files = sorted(self._models_dir.glob(pattern))
        return files[-1] if files else None

    def _load_feature_index(self) -> None:
        """Load features.parquet and build customer_id → numpy row index."""
        feat_path = self._processed_dir / "features.parquet"
        if not feat_path.exists():
            log.warning("  features.parquet not found — ML scoring will return 0.0")
            return

        df = pd.read_parquet(feat_path)
        self._feature_df = df

        if "customer_id" not in df.columns:
            log.warning("  features.parquet has no customer_id column")
            return

        available = [c for c in self._feature_cols if c in df.columns]
        if not available:
            log.warning("  No expected feature columns found in features.parquet")
            return

        # Fill NaN with 0 for inference
        feat_matrix = df[available].fillna(0).values.astype(np.float32)
        self._feature_index = {
            str(row["customer_id"]): feat_matrix[i]
            for i, (_, row) in enumerate(df.iterrows())
        }
        log.info(f"  ✅ Feature index built: {len(self._feature_index):,} customers")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def predict(self, customer_id: str) -> float:
        """Return abuse probability for a single customer."""
        if self._model is None or customer_id not in self._feature_index:
            return 0.0
        row = self._feature_index[customer_id].reshape(1, -1)
        prob = float(self._model.predict_proba(row)[0][1])
        return round(prob, 4)

    def predict_batch(self, customer_ids: list[str]) -> dict[str, float]:
        """Return {customer_id: probability} for a list of customers."""
        if self._model is None:
            return {cid: 0.0 for cid in customer_ids}
        found = [(cid, self._feature_index[cid]) for cid in customer_ids if cid in self._feature_index]
        if not found:
            return {cid: 0.0 for cid in customer_ids}
        ids, rows = zip(*found)
        matrix = np.stack(rows)
        probs = self._model.predict_proba(matrix)[:, 1]
        result = {cid: 0.0 for cid in customer_ids}
        for cid, prob in zip(ids, probs):
            result[cid] = round(float(prob), 4)
        return result

    def explain(self, customer_id: str, top_n: int = 5) -> list[dict]:
        """Return top-N SHAP feature contributions for a customer."""
        if not self._shap_importance or customer_id not in self._feature_index:
            return []
        row = self._feature_index[customer_id]
        available = [c for c in self._feature_cols if c in self._shap_importance]
        explanations = []
        for col in available:
            feat_idx = self._feature_cols.index(col) if col in self._feature_cols else -1
            feat_val = float(row[feat_idx]) if feat_idx >= 0 else 0.0
            importance = float(self._shap_importance.get(col, 0.0))
            explanations.append({
                "feature":    col,
                "value":      feat_val,
                "importance": importance,
                "direction":  "increases_risk" if importance > 0 else "decreases_risk",
            })
        explanations.sort(key=lambda x: abs(x["importance"]), reverse=True)
        return explanations[:top_n]

    def get_risk_level(self, score: float) -> str:
        if score >= 0.80:
            return "CRITICAL"
        if score >= 0.60:
            return "HIGH"
        if score >= 0.30:
            return "MEDIUM"
        return "LOW"

    def get_model_metrics(self) -> dict:
        return self._eval_results

    def get_threshold_sweep(self) -> list[dict]:
        return self._eval_results.get("threshold_sweep", [])

    def get_shap_importance(self) -> dict:
        return self._shap_importance

    def get_meta(self) -> dict:
        return self._meta

    def is_loaded(self) -> bool:
        return self._model is not None
