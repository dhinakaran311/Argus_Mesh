"""
ml/evaluation/evaluate.py
==========================
AbuseRing Sentinel — Held-out Test Set Evaluation

Loads the best trained model and evaluates it on the November
held-out test set. Produces results.json with all metrics needed
for the dashboard and the Razorpay submission.

Outputs:
    ml/evaluation/results.json   — full metrics, confusion matrix, cost sweep

Usage:
    python ml/evaluation/evaluate.py
"""

import os
import sys
import json
import logging
import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = os.path.join(ROOT_DIR, "ml", "models")
EVAL_DIR      = os.path.join(ROOT_DIR, "ml", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FP_COST = 100
FN_COST = 3000


# ===========================================================================
# 1. LOAD MODEL + DATA
# ===========================================================================

def load_model_and_data() -> tuple:
    # Load model metadata
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")
    if not os.path.exists(meta_path):
        log.error("model_meta.json not found — run ml/training/train.py first")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    model_file = os.path.join(MODELS_DIR, meta["model_file"])
    if not os.path.exists(model_file):
        log.error(f"Model file not found: {model_file}")
        sys.exit(1)

    model         = joblib.load(model_file)
    feature_cols  = meta["feature_cols"]
    threshold     = meta["optimal_threshold"]
    log.info(f"Loaded model: {meta['model_file']}")
    log.info(f"Threshold from training (val): {threshold}")

    # Load features
    feat_path = os.path.join(PROCESSED_DIR, "features.parquet")
    feat = pd.read_parquet(feat_path)

    test_df = feat[feat["split"] == "test"].copy()
    log.info(f"Test set: {len(test_df):,} customers | "
             f"{int(test_df['is_abuse'].sum())} abuse ({test_df['is_abuse'].mean()*100:.1f}%)")

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["is_abuse"].values.astype(int)

    return model, threshold, feature_cols, X_test, y_test, test_df, meta


# ===========================================================================
# 2. EVALUATE AT ONE THRESHOLD
# ===========================================================================

def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision  = precision_score(y_true, y_pred, zero_division=0)
    recall     = recall_score(y_true, y_pred, zero_division=0)
    f1         = f1_score(y_true, y_pred, zero_division=0)
    fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    total_cost = fp * FP_COST + fn * FN_COST
    return {
        "threshold":      round(float(threshold), 3),
        "precision":      round(float(precision), 4),
        "recall":         round(float(recall), 4),
        "f1":             round(float(f1), 4),
        "fpr":            round(float(fpr), 4),
        "tp":             int(tp),
        "tn":             int(tn),
        "fp":             int(fp),
        "fn":             int(fn),
        "total_cost_inr": int(total_cost),
        "fp_cost_inr":    int(fp * FP_COST),
        "fn_cost_inr":    int(fn * FN_COST),
    }


# ===========================================================================
# 3. THRESHOLD SWEEP (for dashboard chart)
# ===========================================================================

def full_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    thresholds = np.arange(0.05, 0.97, 0.05)
    return [evaluate_at_threshold(y_true, y_prob, t) for t in thresholds]


def find_optimal_threshold_by_cost(sweep: list[dict]) -> float:
    return min(sweep, key=lambda x: x["total_cost_inr"])["threshold"]


# ===========================================================================
# 4. MAIN EVALUATION
# ===========================================================================

def main() -> None:
    log.info("=" * 60)
    log.info("AbuseRing Sentinel - Held-out Test Set Evaluation")
    log.info("=" * 60)

    model, val_threshold, feature_cols, X_test, y_test, test_df, meta = (
        load_model_and_data()
    )

    # Predict probabilities on held-out test set
    y_prob = model.predict_proba(X_test)[:, 1]

    # AUC on test
    auc = roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else 0.0
    avg_precision = average_precision_score(y_test, y_prob) if y_test.sum() > 0 else 0.0
    log.info(f"\nTest AUC-ROC:         {auc:.4f}")
    log.info(f"Test Avg Precision:   {avg_precision:.4f}")

    # Threshold sweep on test set
    sweep = full_threshold_sweep(y_test, y_prob)

    # Evaluate at the threshold selected on VALIDATION set (no leakage)
    test_metrics = evaluate_at_threshold(y_test, y_prob, val_threshold)

    # Also find optimal threshold on test for reference (would not be used in production)
    test_optimal_t = find_optimal_threshold_by_cost(sweep)
    test_optimal_m = evaluate_at_threshold(y_test, y_prob, test_optimal_t)

    log.info(f"\n--- Results at val-selected threshold ({val_threshold}) ---")
    log.info(f"  Precision:  {test_metrics['precision']:.4f}")
    log.info(f"  Recall:     {test_metrics['recall']:.4f}")
    log.info(f"  F1:         {test_metrics['f1']:.4f}")
    log.info(f"  FPR:        {test_metrics['fpr']:.4f}")
    log.info(f"  Confusion:  TP={test_metrics['tp']}  FP={test_metrics['fp']}  "
             f"FN={test_metrics['fn']}  TN={test_metrics['tn']}")
    log.info(f"  Cost:       Rs.{test_metrics['total_cost_inr']:,} "
             f"(FP=Rs.{test_metrics['fp_cost_inr']:,} + FN=Rs.{test_metrics['fn_cost_inr']:,})")

    # Risk score distribution on test set
    score_percentiles = {
        "p25": float(np.percentile(y_prob, 25)),
        "p50": float(np.percentile(y_prob, 50)),
        "p75": float(np.percentile(y_prob, 75)),
        "p90": float(np.percentile(y_prob, 90)),
        "p95": float(np.percentile(y_prob, 95)),
    }

    # Category distribution at optimal threshold
    y_pred = (y_prob >= val_threshold).astype(int)
    risk_levels = {
        "LOW":      int(np.sum(y_prob < 0.30)),
        "MEDIUM":   int(np.sum((y_prob >= 0.30) & (y_prob < 0.60))),
        "HIGH":     int(np.sum((y_prob >= 0.60) & (y_prob < 0.80))),
        "CRITICAL": int(np.sum(y_prob >= 0.80)),
    }

    # ── Save results.json ───────────────────────────────────────────────────
    results = {
        "evaluation_date":   pd.Timestamp.now().isoformat(),
        "dataset_split":     "test (November 2024 — never seen during training or tuning)",
        "model_version":     meta["model_version"],
        "n_test_samples":    int(len(y_test)),
        "n_abuse_true":      int(y_test.sum()),
        "abuse_rate_test":   round(float(y_test.mean()), 4),

        # Primary metrics (at val-selected threshold)
        "primary_metrics": {
            **test_metrics,
            "auc_roc":        round(auc, 4),
            "avg_precision":  round(avg_precision, 4),
            "note": "Threshold selected on validation set — no test-set leakage",
        },

        # Reference: what optimal test threshold would give
        "optimal_test_threshold_reference": {
            **test_optimal_m,
            "note": "For reference only — NOT used as model threshold (would be leakage)",
        },

        # Business cost
        "business_cost": {
            "fp_cost_per_account_inr": FP_COST,
            "fn_cost_per_account_inr": FN_COST,
            "total_cost_at_threshold_inr": test_metrics["total_cost_inr"],
            "false_positive_cost_inr":     test_metrics["fp_cost_inr"],
            "false_negative_cost_inr":     test_metrics["fn_cost_inr"],
            "cost_note": (
                "FP cost = investigation overhead for flagging a legitimate account. "
                "FN cost = estimated loss from a missed abuse account (avg transaction value)."
            ),
        },

        # Confusion matrix (for dashboard)
        "confusion_matrix": {
            "labels":  ["Legitimate", "Abuse"],
            "matrix":  [
                [test_metrics["tn"], test_metrics["fp"]],
                [test_metrics["fn"], test_metrics["tp"]],
            ],
        },

        # Full threshold sweep (for dashboard chart)
        "threshold_sweep": sweep,

        # Score distribution
        "score_distribution": score_percentiles,

        # Risk level distribution
        "risk_level_distribution": risk_levels,

        # Baseline comparison (from training metadata)
        "baseline_comparison": {
            "logistic_regression_val_f1":  meta.get("baseline_val_metrics", {}).get("f1"),
            "logistic_regression_val_auc": meta.get("baseline_val_auc"),
            "xgboost_val_f1":              meta.get("val_metrics", {}).get("f1"),
            "xgboost_val_auc":             meta.get("val_auc"),
            "xgboost_test_f1":             test_metrics["f1"],
            "xgboost_test_auc":            round(auc, 4),
        },

        # Honest limitations
        "limitations": [
            "Dataset is synthetic — distributions may not match real Razorpay production data.",
            "Risk scores are model estimates, not guaranteed fraud probabilities.",
            "Shared device/IP signals are treated as evidence, not proof of abuse.",
            "Temporal split is simulated; in production, ongoing retraining would be required.",
            "Cluster weights (40% ML / 30% Graph / 20% Behaviour / 10% Velocity) are "
            "experimentally selected on validation data.",
        ],
    }

    results_path = os.path.join(EVAL_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Results saved: {results_path}")

    # Print pitch-ready summary
    m = test_metrics
    log.info(f"""
{'=' * 60}
HELD-OUT TEST SET RESULTS  (November 2024)
{'=' * 60}

  Metric           Value
  ─────────────────────────────────────
  Precision        {m['precision']*100:.1f}%
  Recall           {m['recall']*100:.1f}%
  F1-Score         {m['f1']*100:.1f}%
  AUC-ROC          {auc:.3f}
  FPR              {m['fpr']*100:.1f}%

  Confusion Matrix (Held-out Test):
               Predicted Abuse   Predicted Normal
  Actual Abuse      {m['tp']:<6}            {m['fn']:<6}
  Actual Normal     {m['fp']:<6}            {m['tn']:<6}

  Business Cost @ threshold={val_threshold}:
    FP cost:  Rs.{m['fp_cost_inr']:>8,}  ({m['fp']} legitimate accounts flagged)
    FN cost:  Rs.{m['fn_cost_inr']:>8,}  ({m['fn']} abuse accounts missed)
    Total:    Rs.{m['total_cost_inr']:>8,}

  Risk Distribution (test set, {len(y_test):,} customers):
    CRITICAL (>80%):  {risk_levels['CRITICAL']:>5,}
    HIGH     (60-80%): {risk_levels['HIGH']:>5,}
    MEDIUM   (30-60%): {risk_levels['MEDIUM']:>5,}
    LOW      (<30%):   {risk_levels['LOW']:>5,}
{'=' * 60}
""")


if __name__ == "__main__":
    main()
