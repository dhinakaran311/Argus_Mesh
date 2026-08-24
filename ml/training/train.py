"""
ml/training/train.py
=====================
AbuseRing Sentinel — Model Training Pipeline

Reads data/processed/features.parquet, applies a temporal split,
trains a Logistic Regression baseline and an XGBoost model tuned
with Optuna, then saves the best model artifact.

Usage:
    python ml/training/train.py
    python ml/training/train.py --trials 50 --output ml/models
"""

import os
import sys
import json
import logging
import argparse
import warnings
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import optuna
import shap

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = os.path.join(ROOT_DIR, "ml", "models")
EVAL_DIR      = os.path.join(ROOT_DIR, "ml", "evaluation")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Business cost constants (from plan)
FP_COST = 100     # ₹ — legitimate account flagged
FN_COST = 3000    # ₹ — actual abuse account missed


# ===========================================================================
# 1. LOAD & SPLIT
# ===========================================================================

def load_and_split(feature_dir: str) -> tuple[pd.DataFrame, ...]:
    log.info("Loading features.parquet...")
    path = os.path.join(feature_dir, "features.parquet")
    if not os.path.exists(path):
        log.error("features.parquet not found — run ml/features/feature_engineering.py first")
        sys.exit(1)

    feat = pd.read_parquet(path)
    log.info(f"  Loaded: {len(feat):,} rows x {feat.shape[1]} columns")

    # Load feature names
    names_path = os.path.join(feature_dir, "feature_names.json")
    with open(names_path) as f:
        meta = json.load(f)
    feature_cols = [c for c in meta["feature_cols"] if c in feat.columns]
    label_col    = meta["label_col"]

    log.info(f"  Feature columns: {len(feature_cols)}")
    log.info(f"  Label column:    {label_col}")

    train = feat[feat["split"] == "train"]
    val   = feat[feat["split"] == "val"]
    test  = feat[feat["split"] == "test"]

    log.info(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        abuse = split[label_col].sum()
        log.info(f"  {name} abuse: {abuse} ({abuse/len(split)*100:.1f}%)")

    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train[label_col].values.astype(int)
    X_val   = val[feature_cols].values.astype(np.float32)
    y_val   = val[label_col].values.astype(int)
    X_test  = test[feature_cols].values.astype(np.float32)
    y_test  = test[label_col].values.astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, train, val, test


# ===========================================================================
# 2. METRICS HELPER
# ===========================================================================

def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision  = precision_score(y_true, y_pred, zero_division=0)
    recall     = recall_score(y_true, y_pred, zero_division=0)
    f1         = f1_score(y_true, y_pred, zero_division=0)
    fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    total_cost = fp * FP_COST + fn * FN_COST
    return {
        "threshold": round(threshold, 3),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "fpr":       round(fpr, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_cost_inr": int(total_cost),
        "fp_cost_inr":    int(fp * FP_COST),
        "fn_cost_inr":    int(fn * FN_COST),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    thresholds = np.arange(0.10, 0.96, 0.05)
    return [evaluate_at_threshold(y_true, y_prob, t) for t in thresholds]


def find_optimal_threshold(sweep: list[dict]) -> float:
    """Select threshold with lowest total business cost."""
    best = min(sweep, key=lambda x: x["total_cost_inr"])
    return best["threshold"]


# ===========================================================================
# 3. BASELINE — Logistic Regression
# ===========================================================================

def train_baseline(X_train, y_train, X_val, y_val) -> tuple:
    log.info("\n[Baseline] Training Logistic Regression...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    y_prob_val = pipe.predict_proba(X_val)[:, 1]
    sweep      = threshold_sweep(y_val, y_prob_val)
    best_t     = find_optimal_threshold(sweep)
    metrics    = evaluate_at_threshold(y_val, y_prob_val, best_t)
    auc        = roc_auc_score(y_val, y_prob_val) if y_val.sum() > 0 else 0.0

    log.info(f"  Val  AUC={auc:.4f}  F1={metrics['f1']:.4f}  "
             f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
             f"@threshold={best_t}")
    return pipe, metrics, auc, best_t


# ===========================================================================
# 4. MAIN MODEL — XGBoost + Optuna
# ===========================================================================

def objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective — maximise F1 on validation set."""
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 800),
        "max_depth":          trial.suggest_int("max_depth", 3, 9),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        "tree_method":        "hist",
        "random_state":       42,
        "eval_metric":        "aucpr",
        "use_label_encoder":  False,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_prob = model.predict_proba(X_val)[:, 1]
    sweep  = threshold_sweep(y_val, y_prob)
    best_t = find_optimal_threshold(sweep)
    m      = evaluate_at_threshold(y_val, y_prob, best_t)
    return m["f1"]


def train_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 50) -> tuple:
    log.info(f"\n[XGBoost] Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize", study_name="xgb_abuse_ring")
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    log.info(f"  Best trial F1 (val): {study.best_value:.4f}")
    log.info(f"  Best params: {best_params}")

    # Retrain on train+val with best params
    log.info("  Retraining on Train+Val with best params...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    best_params.update({
        "tree_method":       "hist",
        "random_state":      42,
        "use_label_encoder": False,
    })
    best_params.pop("eval_metric", None)

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_trainval, y_trainval, verbose=False)

    y_prob_val = final_model.predict_proba(X_val)[:, 1]
    sweep      = threshold_sweep(y_val, y_prob_val)
    best_t     = find_optimal_threshold(sweep)
    metrics    = evaluate_at_threshold(y_val, y_prob_val, best_t)
    auc        = roc_auc_score(y_val, y_prob_val) if y_val.sum() > 0 else 0.0

    log.info(f"  Val  AUC={auc:.4f}  F1={metrics['f1']:.4f}  "
             f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
             f"@threshold={best_t}")
    return final_model, metrics, auc, best_t, study


# ===========================================================================
# 5. SHAP EXPLANATION
# ===========================================================================

def compute_shap(model, X_val: np.ndarray, feature_cols: list) -> pd.DataFrame:
    log.info("\nComputing SHAP values on validation set...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # shap_values shape: (n_samples, n_features) for binary XGBoost
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":          feature_cols,
        "mean_abs_shap":    mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    log.info("  Top 10 features by SHAP importance:")
    for _, row in shap_df.head(10).iterrows():
        log.info(f"    {row['feature']:<40} {row['mean_abs_shap']:.4f}")

    return shap_df


# ===========================================================================
# 6. SAVE MODEL + ARTEFACTS
# ===========================================================================

def save_model(
    model,
    baseline,
    best_threshold: float,
    feature_cols:   list,
    xgb_metrics:    dict,
    baseline_metrics: dict,
    xgb_auc:        float,
    baseline_auc:   float,
    shap_df:        pd.DataFrame,
    models_dir:     str,
) -> str:
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path  = os.path.join(models_dir, f"xgboost_{timestamp}.pkl")
    base_path   = os.path.join(models_dir, f"baseline_{timestamp}.pkl")
    meta_path   = os.path.join(models_dir, "model_meta.json")
    shap_path   = os.path.join(models_dir, "shap_importance.json")

    joblib.dump(model,    model_path)
    joblib.dump(baseline, base_path)
    log.info(f"\n  Model saved: {model_path}")

    # Save metadata
    meta = {
        "model_version":       timestamp,
        "model_type":          "XGBoostClassifier",
        "feature_cols":        feature_cols,
        "optimal_threshold":   best_threshold,
        "fp_cost_inr":         FP_COST,
        "fn_cost_inr":         FN_COST,
        "val_metrics":         xgb_metrics,
        "val_auc":             round(xgb_auc, 4),
        "baseline_val_metrics": baseline_metrics,
        "baseline_val_auc":    round(baseline_auc, 4),
        "trained_at":          datetime.now().isoformat(),
        "model_file":          os.path.basename(model_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved:  {meta_path}")

    # SHAP importance
    shap_records = shap_df.to_dict(orient="records")
    with open(shap_path, "w") as f:
        json.dump(shap_records, f, indent=2)
    log.info(f"  SHAP saved:  {shap_path}")

    return model_path


# ===========================================================================
# 7. MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int,   default=50,    help="Optuna trials")
    parser.add_argument("--output",  type=str,   default=MODELS_DIR)
    parser.add_argument("--feature-dir", type=str, default=PROCESSED_DIR)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("AbuseRing Sentinel - Model Training Pipeline")
    log.info("=" * 60)
    log.info(f"  Optuna trials: {args.trials}")
    log.info(f"  FP cost: Rs.{FP_COST}  |  FN cost: Rs.{FN_COST}")

    (X_train, y_train, X_val, y_val, X_test, y_test,
     feature_cols, train_df, val_df, test_df) = load_and_split(args.feature_dir)

    # 1. Baseline
    baseline, baseline_metrics, baseline_auc, baseline_t = train_baseline(
        X_train, y_train, X_val, y_val
    )

    # 2. XGBoost
    xgb_model, xgb_metrics, xgb_auc, best_t, study = train_xgboost(
        X_train, y_train, X_val, y_val, n_trials=args.trials
    )

    # 3. SHAP
    shap_df = compute_shap(xgb_model, X_val, feature_cols)

    # 4. Save
    model_path = save_model(
        xgb_model, baseline, best_t, feature_cols,
        xgb_metrics, baseline_metrics, xgb_auc, baseline_auc, shap_df, args.output
    )

    log.info(f"""
{'=' * 60}
TRAINING COMPLETE
{'=' * 60}
  Baseline LR   AUC={baseline_auc:.4f}  F1={baseline_metrics['f1']:.4f}
  XGBoost       AUC={xgb_auc:.4f}  F1={xgb_metrics['f1']:.4f}

  Optimal threshold (validation): {best_t}

  Next step: python ml/evaluation/evaluate.py
{'=' * 60}
""")


if __name__ == "__main__":
    main()
