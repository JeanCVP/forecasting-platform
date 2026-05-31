"""
Loop 3 — Two-Stage LightGBM Forecaster
=======================================
Addresses the critical finding from Loop 2: 99% zero rate dominates all metrics.

Architecture:
  Stage 1 — LGB Classifier  : P(demand > 0)  [trained on all rows]
  Stage 2 — LGB Regressor   : E[qty | demand > 0]  [trained on non-zero rows only]
  Final forecast             : P(demand > 0) × E[qty | demand > 0]

Loop 3 features (derived on-the-fly from gold):
  lag_nonzero_1/4/52        : binary indicator for each lag
  demand_rate_12            : fraction of last 12 weeks with demand > 0  (= 1 - intermittent_flag)
  zscore_vs_channel         : (rolling_mean_12 - channel_mean12) / (channel_std12 + eps)

Train  : year_week <= 202452
Val    : 202501 – 202513  (mirrors Fold 5)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    active_wape,
    compute_all_metrics,
    demand_f1,
    pinball_loss,
    wape,
)

log = logging.getLogger(__name__)

GOLD_PATH   = Path("data/gold/gold_features.parquet")
MODELS_DIR  = Path("data/models")
REPORTS_DIR = Path("reports")

CATEGORY  = "Sell-in"
TRAIN_END = 202452
VAL_START = 202501
VAL_END   = 202513

EXPERIMENT = "ai_dlc_loop3_two_stage"

BASE_FEATURES = [
    "lag_1", "lag_4", "lag_52",
    "rolling_mean_4", "rolling_mean_12", "rolling_std_12",
    "weeks_since_last_sale", "intermittent_flag",
    "week_sin", "week_cos",
    "inventory_days_of_supply",
]
LOOP3_FEATURES = [
    "lag_nonzero_1", "lag_nonzero_4", "lag_nonzero_52",
    "demand_rate_12",
    "zscore_vs_channel",
]

CLF_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "scale_pos_weight": 99,   # ~99:1 class imbalance
    "verbose": -1,
    "n_jobs": -1,
}

REG_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
}


def _add_loop3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Loop 3 features from existing gold columns."""
    eps = 1e-9

    df["lag_nonzero_1"]  = (df["lag_1"]  > 0).astype("float32")
    df["lag_nonzero_4"]  = (df["lag_4"]  > 0).astype("float32")
    df["lag_nonzero_52"] = (df["lag_52"] > 0).astype("float32")

    # demand_rate_12 = fraction of last 12 weeks with demand (= 1 - intermittent_flag)
    df["demand_rate_12"] = (1.0 - df["intermittent_flag"]).astype("float32")

    # zscore_vs_channel: how does this SKU compare to its channel's mean rolling demand
    ch_stats = (
        df.groupby("Channel")["rolling_mean_12"]
        .agg(ch_mean="mean", ch_std="std")
        .reset_index()
    )
    df = df.merge(ch_stats, on="Channel", how="left")
    df["zscore_vs_channel"] = (
        (df["rolling_mean_12"] - df["ch_mean"]) / (df["ch_std"] + eps)
    ).astype("float32")
    df = df.drop(columns=["ch_mean", "ch_std"])

    return df


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce RAM usage."""
    for col in BASE_FEATURES + LOOP3_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    df["quantity"] = df["quantity"].astype("float32")
    df["year_week"] = df["year_week"].astype("int32")
    return df


def _optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises Youden's J = TPR - FPR on the validation set."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def run_two_stage_training(tune_threshold: bool = True) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & prepare ────────────────────────────────────────────────────────
    log.info("Loading gold features...")
    df = pd.read_parquet(GOLD_PATH)
    df = df[df["Category"] == CATEGORY].copy()
    df["year_week"] = (
        df["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    )
    log.info(f"  Sell-in rows: {len(df):,}")

    log.info("Adding Loop 3 features...")
    df = _add_loop3_features(df)
    df = _optimize_dtypes(df)

    feat_cols = [c for c in BASE_FEATURES + LOOP3_FEATURES if c in df.columns]
    log.info(f"  Features ({len(feat_cols)}): {feat_cols}")

    train = df[df["year_week"] <= TRAIN_END].dropna(subset=["lag_1"])
    val   = df[(df["year_week"] >= VAL_START) & (df["year_week"] <= VAL_END)].dropna(subset=["lag_1"])
    log.info(f"  Train: {len(train):,} rows | Val: {len(val):,} rows")

    X_train = train[feat_cols].fillna(0).values.astype("float32")
    y_train = train["quantity"].values.astype("float32")
    X_val   = val[feat_cols].fillna(0).values.astype("float32")
    y_val   = val["quantity"].values.astype("float32")

    # Binary targets
    y_train_clf = (y_train > 0).astype("float32")
    y_val_clf   = (y_val   > 0).astype("float32")
    demand_rate = float(y_train_clf.mean())
    log.info(f"  Train demand rate: {demand_rate:.4f} ({y_train_clf.sum():.0f} non-zero rows)")

    # ── MLflow experiment ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="loop3_two_stage") as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "train_end": TRAIN_END, "val_start": VAL_START, "val_end": VAL_END,
            "n_features": len(feat_cols), "train_rows": len(train), "val_rows": len(val),
            "demand_rate": round(demand_rate, 4),
        })

        # ── Stage 1: Demand Classifier ────────────────────────────────────────
        log.info("Training Stage 1 — Demand Classifier...")
        dcl = lgb.Dataset(X_train, label=y_train_clf, feature_name=feat_cols, free_raw_data=False)
        dcl_val = lgb.Dataset(X_val, label=y_val_clf, reference=dcl, free_raw_data=False)

        clf_cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)]
        clf_model = lgb.train(
            CLF_PARAMS, dcl, num_boost_round=500,
            valid_sets=[dcl_val], callbacks=clf_cb,
        )

        demand_probs_val = clf_model.predict(X_val).astype("float32")
        from sklearn.metrics import roc_auc_score, f1_score
        clf_auc = float(roc_auc_score(y_val_clf, demand_probs_val))

        # Threshold tuning via Youden's J (optimal for imbalanced classes)
        if tune_threshold and y_val_clf.sum() > 0:
            opt_threshold = _optimal_threshold_youden(y_val_clf, demand_probs_val)
        else:
            opt_threshold = float(demand_rate)   # fallback: use demand rate
        opt_threshold = float(np.clip(opt_threshold, 0.01, 0.5))

        clf_f1 = float(f1_score(y_val_clf, (demand_probs_val >= opt_threshold).astype(int), zero_division=0))
        log.info(
            f"  Stage 1 — AUC={clf_auc:.4f}  opt_thresh={opt_threshold:.4f}  "
            f"F1@thresh={clf_f1:.4f}  iter={clf_model.best_iteration}"
        )

        clf_path = MODELS_DIR / "lgbm_stage1_classifier.txt"
        clf_model.save_model(str(clf_path))

        # ── Stage 2: Quantity Regressor (non-zero rows only) ──────────────────
        log.info("Training Stage 2 — Quantity Regressor (non-zero rows)...")
        nz_mask = y_train > 0
        X_train_nz = X_train[nz_mask]
        y_train_nz = y_train[nz_mask]
        log.info(f"  Non-zero train rows: {len(X_train_nz):,}")

        dreg = lgb.Dataset(X_train_nz, label=y_train_nz, feature_name=feat_cols, free_raw_data=False)
        # Val: use all val rows for early stopping (model sees x → predicts qty)
        dreg_val = lgb.Dataset(X_val, label=y_val, reference=dreg, free_raw_data=False)

        reg_cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)]
        reg_model = lgb.train(
            REG_PARAMS, dreg, num_boost_round=500,
            valid_sets=[dreg_val], callbacks=reg_cb,
        )

        qty_preds_val = np.clip(reg_model.predict(X_val), 0.0, None).astype("float32")
        reg_path = MODELS_DIR / "lgbm_stage2_regressor.txt"
        reg_model.save_model(str(reg_path))

        # ── Combined forecast (threshold-gated — Loop 4) ─────────────────────
        # P × qty_pred when P >= opt_threshold, else 0
        demand_mask = demand_probs_val >= opt_threshold
        final_preds = np.where(demand_mask, qty_preds_val, 0.0).astype("float32")

        # ── Evaluation ───────────────────────────────────────────────────────
        metrics_all  = compute_all_metrics(y_val, final_preds)
        aw           = active_wape(y_val, final_preds)
        df1          = demand_f1(y_val, final_preds, threshold=0.5)
        pl           = pinball_loss(y_val, final_preds, quantile=0.5)

        # Baseline comparison: Seasonal Naïve from Loop 2 had sMAPE=25.03, active_wape unknown
        log.info(
            f"  Two-Stage — sMAPE={metrics_all['smape']:.2f}  "
            f"WAPE={metrics_all['wape']:.2f}  active_WAPE={aw:.2f}  "
            f"MASE={metrics_all['mase']:.4f}  demand_F1={df1:.4f}  "
            f"pinball50={pl:.4f}"
        )

        mlflow.log_metrics({
            "val_smape":       round(metrics_all["smape"], 4),
            "val_wape":        round(metrics_all["wape"], 4),
            "val_active_wape": round(aw, 4),
            "val_mase":        round(metrics_all["mase"], 4),
            "val_bias":        round(metrics_all["bias"], 4),
            "val_mae":         round(metrics_all["mae"], 4),
            "val_rmse":        round(metrics_all["rmse"], 4),
            "val_demand_f1":   round(df1, 4),
            "val_pinball_50":  round(pl, 4),
            "clf_auc":         round(clf_auc, 4),
            "clf_f1_at_thresh": round(clf_f1, 4),
            "clf_opt_threshold": round(opt_threshold, 4),
            "clf_best_iter":   clf_model.best_iteration,
            "reg_best_iter":   reg_model.best_iteration,
        })
        mlflow.log_artifact(str(clf_path))
        mlflow.log_artifact(str(reg_path))

    # ── Feature importance ────────────────────────────────────────────────────
    clf_imp = dict(zip(feat_cols, clf_model.feature_importance(importance_type="gain").tolist()))
    reg_imp = dict(zip(feat_cols, reg_model.feature_importance(importance_type="gain").tolist()))

    # ── Report ────────────────────────────────────────────────────────────────
    result = {
        "pipeline":        "loop3_two_stage",
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id":   run_id,
        "category":        CATEGORY,
        "train_end":       TRAIN_END,
        "val_start":       VAL_START,
        "val_end":         VAL_END,
        "n_features":      len(feat_cols),
        "feature_cols":    feat_cols,
        "train_rows":      len(train),
        "val_rows":        len(val),
        "train_demand_rate": round(demand_rate, 4),
        "stage1_classifier": {
            "auc":           round(clf_auc, 4),
            "f1_at_thresh":  round(clf_f1, 4),
            "opt_threshold": round(opt_threshold, 4),
            "best_iter":     clf_model.best_iteration,
            "model_path":    str(clf_path),
            "top_features":  dict(sorted(clf_imp.items(), key=lambda x: -x[1])[:10]),
        },
        "stage2_regressor": {
            "best_iter":    reg_model.best_iteration,
            "model_path":   str(reg_path),
            "non_zero_train_rows": int(nz_mask.sum()),
            "top_features": dict(sorted(reg_imp.items(), key=lambda x: -x[1])[:10]),
        },
        "val_metrics": {
            **{k: round(v, 4) for k, v in metrics_all.items()},
            "active_wape": round(aw, 4),
            "demand_f1":   round(df1, 4),
            "pinball_50":  round(pl, 4),
        },
        "vs_loop2_baseline": {
            "loop2_smape_seasonal_naive": 25.0294,
            "loop3_smape_two_stage":      round(metrics_all["smape"], 4),
            "improvement_smape_pct": round(
                (25.0294 - metrics_all["smape"]) / 25.0294 * 100, 2
            ),
        },
    }

    out = REPORTS_DIR / "loop3_two_stage_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Report → {out}")

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    r = run_two_stage_training()
    print(f"\n{'='*65}")
    print(f"  LOOP 3 — TWO-STAGE LIGHTGBM")
    print(f"{'='*65}")
    print(f"  Stage 1 Classifier — AUC: {r['stage1_classifier']['auc']}  F1: {r['stage1_classifier']['f1']}")
    print(f"  Stage 2 Regressor  — iter: {r['stage2_regressor']['best_iter']}")
    print()
    m = r["val_metrics"]
    print(f"  Val Metrics (combined forecast):")
    print(f"    sMAPE:       {m['smape']}")
    print(f"    WAPE:        {m['wape']}")
    print(f"    active_WAPE: {m['active_wape']}  ← Loop 3 primary metric")
    print(f"    MASE:        {m['mase']}")
    print(f"    demand_F1:   {m['demand_f1']}")
    print(f"    pinball_50:  {m['pinball_50']}")
    print()
    c = r["vs_loop2_baseline"]
    print(f"  vs Loop 2 Seasonal Naïve:")
    print(f"    Loop 2 sMAPE: {c['loop2_smape_seasonal_naive']}")
    print(f"    Loop 3 sMAPE: {c['loop3_smape_two_stage']}")
    delta = c['improvement_smape_pct']
    print(f"    Change:       {delta:+.2f}%")
    print(f"{'='*65}\n")
