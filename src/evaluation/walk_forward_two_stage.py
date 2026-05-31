"""
Walk-Forward CV — Two-Stage LightGBM (Loop 4)
=============================================
5-fold expanding-window CV que reentrena ambos stages por fold.
Añade threshold tuning via Youden's J por fold.

Folds idénticos a Loop 2 para comparabilidad directa.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from src.evaluation.metrics import (
    active_wape, compute_all_metrics, demand_f1, pinball_loss,
)

log = logging.getLogger(__name__)

GOLD_PATH     = Path("data/gold/gold_features.parquet")
BENCHMARKS_DIR = Path("data/benchmarks")
REPORTS_DIR   = Path("reports")

CATEGORY      = "Sell-in"
EXPERIMENT    = "ai_dlc_loop4_cv_two_stage"
MIN_TRAIN_ROWS = 500   # mínimo de filas no-NaN para entrenar

FOLDS = [
    {"fold": 1, "train_end": 202352, "val_start": 202401, "val_end": 202413},
    {"fold": 2, "train_end": 202413, "val_start": 202414, "val_end": 202426},
    {"fold": 3, "train_end": 202426, "val_start": 202427, "val_end": 202439},
    {"fold": 4, "train_end": 202439, "val_start": 202440, "val_end": 202452},
    {"fold": 5, "train_end": 202452, "val_start": 202501, "val_end": 202513},
]

BASE_FEATURES = [
    "lag_1", "lag_4", "lag_52",
    "rolling_mean_4", "rolling_mean_12", "rolling_std_12",
    "weeks_since_last_sale", "intermittent_flag",
    "week_sin", "week_cos",
    "inventory_days_of_supply",
]
LOOP3_FEATURES = [
    "lag_nonzero_1", "lag_nonzero_4", "lag_nonzero_52",
    "demand_rate_12", "zscore_vs_channel",
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
    "scale_pos_weight": 99,
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
    "min_child_samples": 10,   # Loop 4: lower than Loop 3 (20) — more sensitive to rare demand
    "verbose": -1,
    "n_jobs": -1,
}


def _add_loop3_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    df = df.copy()
    df["lag_nonzero_1"]  = (df["lag_1"]  > 0).astype("float32")
    df["lag_nonzero_4"]  = (df["lag_4"]  > 0).astype("float32")
    df["lag_nonzero_52"] = (df["lag_52"] > 0).astype("float32")
    df["demand_rate_12"] = (1.0 - df["intermittent_flag"]).astype("float32")
    ch_stats = (
        df.groupby("Channel")["rolling_mean_12"]
        .agg(ch_mean="mean", ch_std="std")
        .reset_index()
    )
    df = df.merge(ch_stats, on="Channel", how="left")
    df["zscore_vs_channel"] = (
        (df["rolling_mean_12"] - df["ch_mean"]) / (df["ch_std"] + eps)
    ).astype("float32")
    return df.drop(columns=["ch_mean", "ch_std"])


def _optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return threshold that maximises Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


def _train_and_eval_fold(
    df: pd.DataFrame,
    fold: dict,
    feat_cols: list[str],
    tracker: mlflow.ActiveRun,
) -> dict:
    f          = fold["fold"]
    train_end  = fold["train_end"]
    val_start  = fold["val_start"]
    val_end    = fold["val_end"]

    assert train_end < val_start, f"LEAKAGE: train_end={train_end} >= val_start={val_start}"

    train = df[df["year_week"] <= train_end].dropna(subset=["lag_1"]).copy()
    val   = df[(df["year_week"] >= val_start) & (df["year_week"] <= val_end)].dropna(subset=["lag_1"]).copy()

    if len(train) < MIN_TRAIN_ROWS or len(val) == 0:
        log.warning(f"Fold {f}: skipping — train={len(train)} val={len(val)}")
        return {"fold": f, "skipped": True}

    X_train = train[feat_cols].fillna(0).values.astype("float32")
    y_train = train["quantity"].values.astype("float32")
    X_val   = val[feat_cols].fillna(0).values.astype("float32")
    y_val   = val["quantity"].values.astype("float32")

    y_clf_train = (y_train > 0).astype("float32")
    y_clf_val   = (y_val   > 0).astype("float32")

    demand_rate = float(y_clf_train.mean())
    pos_weight  = max(1.0, (1 - demand_rate) / (demand_rate + 1e-9))
    clf_p = {**CLF_PARAMS, "scale_pos_weight": pos_weight}

    # Stage 1 — Classifier
    dcl     = lgb.Dataset(X_train, label=y_clf_train, feature_name=feat_cols, free_raw_data=False)
    dcl_val = lgb.Dataset(X_val,   label=y_clf_val,   reference=dcl,          free_raw_data=False)
    clf_cb  = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)]
    clf     = lgb.train(clf_p, dcl, num_boost_round=300, valid_sets=[dcl_val], callbacks=clf_cb)

    demand_probs = clf.predict(X_val).astype("float32")
    auc = float(roc_auc_score(y_clf_val, demand_probs)) if y_clf_val.sum() > 0 else 0.0
    opt_thresh = _optimal_threshold_youden(y_clf_val, demand_probs) if y_clf_val.sum() > 0 else 0.05

    # Stage 2 — Regressor (non-zero rows only)
    nz = y_train > 0
    if nz.sum() < 10:
        qty_preds = np.zeros(len(y_val), dtype="float32")
        reg_iter  = 0
    else:
        dreg     = lgb.Dataset(X_train[nz], label=y_train[nz], feature_name=feat_cols, free_raw_data=False)
        dreg_val = lgb.Dataset(X_val,       label=y_val,        reference=dreg,         free_raw_data=False)
        reg_cb   = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)]
        reg      = lgb.train(REG_PARAMS, dreg, num_boost_round=300, valid_sets=[dreg_val], callbacks=reg_cb)
        qty_preds = np.clip(reg.predict(X_val), 0.0, None).astype("float32")
        reg_iter  = reg.best_iteration

    # Combined forecast: P × qty (Loop 3 approach)
    final_preds_naive = demand_probs * qty_preds

    # Threshold-gated forecast: set to 0 if P < opt_thresh
    demand_mask       = demand_probs >= opt_thresh
    final_preds_gated = np.where(demand_mask, qty_preds, 0.0).astype("float32")

    # Metrics for both variants
    m_naive = compute_all_metrics(y_val, final_preds_naive)
    m_gated = compute_all_metrics(y_val, final_preds_gated)

    aw_naive  = active_wape(y_val, final_preds_naive)
    aw_gated  = active_wape(y_val, final_preds_gated)
    f1_naive  = demand_f1(y_val, final_preds_naive, threshold=0.5)
    f1_gated  = demand_f1(y_val, final_preds_gated, threshold=0.5)
    f1_clf    = float(f1_score(y_clf_val, (demand_probs >= opt_thresh).astype(int), zero_division=0))
    pl_gated  = pinball_loss(y_val, final_preds_gated)

    log.info(
        f"  Fold {f} | train={len(train):,} val={len(val):,} | "
        f"AUC={auc:.4f} thresh={opt_thresh:.4f} clf_F1={f1_clf:.4f} | "
        f"gated: active_WAPE={aw_gated:.2f} MASE={m_gated['mase']:.4f} demand_F1={f1_gated:.4f}"
    )

    fold_result = {
        "fold":          f,
        "train_end":     train_end,
        "val_start":     val_start,
        "val_end":       val_end,
        "train_rows":    len(train),
        "val_rows":      len(val),
        "demand_rate":   round(demand_rate, 4),
        "nz_train_rows": int(nz.sum()),
        "clf_auc":       round(auc, 4),
        "clf_f1_at_opt_thresh": round(f1_clf, 4),
        "opt_threshold": round(opt_thresh, 4),
        "clf_best_iter": clf.best_iteration,
        "reg_best_iter": reg_iter,
        "metrics_naive": {k: round(v, 4) for k, v in {**m_naive, "active_wape": aw_naive, "demand_f1": f1_naive}.items()},
        "metrics_gated": {k: round(v, 4) for k, v in {**m_gated, "active_wape": aw_gated, "demand_f1": f1_gated, "pinball_50": pl_gated}.items()},
    }

    if tracker is not None:
        mlflow.log_metrics({
            f"fold{f}_clf_auc":        round(auc, 4),
            f"fold{f}_opt_threshold":  round(opt_thresh, 4),
            f"fold{f}_active_wape":    round(aw_gated, 4),
            f"fold{f}_mase":           round(m_gated["mase"], 4),
            f"fold{f}_demand_f1":      round(f1_gated, 4),
        })

    return fold_result


def run_walk_forward_two_stage_cv() -> dict:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading gold features...")
    df = pd.read_parquet(GOLD_PATH)
    df = df[df["Category"] == CATEGORY].copy()
    df["year_week"] = (
        df["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    )
    log.info(f"  Sell-in rows: {len(df):,}")

    log.info("Adding Loop 3 features...")
    df = _add_loop3_features(df)

    feat_cols = [c for c in BASE_FEATURES + LOOP3_FEATURES if c in df.columns]
    log.info(f"  Features ({len(feat_cols)}): {feat_cols}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    fold_results = []
    t0 = time.time()

    with mlflow.start_run(run_name="loop4_cv_five_folds") as run:
        mlflow.log_params({
            "n_folds": len(FOLDS),
            "n_features": len(feat_cols),
            "reg_min_child_samples": REG_PARAMS["min_child_samples"],
        })

        for fold in FOLDS:
            log.info(f"── Fold {fold['fold']} | train≤{fold['train_end']} val={fold['val_start']}–{fold['val_end']} ──")
            result = _train_and_eval_fold(df, fold, feat_cols, run)
            fold_results.append(result)

        # Aggregate across folds (gated variant — threshold-tuned)
        valid_folds = [r for r in fold_results if not r.get("skipped")]
        agg_keys = ["active_wape", "mase", "smape", "wape", "demand_f1", "bias", "mae"]

        agg = {}
        for k in agg_keys:
            vals = [r["metrics_gated"].get(k, np.nan) for r in valid_folds]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                agg[k] = {"mean": round(float(np.mean(vals)), 4),
                           "std":  round(float(np.std(vals)),  4),
                           "min":  round(float(np.min(vals)),  4),
                           "max":  round(float(np.max(vals)),  4)}

        avg_auc    = np.mean([r["clf_auc"]        for r in valid_folds])
        avg_thresh = np.mean([r["opt_threshold"]  for r in valid_folds])

        mlflow.log_metrics({
            "cv_active_wape_mean": agg.get("active_wape", {}).get("mean", 0),
            "cv_mase_mean":        agg.get("mase",        {}).get("mean", 0),
            "cv_demand_f1_mean":   agg.get("demand_f1",   {}).get("mean", 0),
            "cv_clf_auc_mean":     round(avg_auc, 4),
            "cv_opt_threshold":    round(avg_thresh, 4),
        })

        run_id = run.info.run_id

    elapsed = time.time() - t0
    log.info(f"CV complete in {elapsed:.1f}s")
    log.info(f"  avg active_WAPE: {agg.get('active_wape', {}).get('mean', 'n/a')}")
    log.info(f"  avg MASE:        {agg.get('mase', {}).get('mean', 'n/a')}")
    log.info(f"  avg demand_F1:   {agg.get('demand_f1', {}).get('mean', 'n/a')}")
    log.info(f"  avg AUC:         {round(avg_auc, 4)}")
    log.info(f"  avg opt_thresh:  {round(avg_thresh, 4)}")

    result = {
        "pipeline":      "walk_forward_two_stage_cv",
        "mlflow_run_id": run_id,
        "n_folds":       len(FOLDS),
        "n_features":    len(feat_cols),
        "elapsed_s":     round(elapsed, 1),
        "fold_results":  fold_results,
        "aggregate_metrics_gated": agg,
        "avg_clf_auc":   round(avg_auc, 4),
        "avg_opt_threshold": round(avg_thresh, 4),
        "reg_min_child_samples": REG_PARAMS["min_child_samples"],
    }

    import json
    out = REPORTS_DIR / "loop4_cv_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"CV report → {out}")

    return result


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s | %(levelname)s | %(message)s")
    run_walk_forward_two_stage_cv()
