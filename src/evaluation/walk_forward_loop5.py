"""
Walk-Forward CV — Loop 5 + Segmented Stage 2 (Loop 6)
======================================================
5-fold expanding-window CV usando:
  - 21 features (Base + Loop3 + Loop5)
  - Hiperparámetros Optuna (Huber + log_target=True)
  - Stage 2 segmentado: "regular" (demand_rate_12 > 0.1) vs "sparse" (≤ 0.1)
  - Threshold Youden's J por fold

Fold 5 genera además las predicciones Q10/Q50/Q90 por (Channel, Material, year_week)
para alimentar el dashboard de intervalos de confianza.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.isotonic import IsotonicRegression

from src.evaluation.metrics import (
    active_wape, compute_all_metrics, demand_f1, pinball_loss,
)

log = logging.getLogger(__name__)

GOLD_PATH      = Path("data/gold/gold_features.parquet")
FORECASTS_DIR  = Path("data/forecasts")
REPORTS_DIR    = Path("reports")

CATEGORY       = "Sell-in"
EXPERIMENT     = "ai_dlc_loop6_cv_segmented"
SEGMENT_THRESH = 0.1   # demand_rate_12 > SEGMENT_THRESH → "regular", else "sparse"
MIN_SEGMENT_ROWS = 20  # mínimo para entrenar un segment model separado

FOLDS = [
    {"fold": 1, "train_end": 202352, "val_start": 202401, "val_end": 202413},
    {"fold": 2, "train_end": 202413, "val_start": 202414, "val_end": 202426},
    {"fold": 3, "train_end": 202426, "val_start": 202427, "val_end": 202439},
    {"fold": 4, "train_end": 202439, "val_start": 202440, "val_end": 202452},
    {"fold": 5, "train_end": 202452, "val_start": 202501, "val_end": 202513},
]

# Optuna best params from Loop 5
OPTUNA_PARAMS = {
    "objective":         "huber",
    "alpha":             0.9823086684729874,
    "metric":            "mae",
    "num_leaves":        154,
    "learning_rate":     0.14839058155279436,
    "min_child_samples": 19,
    "feature_fraction":  0.5014767193713563,
    "bagging_fraction":  0.7352211639093376,
    "bagging_freq":      5,
    "verbose":           -1,
    "n_jobs":            -1,
}

BASE_FEATURES = [
    "lag_1", "lag_4", "lag_52",
    "rolling_mean_4", "rolling_mean_12", "rolling_std_12",
    "weeks_since_last_sale", "intermittent_flag",
    "week_sin", "week_cos", "inventory_days_of_supply",
]
LOOP3_FEATURES = [
    "lag_nonzero_1", "lag_nonzero_4", "lag_nonzero_52",
    "demand_rate_12", "zscore_vs_channel",
]
LOOP5_FEATURES = [
    "log_lag_1", "log_lag_4", "log_rolling_mean_12",
    "cv_12", "channel_nonzero_avg",
]

CLF_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "num_leaves": 63, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 50, "scale_pos_weight": 99,
    "verbose": -1, "n_jobs": -1,
}


def _add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    df = df.copy()
    df["lag_nonzero_1"]  = (df["lag_1"]  > 0).astype("float32")
    df["lag_nonzero_4"]  = (df["lag_4"]  > 0).astype("float32")
    df["lag_nonzero_52"] = (df["lag_52"] > 0).astype("float32")
    df["demand_rate_12"] = (1.0 - df["intermittent_flag"]).astype("float32")
    ch_stats = df.groupby("Channel")["rolling_mean_12"].agg(ch_mean="mean", ch_std="std").reset_index()
    df = df.merge(ch_stats, on="Channel", how="left")
    df["zscore_vs_channel"] = ((df["rolling_mean_12"] - df["ch_mean"]) / (df["ch_std"] + eps)).astype("float32")
    df = df.drop(columns=["ch_mean", "ch_std"])
    df["log_lag_1"]           = np.log1p(df["lag_1"].fillna(0)).astype("float32")
    df["log_lag_4"]           = np.log1p(df["lag_4"].fillna(0)).astype("float32")
    df["log_rolling_mean_12"] = np.log1p(df["rolling_mean_12"].fillna(0)).astype("float32")
    df["cv_12"] = (df["rolling_std_12"] / (df["rolling_mean_12"] + eps)).fillna(0).clip(0, 10).astype("float32")
    nz = df[df["quantity"] > 0]
    ch_nz = nz.groupby("Channel")["quantity"].mean().rename("channel_nonzero_avg").reset_index()
    df = df.merge(ch_nz, on="Channel", how="left")
    df["channel_nonzero_avg"] = df["channel_nonzero_avg"].fillna(0).astype("float32")
    return df


def _train_segment_model(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                         feat_cols: list, log_target: bool, label: str) -> lgb.Booster | None:
    if len(X) < MIN_SEGMENT_ROWS:
        log.debug(f"    Segment '{label}': only {len(X)} rows — skip")
        return None
    y_tr = np.log1p(y) if log_target else y
    y_va = np.log1p(y_val) if log_target else y_val
    d_tr = lgb.Dataset(X, label=y_tr, feature_name=feat_cols, free_raw_data=False)
    d_va = lgb.Dataset(X_val, label=y_va, reference=d_tr, free_raw_data=False)
    cb = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=0)]
    return lgb.train(OPTUNA_PARAMS, d_tr, num_boost_round=400, valid_sets=[d_va], callbacks=cb)


def _predict_with_fallback(models: dict, X: np.ndarray, segment_mask: np.ndarray,
                           log_target: bool) -> np.ndarray:
    preds = np.zeros(len(X), dtype="float32")
    for seg, mask in [("regular", segment_mask), ("sparse", ~segment_mask)]:
        if mask.sum() == 0:
            continue
        model = models.get(seg) or models.get("regular") or models.get("sparse")
        if model is None:
            continue
        raw = model.predict(X[mask]).astype("float32")
        preds[mask] = np.expm1(raw).clip(0) if log_target else np.clip(raw, 0, None)
    return preds


def _optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(np.clip(thresholds[int(np.argmax(j))], 0.01, 0.95))


def _train_and_eval_fold(df: pd.DataFrame, fold: dict, feat_cols: list,
                         save_ci: bool = False) -> dict:
    f, train_end, val_start, val_end = (
        fold["fold"], fold["train_end"], fold["val_start"], fold["val_end"]
    )
    assert train_end < val_start

    train = df[df["year_week"] <= train_end].dropna(subset=["lag_1"]).copy()
    val   = df[(df["year_week"] >= val_start) & (df["year_week"] <= val_end)].dropna(subset=["lag_1"]).copy()

    if len(train) < 500 or len(val) == 0:
        return {"fold": f, "skipped": True}

    X_tr = train[feat_cols].fillna(0).values.astype("float32")
    y_tr = train["quantity"].values.astype("float32")
    X_va = val[feat_cols].fillna(0).values.astype("float32")
    y_va = val["quantity"].values.astype("float32")

    y_clf_tr = (y_tr > 0).astype("float32")
    y_clf_va = (y_va > 0).astype("float32")

    # ── Stage 1 with isotonic calibration ────────────────────────────────────
    dcl     = lgb.Dataset(X_tr, label=y_clf_tr, feature_name=feat_cols, free_raw_data=False)
    dcl_val = lgb.Dataset(X_va, label=y_clf_va, reference=dcl,          free_raw_data=False)
    clf_cb  = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)]
    clf     = lgb.train(CLF_PARAMS, dcl, num_boost_round=300, valid_sets=[dcl_val], callbacks=clf_cb)

    raw_probs_tr = clf.predict(X_tr).astype("float32")
    raw_probs_va = clf.predict(X_va).astype("float32")
    cal = IsotonicRegression(out_of_bounds="clip").fit(raw_probs_tr, y_clf_tr)
    cal_probs_va = cal.predict(raw_probs_va).astype("float32")

    auc = float(roc_auc_score(y_clf_va, cal_probs_va)) if y_clf_va.sum() > 0 else 0.0
    opt_thr = _optimal_threshold_youden(y_clf_va, cal_probs_va) if y_clf_va.sum() > 0 else 0.05
    clf_f1 = float(f1_score(y_clf_va, (cal_probs_va >= opt_thr).astype(int), zero_division=0))

    # ── Stage 2 segmented ─────────────────────────────────────────────────────
    feat_dr12 = feat_cols.index("demand_rate_12") if "demand_rate_12" in feat_cols else None
    nz_mask = y_tr > 0
    X_nz    = X_tr[nz_mask]
    y_nz    = y_tr[nz_mask]

    if feat_dr12 is not None:
        reg_mask_tr = X_nz[:, feat_dr12] > SEGMENT_THRESH
        reg_mask_va = X_va[:, feat_dr12] > SEGMENT_THRESH
    else:
        reg_mask_tr = np.ones(len(X_nz), dtype=bool)
        reg_mask_va = np.ones(len(X_va),  dtype=bool)

    models = {}
    for seg, mask_tr in [("regular", reg_mask_tr), ("sparse", ~reg_mask_tr)]:
        m = _train_segment_model(X_nz[mask_tr], y_nz[mask_tr], X_va, y_va, feat_cols, True, seg)
        if m:
            models[seg] = m

    # ── Q10 / Q90 (fold 5 only for CI dashboard data) ─────────────────────────
    q10_preds = np.zeros(len(y_va), dtype="float32")
    q90_preds = np.zeros(len(y_va), dtype="float32")
    if save_ci and len(X_nz) >= MIN_SEGMENT_ROWS:
        for q_val, attr in [(0.1, "q10"), (0.9, "q90")]:
            qp = {**OPTUNA_PARAMS, "objective": "quantile", "alpha": q_val, "metric": "quantile"}
            dq  = lgb.Dataset(X_nz, label=np.log1p(y_nz), feature_name=feat_cols, free_raw_data=False)
            dqv = lgb.Dataset(X_va, label=np.log1p(y_va), reference=dq,           free_raw_data=False)
            qm  = lgb.train(qp, dq, num_boost_round=300,
                            valid_sets=[dqv],
                            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=0)])
            raw_q = qm.predict(X_va).astype("float32")
            q_out = np.expm1(raw_q).clip(0)
            if attr == "q10":
                q10_preds = q_out
            else:
                q90_preds = q_out

    qty_preds = _predict_with_fallback(models, X_va, reg_mask_va, True)

    demand_mask  = cal_probs_va >= opt_thr
    final_preds  = np.where(demand_mask, qty_preds,  0.0).astype("float32")
    final_q10    = np.where(demand_mask, q10_preds,  0.0).astype("float32")
    final_q90    = np.where(demand_mask, q90_preds,  0.0).astype("float32")

    m_all = compute_all_metrics(y_va, final_preds)
    aw    = active_wape(y_va, final_preds)
    df1   = demand_f1(y_va, final_preds)
    pl    = pinball_loss(y_va, final_preds)

    n_reg  = int(reg_mask_tr.sum())
    n_spr  = int((~reg_mask_tr).sum())

    log.info(
        f"  Fold {f} | AUC={auc:.4f} thr={opt_thr:.4f} clf_F1={clf_f1:.4f} | "
        f"seg: regular={n_reg} sparse={n_spr} | "
        f"active_WAPE={aw:.2f} MASE={m_all['mase']:.4f} demand_F1={df1:.4f}"
    )

    # Save CI data for fold 5
    if save_ci:
        FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
        ci_df = val[["Channel", "Material Description", "year_week"]].copy()
        ci_df["quantity_actual"] = y_va
        ci_df["forecast_q50"]    = final_preds
        ci_df["forecast_q10"]    = final_q10
        ci_df["forecast_q90"]    = final_q90
        ci_df["demand_prob"]     = cal_probs_va
        ci_path = FORECASTS_DIR / "loop6_ci_predictions.parquet"
        ci_df.to_parquet(ci_path, index=False)
        log.info(f"  CI predictions saved → {ci_path} ({len(ci_df):,} rows)")

    return {
        "fold": f, "train_end": train_end, "val_start": val_start, "val_end": val_end,
        "train_rows": len(train), "val_rows": len(val),
        "clf_auc": round(auc, 4), "opt_threshold": round(opt_thr, 4), "clf_f1": round(clf_f1, 4),
        "segment_regular_rows": n_reg, "segment_sparse_rows": n_spr,
        "metrics": {
            **{k: round(v, 4) for k, v in m_all.items()},
            "active_wape": round(aw, 4), "demand_f1": round(df1, 4), "pinball_50": round(pl, 4),
        },
    }


def run_walk_forward_loop5_cv() -> dict:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading gold features...")
    df = pd.read_parquet(GOLD_PATH)
    df = df[df["Category"] == CATEGORY].copy()
    df["year_week"] = df["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    df["quantity"]  = df["quantity"].astype("float32")
    log.info(f"  Sell-in rows: {len(df):,}")

    df = _add_all_features(df)
    feat_cols = [c for c in BASE_FEATURES + LOOP3_FEATURES + LOOP5_FEATURES if c in df.columns]
    log.info(f"  Features ({len(feat_cols)}): {feat_cols}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    fold_results = []
    t0 = time.time()

    with mlflow.start_run(run_name="loop6_cv_segmented") as run:
        mlflow.log_params({
            "n_folds": len(FOLDS), "n_features": len(feat_cols),
            "segment_threshold": SEGMENT_THRESH,
            "reg_objective": OPTUNA_PARAMS["objective"],
            "log_target": True,
        })

        for fold in FOLDS:
            log.info(f"── Fold {fold['fold']} | train≤{fold['train_end']} val={fold['val_start']}–{fold['val_end']} ──")
            save_ci = (fold["fold"] == 5)
            result = _train_and_eval_fold(df, fold, feat_cols, save_ci=save_ci)
            fold_results.append(result)

        valid = [r for r in fold_results if not r.get("skipped")]
        agg_keys = ["active_wape", "mase", "smape", "wape", "demand_f1", "bias", "mae"]
        agg = {}
        for k in agg_keys:
            vals = [r["metrics"].get(k, np.nan) for r in valid]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                agg[k] = {"mean": round(float(np.mean(vals)), 4),
                           "std":  round(float(np.std(vals)),  4),
                           "min":  round(float(np.min(vals)),  4),
                           "max":  round(float(np.max(vals)),  4)}

        avg_auc = round(float(np.mean([r["clf_auc"] for r in valid])), 4)
        mlflow.log_metrics({
            "cv_active_wape_mean": agg.get("active_wape", {}).get("mean", 0),
            "cv_mase_mean":        agg.get("mase",        {}).get("mean", 0),
            "cv_demand_f1_mean":   agg.get("demand_f1",   {}).get("mean", 0),
            "cv_clf_auc_mean":     avg_auc,
        })
        run_id = run.info.run_id

    elapsed = time.time() - t0
    log.info(f"CV complete in {elapsed:.1f}s")
    log.info(f"  avg active_WAPE: {agg.get('active_wape', {}).get('mean', '?')}")
    log.info(f"  avg MASE:        {agg.get('mase', {}).get('mean', '?')}")
    log.info(f"  avg demand_F1:   {agg.get('demand_f1', {}).get('mean', '?')}")
    log.info(f"  avg AUC:         {avg_auc}")

    result = {
        "pipeline":          "walk_forward_loop5_segmented",
        "mlflow_run_id":     run_id,
        "n_folds":           len(FOLDS),
        "n_features":        len(feat_cols),
        "elapsed_s":         round(elapsed, 1),
        "segment_threshold": SEGMENT_THRESH,
        "fold_results":      fold_results,
        "aggregate_metrics": agg,
        "avg_clf_auc":       avg_auc,
    }

    out = REPORTS_DIR / "loop6_cv_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"CV report → {out}")
    return result


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_walk_forward_loop5_cv()
