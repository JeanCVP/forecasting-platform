"""
Loop 5 — Enhanced Two-Stage LightGBM
=====================================
Mejoras sobre Loop 4:

1. Log-transform del target en Stage 2
   → Corrige bias de -87.5% (L1 predice mediana en dist. sesgada)

2. Nuevas features de escala (Loop 5)
   → log_lag_1, log_lag_4, log_rolling_mean_12, cv_12, channel_nonzero_avg

3. Calibración isotónica del clasificador (Stage 1)
   → Mejora la calibración de probabilidades para threshold tuning

4. Cuantiles Q10/Q90 en Stage 2
   → Intervalos de confianza para el dashboard

5. Hiperparámetros de Stage 2 vienen del tuning Optuna
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score

from src.evaluation.metrics import (
    active_wape, compute_all_metrics, demand_f1, pinball_loss,
)

log = logging.getLogger(__name__)

GOLD_PATH   = Path("data/gold/gold_features.parquet")
MODELS_DIR  = Path("data/models")
REPORTS_DIR = Path("reports")

CATEGORY  = "Sell-in"
TRAIN_END = 202452
VAL_START = 202501
VAL_END   = 202513

EXPERIMENT = "ai_dlc_loop5_enhanced"

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


def _add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    df = df.copy()

    # Loop 3
    df["lag_nonzero_1"]  = (df["lag_1"]  > 0).astype("float32")
    df["lag_nonzero_4"]  = (df["lag_4"]  > 0).astype("float32")
    df["lag_nonzero_52"] = (df["lag_52"] > 0).astype("float32")
    df["demand_rate_12"] = (1.0 - df["intermittent_flag"]).astype("float32")
    ch_stats = (
        df.groupby("Channel")["rolling_mean_12"]
        .agg(ch_mean="mean", ch_std="std").reset_index()
    )
    df = df.merge(ch_stats, on="Channel", how="left")
    df["zscore_vs_channel"] = (
        (df["rolling_mean_12"] - df["ch_mean"]) / (df["ch_std"] + eps)
    ).astype("float32")
    df = df.drop(columns=["ch_mean", "ch_std"])

    # Loop 5 — scale-aware features
    df["log_lag_1"]           = np.log1p(df["lag_1"].fillna(0)).astype("float32")
    df["log_lag_4"]           = np.log1p(df["lag_4"].fillna(0)).astype("float32")
    df["log_rolling_mean_12"] = np.log1p(df["rolling_mean_12"].fillna(0)).astype("float32")
    df["cv_12"] = (
        df["rolling_std_12"] / (df["rolling_mean_12"] + eps)
    ).fillna(0).clip(0, 10).astype("float32")

    # channel_nonzero_avg: avg non-zero qty per channel
    nz = df[df["quantity"] > 0]
    ch_nz = nz.groupby("Channel")["quantity"].mean().rename("channel_nonzero_avg").reset_index()
    df = df.merge(ch_nz, on="Channel", how="left")
    df["channel_nonzero_avg"] = df["channel_nonzero_avg"].fillna(0).astype("float32")

    return df


def _optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(np.clip(thresholds[int(np.argmax(j))], 0.01, 0.95))


def run_loop5_training(optuna_params: dict | None = None) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & prepare ────────────────────────────────────────────────────────
    log.info("Loading gold features...")
    df = pd.read_parquet(GOLD_PATH)
    df = df[df["Category"] == CATEGORY].copy()
    df["year_week"] = (
        df["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    )
    df["quantity"] = df["quantity"].astype("float32")
    log.info(f"  Sell-in rows: {len(df):,}")

    log.info("Adding all features (Loop 3 + Loop 5)...")
    df = _add_all_features(df)

    feat_cols = [c for c in BASE_FEATURES + LOOP3_FEATURES + LOOP5_FEATURES if c in df.columns]
    log.info(f"  Features ({len(feat_cols)}): {feat_cols}")

    train = df[df["year_week"] <= TRAIN_END].dropna(subset=["lag_1"]).copy()
    val   = df[(df["year_week"] >= VAL_START) & (df["year_week"] <= VAL_END)].dropna(subset=["lag_1"]).copy()
    log.info(f"  Train: {len(train):,} | Val: {len(val):,}")

    X_train = train[feat_cols].fillna(0).values.astype("float32")
    y_train = train["quantity"].values.astype("float32")
    X_val   = val[feat_cols].fillna(0).values.astype("float32")
    y_val   = val["quantity"].values.astype("float32")

    y_clf_train = (y_train > 0).astype("float32")
    y_clf_val   = (y_val   > 0).astype("float32")
    demand_rate = float(y_clf_train.mean())

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="loop5_enhanced") as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "n_features":   len(feat_cols),
            "train_rows":   len(train),
            "val_rows":     len(val),
            "demand_rate":  round(demand_rate, 4),
            "log_target":   str(optuna_params.get("log_target", True) if optuna_params else True),
            "has_optuna":   optuna_params is not None,
        })

        # ── Stage 1: Classifier ───────────────────────────────────────────────
        log.info("Training Stage 1 — Demand Classifier...")
        dcl     = lgb.Dataset(X_train, label=y_clf_train, feature_name=feat_cols, free_raw_data=False)
        dcl_val = lgb.Dataset(X_val,   label=y_clf_val,   reference=dcl,          free_raw_data=False)
        clf_cb  = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
        clf     = lgb.train(CLF_PARAMS, dcl, num_boost_round=500, valid_sets=[dcl_val], callbacks=clf_cb)

        raw_probs_val = clf.predict(X_val).astype("float32")
        clf_auc       = float(roc_auc_score(y_clf_val, raw_probs_val))

        # Isotonic calibration
        log.info("  Calibrating Stage 1 probabilities (isotonic regression)...")
        raw_probs_train = clf.predict(X_train).astype("float32")
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs_train, y_clf_train)
        cal_probs_val = calibrator.predict(raw_probs_val).astype("float32")

        # Optimal threshold on calibrated probs
        opt_threshold = _optimal_threshold_youden(y_clf_val, cal_probs_val)
        clf_f1 = float(f1_score(y_clf_val, (cal_probs_val >= opt_threshold).astype(int), zero_division=0))
        log.info(f"  Stage 1 — AUC={clf_auc:.4f}  opt_thresh={opt_threshold:.4f}  F1@thresh={clf_f1:.4f}")

        clf_path = MODELS_DIR / "loop5_stage1_classifier.txt"
        cal_path = MODELS_DIR / "loop5_stage1_calibrator.pkl"
        clf.save_model(str(clf_path))
        with open(cal_path, "wb") as fp:
            pickle.dump(calibrator, fp)

        # ── Stage 2: Regressor with log-target ───────────────────────────────
        nz_mask = y_train > 0
        X_tr_nz = X_train[nz_mask]
        y_tr_nz = y_train[nz_mask]
        log.info(f"Training Stage 2 — Regressor (log-target, {nz_mask.sum():,} non-zero rows)...")

        # Determine if using log_target from Optuna or default True
        use_log_target = True
        if optuna_params is not None:
            use_log_target = optuna_params.get("log_target", True)

        y_tr_reg = np.log1p(y_tr_nz) if use_log_target else y_tr_nz

        # Build Stage 2 params from Optuna or defaults
        if optuna_params:
            reg_params = {
                "objective":          optuna_params.get("objective", "regression_l1"),
                "metric":             "mae",
                "num_leaves":         optuna_params.get("num_leaves", 127),
                "learning_rate":      optuna_params.get("learning_rate", 0.03),
                "min_child_samples":  optuna_params.get("min_child_samples", 10),
                "feature_fraction":   optuna_params.get("feature_fraction", 0.8),
                "bagging_fraction":   optuna_params.get("bagging_fraction", 0.8),
                "bagging_freq":       5,
                "verbose":            -1,
                "n_jobs":             -1,
            }
            if optuna_params.get("objective") == "huber":
                reg_params["alpha"] = optuna_params.get("huber_alpha", 0.9)
        else:
            reg_params = {
                "objective": "regression_l1",
                "metric": "mae",
                "num_leaves": 127,
                "learning_rate": 0.03,
                "min_child_samples": 10,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "n_jobs": -1,
            }

        y_va_reg_label = np.log1p(y_val) if use_log_target else y_val
        dreg     = lgb.Dataset(X_tr_nz, label=y_tr_reg,      free_raw_data=False, feature_name=feat_cols)
        dreg_val = lgb.Dataset(X_val,   label=y_va_reg_label, reference=dreg,     free_raw_data=False)
        reg_cb   = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
        reg      = lgb.train(reg_params, dreg, num_boost_round=600, valid_sets=[dreg_val], callbacks=reg_cb)

        raw_qty = reg.predict(X_val).astype("float32")
        qty_preds = np.expm1(raw_qty).clip(0) if use_log_target else np.clip(raw_qty, 0, None)

        reg_path = MODELS_DIR / "loop5_stage2_regressor.txt"
        reg.save_model(str(reg_path))

        # ── Quantile models for confidence intervals (Q10, Q90) ──────────────
        log.info("Training Stage 2 quantile models (Q10, Q90)...")
        q10_preds = np.zeros(len(y_val), dtype="float32")
        q90_preds = np.zeros(len(y_val), dtype="float32")
        for q, attr in [(0.1, "q10"), (0.9, "q90")]:
            qp = {**reg_params, "objective": "quantile", "alpha": q, "metric": "quantile"}
            y_q_label = np.log1p(y_tr_nz) if use_log_target else y_tr_nz
            dq  = lgb.Dataset(X_tr_nz, label=y_q_label, free_raw_data=False, feature_name=feat_cols)
            dqv = lgb.Dataset(X_val, label=y_va_reg_label, reference=dq, free_raw_data=False)
            qm  = lgb.train(qp, dq, num_boost_round=400, valid_sets=[dqv],
                            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)])
            raw_q = qm.predict(X_val).astype("float32")
            q_out = np.expm1(raw_q).clip(0) if use_log_target else np.clip(raw_q, 0, None)
            qm.save_model(str(MODELS_DIR / f"loop5_stage2_{attr}.txt"))
            if q == 0.1:
                q10_preds = q_out
            else:
                q90_preds = q_out
        log.info("  Quantile models trained.")

        # ── Combined forecast (threshold-gated) ───────────────────────────────
        demand_mask = cal_probs_val >= opt_threshold
        final_preds  = np.where(demand_mask, qty_preds,  0.0).astype("float32")
        final_q10    = np.where(demand_mask, q10_preds,  0.0).astype("float32")
        final_q90    = np.where(demand_mask, q90_preds,  0.0).astype("float32")

        # ── Evaluation ────────────────────────────────────────────────────────
        m   = compute_all_metrics(y_val, final_preds)
        aw  = active_wape(y_val, final_preds)
        df1 = demand_f1(y_val, final_preds, threshold=0.5)
        pl  = pinball_loss(y_val, final_preds, quantile=0.5)
        pl10 = pinball_loss(y_val, final_q10, quantile=0.1)
        pl90 = pinball_loss(y_val, final_q90, quantile=0.9)

        log.info(
            f"  Loop 5 — sMAPE={m['smape']:.2f}  WAPE={m['wape']:.2f}  "
            f"active_WAPE={aw:.2f}  MASE={m['mase']:.4f}  "
            f"demand_F1={df1:.4f}  bias={m['bias']:.2f}"
        )
        log.info(f"  Quantile coverage — Q10 pinball={pl10:.4f}  Q90 pinball={pl90:.4f}")

        mlflow.log_metrics({
            "val_smape":         round(m["smape"],  4),
            "val_wape":          round(m["wape"],   4),
            "val_active_wape":   round(aw,          4),
            "val_mase":          round(m["mase"],   4),
            "val_bias":          round(m["bias"],   4),
            "val_mae":           round(m["mae"],    4),
            "val_demand_f1":     round(df1,         4),
            "val_pinball_50":    round(pl,          4),
            "val_pinball_10":    round(pl10,        4),
            "val_pinball_90":    round(pl90,        4),
            "clf_auc":           round(clf_auc,     4),
            "clf_f1_at_thresh":  round(clf_f1,      4),
            "clf_opt_threshold": round(opt_threshold, 4),
            "reg_best_iter":     reg.best_iteration,
            "use_log_target":    int(use_log_target),
        })

    # ── Feature importances ───────────────────────────────────────────────────
    clf_imp = dict(zip(feat_cols, clf.feature_importance("gain").tolist()))
    reg_imp = dict(zip(feat_cols, reg.feature_importance("gain").tolist()))

    result = {
        "pipeline":      "loop5_enhanced_two_stage",
        "run_at":        datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": run_id,
        "n_features":    len(feat_cols),
        "feature_cols":  feat_cols,
        "use_log_target": use_log_target,
        "optuna_params": optuna_params,
        "stage1_classifier": {
            "auc":           round(clf_auc,       4),
            "f1_at_thresh":  round(clf_f1,        4),
            "opt_threshold": round(opt_threshold, 4),
            "best_iter":     clf.best_iteration,
            "calibrated":    True,
            "top_features":  dict(sorted(clf_imp.items(), key=lambda x: -x[1])[:10]),
        },
        "stage2_regressor": {
            "best_iter":        reg.best_iteration,
            "log_target":       use_log_target,
            "nz_train_rows":    int(nz_mask.sum()),
            "params":           {k: v for k, v in reg_params.items() if k != "verbose"},
            "top_features":     dict(sorted(reg_imp.items(), key=lambda x: -x[1])[:10]),
        },
        "val_metrics": {
            **{k: round(v, 4) for k, v in m.items()},
            "active_wape": round(aw,  4),
            "demand_f1":   round(df1, 4),
            "pinball_50":  round(pl,  4),
            "pinball_10":  round(pl10, 4),
            "pinball_90":  round(pl90, 4),
        },
        "vs_loop4": {
            "loop4_active_wape": 93.7601,
            "loop5_active_wape": round(aw,       4),
            "loop4_mase":        0.6169,
            "loop5_mase":        round(m["mase"], 4),
            "loop4_bias":        -87.5074,
            "loop5_bias":        round(m["bias"], 4),
            "active_wape_improvement_pct": round((93.7601 - aw) / 93.7601 * 100, 2),
        },
    }

    out = REPORTS_DIR / "loop5_enhanced_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Report → {out}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    r = run_loop5_training()
    vm = r["val_metrics"]
    c  = r["vs_loop4"]
    print(f"\n{'='*65}")
    print(f"  LOOP 5 — ENHANCED TWO-STAGE LIGHTGBM")
    print(f"{'='*65}")
    print(f"  active_WAPE: {c['loop4_active_wape']} → {c['loop5_active_wape']}  ({c['active_wape_improvement_pct']:+.1f}%)")
    print(f"  MASE:        {c['loop4_mase']} → {c['loop5_mase']}")
    print(f"  bias%:       {c['loop4_bias']:.1f} → {c['loop5_bias']:.1f}")
    print(f"  demand_F1:   {vm['demand_f1']}")
    print(f"  Q10/Q90:     pinball_10={vm['pinball_10']}  pinball_90={vm['pinball_90']}")
    print(f"{'='*65}\n")
