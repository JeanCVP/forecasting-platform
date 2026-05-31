"""
Optuna Hyperparameter Tuner — Loop 5
=====================================
Optimiza Stage 2 (regresor de cantidad) minimizando active_WAPE en fold 5.

Espacio de búsqueda:
  - objective  : regression_l1 / regression_l2 / huber
  - log_target : bool — entrenar sobre log(1+y) para reducir bias de -87%
  - num_leaves : 31 – 255
  - learning_rate: 0.01 – 0.15
  - min_child_samples: 5 – 50
  - feature_fraction: 0.5 – 1.0
  - bagging_fraction: 0.5 – 1.0

Fold de tuning: fold 5 (train ≤ 202452, val 202501–202513)
"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.evaluation.metrics import active_wape

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)

GOLD_PATH  = Path("data/gold/gold_features.parquet")
CATEGORY   = "Sell-in"
TRAIN_END  = 202452
VAL_START  = 202501
VAL_END    = 202513

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


def _add_all_features(df):
    import pandas as pd
    eps = 1e-9
    df = df.copy()

    # Loop 3 features
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

    # Loop 5 features
    df["log_lag_1"]           = np.log1p(df["lag_1"].fillna(0)).astype("float32")
    df["log_lag_4"]           = np.log1p(df["lag_4"].fillna(0)).astype("float32")
    df["log_rolling_mean_12"] = np.log1p(df["rolling_mean_12"].fillna(0)).astype("float32")
    df["cv_12"] = (
        df["rolling_std_12"] / (df["rolling_mean_12"] + eps)
    ).fillna(0).clip(0, 10).astype("float32")

    # channel_nonzero_avg: average non-zero qty per channel (from training rows only)
    nz = df[df["quantity"] > 0]
    ch_nz = nz.groupby("Channel")["quantity"].mean().rename("channel_nonzero_avg").reset_index()
    df = df.merge(ch_nz, on="Channel", how="left")
    df["channel_nonzero_avg"] = df["channel_nonzero_avg"].fillna(0).astype("float32")

    return df


def _build_datasets(df, feat_cols):
    train = df[df["year_week"] <= TRAIN_END].dropna(subset=["lag_1"])
    val   = df[(df["year_week"] >= VAL_START) & (df["year_week"] <= VAL_END)].dropna(subset=["lag_1"])

    X_tr = train[feat_cols].fillna(0).values.astype("float32")
    y_tr = train["quantity"].values.astype("float32")
    X_va = val[feat_cols].fillna(0).values.astype("float32")
    y_va = val["quantity"].values.astype("float32")
    return X_tr, y_tr, X_va, y_va


def run_optuna_tuning(n_trials: int = 30) -> dict:
    import pandas as pd

    log.info("Loading gold for Optuna tuning...")
    df = pd.read_parquet(GOLD_PATH)
    df = df[df["Category"] == CATEGORY].copy()
    df["year_week"] = (
        df["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    )
    df["quantity"] = df["quantity"].astype("float32")

    log.info("Adding all features...")
    df = _add_all_features(df)

    feat_cols = [c for c in BASE_FEATURES + LOOP3_FEATURES + LOOP5_FEATURES if c in df.columns]
    log.info(f"  Features for tuning ({len(feat_cols)}): {feat_cols}")

    X_tr, y_tr, X_va, y_va = _build_datasets(df, feat_cols)
    nz_mask = y_tr > 0
    X_tr_nz = X_tr[nz_mask]
    y_tr_nz = y_tr[nz_mask]

    log.info(f"  Non-zero train rows: {nz_mask.sum():,} | Val rows: {len(y_va):,}")

    def objective(trial: optuna.Trial) -> float:
        log_target = trial.suggest_categorical("log_target", [True, False])
        params = {
            "objective":       trial.suggest_categorical("objective", ["regression_l1", "regression_l2", "huber"]),
            "metric":          "mae",
            "num_leaves":      trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":    5,
            "verbose":         -1,
            "n_jobs":          -1,
        }
        if params["objective"] == "huber":
            params["alpha"] = trial.suggest_float("huber_alpha", 0.7, 0.99)

        y_train_use = np.log1p(y_tr_nz) if log_target else y_tr_nz

        d_tr = lgb.Dataset(X_tr_nz, label=y_train_use, free_raw_data=False)
        d_va = lgb.Dataset(X_va,    label=np.log1p(y_va) if log_target else y_va, reference=d_tr, free_raw_data=False)

        cb = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=0)]
        model = lgb.train(params, d_tr, num_boost_round=400, valid_sets=[d_va], callbacks=cb)

        raw_preds = model.predict(X_va).astype("float32")
        preds = np.expm1(raw_preds).clip(0) if log_target else np.clip(raw_preds, 0, None)

        return active_wape(y_va, preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name="loop5_stage2_tuning",
    )

    log.info(f"Running {n_trials} Optuna trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best  = study.best_trial
    log.info(f"Best trial #{best.number}: active_WAPE={best.value:.4f}")
    log.info(f"  Params: {best.params}")

    return {
        "best_active_wape": round(best.value, 4),
        "best_params":      best.params,
        "n_trials":         n_trials,
        "n_features":       len(feat_cols),
        "feature_cols":     feat_cols,
        "all_trials": [
            {"number": t.number, "value": round(t.value, 4), "params": t.params}
            for t in sorted(study.trials, key=lambda t: t.value)[:10]
        ],
    }
