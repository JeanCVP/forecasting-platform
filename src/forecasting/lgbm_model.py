"""
LightGBM Global Forecasting Model
Trains a single global model across all SKUs using gold features.
Train: year_week <= 202452  |  Val: 202501–202513  (mirrors Fold 5 of walk-forward CV)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GOLD_PATH = Path("data/gold/gold_features.parquet")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("data/models")

TRAIN_END = 202452
VAL_START = 202501
VAL_END = 202513
CATEGORY = "Sell-in"

FEATURE_COLS = [
    "lag_1", "lag_4", "lag_52",
    "rolling_mean_4", "rolling_mean_12", "rolling_std_12",
    "weeks_since_last_sale", "intermittent_flag",
    "week_sin", "week_cos",
    "inventory_days_of_supply",
]


def _smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(forecast) + 1e-9
    return float(np.mean(2.0 * np.abs(actual - forecast) / denom) * 100)


def _wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    total = np.sum(np.abs(actual))
    if total < 1e-9:
        return 0.0
    return float(np.sum(np.abs(actual - forecast)) / total * 100)


def run_lgbm_training() -> dict:
    import lightgbm as lgb

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading gold features...")
    df = pd.read_parquet(GOLD_PATH)
    df = df[df["Category"] == CATEGORY].copy()
    df["year_week"] = df["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    log.info(f"  Sell-in rows: {len(df):,}")

    # Features: use available subset
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    log.info(f"  Features ({len(feat_cols)}): {feat_cols}")

    train = df[df["year_week"] <= TRAIN_END].copy()
    val = df[(df["year_week"] >= VAL_START) & (df["year_week"] <= VAL_END)].copy()

    # Drop rows where all lag features are NaN (start of each SKU history)
    lag_cols = [c for c in feat_cols if c.startswith("lag_")]
    train = train.dropna(subset=lag_cols[:1])  # at minimum lag_1 must exist
    val = val.dropna(subset=lag_cols[:1])

    log.info(f"  Train: {len(train):,} rows | Val: {len(val):,} rows")

    X_train = train[feat_cols].fillna(0).values
    y_train = train["quantity"].values.astype(float)
    X_val = val[feat_cols].fillna(0).values
    y_val = val["quantity"].values.astype(float)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression_l1",  # MAE — more robust to intermittent zeros
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

    log.info("Training LightGBM...")
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=50)]
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    preds = np.clip(model.predict(X_val), 0.0, None)
    smape_val = _smape(y_val, preds)
    wape_val = _wape(y_val, preds)
    mae_val = float(np.mean(np.abs(y_val - preds)))

    log.info(f"  Val sMAPE={smape_val:.2f}  WAPE={wape_val:.2f}  MAE={mae_val:.4f}")
    log.info(f"  Best iteration: {model.best_iteration}")

    model_path = MODELS_DIR / "lgbm_global.txt"
    model.save_model(str(model_path))
    log.info(f"  Model saved → {model_path}")

    result = {
        "pipeline": "lgbm_training",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "category": CATEGORY,
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "train_rows": len(train),
        "val_rows": len(val),
        "n_features": len(feat_cols),
        "best_iteration": model.best_iteration,
        "smape": round(smape_val, 4),
        "wape": round(wape_val, 4),
        "mae": round(mae_val, 4),
        "model_path": str(model_path),
    }

    out = REPORTS_DIR / "lgbm_report.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"LightGBM report → {out}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_lgbm_training()
