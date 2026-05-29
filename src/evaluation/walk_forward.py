"""
Walk-Forward Cross-Validation Engine
Expanding window, 5 folds, strict temporal cutoff, no future leakage.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.forecasting.baselines import ALL_MODELS, BaseForecaster
from src.evaluation.metrics import compute_all_metrics
from src.runtime.tracker import RunTracker

log = logging.getLogger(__name__)

SILVER_PATH = Path("data/silver/silver_dataset.parquet")
BENCHMARKS_DIR = Path("data/benchmarks")
REPORTS_DIR = Path("reports")

HORIZON = 13
MIN_TRAIN_WEEKS = 26
CATEGORY = "Sell-in"
EXPERIMENT = "ai_dlc_loop2_baselines"

FOLDS = [
    {"fold": 1, "train_end": 202352, "val_start": 202401, "val_end": 202413},
    {"fold": 2, "train_end": 202413, "val_start": 202414, "val_end": 202426},
    {"fold": 3, "train_end": 202426, "val_start": 202427, "val_end": 202439},
    {"fold": 4, "train_end": 202439, "val_start": 202440, "val_end": 202452},
    {"fold": 5, "train_end": 202452, "val_start": 202501, "val_end": 202513},
]


def _assert_no_leakage(train_yw: int, val_yw_min: int) -> None:
    if train_yw >= val_yw_min:
        raise RuntimeError(
            f"LEAKAGE DETECTED: train_end={train_yw} >= val_start={val_yw_min}"
        )


def _evaluate_fold(
    df_sellin: pd.DataFrame,
    fold: dict,
    models: dict[str, BaseForecaster],
    tracker: RunTracker,
) -> list[dict]:

    f = fold["fold"]
    train_end = int(fold["train_end"])
    val_start = int(fold["val_start"])
    val_end = int(fold["val_end"])

    _assert_no_leakage(train_end, val_start)

    log.info(f"Fold {f}: train≤{train_end} | val {val_start}–{val_end}")

    df_sellin = df_sellin.copy()

    df_sellin["year_week"] = (
        df_sellin["year_week"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .astype(int)
    )

    train_df = df_sellin[df_sellin["year_week"] <= train_end]
    val_df = df_sellin[
        (df_sellin["year_week"] >= val_start)
        & (df_sellin["year_week"] <= val_end)
    ]

    train_skus = set(zip(train_df["Channel"], train_df["Material Description"]))
    val_skus = set(zip(val_df["Channel"], val_df["Material Description"]))
    common_skus = list(train_skus & val_skus)

    if len(common_skus) == 0:
        log.warning(f"Fold {f}: no common SKUs between train/val")
        return []

    results = []

    for model_name, model in models.items():

        log.info(f"Running model → {model_name}")

        with tracker.start_run(
            experiment=EXPERIMENT,
            run_name=f"fold{f}_{model_name}",
            params={
                "fold": f,
                "model": model_name,
                "horizon": HORIZON,
                "train_end": train_end,
                "val_start": val_start,
                "min_train_weeks": MIN_TRAIN_WEEKS,
            },
            tags={"stage": "walk_forward_cv", "category": CATEGORY},
        ) as run:

            fold_actuals_all = []
            fold_preds_all = []
            model_results = []

            for idx, (ch, mat) in enumerate(common_skus, start=1):

                sku_train = (
                    train_df[
                        (train_df["Channel"] == ch)
                        & (train_df["Material Description"] == mat)
                    ]
                    .sort_values("year_week")["quantity"]
                    .values.astype(float)
                )

                sku_val = (
                    val_df[
                        (val_df["Channel"] == ch)
                        & (val_df["Material Description"] == mat)
                    ]
                    .sort_values("year_week")["quantity"]
                    .values.astype(float)
                )

                if len(sku_train) < MIN_TRAIN_WEEKS:
                    continue
                if len(sku_val) == 0:
                    continue

                try:
                    preds = model.fit_predict(
                        sku_train,
                        horizon=len(sku_val),
                    )

                    preds = np.asarray(preds, dtype=float)

                    if len(preds) != len(sku_val):
                        continue
                    if np.isnan(preds).any() or np.isinf(preds).any():
                        continue

                    fold_actuals_all.append(sku_val)
                    fold_preds_all.append(preds)

                    model_results.append((ch, mat))

                except Exception as e:
                    log.debug(f"[{model_name}] SKU failure ({ch},{mat}) → {e}")
                    continue

            if len(fold_actuals_all) == 0:
                log.warning(f"Fold {f}/{model_name}: no valid SKUs")
                continue

            actuals = np.concatenate(fold_actuals_all)
            preds = np.concatenate(fold_preds_all)

            if actuals.size == 0 or preds.size == 0:
                continue
            if actuals.shape != preds.shape:
                continue

            actuals = np.nan_to_num(actuals)
            preds = np.nan_to_num(preds)

            try:
                metrics = compute_all_metrics(actuals, preds)
            except Exception as e:
                log.exception(f"Metric failure fold={f} model={model_name}: {e}")
                continue

            if not metrics:
                continue

            metrics = {
                k: (0.0 if v is None else v)
                for k, v in metrics.items()
            }

            metrics["n_skus"] = len(model_results)
            metrics["fold"] = f

            run.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )

            results.append({
                "fold": f,
                "model": model_name,
                "train_end": train_end,
                "val_start": val_start,
                "n_skus": len(model_results),
                **{
                    k: round(float(v), 4)
                    for k, v in metrics.items()
                    if k != "fold"
                },
            })

    return results


def run_walk_forward_cv() -> dict[str, Any]:

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    df = pd.read_parquet(SILVER_PATH)
    df_sellin = df[df["Category"] == CATEGORY].copy()

    tracker = RunTracker(use_mlflow=True)

    all_results = []

    for fold in FOLDS:
        log.info(f"Fold {fold['fold']}/5")

        all_results.extend(
            _evaluate_fold(df_sellin, fold, ALL_MODELS, tracker)
        )

    results_df = pd.DataFrame(all_results)

    raw_path = BENCHMARKS_DIR / "cv_raw_results.csv"
    results_df.to_csv(raw_path, index=False)

    return {
        "n_folds": len(FOLDS),
        "raw_results_path": str(raw_path),
        "n_rows": int(len(results_df)) if results_df is not None else 0
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_walk_forward_cv()