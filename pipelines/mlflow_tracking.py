"""
MLflow Foundation (with sqlite-backed local tracking fallback).
Baseline: Seasonal Naïve + Rolling Average.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPORTS_DIR = Path("reports")
GOLD_PATH = Path("data/gold/gold_features.parquet")
GOLD_CSV = Path("data/gold/gold_features.csv")
SILVER_PATH = Path("data/silver/silver_dataset.parquet")
SILVER_CSV = Path("data/silver/silver_dataset.csv")
MLFLOW_DB = Path("mlruns/mlflow_runs.db")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)])


# ─── Lightweight MLflow-compatible tracker (sqlite) ──────────────────────────
class RunTracker:
    """SQLite-backed run tracker when MLflow server unavailable."""

    def __init__(self, db_path: Path = MLFLOW_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, experiment TEXT,
                run_name TEXT, status TEXT,
                start_time TEXT, end_time TEXT,
                tags TEXT, params TEXT, metrics TEXT, artifacts TEXT
            )""")
        self.conn.commit()

    def log_run(self, experiment: str, run_name: str, params: dict,
                metrics: dict, tags: dict, artifact_paths: list[str] = None) -> str:
        run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, experiment, run_name, "FINISHED",
             datetime.now(timezone.utc).isoformat(),
             datetime.now(timezone.utc).isoformat(),
             json.dumps(tags), json.dumps(params),
             json.dumps({k: round(float(v), 4) for k, v in metrics.items()}),
             json.dumps(artifact_paths or []))
        )
        self.conn.commit()
        return run_id

    def get_runs(self, experiment: str) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM runs WHERE experiment=?", (experiment,))
        rows = cur.fetchall()
        cols = ["run_id","experiment","run_name","status","start_time",
                "end_time","tags","params","metrics","artifacts"]
        return [dict(zip(cols, r)) for r in rows]


def _try_mlflow(tracker: RunTracker, experiment: str, run_name: str,
                params: dict, metrics: dict, tags: dict, artifact_path: str) -> str:
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(tags)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if Path(artifact_path).exists():
                mlflow.log_artifact(artifact_path)
        log.info(f"  [MLflow] Run logged: {run.info.run_id}")
        return run.info.run_id
    except Exception as e:
        log.warning(f"  MLflow unavailable ({e}), using local SQLite tracker")
        return tracker.log_run(experiment, run_name, params, metrics, tags, [artifact_path])


# ─── Baselines ────────────────────────────────────────────────────────────────
def seasonal_naive(series: np.ndarray, horizon=13, season=52) -> np.ndarray:
    if len(series) < season:
        return np.full(horizon, series[-1] if len(series) else 0.0)
    return np.array([series[-season + (h % season)] for h in range(horizon)])


def rolling_average(series: np.ndarray, horizon=13, window=12) -> np.ndarray:
    w = min(window, len(series)) if len(series) > 0 else 1
    return np.full(horizon, float(np.mean(series[-w:])) if len(series) else 0.0)


def _metrics(actuals: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    mask = actuals > 0
    if mask.sum() == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "bias": 0.0}
    err = actuals[mask] - preds[mask]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err / actuals[mask]))) * 100,
        "bias": float(np.mean(err)),
    }


def run_baseline_forecast() -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = RunTracker()
    EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "ai_dlc_demand_forecast")

    # Load data
    for p in [GOLD_PATH, GOLD_CSV, SILVER_PATH, SILVER_CSV]:
        if p.exists():
            df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
            log.info(f"Loaded: {p} ({len(df):,} rows)")
            break
    else:
        raise FileNotFoundError("No gold/silver dataset found")

    df_sellin = df[df["Category"] == "Sell-in"].sort_values(
        ["Channel", "Material Description", "year", "week_num"]
    )

    all_metrics: dict[str, list] = {"seasonal_naive": [], "rolling_avg": []}

    skus = df_sellin[["Channel", "Material Description"]].drop_duplicates().head(500)
    log.info(f"Evaluating {len(skus)} SKUs...")

    for _, row in skus.iterrows():
        ts = df_sellin[
            (df_sellin["Channel"] == row["Channel"]) &
            (df_sellin["Material Description"] == row["Material Description"])
        ]["quantity"].values.astype(float)
        if len(ts) < 26:
            continue
        train, test = ts[:-13], ts[-13:]
        all_metrics["seasonal_naive"].append(_metrics(test, seasonal_naive(train)))
        all_metrics["rolling_avg"].append(_metrics(test, rolling_average(train)))

    results: dict[str, Any] = {}
    for model_name, metric_list in all_metrics.items():
        if not metric_list:
            continue
        agg = {k: float(np.mean([m[k] for m in metric_list])) for k in ["mae", "rmse", "mape", "bias"]}
        results[model_name] = agg

        artifact_path = str(REPORTS_DIR / f"baseline_{model_name}.json")
        with open(artifact_path, "w") as f:
            json.dump({"model": model_name, "metrics": agg, "n_skus": len(metric_list)}, f, indent=2)

        run_id = _try_mlflow(
            tracker, EXPERIMENT, f"baseline_{model_name}",
            params={"forecast_horizon": 13, "n_skus": len(metric_list), "category": "Sell-in"},
            metrics=agg,
            tags={"model_type": "baseline", "algorithm": model_name},
            artifact_path=artifact_path,
        )
        log.info(f"[{model_name}] MAE={agg['mae']:.2f} | RMSE={agg['rmse']:.2f} | MAPE={agg['mape']:.1f}% | run_id={run_id}")

    baseline_report = {
        "pipeline": "baseline_forecast",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "models": results,
        "n_skus_evaluated": len(all_metrics.get("seasonal_naive", [])),
        "tracking_backend": "mlflow_or_sqlite",
    }
    with open(REPORTS_DIR / "baseline_report.json", "w") as f:
        json.dump(baseline_report, f, indent=2, default=str)

    return baseline_report


if __name__ == "__main__":
    run_baseline_forecast()
