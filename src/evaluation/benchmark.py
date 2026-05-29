"""
Benchmark Report — aggregates walk-forward CV results and identifies the best baseline.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CV_RESULTS_PATH = Path("data/benchmarks/cv_raw_results.csv")
REPORTS_DIR = Path("reports")
RANK_METRIC = "smape"


def generate_benchmark_report() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CV_RESULTS_PATH.exists():
        log.warning(f"CV results not found: {CV_RESULTS_PATH}")
        return {"status": "skipped", "reason": "cv_raw_results.csv not found"}

    df = pd.read_csv(CV_RESULTS_PATH)
    log.info(f"Loaded CV results: {len(df)} rows, {df['model'].nunique()} models, {df['fold'].nunique()} folds")

    # Aggregate metrics across folds (mean)
    agg = (
        df.groupby("model")[["smape", "wape", "mase", "bias", "slp", "mae", "rmse"]]
        .mean()
        .round(4)
        .reset_index()
        .sort_values(RANK_METRIC)
    )

    best_row = agg.iloc[0]
    best_model = str(best_row["model"])
    best_smape = float(best_row[RANK_METRIC])

    log.info(f"Best model: {best_model} (sMAPE={best_smape:.2f})")
    for _, row in agg.iterrows():
        log.info(f"  {row['model']:30s}  sMAPE={row['smape']:.2f}  WAPE={row['wape']:.2f}  MASE={row['mase']:.3f}")

    model_summary = agg.set_index("model").to_dict(orient="index")

    report = {
        "pipeline": "benchmark_report",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "rank_metric": RANK_METRIC,
        "best_model": best_model,
        "best_smape": best_smape,
        "n_folds": int(df["fold"].nunique()),
        "n_models": int(df["model"].nunique()),
        "model_summary": model_summary,
    }

    out = REPORTS_DIR / "benchmark_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Benchmark report → {out}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_benchmark_report()
