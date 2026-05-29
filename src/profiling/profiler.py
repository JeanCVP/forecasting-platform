"""
Runtime Profiler — measures memory footprint, file sizes, and inference speed.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


def _file_mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 2) if path.exists() else 0.0


def _dataset_stats(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    try:
        import pandas as pd
        df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
        mem_mb = round(df.memory_usage(deep=True).sum() / 1e6, 2)
        return {
            "exists": True,
            "rows": len(df),
            "columns": len(df.columns),
            "file_mb": _file_mb(path),
            "memory_mb": mem_mb,
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def _profile_baseline_inference() -> dict:
    """Measure wall-clock time for one fold of SeasonalNaive over 1000 SKUs."""
    from src.forecasting.baselines import SeasonalNaive
    model = SeasonalNaive(season=52)
    rng = np.random.default_rng(42)
    series_list = [rng.poisson(lam=0.5, size=104).astype(float) for _ in range(1000)]

    t0 = time.perf_counter()
    for s in series_list:
        model.fit_predict(s, horizon=13)
    elapsed = time.perf_counter() - t0

    return {
        "n_skus": 1000,
        "horizon": 13,
        "total_s": round(elapsed, 3),
        "ms_per_sku": round(elapsed / 1000 * 1000, 3),
    }


def run_profiling() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Profiling dataset footprints...")
    datasets = {
        "silver": _dataset_stats(Path("data/silver/silver_dataset.parquet")),
        "gold":   _dataset_stats(Path("data/gold/gold_features.parquet")),
    }

    log.info("Profiling baseline inference speed...")
    inference = _profile_baseline_inference()
    log.info(f"  SeasonalNaive: {inference['ms_per_sku']:.3f} ms/SKU")

    existing_reports = {}
    for name in ["ingestion_report", "dq_report", "feature_report", "validation_report",
                 "benchmark_report", "lgbm_report"]:
        p = REPORTS_DIR / f"{name}.json"
        existing_reports[name] = p.exists()

    result = {
        "pipeline": "profiling",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "baseline_inference_1k_skus": inference,
        "reports_present": existing_reports,
    }

    out = REPORTS_DIR / "profiling_report.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Profiling report → {out}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_profiling()
