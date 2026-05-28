"""Prefect Orchestration — with graceful fallback for no-Prefect env."""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)])

DATA_DIR = os.getenv("DATA_RAW_DIR", "/mnt/project")

try:
    from prefect import flow, task
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    def flow(**kw): return lambda f: f
    def task(**kw): return lambda f: f


@task(name="ingest_bronze", retries=2, retry_delay_seconds=10) if PREFECT_AVAILABLE else lambda f: f
def task_ingest(data_dir=DATA_DIR):
    from src.ingestion.ingest import run_ingestion
    log.info("=== TASK: Bronze Ingestion ===")
    return run_ingestion(data_dir=data_dir)


@task(name="clean_silver") if PREFECT_AVAILABLE else lambda f: f
def task_clean():
    from src.transformation.clean import run_cleaning
    log.info("=== TASK: Silver Cleaning ===")
    df = run_cleaning()
    return len(df)


@task(name="validate_data") if PREFECT_AVAILABLE else lambda f: f
def task_validate():
    from src.validation.validators import run_all_validators
    log.info("=== TASK: Data Validation ===")
    return run_all_validators()


@task(name="feature_engineering") if PREFECT_AVAILABLE else lambda f: f
def task_features():
    from src.transformation.feature_engineering import run_feature_engineering
    log.info("=== TASK: Feature Engineering ===")
    df = run_feature_engineering()
    return len(df)


@task(name="baseline_forecast") if PREFECT_AVAILABLE else lambda f: f
def task_baseline():
    from pipelines.mlflow_tracking import run_baseline_forecast
    log.info("=== TASK: Baseline Forecasting ===")
    return run_baseline_forecast()


def training_flow(data_dir=DATA_DIR):
    log.info("╔══════════════════════════════╗")
    log.info("║  AI-DLC Training Flow Start  ║")
    log.info("╚══════════════════════════════╝")

    ingestion = task_ingest(data_dir=data_dir)
    n_silver = task_clean()
    validation = task_validate()
    n_gold = task_features()
    baseline = task_baseline()

    summary = {
        "status": "success",
        "silver_rows": n_silver,
        "gold_rows": n_gold,
        "validation_passed": validation["overall_passed"],
        "validators_passed": f"{validation['validators_passed']}/{validation['validators_run']}",
        "baseline_models": list(baseline.get("models", {}).keys()),
    }

    log.info("╔══════════════════════════════╗")
    log.info("║  AI-DLC Training Flow Done ✓ ║")
    log.info("╚══════════════════════════════╝")
    log.info(f"Summary: {summary}")
    return summary


if __name__ == "__main__":
    result = training_flow(data_dir=DATA_DIR)
