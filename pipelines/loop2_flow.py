"""
Prefect Orchestration Layer — Loop 2
Full DAG: validate → baselines (5 models) → walk-forward CV → benchmark → LightGBM
Graceful fallback: runs as plain Python if Prefect not installed.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

# ─── Prefect import with fallback ─────────────────────────────────────────────
try:
    from prefect import flow, task
    PREFECT_AVAILABLE = True
    log.info("Prefect: AVAILABLE")
except ImportError:
    PREFECT_AVAILABLE = False
    log.info("Prefect: not installed — running in sequential mode")

    def flow(fn=None, **kw):
        return fn if fn else (lambda f: f)

    def task(fn=None, **kw):
        return fn if fn else (lambda f: f)


REPORTS_DIR = Path("reports")


# ─── DAG ──────────────────────────────────────────────────────────────────────
def _emit_dag_json():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dag = {
        "pipeline": "loop2_training_flow",
        "prefect_available": PREFECT_AVAILABLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out = REPORTS_DIR / "prefect_flow_dag.json"
    with open(out, "w") as f:
        json.dump(dag, f, indent=2)

    log.info(f"DAG saved → {out}")
    return dag


# ─── Tasks ────────────────────────────────────────────────────────────────────
@task if PREFECT_AVAILABLE else (lambda f: f)
def task_validate_runtime():
    from src.orchestration.runtime_validator import validate_runtime
    return validate_runtime()


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_walk_forward_cv():
    from src.evaluation.walk_forward import run_walk_forward_cv
    return run_walk_forward_cv()


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_benchmark_report():
    from src.evaluation.benchmark import generate_benchmark_report
    return generate_benchmark_report()


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_lgbm():
    from src.forecasting.lgbm_model import run_lgbm_training
    return run_lgbm_training()


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_profiling():
    from src.profiling.profiler import run_profiling
    return run_profiling()


# ─── FLOW ─────────────────────────────────────────────────────────────────────
@flow(name="ai_dlc_loop2_pipeline") if PREFECT_AVAILABLE else (lambda f: f)
def loop2_flow() -> dict[str, Any]:

    t_start = time.time()
    _emit_dag_json()

    log.info("╔══════════════════════════════════════════╗")
    log.info("║      AI-DLC LOOP 2 — STARTING            ║")
    log.info("║  Priority: Stability > Accuracy          ║")
    log.info("╚══════════════════════════════════════════╝")

    # ── Stage 1 ─────────────────────────────────────────────
    log.info("── Stage 1: Runtime Validation ──")
    rt_result = task_validate_runtime()

    log.info(f"   Runtime valid: {rt_result.get('all_checks_passed')}")

    # ── Stage 2 ─────────────────────────────────────────────
    log.info("── Stage 2: Walk-Forward CV (5 folds × 5 baselines) ──")

    cv_result = task_walk_forward_cv() or {}

    n_folds = cv_result.get("n_folds", 0)
    n_models = cv_result.get("n_models", 0)
    n_skus = cv_result.get("n_skus_evaluated") or 0

    log.info(
        f"   CV complete: {n_folds} folds, "
        f"{n_models} models, "
        f"{n_skus:,} SKUs"
    )

    # ── Stage 3 ─────────────────────────────────────────────
    log.info("── Stage 3: Benchmark Report ──")
    bench_result = task_benchmark_report() or {}

    log.info(
        f"   Best baseline: {bench_result.get('best_model', 'n/a')} "
        f"(sMAPE={bench_result.get('best_smape', 'n/a')})"
    )

    # ── Stage 4 ─────────────────────────────────────────────
    log.info("── Stage 4: LightGBM Global Model ──")
    lgbm_result = task_lgbm() or {}

    log.info(f"   LightGBM sMAPE: {lgbm_result.get('smape', 'n/a')}")

    # ── Stage 5 ─────────────────────────────────────────────
    log.info("── Stage 5: Performance Profiling ──")
    prof_result = task_profiling() or {}

    elapsed = time.time() - t_start

    summary = {
        "status": "success",
        "total_runtime_s": round(elapsed, 1),
        "prefect_mode": PREFECT_AVAILABLE,
        "runtime_validation": rt_result,
        "cv_summary": cv_result,
        "benchmark_summary": bench_result,
        "lgbm_summary": lgbm_result,
        "profiling_summary": prof_result,
    }

    out = REPORTS_DIR / "loop2_flow_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("╔══════════════════════════════════════════╗")
    log.info(f"║  AI-DLC LOOP 2 COMPLETE ✅ {elapsed:.0f}s        ║")
    log.info("╚══════════════════════════════════════════╝")

    return summary


if __name__ == "__main__":
    result = loop2_flow()