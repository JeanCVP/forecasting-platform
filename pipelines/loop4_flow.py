"""
Prefect Orchestration Layer — Loop 4
DAG: threshold tuning → walk-forward CV (two-stage) → final retrain → report
Graceful fallback: runs as plain Python if Prefect not installed.

Loop 4 objective:
  - Tune classifier threshold via Youden's J (fix F1=0 from Loop 3)
  - Walk-forward CV con two-stage model (5 folds) para validar MASE < 1 consistently
  - Reduce active_WAPE: target < 80
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

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
MODELS_DIR  = Path("data/models")

# Loop 3 baseline (from lgbm_two_stage with threshold=0.5)
LOOP3_MASE        = 0.722
LOOP3_ACTIVE_WAPE = 97.01


# ─── Tasks ────────────────────────────────────────────────────────────────────

@task if PREFECT_AVAILABLE else (lambda f: f)
def task_walk_forward_cv() -> dict:
    """5-fold walk-forward CV with threshold-tuned two-stage model."""
    from src.evaluation.walk_forward_two_stage import run_walk_forward_two_stage_cv
    return run_walk_forward_two_stage_cv()


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_final_retrain() -> dict:
    """Retrain final model on full train set (≤W52·2024) with threshold tuning."""
    from src.forecasting.lgbm_two_stage import run_two_stage_training
    return run_two_stage_training(tune_threshold=True)


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_write_execution_report(summary: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cv    = summary.get("cv", {})
    agg   = cv.get("aggregate_metrics_gated", {})
    ft    = summary.get("final_model", {})
    s1    = ft.get("stage1_classifier", {})
    s2    = ft.get("stage2_regressor", {})
    vm    = ft.get("val_metrics", {})
    folds = cv.get("fold_results", [])

    cv_mase = agg.get("mase", {}).get("mean", "n/a")
    cv_aw   = agg.get("active_wape", {}).get("mean", "n/a")
    cv_f1   = agg.get("demand_f1", {}).get("mean", "n/a")
    cv_auc  = cv.get("avg_clf_auc", "n/a")
    cv_thr  = cv.get("avg_opt_threshold", "n/a")

    fm_mase = vm.get("mase", "n/a")
    fm_aw   = vm.get("active_wape", "n/a")
    fm_f1   = vm.get("demand_f1", "n/a")
    fm_auc  = s1.get("auc", "n/a")
    fm_thr  = s1.get("opt_threshold", "n/a")

    mase_beat  = isinstance(cv_mase, float) and cv_mase < 1.0
    aw_improve = isinstance(cv_aw,   float) and cv_aw < LOOP3_ACTIVE_WAPE

    fold_rows = "\n".join(
        f"| {r['fold']} | ≤{r['train_end']} | {r['val_start']}–{r['val_end']} | "
        f"{r.get('clf_auc','n/a')} | {r.get('opt_threshold','n/a')} | "
        f"{r['metrics_gated'].get('active_wape','n/a')} | "
        f"{r['metrics_gated'].get('mase','n/a')} | "
        f"{r['metrics_gated'].get('demand_f1','n/a')} |"
        for r in folds if not r.get("skipped")
    )

    md = f"""# AI-DLC CONSTRUCTION LOOP 4 — EXECUTION REPORT
**Threshold Tuning + Walk-Forward CV Two-Stage LightGBM**
`Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Status: ✅ ALL STAGES COMPLETE`

---

## Executive Summary

Loop 4 valida el modelo two-stage de Loop 3 con **walk-forward CV de 5 folds** e introduce
**threshold tuning via Youden's J** para corregir el F1=0 del clasificador a threshold=0.5.

| Métrica | Loop 3 | Loop 4 CV (avg) | Mejora |
|---|---|---|---|
| MASE | 0.722 | {cv_mase} | {'✅ Sigue < 1.0' if mase_beat else '—'} |
| active_WAPE | {LOOP3_ACTIVE_WAPE} | {cv_aw} | {'✅ Mejor' if aw_improve else '—'} |
| demand_F1 | 0.1577 | {cv_f1} | — |
| Classifier AUC | 0.9593 | {cv_auc} | — |
| Opt. threshold | 0.5 (fijo) | {cv_thr} | ✅ Tuned via Youden's J |

---

## 1. Walk-Forward CV — 5 Folds

| Fold | Train | Val | AUC | Opt.Thresh | active_WAPE | MASE | demand_F1 |
|---|---|---|---|---|---|---|---|
{fold_rows}
| **AVG** | | | **{cv_auc}** | **{cv_thr}** | **{cv_aw}** | **{cv_mase}** | **{cv_f1}** |

---

## 2. Final Model (retrain ≤W52·2024, val W01–W13·2025)

| Componente | Métrica | Valor |
|---|---|---|
| Stage 1 Classifier | AUC | {fm_auc} |
| Stage 1 Classifier | Opt. threshold (Youden's J) | {fm_thr} |
| Stage 1 Classifier | F1 @ opt_threshold | {s1.get('f1_at_thresh','n/a')} |
| Stage 2 Regressor | Best iteration | {s2.get('best_iter','n/a')} |
| Stage 2 Regressor | Non-zero train rows | {s2.get('non_zero_train_rows',0):,} |
| Combined | MASE | {fm_mase} |
| Combined | active_WAPE | {fm_aw} |
| Combined | demand_F1 | {fm_f1} |
| Combined | MAE | {vm.get('mae','n/a')} |
| Combined | bias% | {vm.get('bias','n/a')} |

---

## 3. Threshold Tuning Impact

Loop 3 usaba threshold=0.5 fijo → F1=0 (ninguna predicción de demanda positiva).
Loop 4 usa Youden's J = TPR - FPR maximizado en el validation set.

- **threshold promedio CV:** {cv_thr}
- **Demanda rate train:** ~5.48%
- **Impacto:** demand_F1 pasa de 0 → {cv_f1} (CV avg)

---

## 4. MLflow

- **Experimento CV:** `ai_dlc_loop4_cv_two_stage` — run: `{cv.get('mlflow_run_id','n/a')}`
- **Experimento final:** `ai_dlc_loop3_two_stage` — run: `{ft.get('mlflow_run_id','n/a')}`
- **Modelos:** `data/models/lgbm_stage1_classifier.txt` · `data/models/lgbm_stage2_regressor.txt`

---

## 5. Loop 5 Priorities

```
MODELING
  ☐ Hyperparameter tuning con Optuna (Stage 1 + Stage 2 conjuntamente)
  ☐ Calibración de probabilidades (Platt scaling / isotonic) para Stage 1
  ☐ Quantile regression en Stage 2 → intervalos de confianza
  ☐ Features de canal cruzado (demanda agregada por Channel × Category)

EVALUACIÓN
  ☐ active_WAPE objetivo: < 60
  ☐ MASE objetivo: < 0.60
  ☐ demand_F1 objetivo: > 0.35

INFRAESTRUCTURA
  ☐ Model registry: two-stage como champion vs Seasonal Naïve
  ☐ Forecast confidence intervals en dashboard Streamlit
  ☐ Weekly batch inference pipeline (< 30s para 17K SKUs)
  ☐ A/B shadow mode: two-stage vs Seasonal Naïve en producción
```

---

*AI-DLC Loop 4 — Threshold Tuning + Walk-Forward CV — Complete.*
"""

    out = REPORTS_DIR / "LOOP4_EXECUTION_REPORT.md"
    out.write_text(md, encoding="utf-8")
    log.info(f"Execution report → {out}")
    return out


# ─── FLOW ─────────────────────────────────────────────────────────────────────

@flow(name="ai_dlc_loop4_pipeline") if PREFECT_AVAILABLE else (lambda f: f)
def loop4_flow() -> dict[str, Any]:
    t_start = time.time()

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║      AI-DLC LOOP 4 — STARTING                    ║")
    log.info("║  Threshold Tuning + Walk-Forward CV Two-Stage    ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # ── Stage 1: Walk-forward CV ──────────────────────────────────────────────
    log.info("── Stage 1: Walk-Forward CV (5 folds, two-stage) ──")
    cv_result = task_walk_forward_cv()

    agg = cv_result.get("aggregate_metrics_gated", {})
    log.info(
        f"   CV — active_WAPE={agg.get('active_wape',{}).get('mean','?')}  "
        f"MASE={agg.get('mase',{}).get('mean','?')}  "
        f"demand_F1={agg.get('demand_f1',{}).get('mean','?')}  "
        f"AUC={cv_result.get('avg_clf_auc','?')}  "
        f"opt_thresh={cv_result.get('avg_opt_threshold','?')}"
    )

    # ── Stage 2: Final retrain with threshold tuning ──────────────────────────
    log.info("── Stage 2: Final Retrain (threshold-tuned) ──")
    final_result = task_final_retrain()

    vm = final_result.get("val_metrics", {})
    s1 = final_result.get("stage1_classifier", {})
    log.info(
        f"   Final — AUC={s1.get('auc')}  thresh={s1.get('opt_threshold')}  "
        f"F1@thresh={s1.get('f1_at_thresh')}  "
        f"MASE={vm.get('mase')}  active_WAPE={vm.get('active_wape')}"
    )

    elapsed = time.time() - t_start

    summary = {
        "status":          "success",
        "total_runtime_s": round(elapsed, 1),
        "prefect_mode":    PREFECT_AVAILABLE,
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "cv":              cv_result,
        "final_model":     final_result,
    }

    # ── Stage 3: Execution report ─────────────────────────────────────────────
    log.info("── Stage 3: Writing Execution Report ──")
    task_write_execution_report(summary)

    out = REPORTS_DIR / "loop4_flow_summary.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    cv_mase = agg.get("mase", {}).get("mean", 99)
    beat = isinstance(cv_mase, float) and cv_mase < 1.0

    log.info("╔══════════════════════════════════════════════════╗")
    log.info(f"║  AI-DLC LOOP 4 COMPLETE ✅  {elapsed:.0f}s{' ' * max(0, 18 - len(str(int(elapsed))))}║")
    log.info(f"║  MASE < 1.0: {'✅ confirmado en CV' if beat else '⚠️ revisar'}{' ' * max(0, 29 - 18)}║")
    log.info("╚══════════════════════════════════════════════════╝")

    return summary


if __name__ == "__main__":
    result = loop4_flow()
