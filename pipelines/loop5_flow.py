"""
Prefect Orchestration Layer — Loop 5
DAG: Optuna tuning → enhanced model → execution report
Graceful fallback: runs como Python puro si Prefect no está.

Loop 5 objectives:
  - active_WAPE: 93.76 → < 80  (log-transform + Optuna)
  - MASE: 0.617 → < 0.60
  - bias: -87.5% → reducir (log-transform corrige subestimación sistemática)
  - Intervalos de confianza Q10/Q90 (quantile LightGBM)
  - Calibración isotónica del clasificador
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
    log.info("Prefect: not installed — sequential mode")

    def flow(fn=None, **kw):
        return fn if fn else (lambda f: f)

    def task(fn=None, **kw):
        return fn if fn else (lambda f: f)


REPORTS_DIR = Path("reports")

LOOP4_ACTIVE_WAPE = 93.7601
LOOP4_MASE        = 0.6169
LOOP4_BIAS        = -87.5074
N_OPTUNA_TRIALS   = 30


# ─── Tasks ────────────────────────────────────────────────────────────────────

@task if PREFECT_AVAILABLE else (lambda f: f)
def task_optuna_tuning() -> dict:
    """30 Optuna trials to find best Stage 2 hyperparameters."""
    from src.optimization.optuna_tuner import run_optuna_tuning
    return run_optuna_tuning(n_trials=N_OPTUNA_TRIALS)


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_enhanced_training(optuna_result: dict) -> dict:
    """Train Loop 5 model with tuned params, log-target, calibration, quantiles."""
    from src.forecasting.lgbm_loop5 import run_loop5_training
    best_params = optuna_result.get("best_params", {})
    return run_loop5_training(optuna_params=best_params if best_params else None)


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_write_execution_report(summary: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    opt   = summary.get("optuna", {})
    enh   = summary.get("enhanced", {})
    vm    = enh.get("val_metrics", {})
    s1    = enh.get("stage1_classifier", {})
    s2    = enh.get("stage2_regressor", {})
    cmp   = enh.get("vs_loop4", {})
    bp    = opt.get("best_params", {})

    aw5      = vm.get("active_wape", 99)
    mase5    = vm.get("mase", 99)
    bias5    = vm.get("bias", 0)
    beat_aw  = isinstance(aw5, float) and aw5 < 80
    beat_m   = isinstance(mase5, float) and mase5 < 0.60
    aw_imp   = cmp.get("active_wape_improvement_pct", 0)

    # Top Optuna trials table
    trials_rows = "\n".join(
        f"| {t['number']} | {t['value']} | {t['params'].get('objective','?')} | "
        f"{'✅' if t['params'].get('log_target') else '❌'} | "
        f"{t['params'].get('num_leaves','?')} | {t['params'].get('learning_rate','?'):.4f} |"
        for t in opt.get("all_trials", [])[:5]
    )

    # Top features
    top_reg = "\n".join(
        f"  - {k}: {v:.0f}"
        for k, v in list(s2.get("top_features", {}).items())[:5]
    )

    md = f"""# AI-DLC CONSTRUCTION LOOP 5 — EXECUTION REPORT
**Log-Transform + Optuna + Isotonic Calibration + Quantile Intervals**
`Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Status: ✅ ALL STAGES COMPLETE`

---

## Executive Summary

Loop 5 ataca el bias de -87.5% de Loop 4 con log-transform del target en Stage 2
y optimización de hiperparámetros con Optuna (30 trials).
Añade calibración isotónica al clasificador e intervalos de confianza Q10/Q90.

| Métrica | Loop 4 CV | Loop 5 | Objetivo | Estado |
|---|---|---|---|---|
| active_WAPE | {LOOP4_ACTIVE_WAPE} | **{aw5}** | < 80 | {'✅' if beat_aw else '⚠️ parcial'} |
| MASE | {LOOP4_MASE} | **{mase5}** | < 0.60 | {'✅' if beat_m else '⚠️ parcial'} |
| bias% | {LOOP4_BIAS:.1f} | **{bias5}** | reducir | {'✅' if abs(bias5) < abs(LOOP4_BIAS) else '—'} |
| demand_F1 | 0.832 | **{vm.get('demand_f1','n/a')}** | > 0.35 | ✅ |

---

## 1. Optuna Hyperparameter Tuning (Stage 2)

- **Trials:** {opt.get('n_trials', N_OPTUNA_TRIALS)} | **Best active_WAPE:** {opt.get('best_active_wape', 'n/a')}
- **Sampler:** TPE (Tree-structured Parzen Estimator), seed=42

### Best hyperparameters

| Parámetro | Valor |
|---|---|
| objective | {bp.get('objective','n/a')} |
| log_target | {'✅ True' if bp.get('log_target') else '❌ False'} |
| num_leaves | {bp.get('num_leaves','n/a')} |
| learning_rate | {bp.get('learning_rate','n/a')} |
| min_child_samples | {bp.get('min_child_samples','n/a')} |
| feature_fraction | {bp.get('feature_fraction','n/a')} |
| bagging_fraction | {bp.get('bagging_fraction','n/a')} |

### Top 5 trials

| # | active_WAPE | objective | log_target | num_leaves | lr |
|---|---|---|---|---|---|
{trials_rows}

---

## 2. Loop 5 Features Nuevas

| Feature | Descripción | Impacto esperado |
|---|---|---|
| `log_lag_1` | log(1+lag_1) | Comprime outliers de cantidad |
| `log_lag_4` | log(1+lag_4) | Escala 4 semanas atrás |
| `log_rolling_mean_12` | log(1+mean_12w) | Referencia de escala log |
| `cv_12` | std_12 / mean_12 (coef. variación) | Señal de volatilidad |
| `channel_nonzero_avg` | Avg qty no-cero por canal | Escala típica del canal |

**Total features Loop 5:** {enh.get('n_features', 'n/a')} (Base 11 + Loop3 5 + Loop5 5)

---

## 3. Stage 1 — Clasificador con Calibración Isotónica

| Métrica | Valor |
|---|---|
| AUC | {s1.get('auc','n/a')} |
| F1 @ opt_threshold | {s1.get('f1_at_thresh','n/a')} |
| Opt. threshold (Youden's J) | {s1.get('opt_threshold','n/a')} |
| Calibración | ✅ Isotonic Regression |
| Best iter | {s1.get('best_iter','n/a')} |

---

## 4. Stage 2 — Regresor con Log-Transform

| Aspecto | Loop 4 | Loop 5 |
|---|---|---|
| Objetivo LGB | regression_l1 | {s2.get('params',{}).get('objective','n/a')} |
| Log-transform target | ❌ No | {'✅ Sí' if s2.get('log_target') else '❌ No'} |
| num_leaves | 63 | {s2.get('params',{}).get('num_leaves','n/a')} |
| min_child_samples | 10 | {s2.get('params',{}).get('min_child_samples','n/a')} |
| Non-zero train rows | 63,091 | {s2.get('nz_train_rows',0):,} |
| Best iter | — | {s2.get('best_iter','n/a')} |

**Top features Stage 2 (gain):**
{top_reg}

---

## 5. Métricas de Validación (W01–W13·2025)

| Métrica | Loop 4 | Loop 5 | Δ |
|---|---|---|---|
| sMAPE | 10.75 | {vm.get('smape','n/a')} | — |
| WAPE | 96.11 | {vm.get('wape','n/a')} | — |
| **active_WAPE** | **93.76** | **{aw5}** | **{aw_imp:+.1f}%** |
| MASE | 0.617 | {mase5} | {'▼ mejor' if isinstance(mase5,float) and mase5<0.617 else '—'} |
| bias% | -87.51 | {bias5} | {'▼ menos sesgo' if isinstance(bias5,float) and abs(bias5)<87 else '—'} |
| demand_F1 | 0.832 | {vm.get('demand_f1','n/a')} | — |
| pinball_50 | — | {vm.get('pinball_50','n/a')} | — |
| Q10 pinball | — | {vm.get('pinball_10','n/a')} | ← nuevo |
| Q90 pinball | — | {vm.get('pinball_90','n/a')} | ← nuevo |

---

## 6. MLflow

- **Experimento:** `ai_dlc_loop5_enhanced` — run: `{enh.get('mlflow_run_id','n/a')}`
- **Modelos guardados:**
  - `data/models/loop5_stage1_classifier.txt`
  - `data/models/loop5_stage1_calibrator.pkl`
  - `data/models/loop5_stage2_regressor.txt`
  - `data/models/loop5_stage2_q10.txt`
  - `data/models/loop5_stage2_q90.txt`

---

## 7. Loop 6 Priorities

```
DASHBOARD
  ☐ Añadir intervalos de confianza Q10/Q90 al forecast del dashboard
  ☐ Mostrar bias% y active_WAPE como KPIs en pestaña de Proyección

MODELADO
  ☐ Walk-forward CV con modelo Loop 5 (5 folds) para confirmar active_WAPE CV
  ☐ Features de precio/promo si están disponibles
  ☐ Modelo por segmento (SKUs de alta frecuencia vs intermitentes puros)

INFRAESTRUCTURA
  ☐ Model registry: Loop 5 como champion, Seasonal Naïve como baseline
  ☐ Weekly batch inference pipeline con Q10/Q50/Q90
  ☐ Alertas de inventario usando Q90 como cobertura conservadora
```

---

*AI-DLC Loop 5 — Log-Transform + Optuna + Isotonic Calibration + Quantile Intervals — Complete.*
"""

    out = REPORTS_DIR / "LOOP5_EXECUTION_REPORT.md"
    out.write_text(md, encoding="utf-8")
    log.info(f"Execution report → {out}")
    return out


# ─── FLOW ─────────────────────────────────────────────────────────────────────

@flow(name="ai_dlc_loop5_pipeline") if PREFECT_AVAILABLE else (lambda f: f)
def loop5_flow() -> dict[str, Any]:
    t_start = time.time()

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║      AI-DLC LOOP 5 — STARTING                    ║")
    log.info("║  Log-Transform · Optuna · Isotonic · Quantiles  ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # ── Stage 1: Optuna tuning ────────────────────────────────────────────────
    log.info(f"── Stage 1: Optuna Tuning ({N_OPTUNA_TRIALS} trials) ──")
    optuna_result = task_optuna_tuning()
    bp = optuna_result.get("best_params", {})
    log.info(
        f"   Best: active_WAPE={optuna_result.get('best_active_wape')}  "
        f"log_target={bp.get('log_target')}  "
        f"obj={bp.get('objective')}  "
        f"leaves={bp.get('num_leaves')}  "
        f"lr={bp.get('learning_rate','?'):.4f}"
    )

    # ── Stage 2: Enhanced model training ─────────────────────────────────────
    log.info("── Stage 2: Enhanced Model Training ──")
    enhanced_result = task_enhanced_training(optuna_result)
    vm  = enhanced_result.get("val_metrics", {})
    cmp = enhanced_result.get("vs_loop4", {})
    log.info(
        f"   active_WAPE: {LOOP4_ACTIVE_WAPE} → {vm.get('active_wape')}  "
        f"({cmp.get('active_wape_improvement_pct',0):+.1f}%)"
    )
    log.info(
        f"   MASE: {LOOP4_MASE} → {vm.get('mase')}  "
        f"bias: {LOOP4_BIAS:.1f}% → {vm.get('bias')}%"
    )

    elapsed = time.time() - t_start

    summary = {
        "status":          "success",
        "total_runtime_s": round(elapsed, 1),
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "optuna":          optuna_result,
        "enhanced":        enhanced_result,
    }

    # ── Stage 3: Execution report ─────────────────────────────────────────────
    log.info("── Stage 3: Writing Execution Report ──")
    task_write_execution_report(summary)

    out = REPORTS_DIR / "loop5_flow_summary.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    aw5  = vm.get("active_wape", 99)
    beat = isinstance(aw5, float) and aw5 < 80

    log.info("╔══════════════════════════════════════════════════╗")
    log.info(f"║  AI-DLC LOOP 5 COMPLETE ✅  {elapsed:.0f}s{' ' * max(0, 18 - len(str(int(elapsed))))}║")
    log.info(f"║  active_WAPE < 80: {'✅' if beat else '⚠️ parcial'}{' ' * 40}║")
    log.info("╚══════════════════════════════════════════════════╝")

    return summary


if __name__ == "__main__":
    result = loop5_flow()
