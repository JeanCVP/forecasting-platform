"""
Prefect Orchestration Layer — Loop 7
DAG: bias correction → recursive forecast (71 semanas Q10/Q50/Q90) → inventory Q90 → reporte
Graceful fallback: runs como Python puro si Prefect no está.

Loop 7 objectives:
  - Forecast recursivo W34·2025 → W52·2026 con intervalos Q10/Q50/Q90
  - Bias correction multiplicativa por segmento (regular 1.88x, sparse 5.0x cap)
  - Alertas de inventario con Q90 como demanda pesimista
  - Reducir active_WAPE adicional vía bias correction en punto de predicción
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
LOOP6_ACTIVE_WAPE = 72.8447
LOOP6_MASE        = 0.4952


# ─── Tasks ────────────────────────────────────────────────────────────────────

@task if PREFECT_AVAILABLE else (lambda f: f)
def task_bias_correction() -> dict:
    """Compute and save per-segment bias correction factors from fold-5 CI data."""
    from src.forecasting.bias_corrector import compute_bias_factors, save_bias_factors
    factors = compute_bias_factors()
    save_bias_factors(factors)
    return factors


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_recursive_forecast() -> dict:
    """Generate recursive Q10/Q50/Q90 forecasts for W34·2025 → W52·2026."""
    from src.forecasting.recursive_forecaster import run_recursive_forecast
    return run_recursive_forecast(horizon=71)


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_inventory_risk() -> dict:
    """Re-run inventory risk scorer using Loop 7 Q90 for conservative alerts."""
    from src.inventory.risk_scorer import run_inventory_risk_scoring
    return run_inventory_risk_scoring()


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_write_execution_report(summary: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    bc   = summary.get("bias_factors", {})
    fc   = summary.get("recursive_forecast", {})
    inv  = summary.get("inventory_risk", {})

    md = f"""# AI-DLC CONSTRUCTION LOOP 7 — EXECUTION REPORT
**Forecast Recursivo Q10/Q50/Q90 + Bias Correction + Inventario Conservador**
`Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Status: ✅ ALL STAGES COMPLETE`

---

## Executive Summary

Loop 7 cierra la brecha entre el modelo de validación (Loop 6) y la producción:
genera **forecasts recursivos** para las 71 semanas futuras (W34·2025→W52·2026)
con **bias correction** por segmento e **intervalos de confianza Q10/Q90**.

| Componente | Resultado |
|---|---|
| Bias correction (regular) | {bc.get('regular', 'n/a')}x |
| Bias correction (sparse) | {bc.get('sparse', 'n/a')}x (capped {bc.get('sparse', 'n/a')} ≤ 5.0) |
| Forecast horizon | {fc.get('forecast_start','n/a')} → {fc.get('forecast_end','n/a')} ({fc.get('horizon_weeks','n/a')} semanas) |
| SKUs con demanda prevista | {fc.get('n_skus_active', 0):,} |
| Total Q50 (71 semanas) | {fc.get('total_q50', 0):,.0f} unidades |
| Avg semanal Q50 | {fc.get('avg_weekly_q50', 0):,.0f} unidades/semana |
| Runtime | {fc.get('elapsed_s', 0)}s |

---

## 1. Bias Correction

Calculado de las predicciones fold-5 (loop6_ci_predictions.parquet):

| Segmento | Criterio | Factor | Interpretación |
|---|---|---|---|
| regular | demand_prob > 0.3 | {bc.get('regular','n/a')}x | Modelo sub-predice ~{round((bc.get('regular',1)-1)*100)}% en alta demanda |
| sparse  | demand_prob ≤ 0.3 | {bc.get('sparse','n/a')}x (cap 5.0) | Modelo sub-predice fuertemente en baja demanda |
| global  | todos | {bc.get('global','n/a')}x | Factor global de referencia |

---

## 2. Forecast Recursivo

- **Modelo:** Loop 5 (Stage 1 + Stage 2 calibrado, opt_threshold=0.0506)
- **Algoritmo:** vectorizado por semana (17K SKUs en batch por semana)
- **Propagación:** Q50 de la semana t se usa como lag_1 para t+1
- **Intervalos:** Q10/Q90 del regresor cuantil Loop 5, NO corregidos por bias
  (representan el rango natural, no el punto central)

### Archivo de salida
- `data/forecasts/loop7_recursive_forecasts.parquet`
- Columnas: Channel, Material Description, year_week, forecast_q10, forecast_q50, forecast_q90

---

## 3. Alertas de Inventario con Q90

El risk scorer ahora usa `avg_weekly_demand_q90` para calcular `weeks_of_supply_conservative`.
Esta métrica refleja el **escenario pesimista** de demanda:

| Antes (Loop 6) | Después (Loop 7) |
|---|---|
| weeks_of_supply = inv / avg_Q50 | weeks_of_supply_conservative = inv / avg_Q90 |
| Risk level basado en Q50 | **Risk level basado en Q90** |
| Puede subestimar necesidad de reposición | ✅ Más conservador, reduce stockouts |

Distribución de riesgo (Loop 7):
- CRITICAL: {inv.get('risk_distribution', {}).get('CRITICAL', 'n/a')}
- HIGH:     {inv.get('risk_distribution', {}).get('HIGH', 'n/a')}
- MEDIUM:   {inv.get('risk_distribution', {}).get('MEDIUM', 'n/a')}
- LOW:      {inv.get('risk_distribution', {}).get('LOW', 'n/a')}

---

## 4. Loop 8 Priorities

```
DASHBOARD
  ☐ Reemplazar gráfico de Proyección con Loop 7 recursive forecasts (Q10/Q50/Q90)
  ☐ Selector de percentil (Q10/Q50/Q90) para planificación de inventario
  ☐ Comparar Loop 7 Q50 vs Seasonal Naïve en el dashboard

MODELADO
  ☐ Calibración de intervalos: verificar cobertura empírica Q10/Q90 en CV
  ☐ Walk-forward CV del modelo recursivo (simular producción real)
  ☐ Modelo por segmento de canal (top 10 canales vs long tail)

INFRAESTRUCTURA
  ☐ Weekly refresh pipeline: gold → features → recursive forecast → alerts
  ☐ Scheduler semanal (Prefect + CronCreate)
```

---

*AI-DLC Loop 7 — Recursive Forecast Q10/Q50/Q90 + Bias Correction + Conservative Inventory — Complete.*
"""

    out = REPORTS_DIR / "LOOP7_EXECUTION_REPORT.md"
    out.write_text(md, encoding="utf-8")
    log.info(f"Execution report → {out}")
    return out


# ─── FLOW ─────────────────────────────────────────────────────────────────────

@flow(name="ai_dlc_loop7_pipeline") if PREFECT_AVAILABLE else (lambda f: f)
def loop7_flow() -> dict[str, Any]:
    t_start = time.time()

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║      AI-DLC LOOP 7 — STARTING                    ║")
    log.info("║  Recursive Forecast Q10/Q50/Q90 + Bias Correct  ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # ── Stage 1: Bias correction ──────────────────────────────────────────────
    log.info("── Stage 1: Bias Correction ──")
    bc_factors = task_bias_correction()
    log.info(f"   Factors — global={bc_factors['global']}  regular={bc_factors['regular']}  sparse={bc_factors['sparse']}")

    # ── Stage 2: Recursive forecast ───────────────────────────────────────────
    log.info("── Stage 2: Recursive Forecast (71 weeks Q10/Q50/Q90) ──")
    fc_result = task_recursive_forecast()
    log.info(
        f"   {fc_result['forecast_start']} → {fc_result['forecast_end']} | "
        f"SKUs={fc_result['n_skus_active']:,} | "
        f"total_Q50={fc_result['total_q50']:,.0f} | "
        f"{fc_result['elapsed_s']}s"
    )

    # ── Stage 3: Inventory risk with Q90 ─────────────────────────────────────
    log.info("── Stage 3: Inventory Risk (Q90 conservative) ──")
    inv_result = task_inventory_risk()
    dist = inv_result.get("risk_distribution", {})
    log.info(f"   CRITICAL={dist.get('CRITICAL',0)}  HIGH={dist.get('HIGH',0)}  MEDIUM={dist.get('MEDIUM',0)}  LOW={dist.get('LOW',0)}")

    elapsed = time.time() - t_start

    summary = {
        "status":          "success",
        "total_runtime_s": round(elapsed, 1),
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "bias_factors":    bc_factors,
        "recursive_forecast": fc_result,
        "inventory_risk":  inv_result,
    }

    # ── Stage 4: Execution report ─────────────────────────────────────────────
    log.info("── Stage 4: Writing Execution Report ──")
    task_write_execution_report(summary)

    out = REPORTS_DIR / "loop7_flow_summary.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("╔══════════════════════════════════════════════════╗")
    log.info(f"║  AI-DLC LOOP 7 COMPLETE ✅  {elapsed:.0f}s{' ' * max(0, 18 - len(str(int(elapsed))))}║")
    log.info(f"║  {fc_result['n_skus_active']:,} SKUs · Q10/Q50/Q90 · bias corrected  ║")
    log.info("╚══════════════════════════════════════════════════╝")

    return summary


if __name__ == "__main__":
    result = loop7_flow()
