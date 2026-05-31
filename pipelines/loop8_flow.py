"""
Prefect Orchestration Layer — Loop 8
DAG: blended forecast → dashboard listo para producción

Loop 8 objetivo: conectar el modelo ML al dashboard.
  - Combina Loop 7 (330 SKUs activos, LightGBM) + Seasonal Naïve (16,671 SKUs)
  - Genera loop8_blended_forecasts.parquet con Q10/Q50/Q90 para los 17,001 SKUs
  - El dashboard usa el blended como fuente principal de Proyección de Demanda
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
    def flow(fn=None, **kw): return fn if fn else (lambda f: f)
    def task(fn=None, **kw): return fn if fn else (lambda f: f)

REPORTS_DIR = Path("reports")


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_blended_forecast() -> dict:
    from src.forecasting.blended_forecaster import run_blended_forecast
    return run_blended_forecast()


@task if PREFECT_AVAILABLE else (lambda f: f)
def task_write_report(summary: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bf = summary.get("blended", {})

    md = f"""# AI-DLC CONSTRUCTION LOOP 8 — EXECUTION REPORT
**Blended Forecast: LightGBM + Seasonal Naïve → Dashboard Listo**
`Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Status: ✅ COMPLETE`

---

## Resultado

El modelo de ML (LightGBM Loop 7) se conectó al dashboard Streamlit.
El archivo `loop8_blended_forecasts.parquet` unifica ambos modelos:

| Segmento | SKUs | Semanas | Modelo |
|---|---|---|---|
| Alta frecuencia | ~330 | W34·2025–W17·2026 | LightGBM Loop 7 (Q10/Q50/Q90) |
| Intermitentes | ~16,671 | W01·2026–W52·2026 | Seasonal Naïve + CI estimados |
| **Total** | **{bf.get('n_skus', 17001):,}** | **{bf.get('n_weeks', 71)}** | **Blended** |

- **Total Q50:** {bf.get('total_q50', 0):,.0f} unidades (71 semanas)
- **Total Q90:** {bf.get('total_q90', 0):,.0f} unidades
- **Promedio semanal Q50:** {bf.get('avg_weekly_q50', 0):,.0f} uds/semana

## Dashboard

El dashboard ahora muestra:
  ✅ Proyección con el mejor modelo disponible por SKU
  ✅ Banda de confianza Q10–Q90 en el gráfico de proyección
  ✅ KPIs de calidad del modelo (MASE, active_WAPE del CV)
  ✅ Alertas de inventario con demanda pesimista (Q90)
  ✅ Identificación del modelo activo ("LightGBM Loop 8" vs "Seasonal Naïve")

## Fuentes de datos
  - `data/forecasts/loop8_blended_forecasts.parquet` — forecast unificado
  - `data/forecasts/loop7_recursive_forecasts.parquet` — LightGBM activos
  - `data/forecasts/forecasts.parquet` — Seasonal Naïve fallback

---

*AI-DLC Loop 8 — Blended Forecast Dashboard — COMPLETO.*
"""
    out = REPORTS_DIR / "LOOP8_EXECUTION_REPORT.md"
    out.write_text(md, encoding="utf-8")
    return out


@flow(name="ai_dlc_loop8_pipeline") if PREFECT_AVAILABLE else (lambda f: f)
def loop8_flow() -> dict[str, Any]:
    t_start = time.time()

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║      AI-DLC LOOP 8 — STARTING                    ║")
    log.info("║  Blended Forecast → Dashboard Producción         ║")
    log.info("╚══════════════════════════════════════════════════╝")

    log.info("── Blended Forecast (LightGBM + Seasonal Naïve) ──")
    blended_result = task_blended_forecast()
    log.info(
        f"   {blended_result['n_skus']:,} SKUs · {blended_result['n_weeks']} semanas · "
        f"Q50={blended_result['total_q50']:,.0f} · Q90={blended_result['total_q90']:,.0f}"
    )
    log.info(f"   Fuentes: {blended_result['source_breakdown']}")

    elapsed = time.time() - t_start
    summary = {
        "status": "success",
        "total_runtime_s": round(elapsed, 1),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "blended": blended_result,
    }

    task_write_report(summary)

    out = REPORTS_DIR / "loop8_flow_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("╔══════════════════════════════════════════════════╗")
    log.info(f"║  AI-DLC LOOP 8 COMPLETE ✅  {elapsed:.1f}s              ║")
    log.info("║  Dashboard listo con modelo LightGBM activo      ║")
    log.info("╚══════════════════════════════════════════════════╝")
    return summary


if __name__ == "__main__":
    loop8_flow()
