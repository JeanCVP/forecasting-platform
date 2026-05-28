# Monitoring Strategy — v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** MONITOR-ITER3-001

---

## 1. Stack de Observabilidad

```
OBSERVABILIDAD COMPLETA DEL SISTEMA

┌──────────────────────────────────────────────────────────────────────┐
│                     MONITORING STACK                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  SEÑAL 1: PIPELINE HEALTH          Herramienta: Prefect UI           │
│  ¿Completaron todos los flows?     Retención: 90 días               │
│  ¿Cuánto tardaron?                 Alerta: flow FAILED o > 2h       │
│                                                                       │
│  SEÑAL 2: DATA QUALITY             Herramienta: GE + custom          │
│  ¿Pasaron todos los gates?         Retención: Permanente (audit)    │
│  ¿Cuál es el DQ Score?            Alerta: score < 85               │
│                                                                       │
│  SEÑAL 3: FEATURE DRIFT            Herramienta: PSI custom           │
│  ¿Cambiaron las distribuciones?    Retención: 24 meses (parquet)   │
│  ¿Qué features están en rojo?     Alerta: PSI > 0.20              │
│                                                                       │
│  SEÑAL 4: FORECAST ACCURACY        Herramienta: MLflow + custom      │
│  ¿Cuál es el MAPE de esta semana? Retención: Permanente            │
│  ¿Hay bias sistemático?           Alerta: sMAPE > 30% o bias > 15%│
│                                                                       │
│  SEÑAL 5: INVENTORY RISK           Herramienta: custom scorer        │
│  ¿Cuántas series en riesgo?       Retención: 12 meses              │
│  ¿Aumentó el sobrestock?          Alerta: > 25% series en riesgo  │
│                                                                       │
│  SEÑAL 6: MODEL REGISTRY           Herramienta: MLflow               │
│  ¿Cuándo fue el último retrain?   Alerta: > 6 semanas sin retrain  │
│  ¿Champion sigue vigente?                                            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Monitor Flow Principal

```python
# flows/monitor_flow.py

from prefect import flow, task
import polars as pl
import mlflow
from src.monitoring.psi import compute_psi_all_features
from src.monitoring.concept_drift import detect_concept_drift
from src.monitoring.config import (
    MONITORED_FEATURES_PRIMARY, REFERENCE_WINDOW_WEEKS, CURRENT_WINDOW_WEEKS
)
from src.audit.logger import audit
from src.utils import get_week_list_before

@task
def load_monitoring_data(current_week: str) -> dict:
    feature_store = pl.read_parquet("data/gold/feature_store.parquet")
    accuracy_log  = pl.read_parquet("data/gold/accuracy_log.parquet")
    forecast_out  = pl.read_parquet("data/gold/forecast_output.parquet")

    # Reference: last 26 weeks before cutoff
    all_weeks = sorted(feature_store["yearweek"].unique().to_list())
    cw_idx    = all_weeks.index(current_week) if current_week in all_weeks else -1
    ref_weeks = all_weeks[max(0, cw_idx - REFERENCE_WINDOW_WEEKS): cw_idx]
    cur_weeks = all_weeks[max(0, cw_idx - CURRENT_WINDOW_WEEKS):  cw_idx + 1]

    return {
        "feature_store": feature_store,
        "accuracy_log":  accuracy_log,
        "forecast_out":  forecast_out,
        "ref_weeks":     ref_weeks,
        "cur_weeks":     cur_weeks,
    }


@task
def compute_accuracy_metrics(data: dict, current_week: str) -> dict:
    """Compute this week's forecast accuracy vs newly arrived actuals."""
    fc = data["forecast_out"]
    fs = data["feature_store"]

    # Join forecast (H+1 from last week) with actuals (current week)
    h1_fc = fc.filter(
        (pl.col("yearweek") == current_week) &
        (pl.col("horizon_weeks") == 1)
    )
    actuals = fs.filter(pl.col("yearweek") == current_week).select([
        "channel","material","sell_in","segment","product_family"
    ])

    eval_df = h1_fc.join(actuals, on=["channel","material"], how="inner")

    if len(eval_df) == 0:
        return {"error": "No matching forecast-actual pairs"}

    actual   = eval_df["sell_in"].to_numpy()
    forecast = eval_df["sell_in_p50"].to_numpy()

    from src.ml.metrics import smape, bias_pct, hit_rate

    metrics = {
        "yearweek":        current_week,
        "smape_overall":   smape(actual, forecast),
        "bias_pct":        bias_pct(actual, forecast),
        "hit_rate_25pct":  hit_rate(actual, forecast),
        "n_series":        len(eval_df),
    }

    for seg in ["regular","intermittent","rare"]:
        seg_df = eval_df.filter(pl.col("segment") == seg)
        if len(seg_df) > 0:
            metrics[f"smape_{seg}"] = smape(
                seg_df["sell_in"].to_numpy(),
                seg_df["sell_in_p50"].to_numpy()
            )

    for family in eval_df["product_family"].unique().to_list():
        fam_df = eval_df.filter(pl.col("product_family") == family)
        key = family.lower().replace(" ","_").replace(".","")
        metrics[f"smape_family_{key}"] = smape(
            fam_df["sell_in"].to_numpy(),
            fam_df["sell_in_p50"].to_numpy()
        )

    # Append to accuracy log
    new_row = pl.DataFrame([metrics])
    try:
        existing = pl.read_parquet("data/gold/accuracy_log.parquet")
        updated  = pl.concat([existing, new_row])
    except FileNotFoundError:
        updated = new_row
    updated.write_parquet("data/gold/accuracy_log.parquet")

    # Log to MLflow monitoring experiment
    mlflow.set_experiment("forecast_monitoring")
    with mlflow.start_run(run_name=f"monitor_{current_week}"):
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if isinstance(v, float)})
        mlflow.set_tag("yearweek", current_week)

    return metrics


@task
def compute_drift_signals(data: dict, current_week: str) -> dict:
    """Compute PSI for all monitored features."""
    psi_results = compute_psi_all_features(
        feature_store=data["feature_store"],
        reference_weeks=data["ref_weeks"],
        current_weeks=data["cur_weeks"],
        monitored_features=MONITORED_FEATURES_PRIMARY,
    )

    # Save to drift log
    psi_results = psi_results.with_columns([
        pl.lit(current_week).alias("yearweek"),
        pl.lit("data").alias("drift_type"),
        pl.now().alias("computed_at"),
    ])
    try:
        existing = pl.read_parquet("data/gold/drift_log.parquet")
        updated  = pl.concat([existing, psi_results])
    except FileNotFoundError:
        updated = psi_results
    updated.write_parquet("data/gold/drift_log.parquet")

    return psi_results.to_dicts()


@task
def evaluate_alerts(
    accuracy_metrics: dict,
    drift_signals: list,
    thresholds: dict,
) -> dict:
    alerts = {"critical": [], "warning": [], "info": []}

    # Accuracy alerts
    smape = accuracy_metrics.get("smape_overall", 0)
    bias  = abs(accuracy_metrics.get("bias_pct", 0))

    if smape > thresholds["smape_alert_regular"]:
        alerts["critical"].append(
            f"🔴 sMAPE = {smape:.1f}% > {thresholds['smape_alert_regular']}%")
    elif smape > thresholds["smape_alert_regular"] * 0.7:
        alerts["warning"].append(
            f"🟡 sMAPE = {smape:.1f}% (approaching threshold)")

    if bias > thresholds["bias_alert_pct"]:
        alerts["critical"].append(
            f"🔴 Bias = {bias:.1f}% > {thresholds['bias_alert_pct']}%")

    # Drift alerts
    red_features    = [d["feature"] for d in drift_signals if d["status"] == "red"]
    yellow_features = [d["feature"] for d in drift_signals if d["status"] == "yellow"]

    if red_features:
        alerts["critical"].append(f"🔴 Feature drift RED: {red_features}")
    if yellow_features:
        alerts["warning"].append(f"🟡 Feature drift YELLOW: {yellow_features}")

    # Determine retraining action
    n_critical = len(alerts["critical"])
    n_warning  = len(alerts["warning"])

    if n_critical >= 1:
        action = "RETRAIN_IMMEDIATE"
    elif n_warning >= 3:
        action = "RETRAIN_SCHEDULED"
    else:
        action = "MONITOR_ONLY"

    return {
        "alerts":   alerts,
        "n_critical": n_critical,
        "n_warning":  n_warning,
        "action":   action,
    }


@task
def send_weekly_report(
    accuracy_metrics: dict,
    drift_signals: list,
    alert_result: dict,
    current_week: str,
):
    psi_summary = {d["feature"]: d["status"] for d in drift_signals}
    red_count    = sum(1 for s in psi_summary.values() if s == "red")
    yellow_count = sum(1 for s in psi_summary.values() if s == "yellow")

    report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI-DLC Weekly Forecast Report — W{current_week}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 FORECAST ACCURACY (vs actuals this week)
  sMAPE Overall:      {accuracy_metrics.get('smape_overall', 0):.1f}%  {'✅' if accuracy_metrics.get('smape_overall',99) < 20 else '🟡' if accuracy_metrics.get('smape_overall',99) < 30 else '🔴'}
  sMAPE Regular:      {accuracy_metrics.get('smape_regular', 0):.1f}%
  sMAPE Intermittent: {accuracy_metrics.get('smape_intermittent', 0):.1f}%
  Bias:               {accuracy_metrics.get('bias_pct', 0):+.1f}%  {'✅' if abs(accuracy_metrics.get('bias_pct',99)) < 8 else '🔴'}
  Hit Rate (±25%):    {accuracy_metrics.get('hit_rate_25pct', 0):.1f}%
  Series evaluated:   {accuracy_metrics.get('n_series', 0)}

📉 DRIFT SIGNALS
  🟢 Green features:   {sum(1 for s in psi_summary.values() if s=='green')}
  🟡 Yellow features:  {yellow_count}
  🔴 Red features:     {red_count}
  {('Red features: ' + str([d['feature'] for d in drift_signals if d['status']=='red'])) if red_count > 0 else ''}

⚠️  ALERTS ({len(alert_result['alerts']['critical'])} critical, {len(alert_result['alerts']['warning'])} warning)
{chr(10).join(alert_result['alerts']['critical'] + alert_result['alerts']['warning']) or '  None'}

🔁 RECOMMENDED ACTION: {alert_result['action']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """.strip()

    print(report)
    # Production: send via email/Slack
    return report


@flow(name="monitor_flow", log_prints=True)
def monitor_flow(current_week: str = None):
    from src.utils import get_current_week
    if current_week is None:
        current_week = get_current_week()

    params = load_params("params.yaml")
    data   = load_monitoring_data(current_week)

    accuracy_metrics = compute_accuracy_metrics(data, current_week)
    drift_signals    = compute_drift_signals(data, current_week)
    alert_result     = evaluate_alerts(
        accuracy_metrics, drift_signals, params["monitoring"]
    )

    send_weekly_report(accuracy_metrics, drift_signals, alert_result, current_week)

    # Trigger retraining if needed
    if alert_result["action"] in ("RETRAIN_IMMEDIATE", "RETRAIN_SCHEDULED"):
        from flows.train_flow import train_flow
        train_flow(
            force=True,
            reason=f"monitoring_alert_w{current_week}",
        )

    audit.log_pipeline_step(
        event_type="MONITORING",
        event_subtype="WEEKLY_MONITOR_COMPLETE",
        input_path="data/gold/forecast_output.parquet",
        output_path="data/gold/accuracy_log.parquet",
        row_count_in=data["forecast_out"].height,
        row_count_out=1,
        status="SUCCESS",
        params_json=f'{{"action": "{alert_result[\"action\"]}", "week": "{current_week}"}}'
    )

    return alert_result
```

---

## 3. Umbrales Consolidados

| Señal | Métrica | 🟢 Verde | 🟡 Warning | 🔴 Crítico |
|---|---|---|---|---|
| Accuracy | sMAPE regular | < 20% | 20–30% | > 30% |
| Accuracy | sMAPE intermittent | < 35% | 35–50% | > 50% |
| Accuracy | Bias | ±8% | ±8–15% | > ±15% |
| Accuracy | Hit rate ±25% | > 65% | 55–65% | < 55% |
| Data drift | PSI max | < 0.10 | 0.10–0.20 | > 0.20 |
| Concept drift | MAPE delta 4w | < 2pp | 2–5pp | > 5pp |
| Data quality | DQ Score | > 90 | 80–90 | < 80 |
| Pipeline | Flow duration | < 1.5h | 1.5–2h | > 2h / FAILED |
| Inventory | % series en riesgo | < 10% | 10–25% | > 25% |
| Model | Semanas sin retrain | < 4 | 4–6 | > 6 |
| Coverage | % series con forecast | 100% | 99–100% | < 99% |

---

## 4. Monitoring Parquet Schemas

```sql
-- accuracy_log.parquet
CREATE TABLE accuracy_log (
    yearweek              CHAR(6) PRIMARY KEY,
    computed_at           TIMESTAMP,
    smape_overall         FLOAT,
    smape_regular         FLOAT,
    smape_intermittent    FLOAT,
    smape_rare            FLOAT,
    bias_pct              FLOAT,
    hit_rate_25pct        FLOAT,
    n_series              INT,
    model_version         VARCHAR(50),
    action_triggered      VARCHAR(30)
);

-- drift_log.parquet
CREATE TABLE drift_log (
    yearweek              CHAR(6),
    computed_at           TIMESTAMP,
    drift_type            VARCHAR(20),
    feature               VARCHAR(100),
    psi                   FLOAT,
    status                VARCHAR(10),
    ref_mean              FLOAT,
    cur_mean              FLOAT,
    mean_shift_pct        FLOAT,
    action_triggered      VARCHAR(30),
    PRIMARY KEY (yearweek, drift_type, feature)
);
```

---

## 5. Reporte Semanal Automatizado

El `monitor_flow` genera y distribuye automáticamente cada lunes a las 10:30 COT:

- **Para el equipo de Demand Planning:** sMAPE, bias, hit rate, top series con mayor error
- **Para Supply Chain:** Inventory risk summary, series en riesgo crítico
- **Para el equipo ML:** Drift signals detallados, feature PSI, concept drift trend
- **Para dirección:** Executive KPIs (1 párrafo + 3 métricas clave)

Formato: Email HTML + notificación Slack + parquet log para historización.

---

*AI-DLC Traceability ID: MONITOR-ITER3-001 | Version: 3.0*
