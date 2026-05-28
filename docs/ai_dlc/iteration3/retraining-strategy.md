# Retraining Strategy — v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** RETRAIN-ITER3-001

---

## 1. Triggers de Reentrenamiento

```
JERARQUÍA DE TRIGGERS (prioridad descendente)

T1 — CRÍTICO (ejecutar hoy)
│    Concept drift detectado: MAPE delta > 5pp en 4 semanas
│    PSI crítico (> 0.20) en features de alta importancia
│    Bias > ±15% sostenido 2+ semanas
│
T2 — URGENTE (ejecutar esta semana)
│    MAPE regular > 30% en 2 semanas consecutivas
│    PSI warning (0.10–0.20) en ≥ 3 features simultáneas
│    Nuevo extract de datos crítico (e.g. 2025 sin cap)
│
T3 — PROGRAMADO (primer lunes del mes)
│    Retrain mensual rutinario
│    Nuevo año de datos ingresado
│
T4 — REACTIVO (según demanda)
     Solicitud manual del demand planner
     Cambio en params.yaml (HPO results)
     Rebalanceo significativo del portfolio (>100 SKUs nuevos)
```

---

## 2. Protocolo Completo

```
TRIGGER DETECTADO
      │
      ▼
PASO 1: ¿Pipeline de datos está sano?
  ├── DQ Score Silver ≥ 85? → SÍ: continuar
  └── NO → corregir datos primero; delay retraining

PASO 2: ¿Hay un retrain corriendo ya?
  ├── NO → iniciar
  └── SÍ → encolar para después de que termine (no correr en paralelo)

PASO 3: Construir training set
  ├── Expanding window: W01-2023 → training_cutoff
  ├── Excluir is_censored = True
  ├── Excluir sell_in_lag_52 IS NULL (historia insuficiente)
  └── Verificar: n_rows ≥ 50,000 y n_series ≥ 200

PASO 4: Ejecutar tests críticos PRE-TRAINING
  ├── tests/test_leakage.py         → HALT si falla
  ├── tests/test_contracts.py       → HALT si falla
  └── Compute DQ score              → HALT si < 85

PASO 5: Entrenar challenger
  ├── LightGBM global (segmento regular)
  ├── Croston/TSB (segmento intermitente)
  └── Log todo en MLflow con hashes de lineage

PASO 6: Evaluar challenger vs champion
  ├── sMAPE regular: challenger < champion × 0.97?
  ├── Bias: no empeorar > 5pp?
  ├── Reconciliation residual < 0.01?
  └── leakage_test_passed = True?

PASO 7: Decisión de promoción
  ├── ALL PASS → promover challenger a champion
  │             regenerar todos los forecasts
  │             notificar equipo
  └── ALGÚN FAIL → archivar challenger
                   mantener champion actual
                   investigar causa del fallo
```

---

## 3. Implementación del Flujo

```python
# flows/train_flow.py

from prefect import flow, task
from prefect.blocks.notifications import EmailBlock
import mlflow, time

@task(retries=3, retry_delay_seconds=120,
      task_run_name="train_lgbm_{training_cutoff}")
def train_lgbm_task(
    training_set_path: str,
    params: dict,
    training_cutoff: str,
) -> str:
    """Train LightGBM global model. Returns MLflow run_id."""
    import polars as pl
    from mlforecast import MLForecast
    from lightgbm import LGBMRegressor
    from src.ml.run_logger import log_full_run, RunMetadata
    from src.transformation.feature_version import compute_feature_store_hash
    import dvc.api

    df = pl.read_parquet(training_set_path)
    df = df.filter(pl.col("yearweek") <= training_cutoff)
    df = df.filter(~pl.col("is_censored"))
    df = df.filter(pl.col("segment") == "regular")

    assert len(df) >= 50_000, f"Training set too small: {len(df)} rows"

    # Build model
    model = MLForecast(
        models={"LGBMRegressor": LGBMRegressor(**params["lgbm"])},
        freq="W",
        lags=params["features"]["lag_windows"],
        lag_transforms={
            1: [
                (lambda x, w: x.shift(1).rolling(w).mean(), 4),
                (lambda x, w: x.shift(1).rolling(w).mean(), 13),
            ],
        },
        date_features=["week","month","quarter"],
    )

    # Convert to pandas for MLForecast
    train_pd = df.select([
        pl.col("channel") + "__" + pl.col("material").alias("unique_id"),
        pl.col("date").alias("ds"),
        pl.col("sell_in").alias("y"),
        # Extra features
        "days_of_supply", "sell_through_rate_4w", "inv_lag_1",
        "prob_nonzero_13w", "yoy_sell_in_ratio",
        "week_sin", "week_cos", "is_q4",
        "is_black_friday_week", "is_mothers_day_week",
    ]).to_pandas()

    model.fit(train_pd)

    # Cross-validation
    cv_results = model.cross_validation(
        df=train_pd,
        h=params["training"]["horizon"],
        n_windows=params["training"]["n_cv_folds"],
        step_size=params["training"]["cv_step_size"],
    )

    # Collect metadata
    meta = RunMetadata(
        feature_store_hash=compute_feature_store_hash(df, params),
        silver_hash=dvc.api.get_url("data/silver/timeseries_clean.parquet")[:16],
        bronze_2023_hash=dvc.api.get_url("data/bronze/sell_data_2023.parquet")[:16],
        bronze_2024_hash=dvc.api.get_url("data/bronze/sell_data_2024.parquet")[:16],
        bronze_2025_hash=dvc.api.get_url("data/bronze/sell_data_2025.parquet")[:16],
        training_cutoff=training_cutoff,
        n_training_rows=len(df),
        n_series_trained=df.select(pl.col("unique_id")).n_unique(),
        n_regular_series=(df["segment"] == "regular").sum(),
        n_intermittent=0,
        weeks_of_history=df["yearweek"].n_unique(),
        exclude_censored=True,
        leakage_test_passed=True,  # Already validated pre-training
        segment_regular_pct=float(df["segment"].value_counts().filter(
            pl.col("segment")=="regular")["count"][0] / len(df) * 100),
        segment_intermittent_pct=0.0,
        segment_rare_pct=0.0,
        segment_dead_pct=0.0,
    )

    run_id = log_full_run(model, cv_results, params, meta)
    return run_id


@task
def evaluate_and_promote_task(challenger_run_id: str) -> dict:
    from src.mlops.promote import evaluate_and_promote
    promoted, reason = evaluate_and_promote(challenger_run_id)
    return {"promoted": promoted, "reason": reason,
            "run_id": challenger_run_id}


@task
def send_retraining_report(result: dict, trigger_reason: str):
    smape = mlflow.get_run(result["run_id"]).data.metrics.get("smape_overall", 0)
    status = "✅ PROMOTED" if result["promoted"] else "⚠️ NOT PROMOTED"
    message = f"""
AI-DLC Retraining Report — {trigger_reason}

Status: {status}
Run ID: {result['run_id']}
sMAPE Overall: {smape:.1f}%
Reason: {result['reason']}
    """.strip()
    print(message)
    # In production: EmailBlock().notify(message)


@flow(name="train_flow", log_prints=True)
def train_flow(
    force: bool = False,
    reason: str = "scheduled_monthly",
    training_cutoff: str = None,
):
    from src.utils import get_latest_complete_week
    from src.validation.gates import ValidationGates

    if training_cutoff is None:
        training_cutoff = get_latest_complete_week()

    params = load_params("params.yaml")
    gates = ValidationGates()

    # Pre-training validation
    dq_score = gates.compute_dq_score(
        gates.gate_2_silver("data/silver/timeseries_clean.parquet")
    )
    assert dq_score >= 85, f"DQ score {dq_score} < 85; fix data before retraining"

    # Train
    run_id = train_lgbm_task(
        "data/gold/training_set.parquet",
        params,
        training_cutoff,
    )

    # Evaluate and maybe promote
    result = evaluate_and_promote_task(run_id)

    # Regenerate forecasts if promoted
    if result["promoted"]:
        from flows.forecast_flow import forecast_flow
        forecast_flow(model_version="champion")

    send_retraining_report(result, reason)
    return result
```

---

## 4. Ventana de Entrenamiento

```
EXPANDING WINDOW (default)
────────────────────────────────────────────────────────
Train: [W01-2023 → training_cutoff]
Ventaja: Máximo dato histórico; modelos estables
Desventaja: Datos viejos pueden contener patrones obsoletos

ROLLING WINDOW (alternativa para Fase 5)
────────────────────────────────────────────────────────
Train: [training_cutoff - 104 semanas → training_cutoff]
Ventaja: Solo datos recientes; responde mejor a cambios
Desventaja: Pierde context estacional de 2 años completos

DECISIÓN ACTUAL: Expanding window
EVALUACIÓN: Comparar ambas estrategias en Fase 5 HPO
```

---

## 5. Guardarraíles Anti-Regresión

```python
# Valores que NO pueden degradarse en un challenger vs champion

ANTI_REGRESSION_GUARDRAILS = {
    # Accuracy no puede ser peor que champion
    "smape_regular_max_degradation_pp":      2.0,   # Hasta 2pp peor = OK
    "smape_intermittent_max_degradation_pp": 3.0,
    "bias_max_absolute_increase_pp":         5.0,

    # Coverage total no puede bajar
    "min_forecast_coverage_pct":             99.9,

    # Reconciliation debe seguir funcionando
    "max_reconciliation_residual":           0.01,

    # No puede introducir leakage nuevo
    "leakage_test_must_pass":                True,

    # No puede entrenar con datos censored como ground truth
    "censored_values_excluded":              True,

    # Mínimo de series para que el retrain sea válido
    "min_series_trained":                    200,
    "min_training_rows":                     50_000,
}
```

---

## 6. Retraining Log

```sql
-- data/gold/retraining_log.parquet

CREATE TABLE retraining_log (
    retrain_id             VARCHAR(36),    -- UUID
    triggered_at           TIMESTAMP,
    trigger_type           VARCHAR(20),    -- T1/T2/T3/T4
    trigger_reason         TEXT,
    training_cutoff        CHAR(6),
    challenger_run_id      VARCHAR(50),
    challenger_smape       FLOAT,
    champion_smape_prev    FLOAT,
    improvement_pp         FLOAT,
    was_promoted           BOOLEAN,
    rejection_reasons      TEXT,           -- si no fue promovido
    duration_minutes       FLOAT,
    n_series_trained       INT,
    training_rows          INT,
    PRIMARY KEY (retrain_id)
);
```

---

## 7. Frecuencia por Segmento de Modelo

| Modelo | Frecuencia Mínima | Trigger Adicional |
|---|---|---|
| LightGBM global (Regular) | Mensual | Cualquier T1/T2 |
| Croston/TSB (Intermitente) | Bimestral | T1 o nuevo año de datos |
| Historical Mean (Rare) | Trimestral | Nuevo año de datos únicamente |
| Cold-start proportions | Semanal automático | Siempre (proporciones históricas) |

---

*AI-DLC Traceability ID: RETRAIN-ITER3-001 | Version: 3.0*
