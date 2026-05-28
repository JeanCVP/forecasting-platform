# MLOps Architecture — v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** MLOPS-ARCH-ITER3-001

---

## 1. Arquitectura End-to-End

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      MLOps ARCHITECTURE — ITER3                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  SOURCE: ERP Export (weekly CSV)                                 │    │
│  └──────────────────────────┬─────────────────────────────────────┘    │
│                              │                                           │
│  ┌───────────────────────────▼──────────────────────────────────────┐   │
│  │  ORCHESTRATION: Prefect 2.x                                      │   │
│  │                                                                  │   │
│  │  ingest_flow ──▶ clean_flow ──▶ feature_flow                    │   │
│  │                                     │                            │   │
│  │                         ┌───────────┤                            │   │
│  │                         ▼           ▼                            │   │
│  │                   forecast_flow  train_flow (si trigger)         │   │
│  │                         │           │                            │   │
│  │                         └─────┬─────┘                            │   │
│  │                               ▼                                  │   │
│  │                         monitor_flow                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐   │
│  │  DVC             │  │  MLflow           │  │  Audit Log            │   │
│  │  Dataset Version │  │  Experiment Track │  │  (Parquet append-only)│   │
│  │  Pipeline Repro  │  │  Model Registry   │  │                       │   │
│  └─────────────────┘  └──────────────────┘  └───────────────────────┘   │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  VALIDATION & QUALITY GATES (Great Expectations)                 │   │
│  │  Gate-0 (Source) ──▶ Gate-1 (Bronze) ──▶ Gate-2 (Silver)        │   │
│  │  Gate-3 (Gold/Leakage) ──▶ Gate-4 (Forecast sanity)             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  MONITORING: PSI Drift + MAPE Tracking + Alert System            │   │
│  │  Weekly accuracy report → Email/Slack                            │   │
│  │  Retraining trigger → train_flow (automated)                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  BI SERVING: Streamlit (prototype) → Power BI (production)       │   │
│  │  All via semantic MetricEngine — no inline KPI computation        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Flujos Prefect — Definición Completa

### Flow: `ingest_flow`
```python
# flows/ingest_flow.py
from prefect import flow, task
from prefect.blocks.notifications import EmailBlock
from src.ingestion.ingest import ingest_year
from src.validation.gates import ValidationGates

@task(retries=2, retry_delay_seconds=60,
      task_run_name="ingest_year_{year}")
def ingest_task(year: int) -> dict:
    return ingest_year(f"data/raw/{year}.csv", year)

@task
def validate_bronze(year: int) -> bool:
    gates = ValidationGates()
    result = gates.gate_0_source(f"data/raw/{year}.csv", year)
    if not result:
        raise ValueError(f"Bronze gate failed for year {year}")
    return True

@flow(name="ingest_flow", log_prints=True)
def ingest_flow(years: list[int] = [2023, 2024, 2025]):
    for year in years:
        validate_bronze(year)
        ingest_task(year)
```

### Flow: `weekly_pipeline_flow` (Orquestador maestro)
```python
# flows/weekly_pipeline_flow.py
from prefect import flow
from prefect.deployments import run_deployment

@flow(name="weekly_pipeline", log_prints=True)
def weekly_pipeline_flow():
    """Runs every Monday 08:00 COT."""

    # 1. Ingest (only new files)
    ingest_result = run_deployment(
        "ingest_flow/production",
        parameters={"years": detect_new_files()},
        timeout=600
    )

    # 2. Clean (Silver)
    clean_result = run_deployment("clean_flow/production", timeout=1200)

    # 3. Features (Gold)
    feature_result = run_deployment("feature_flow/production", timeout=1800)

    # 4. Forecast (always)
    forecast_result = run_deployment("forecast_flow/production", timeout=3600)

    # 5. Monitor (after forecast)
    monitor_result = run_deployment(
        "monitor_flow/production",
        parameters={"current_week": get_current_week()},
        timeout=1800
    )

    # 6. Conditional retraining
    if monitor_result.result()["action"] == "RETRAIN_IMMEDIATE":
        run_deployment("train_flow/production",
                      parameters={"force": True, "reason": "monitoring_alert"})
```

---

## 3. Model Lifecycle Management

```
ESTADOS DEL MODELO EN MLFLOW

  STAGING      → "challenger"
  │              Trained, being evaluated
  │              Shadow inference running
  │              Not serving production forecasts
  │
  ├── PASS all gates
  │
  ▼
  PRODUCTION   → "champion"
  │              Serving all forecasts
  │              Monitored weekly
  │              Triggers retraining if MAPE degrades
  │
  ├── MAPE breach or new champion
  │
  ▼
  ARCHIVED
               12-month retention
               Available for rollback
               Not serving forecasts
```

### Promotion Logic

```python
def evaluate_and_promote(
    challenger_run_id: str,
    min_improvement_pct: float = 3.0,
    max_bias_degradation: float = 5.0,
) -> tuple[bool, str]:
    """
    Promote challenger to Production if it passes all gates.
    """
    client = MlflowClient()
    challenger = client.get_run(challenger_run_id)
    cm = challenger.data.metrics

    try:
        champion = client.get_model_version_by_alias("sell_in_lgbm_global","champion")
        champ_run = client.get_run(champion.run_id)
        pm = champ_run.data.metrics
    except Exception:
        # No champion yet — promote automatically
        return True, "no_existing_champion"

    reasons = []

    # Check 1: sMAPE improvement
    smape_improvement = (pm["smape_regular"] - cm["smape_regular"]) / pm["smape_regular"] * 100
    if smape_improvement < min_improvement_pct:
        reasons.append(f"sMAPE improvement {smape_improvement:.1f}% < {min_improvement_pct}%")

    # Check 2: Bias not worse
    bias_delta = abs(cm["bias_pct"]) - abs(pm["bias_pct"])
    if bias_delta > max_bias_degradation:
        reasons.append(f"Bias degraded by {bias_delta:.1f}pp")

    # Check 3: No leakage
    if challenger.data.tags.get("leakage_test_passed") != "True":
        reasons.append("Leakage test not passed")

    # Check 4: Reconciliation valid
    if float(cm.get("reconciliation_residual", 0)) > 0.01:
        reasons.append("Reconciliation residual > 0.01")

    if reasons:
        return False, "; ".join(reasons)

    # Promote
    new_version = client.create_model_version(
        name="sell_in_lgbm_global",
        source=f"runs:/{challenger_run_id}/lgbm_model",
        run_id=challenger_run_id,
    )
    client.set_registered_model_alias("sell_in_lgbm_global", "champion", new_version.version)
    client.set_registered_model_alias("sell_in_lgbm_global", "challenger", None)

    return True, f"promoted_version_{new_version.version}"
```

---

## 4. CI/CD Gates

```yaml
# .github/workflows/ml_ci.yml

name: ML Quality Gates

on: [push, pull_request]

jobs:
  leakage_test:
    name: "🔴 CRITICAL: Leakage Detection"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: pytest tests/test_leakage.py -v --tb=short
      # BLOCKS MERGE if fails

  feature_tests:
    name: "Feature Engineering Tests"
    runs-on: ubuntu-latest
    needs: leakage_test
    steps:
      - run: pytest tests/test_features.py -v --tb=short

  reconciliation_test:
    name: "Hierarchical Reconciliation Test"
    runs-on: ubuntu-latest
    needs: feature_tests
    steps:
      - run: pytest tests/test_reconciliation.py -v --tb=short

  data_contract_test:
    name: "Data Contract Validation"
    runs-on: ubuntu-latest
    needs: leakage_test
    steps:
      - run: pytest tests/test_contracts.py -v --tb=short

  lint:
    name: "Code Quality"
    runs-on: ubuntu-latest
    steps:
      - run: ruff check src/ tests/ dashboards/
      - run: mypy src/ --ignore-missing-imports
```

---

## 5. Entornos

| Entorno | Propósito | Datos | Modelos |
|---|---|---|---|
| **development** | Desarrollo y experimentación | Sample 10% | Cualquier experimento |
| **staging** | Validación pre-producción | 100% histórico | Challengers |
| **production** | Serving semanal | 100% + nuevos datos | Champion únicamente |

```python
# src/config.py

import os

ENV = os.getenv("AIDLC_ENV", "development")

CONFIG = {
    "development": {
        "data_path":    "data/sample/",
        "mlflow_uri":   "sqlite:///mlruns/dev.db",
        "log_level":    "DEBUG",
        "enable_audit": False,
    },
    "staging": {
        "data_path":    "data/",
        "mlflow_uri":   "sqlite:///mlruns/staging.db",
        "log_level":    "INFO",
        "enable_audit": True,
    },
    "production": {
        "data_path":    "data/",
        "mlflow_uri":   "sqlite:///mlruns/prod.db",
        "log_level":    "WARNING",
        "enable_audit": True,
    },
}

def get_config() -> dict:
    return CONFIG[ENV]
```

---

## 6. Makefile Completo

```makefile
# Makefile — shortcuts de desarrollo y operaciones

.PHONY: all ingest clean features train forecast monitor test lint setup

# ── PIPELINE COMPLETO ──────────────────────────────────────────────────
all:
	dvc repro

# ── PASOS INDIVIDUALES ─────────────────────────────────────────────────
ingest:
	python src/ingestion/ingest.py --years 2023 2024 2025

clean:
	python src/transformation/clean.py

features:
	python src/transformation/feature_engineering.py

segment:
	python src/ml/segmentation.py

train:
	python src/ml/train_global.py --force

forecast:
	python src/serving/forecast_runner.py

risk:
	python src/product/inventory_risk_scorer.py

monitor:
	python flows/monitor_flow.py --week $(shell python -c "from src.utils import get_current_week; print(get_current_week())")

# ── TESTS ──────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-critical:
	pytest tests/test_leakage.py tests/test_contracts.py tests/test_reconciliation.py -v

test-leakage:
	pytest tests/test_leakage.py -v

# ── CALIDAD ────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ dashboards/
	mypy src/ --ignore-missing-imports

# ── UI ─────────────────────────────────────────────────────────────────
dashboard:
	streamlit run dashboards/app.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns/prod.db --port 5000

prefect-ui:
	prefect server start

# ── SETUP ──────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	dvc pull --run-cache
	prefect server start &
	mlflow server --backend-store-uri sqlite:///mlruns/prod.db &

# ── ROLLBACK ───────────────────────────────────────────────────────────
rollback-data:
	@echo "Rolling back to dataset version $(VERSION)"
	git checkout dataset-$(VERSION)
	dvc checkout data/

rollback-model:
	@echo "Rolling back champion model"
	python src/ml/rollback_champion.py
```

---

## 7. requirements.txt

```
# Core
polars>=0.20.0
pandas>=2.0.0
duckdb>=0.10.0
pyarrow>=14.0.0
numpy>=1.26.0

# ML
mlforecast>=0.13.0
statsforecast>=1.6.0
lightgbm>=4.0.0
catboost>=1.2.0
hierarchicalforecast>=0.3.0
optuna>=3.5.0

# MLOps
mlflow>=2.10.0
prefect>=2.16.0
dvc>=3.40.0

# Validation
great-expectations>=0.18.0

# BI
streamlit>=1.30.0

# Utils
joblib>=1.3.0
pyyaml>=6.0.0
python-dotenv>=1.0.0
ruff>=0.2.0
mypy>=1.8.0
pytest>=8.0.0
```

---

*AI-DLC Traceability ID: MLOPS-ARCH-ITER3-001 | Version: 3.0*
