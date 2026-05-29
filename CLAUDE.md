# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (uses uv with .venv)
uv sync

# Verify environment
python test_env.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_leakage.py

# Run a single test by name
pytest tests/test_leakage.py::test_silver_temporal_ordering

# Lint
ruff check .

# Format
ruff format .

# Run the full Loop 2 pipeline (validate → CV → benchmark → LightGBM → profiling)
python pipelines/loop2_flow.py

# Run the Loop 1 training flow (ingest → clean → validate → features → baseline)
python pipelines/training_flow.py

# Run individual pipeline stages
python src/ingestion/ingest.py --data-dir data/raw
python src/transformation/clean.py
python src/transformation/feature_engineering.py --sample 50000  # fast mode
python src/validation/validators.py

# Start MLflow UI (local SQLite backend)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Start services via Docker (MLflow on :5000, Prefect on :4200)
docker compose up
```

## Architecture

This is a **weekly demand forecasting platform** for SKUs with highly intermittent demand (~1% non-zero rate). The data model is: `(Channel, Material Description, Category) × year_week → quantity`. Three categories exist: `Sell-in`, `Cust. Sales`, and `Channel Inv.`

### Medallion lakehouse: raw → bronze → silver → gold

| Layer | Path | Description |
|---|---|---|
| Raw | `data/raw/{2023,2024,2025}.csv` | Wide-format CSVs: ID cols + week columns |
| Bronze | `data/bronze/bronze_{year}.parquet` | Raw parquet, no transforms |
| Silver | `data/silver/silver_dataset.parquet` | Long-format (melted), cleaned, deduped |
| Gold | `data/gold/gold_features.parquet` | All lag/rolling/seasonal features added |
| Benchmarks | `data/benchmarks/cv_raw_results.csv` | Walk-forward CV results |

### Pipeline stages (`src/`)

- **`src/ingestion/ingest.py`** — reads raw CSVs as strings (preserving comma-formatted numerics), validates schema, saves bronze parquet, writes `reports/ingestion_report.json` and `reports/schema_validation.json`.
- **`src/transformation/clean.py`** — melts wide→long, fixes comma-formatted numbers, deduplicates by summing, detects truncation and anomalies (z-score > 4σ by channel/category), saves silver, writes `reports/dq_report.json` and `reports/anomaly_report.json`.
- **`src/transformation/feature_engineering.py`** — computes per-SKU lag features (lag_1/4/52), rolling stats (mean_4/12, std_12), `weeks_since_last_sale`, `intermittent_flag`, seasonal sin/cos encoding, and inventory days-of-supply (joining Sell-in ÷ Channel Inv.). **All features are strictly shifted (no leakage)**. Saves gold, writes `reports/feature_report.json`.
- **`src/validation/validators.py`** — 7 validators: Schema, Null, Temporal, Leakage, Duplicate, Range, InventoryConsistency. `run_all_validators()` operates on silver. Writes `reports/validation_report.json`.
- **`src/forecasting/baselines.py`** — 5 intermittent-demand forecasters: `SeasonalNaive`, `RollingSeasonalMean`, `Croston`, `SBA`, `TSB`. All implement `BaseForecaster.fit_predict(series, horizon)` and are stateless.
- **`src/evaluation/metrics.py`** — 7 metrics: sMAPE, WAPE, MASE (lag-52 seasonal naïve denominator), bias%, SLP (service level proxy), MAE, RMSE. All handle zeros/NaNs.
- **`src/evaluation/walk_forward.py`** — 5-fold expanding-window CV on `Sell-in` category (horizons of 13 weeks). Uses `src/runtime/tracker.py` for dual SQLite+MLflow logging.
- **`src/runtime/tracker.py`** — `RunTracker` is the canonical experiment tracker: wraps real MLflow when available, falls back to SQLite (`mlruns/mlflow_runs.db`). Yields `ActiveRun` context managers with `log_metric`, `log_metrics`, `log_artifact`, `log_dict`.
- **`src/orchestration/runtime_validator.py`** — pre-flight checks before Loop 2: silver exists, MLflow SQLite writable, reports dir exists.

### Pipeline orchestration (`pipelines/`)

- **`pipelines/training_flow.py`** — Loop 1 DAG: ingest → clean → validate → features → baseline. Runs as plain Python or Prefect (auto-detected).
- **`pipelines/loop2_flow.py`** — Loop 2 DAG: runtime validation → walk-forward CV → benchmark → LightGBM → profiling. Both pipelines have a graceful Prefect fallback that runs sequentially if Prefect is not installed.

### Experiment tracking

Two tracking backends coexist:
1. **MLflow** (real) — writes to `mlflow.db` (SQLite); URI configured via `MLFLOW_TRACKING_URI`. Walk-forward CV uses `mlflow.set_tracking_uri("sqlite:///mlflow.db")` directly.
2. **`RunTracker`** (custom SQLite) — writes to `mlruns/mlflow_runs.db`. Used by `walk_forward.py` and proxies calls to real MLflow when available.

### Key invariants

- **Leakage prevention is critical**: all lag/rolling features in `feature_engineering.py` use shifted windows; `tests/test_leakage.py` enforces this and must pass before any training run.
- **Year-week format**: 6-digit integer `YYYYWW` (e.g., `202314`). Comparisons use integer arithmetic; string `.0` suffixes are stripped in walk-forward evaluation.
- **Intermittent demand**: ~1% non-zero rate drives model choice (Croston/SBA/TSB) and metric choice (sMAPE/WAPE over MAPE).
- **All pipelines are runnable both as scripts** (`python <file>.py`) and **as imports** (each has a `run_*()` entry point function).
- Reports are always written to `reports/` as JSON.
