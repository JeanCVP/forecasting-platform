# Dataset Versioning — Especificación v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** VERSIONING-ITER3-001

---

## 1. Principios de Versionado

```
REGLAS INMUTABLES
────────────────────────────────────────────────────────────────
1. data/raw/     — NUNCA se modifica. Las correcciones son archivos nuevos.
2. data/bronze/  — NUNCA se modifica post-ingesta.
3. data/silver/  — Rebuilde completo desde Bronze. No edición in-place.
4. data/gold/    — Rebuilde completo desde Silver. No edición in-place.
5. Todo archivo en data/ tiene un .dvc sidecar con SHA-256.
6. Todo MLflow run tiene tags con los hashes de los datos usados.
────────────────────────────────────────────────────────────────
Consecuencia: cualquier commit de Git + dvc checkout reproduce
exactamente el estado de los datos en ese momento.
────────────────────────────────────────────────────────────────
```

---

## 2. DVC Pipeline Completo

```yaml
# dvc.yaml — Pipeline completo de transformación

stages:

  # ─── INGESTA ──────────────────────────────────────────────────
  ingest_2023:
    cmd: python src/ingestion/ingest.py --year 2023
    deps:
      - data/raw/2023.csv
      - src/ingestion/ingest.py
      - src/ingestion/schema_validators.py
    outs:
      - data/bronze/sell_data_2023.parquet
    metrics:
      - reports/bronze_2023_metrics.json:
          cache: false

  ingest_2024:
    cmd: python src/ingestion/ingest.py --year 2024
    deps:
      - data/raw/2024.csv
      - src/ingestion/ingest.py
    outs:
      - data/bronze/sell_data_2024.parquet
    metrics:
      - reports/bronze_2024_metrics.json:
          cache: false

  ingest_2025:
    cmd: python src/ingestion/ingest.py --year 2025
    deps:
      - data/raw/2025.csv
      - src/ingestion/ingest.py
    outs:
      - data/bronze/sell_data_2025.parquet
    metrics:
      - reports/bronze_2025_metrics.json:
          cache: false

  # ─── SILVER ───────────────────────────────────────────────────
  clean:
    cmd: python src/transformation/clean.py
    deps:
      - data/bronze/sell_data_2023.parquet
      - data/bronze/sell_data_2024.parquet
      - data/bronze/sell_data_2025.parquet
      - src/transformation/clean.py
      - src/audit/logger.py
    outs:
      - data/silver/timeseries_clean.parquet
      - data/silver/dim_channel.parquet
      - data/silver/dim_material.parquet
      - data/silver/dim_calendar.parquet
    metrics:
      - reports/silver_metrics.json:
          cache: false

  # ─── GOLD ─────────────────────────────────────────────────────
  features:
    cmd: python src/transformation/feature_engineering.py
    deps:
      - data/silver/timeseries_clean.parquet
      - data/silver/dim_calendar.parquet
      - src/transformation/feature_engineering.py
      - params.yaml
    params:
      - features.lag_windows
      - features.rolling_windows
      - features.calendar_events
    outs:
      - data/gold/feature_store.parquet
      - data/gold/series_registry.parquet
      - data/gold/training_set.parquet
    metrics:
      - reports/gold_metrics.json:
          cache: false

  # ─── SEGMENTACIÓN ─────────────────────────────────────────────
  segment:
    cmd: python src/ml/segmentation.py
    deps:
      - data/gold/training_set.parquet
      - src/ml/segmentation.py
      - params.yaml
    params:
      - segmentation.adi_threshold
      - segmentation.cv2_threshold
      - segmentation.min_active_weeks
    outs:
      - data/gold/series_registry.parquet

  # ─── ENTRENAMIENTO (manual o triggered) ───────────────────────
  train:
    cmd: python src/ml/train_global.py
    deps:
      - data/gold/training_set.parquet
      - data/gold/series_registry.parquet
      - src/ml/train_global.py
      - src/ml/train_intermittent.py
      - params.yaml
    params:
      - lgbm
      - training
      - validation
    outs:
      - models/champion_metadata.json
    metrics:
      - reports/model_metrics.json:
          cache: false

  # ─── FORECAST ─────────────────────────────────────────────────
  forecast:
    cmd: python src/serving/forecast_runner.py
    deps:
      - data/gold/feature_store.parquet
      - data/gold/series_registry.parquet
      - models/champion_metadata.json
      - src/serving/forecast_runner.py
    outs:
      - data/gold/forecast_output.parquet
    metrics:
      - reports/forecast_metrics.json:
          cache: false

  # ─── INVENTORY RISK ───────────────────────────────────────────
  risk:
    cmd: python src/product/inventory_risk_scorer.py
    deps:
      - data/gold/feature_store.parquet
      - data/gold/forecast_output.parquet
      - src/product/inventory_risk_scorer.py
    outs:
      - data/gold/inventory_risk.parquet
```

---

## 3. params.yaml — Configuración Versionada

```yaml
# params.yaml — Todos los hiperparámetros y configuraciones

# ─── FEATURES ─────────────────────────────────────────────────────────
features:
  lag_windows: [1, 2, 4, 8, 13, 26, 52, 51, 53]
  rolling_windows: [4, 13, 26]
  calendar_events:
    mothers_day_weeks: [18, 19]
    black_friday_weeks: [47, 48]
    christmas_weeks: [50, 51, 52]
    back_to_school_weeks: [30, 31, 32, 33, 34]
    new_year_restock_weeks: [1, 2, 3]
    dia_sin_iva_weeks: []  # PENDIENTE: input del equipo comercial

# ─── SEGMENTACIÓN ─────────────────────────────────────────────────────
segmentation:
  adi_threshold: 1.32           # Syntetos-Boylan criterion
  cv2_threshold: 0.49
  min_active_weeks_regular: 13  # Backup criterio por conteo
  min_active_weeks_intermittent: 4
  evaluation_window_weeks: 52   # Últimas 52 semanas para clasificar

# ─── LIGHTGBM ─────────────────────────────────────────────────────────
lgbm:
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 6
  num_leaves: 31
  min_data_in_leaf: 20
  lambda_l1: 0.1
  lambda_l2: 0.1
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 5
  verbose: -1
  n_jobs: -1

# ─── TRAINING ─────────────────────────────────────────────────────────
training:
  training_cutoff: "202533"     # Última semana disponible con datos reales
  exclude_censored: true
  min_train_rows: 50000
  horizon: 19
  n_cv_folds: 8
  cv_step_size: 4

# ─── VALIDACIÓN ───────────────────────────────────────────────────────
validation:
  target_smape_regular: 20.0
  target_smape_intermittent: 35.0
  target_bias_pct: 8.0
  min_dq_score: 85

# ─── MONITORING ───────────────────────────────────────────────────────
monitoring:
  smape_alert_regular: 30.0
  smape_alert_intermittent: 50.0
  bias_alert_pct: 15.0
  psi_warning_threshold: 0.10
  psi_critical_threshold: 0.20
  retraining_smape_delta: 5.0   # Si MAPE sube > 5pp → retrain

# ─── INVENTORY RISK ───────────────────────────────────────────────────
inventory_risk:
  dos_target: 45.0
  dos_critical_overstock: 90.0
  dos_warning_overstock: 60.0
  dos_warning_stockout: 14.0
  dos_critical_stockout: 7.0
```

---

## 4. Convención de Versiones de Dataset

```
FORMATO:  v{MAJOR}.{MINOR}.{PATCH}
Git Tag:  dataset-v{version}-{YYYYMMDD}

MAJOR: Nuevo año de datos (e.g., 2026.csv llega → v2.0.0)
MINOR: Corrección de datos existentes (e.g., 2025 sin cap → v1.1.0)
PATCH: Actualización de calendario, corrección menor → v1.0.1

VERSIÓN ACTUAL: v1.0.0 (datos 2023-2025 con cap conocido)
VERSIÓN TARGET: v1.1.0 (cuando llegue extract 2025 sin cap)
```

---

## 5. Procedimiento de Rollback

```bash
# Listar versiones disponibles
git tag | grep dataset | sort -r

# Rollback a versión específica
git checkout dataset-v1.0.0-20260521
dvc checkout data/silver/  # Restaura Silver exactamente como era
dvc checkout data/gold/    # Restaura Gold exactamente como era

# Reentrenar con datos históricos (para reproducir modelo antiguo)
dvc repro train

# Verificar que el modelo reproducido da las mismas métricas
python src/ml/evaluate.py --expected-smape 22.4
```

---

*AI-DLC Traceability ID: VERSIONING-ITER3-001 | Version: 3.0*
