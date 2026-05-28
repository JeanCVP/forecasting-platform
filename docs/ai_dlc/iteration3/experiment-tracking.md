# Experiment Tracking — v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** EXPTRACK-ITER3-001

---

## 1. Configuración MLflow

```
TOPOLOGÍA

  Tracking Server:    sqlite:///mlruns/prod.db      (Phase 1–2)
                      postgresql://...              (Phase 5+ producción)
  Artifact Store:     ./mlruns/artifacts/           (Phase 1–2)
                      s3://bucket/mlflow-artifacts/ (Phase 5+)
  Model Registry:     mismo backend que tracking
  UI:                 http://localhost:5000          (local)
```

```python
# src/ml/mlflow_config.py

import mlflow, os

EXPERIMENTS = {
    "global_lgbm":        "sell_in_lgbm_global",
    "global_catboost":    "sell_in_catboost_global",
    "intermittent":       "sell_in_intermittent",
    "baselines":          "sell_in_baselines",
    "hpo":                "sell_in_hpo",
    "monitoring":         "forecast_monitoring",
    "reconciliation":     "hierarchical_reconciliation",
}

REGISTERED_MODELS = {
    "global":       "sell_in_lgbm_global",
    "intermittent": "sell_in_intermittent_ensemble",
}

def setup_mlflow(env: str = "production"):
    uri = os.getenv("MLFLOW_TRACKING_URI",
                    f"sqlite:///mlruns/{env}.db")
    mlflow.set_tracking_uri(uri)
    for name in EXPERIMENTS.values():
        mlflow.set_experiment(name)
```

---

## 2. Template de Logging Estándar

Todo run de entrenamiento registra exactamente los mismos campos, sin excepción:

```python
# src/ml/run_logger.py

import mlflow, json, time
from dataclasses import dataclass
from src.transformation.feature_version import compute_feature_store_hash
import dvc.api

@dataclass
class RunMetadata:
    # Data lineage
    feature_store_hash:  str
    silver_hash:         str
    bronze_2023_hash:    str
    bronze_2024_hash:    str
    bronze_2025_hash:    str
    training_cutoff:     str     # e.g. "202533"
    n_training_rows:     int
    n_series_trained:    int
    n_regular_series:    int
    n_intermittent:      int
    weeks_of_history:    int
    exclude_censored:    bool
    leakage_test_passed: bool
    # Segmentation
    segment_regular_pct:      float
    segment_intermittent_pct: float
    segment_rare_pct:         float
    segment_dead_pct:         float


def log_full_run(
    model,
    cv_results,
    params: dict,
    meta: RunMetadata,
    experiment_name: str = "sell_in_lgbm_global",
) -> str:
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"lgbm_{meta.training_cutoff}_{int(time.time())}"
    ) as run:

        # ── TAGS (data lineage + governance) ──────────────────
        mlflow.set_tags({
            "feature_store_hash":        meta.feature_store_hash,
            "silver_hash":               meta.silver_hash,
            "bronze_2023_hash":          meta.bronze_2023_hash,
            "bronze_2024_hash":          meta.bronze_2024_hash,
            "bronze_2025_hash":          meta.bronze_2025_hash,
            "training_cutoff":           meta.training_cutoff,
            "n_training_rows":           str(meta.n_training_rows),
            "n_series_trained":          str(meta.n_series_trained),
            "weeks_of_history":          str(meta.weeks_of_history),
            "exclude_censored":          str(meta.exclude_censored),
            "leakage_test_passed":       str(meta.leakage_test_passed),
            "reconciliation_method":     "mint_shrink",
            "hierarchy_level":           "family_channel",
            "segmentation_criterion":    "syntetos_boylan",
            "model_type":                "lgbm_global",
            "framework":                 "mlforecast",
            "aidlc_iteration":           "3",
        })

        # ── PARAMS (hiperparámetros) ───────────────────────────
        mlflow.log_params({k: v for k, v in params.get("lgbm", {}).items()})
        mlflow.log_params({
            "training_cutoff":   meta.training_cutoff,
            "n_cv_folds":        params["training"]["n_cv_folds"],
            "cv_step_size":      params["training"]["cv_step_size"],
            "horizon":           params["training"]["horizon"],
            "lag_windows":       str(params["features"]["lag_windows"]),
            "rolling_windows":   str(params["features"]["rolling_windows"]),
        })

        # ── METRICS (overall) ─────────────────────────────────
        from src.ml.metrics import smape, mase, bias_pct, hit_rate
        import numpy as np

        actual   = cv_results["y"].to_numpy()
        forecast = cv_results["LGBMRegressor"].to_numpy()

        mlflow.log_metrics({
            "smape_overall":         smape(actual, forecast),
            "mase_overall":          mase(actual, forecast),
            "bias_pct_overall":      bias_pct(actual, forecast),
            "hit_rate_25pct":        hit_rate(actual, forecast, threshold=0.25),
            "hit_rate_10pct":        hit_rate(actual, forecast, threshold=0.10),
        })

        # ── METRICS (by horizon) ──────────────────────────────
        for h in [1, 2, 4, 8, 13, 19]:
            h_df = cv_results[cv_results["h"] == h]
            if len(h_df) == 0:
                continue
            a = h_df["y"].to_numpy()
            f = h_df["LGBMRegressor"].to_numpy()
            mlflow.log_metrics({
                f"smape_h{h}":    smape(a, f),
                f"bias_pct_h{h}": bias_pct(a, f),
            })

        # ── METRICS (by product family) ───────────────────────
        for family in cv_results["product_family"].unique():
            fam = cv_results[cv_results["product_family"] == family]
            key = family.lower().replace(" ","_").replace(".","")
            mlflow.log_metric(f"smape_family_{key}",
                              smape(fam["y"].to_numpy(),
                                    fam["LGBMRegressor"].to_numpy()))

        # ── METRICS (segmentation) ─────────────────────────────
        mlflow.log_metrics({
            "pct_regular_series":      meta.segment_regular_pct,
            "pct_intermittent_series": meta.segment_intermittent_pct,
            "pct_rare_series":         meta.segment_rare_pct,
            "pct_dead_series":         meta.segment_dead_pct,
        })

        # ── ARTIFACTS ─────────────────────────────────────────
        # Feature importance
        import matplotlib.pyplot as plt
        lgbm_model = model.models_["LGBMRegressor"]
        importance = lgbm_model.feature_importances_
        features   = model.ts.static_features_ if hasattr(model, 'ts') else []
        fig, ax = plt.subplots(figsize=(10, 8))
        sorted_idx = importance.argsort()[-20:]
        ax.barh(range(20), importance[sorted_idx])
        ax.set_title("Top 20 Feature Importances")
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close()

        # CV results table
        cv_results.to_parquet("/tmp/cv_results.parquet", index=False)
        mlflow.log_artifact("/tmp/cv_results.parquet")

        # Horizon decay chart
        horizons = [1,2,4,8,13,19]
        h_smapes = []
        for h in horizons:
            h_df = cv_results[cv_results["h"] == h]
            if len(h_df) > 0:
                h_smapes.append(smape(h_df["y"].to_numpy(),
                                      h_df["LGBMRegressor"].to_numpy()))
            else:
                h_smapes.append(None)
        fig2, ax2 = plt.subplots()
        ax2.plot(horizons, h_smapes, marker='o')
        ax2.axhline(y=20, color='green', linestyle='--', label='Target 20%')
        ax2.set_xlabel("Forecast Horizon (weeks)")
        ax2.set_ylabel("sMAPE %")
        ax2.set_title("Accuracy Degradation by Horizon")
        ax2.legend()
        mlflow.log_figure(fig2, "horizon_decay.png")
        plt.close()

        # Serialized model (MLForecast object)
        import joblib
        joblib.dump(model, "/tmp/mlforecast_model.pkl")
        mlflow.log_artifact("/tmp/mlforecast_model.pkl", artifact_path="mlforecast")

        # Register in Model Registry
        mlflow.lightgbm.log_model(
            lgbm_model,
            artifact_path="lgbm_model",
            registered_model_name="sell_in_lgbm_global",
        )

        return run.info.run_id
```

---

## 3. Experimentos por Tipo de Run

| Experimento MLflow | Tipo | Frecuencia | Trigger |
|---|---|---|---|
| `sell_in_lgbm_global` | Entrenamiento global | Mensual + alertas | scheduled / monitor |
| `sell_in_catboost_global` | Challenger alternativo | Trimestral | HPO schedule |
| `sell_in_intermittent` | Croston/TSB | Bimestral | scheduled |
| `sell_in_baselines` | Naive/MA | Por versión de datos | data update |
| `sell_in_hpo` | Búsqueda hyperparámetros | Trimestral | scheduled |
| `forecast_monitoring` | Accuracy semanal | Semanal | monitor_flow |
| `hierarchical_reconciliation` | Residuos de reconciliación | Semanal | forecast_flow |

---

## 4. Vistas de Comparación en MLflow UI

```python
# Configuración recomendada de columnas en MLflow UI

COMPARISON_VIEW_COLUMNS = [
    # Identidad
    "run_name", "training_cutoff", "n_series_trained",
    # Accuracy primaria
    "smape_overall", "smape_h1", "smape_h4", "smape_h19",
    "bias_pct_overall", "hit_rate_25pct",
    # Por familia
    "smape_family_mobile", "smape_family_led_tv", "smape_family_qled_tv",
    # Hiperparámetros
    "n_estimators", "learning_rate", "num_leaves",
    # Lineage
    "feature_store_hash", "training_cutoff",
]

# Para HPO: ordenar por smape_overall ASC
# Para comparar versiones: ordenar por start_time DESC
```

---

## 5. Carga del Champion para Inferencia

```python
# src/serving/load_model.py

import mlflow
from mlflow.tracking import MlflowClient

def load_champion() -> tuple:
    """Load current champion model and its metadata."""
    client = MlflowClient()

    # Get champion version
    champion_mv = client.get_model_version_by_alias(
        "sell_in_lgbm_global", "champion"
    )
    run_id = champion_mv.run_id
    run    = client.get_run(run_id)

    # Load MLForecast object (includes lag config)
    import joblib
    mlforecast_path = client.download_artifacts(
        run_id, "mlforecast/mlforecast_model.pkl"
    )
    mlforecast_model = joblib.load(mlforecast_path)

    metadata = {
        "run_id":               run_id,
        "version":              champion_mv.version,
        "feature_store_hash":   run.data.tags.get("feature_store_hash"),
        "training_cutoff":      run.data.tags.get("training_cutoff"),
        "smape_regular":        run.data.metrics.get("smape_overall"),
        "leakage_test_passed":  run.data.tags.get("leakage_test_passed"),
    }

    return mlforecast_model, metadata
```

---

## 6. Experiment Reproducibility Checklist

Antes de declarar un run como reproducible, verificar:

```python
REPRODUCIBILITY_CHECKLIST = {
    "feature_store_hash_logged":   "mlflow tag 'feature_store_hash' presente",
    "silver_hash_logged":          "mlflow tag 'silver_hash' presente",
    "params_complete":             "todos los hiperparámetros en mlflow params",
    "training_cutoff_logged":      "mlflow tag 'training_cutoff' presente",
    "dvc_pipeline_reproducible":   "dvc repro produce mismo feature_store hash",
    "leakage_test_passed":         "mlflow tag 'leakage_test_passed' = True",
    "cv_results_artifact":         "cv_results.parquet en artifacts",
    "model_artifact":              "mlforecast_model.pkl en artifacts",
    "git_commit_tagged":           "git tag dataset-vX.Y.Z presente",
}

def verify_reproducibility(run_id: str) -> dict:
    client = MlflowClient()
    run = client.get_run(run_id)
    results = {}
    for check, description in REPRODUCIBILITY_CHECKLIST.items():
        if "hash" in check or "cutoff" in check or "passed" in check:
            results[check] = check.replace("_logged","").replace("_passed","") in str(run.data.tags)
        elif "artifact" in check:
            artifacts = [a.path for a in client.list_artifacts(run_id)]
            results[check] = any(check.split("_")[0] in a for a in artifacts)
        else:
            results[check] = True  # manual check required
    return results
```

---

*AI-DLC Traceability ID: EXPTRACK-ITER3-001 | Version: 3.0*
