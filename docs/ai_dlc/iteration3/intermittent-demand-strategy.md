# Intermittent Demand Strategy
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** INTERMITTENT-ITER3-001

---

## 1. Mapa de Demanda Intermitente en el Dataset

```
DISTRIBUCIÓN REAL DEL DATASET (2024, Sell-in)

                CV² < 0.49          CV² ≥ 0.49
               ┌───────────────┬────────────────┐
ADI < 1.32     │   SMOOTH      │    ERRATIC     │
(regular)      │   ~739 series │    ~412 series │
               │   (6%)        │    (3%)        │
               ├───────────────┼────────────────┤
ADI ≥ 1.32     │ INTERMITTENT  │    LUMPY       │
(sparse)       │   ~1,189 ser. │    ~3,864 ser. │
               │   (10%)       │    (31%)       │
               └───────────────┴────────────────┘
               DEAD / RARE: ~6,209 series (50%)

MODELO ASIGNADO:
  Smooth      → LightGBM global
  Erratic     → LightGBM global (con features de volatilidad)
  Intermittent→ CrostonOptimized
  Lumpy       → TSB (Teunter-Syntetos-Babai)
  Rare        → Historical Mean + Family Analog
  Dead/New    → Cold-start (zero o analog proportion)
```

---

## 2. Modelo Croston — Para Series Intermitentes

Croston separa el proceso de demanda en dos componentes:

```
Serie original:  0, 0, 25, 0, 0, 0, 30, 0, 15, 0, 0, 0, 40, ...

Componente 1 — Tamaño de demanda cuando es positiva:
  d = [25, 30, 15, 40, ...]
  Suavizado con SES: d̂(t) = α·d(t) + (1-α)·d̂(t-1)

Componente 2 — Intervalo entre demandas:
  i = [3, 4, 3, ...] semanas
  Suavizado con SES: î(t) = α·i(t) + (1-α)·î(t-1)

Forecast: F(t) = d̂ / î
```

```python
# src/forecasting/intermittent.py

from statsforecast import StatsForecast
from statsforecast.models import (
    CrostonOptimized,   # Croston con α óptimo por MLE
    TSB,                # Teunter-Syntetos-Babai (mejor para lumpy)
    IMAPA,              # Intermittent MAPA
    HistoricAverage,    # Fallback
)
import polars as pl

def train_intermittent_models(
    series_registry: pl.DataFrame,
    training_set: pl.DataFrame,
    horizon: int = 19,
    freq: str = "W"
) -> dict:
    """
    Train intermittent demand models for non-regular series.
    Returns dict of {segment: fitted_model}
    """

    # Separate series by quadrant
    intermittent_ids = series_registry.filter(
        pl.col("quadrant") == "intermittent"
    )["channel"].zip_with(series_registry.filter(
        pl.col("quadrant") == "intermittent")["material"]
    )

    lumpy_ids = series_registry.filter(
        pl.col("quadrant") == "lumpy"
    )["channel"].zip_with(series_registry.filter(
        pl.col("quadrant") == "lumpy")["material"]
    )

    # --- CrostonOptimized for Intermittent ---
    sf_croston = StatsForecast(
        models=[
            CrostonOptimized(),
            IMAPA(),
        ],
        freq=freq,
        n_jobs=-1,
        verbose=False,
    )

    # --- TSB for Lumpy ---
    sf_tsb = StatsForecast(
        models=[
            TSB(alpha_d=0.2, alpha_p=0.1),
            CrostonOptimized(),          # Challenger
        ],
        freq=freq,
        n_jobs=-1,
        verbose=False,
    )

    # Prepare long-format data for StatsForecast
    def prepare_sf_df(ids, data):
        return (
            data
            .with_columns([
                (pl.col("channel") + "__" + pl.col("material"))
                  .alias("unique_id"),
                pl.col("date").alias("ds"),
                pl.col("sell_in").alias("y"),
            ])
            .select(["unique_id","ds","y"])
        )

    intermittent_data = prepare_sf_df(intermittent_ids, training_set)
    lumpy_data = prepare_sf_df(lumpy_ids, training_set)

    forecasts = {}
    if len(intermittent_data) > 0:
        sf_croston.fit(intermittent_data)
        forecasts["intermittent"] = sf_croston.predict(h=horizon)

    if len(lumpy_data) > 0:
        sf_tsb.fit(lumpy_data)
        forecasts["lumpy"] = sf_tsb.predict(h=horizon)

    return forecasts
```

---

## 3. TSB — Para Series Lumpy (Alta Variabilidad + Baja Frecuencia)

TSB actualiza dos componentes: probabilidad de demanda (`p`) y tamaño esperado (`v`):

```
p̂(t) = α_p · 1(d(t)>0)  + (1 - α_p) · p̂(t-1)   ← prob. de demanda
v̂(t) = α_d · d(t)        + (1 - α_d) · v̂(t-1)   ← tamaño promedio

Forecast: F(t) = p̂(t) · v̂(t)
```

**Ventaja sobre Croston:** Croston no actualiza la estimación de probabilidad en semanas con demanda cero. TSB sí lo hace, lo que lo hace más responsivo a cambios de régimen (e.g., SKU en fase de retiro).

---

## 4. Selección de Modelo por Validación Cruzada

```python
def select_best_intermittent_model(
    series_data: pl.DataFrame,
    n_cv_folds: int = 4,
    horizon: int = 4
) -> str:
    """
    Compare CrostonOptimized vs TSB vs IMAPA via CV.
    Returns name of best model by sMAPE.
    """

    models = [
        CrostonOptimized(),
        TSB(alpha_d=0.2, alpha_p=0.1),
        IMAPA(),
    ]

    sf = StatsForecast(models=models, freq="W", n_jobs=-1)
    cv_results = sf.cross_validation(
        df=series_data,
        h=horizon,
        n_windows=n_cv_folds,
        step_size=horizon,
        level=[],
    )

    # Compute sMAPE per model
    smapes = {}
    for model_name in ["CrostonOptimized","TSB","IMAPA"]:
        col = model_name
        if col in cv_results.columns:
            actual = cv_results["y"].to_numpy()
            pred = cv_results[col].to_numpy()
            denom = (np.abs(actual) + np.abs(pred)) / 2
            mask = denom > 0
            smapes[model_name] = float(
                np.mean(np.abs(actual[mask] - pred[mask]) / denom[mask]) * 100
            )

    return min(smapes, key=smapes.get)
```

---

## 5. Historical Mean para Series Raras (1–3 semanas activas)

```python
def rare_series_forecast(
    series: np.ndarray,
    channel: str,
    material: str,
    product_family: str,
    training_set: pl.DataFrame,
    horizon: int = 19,
    family_weight: float = 0.4
) -> np.ndarray:
    """
    Forecast for rare series (1-3 active weeks).

    Strategy:
    - 60% weight: own historical mean (when active)
    - 40% weight: family-channel average
    - Apply zero-inflation: forecast * prob_nonzero
    """
    # Own signal
    own_mean = float(series[series > 0].mean()) if (series > 0).any() else 0.0

    # Family-channel analog
    family_series = (
        training_set
        .filter(
            (pl.col("product_family") == product_family) &
            (pl.col("channel") == channel) &
            (pl.col("segment") != "dead")
        )["sell_in"]
        .to_numpy()
    )
    family_mean = float(family_series[family_series > 0].mean()) if (
        family_series > 0
    ).any() else 0.0

    # Blended estimate
    blended = (1 - family_weight) * own_mean + family_weight * family_mean

    # Apply probability of non-zero (recency-weighted)
    prob_nonzero = float((series[-13:] > 0).mean()) if len(series) >= 13 else float(
        (series > 0).mean()
    )
    adjusted = blended * prob_nonzero

    # Flat forecast for all horizon weeks
    return np.full(horizon, max(adjusted, 0.0))
```

---

## 6. Métricas Apropiadas para Demanda Intermitente

| Métrica | ¿Válida para Intermitente? | Por Qué |
|---|---|---|
| MAPE | ❌ | División por cero cuando actual=0 |
| sMAPE | ✅ | Simétrico; definido para cero |
| MASE | ✅ | Escala con baseline naive estacional |
| MAE | ✅ Parcial | En unidades; comparable solo dentro de mismo SKU |
| RMSE | ⚠️ | Penaliza errores grandes; sesgado en lumpy |
| Pinball Loss | ✅ | Para evaluación de intervalos de predicción |

---

## 7. Targets de Accuracy por Cuadrante

| Cuadrante | Modelo | sMAPE Target | Nota |
|---|---|---|---|
| Smooth (regular) | LightGBM | ≤ 20% | Bien definido |
| Erratic | LightGBM | ≤ 28% | Alta varianza inherente |
| Intermittent | Croston | ≤ 40% | Difícil por definición |
| Lumpy | TSB | ≤ 45% | El más difícil |
| Rare | Hist. Mean | ≤ 60% | Señal insuficiente |

---

*AI-DLC Traceability ID: INTERMITTENT-ITER3-001 | Version: 3.0*
