# Forecasting Strategy — v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** FCST-STRAT-ITER3-001

---

## 1. Arquitectura de Estrategia

```
FORECASTING STRATEGY — TRES EJES

EJE 1: SEGMENTACIÓN         EJE 2: JERARQUÍA          EJE 3: MODELOS
───────────────────         ────────────────          ──────────────
Clasificar series por        Forecast en L2            Modelo correcto
patrón de demanda           (Family×Channel)           por segmento
usando ADI / CV²            Disagregar a L3

Regular     → LightGBM      L2 directo    → LightGBM   Global ML
Intermittent→ Croston/TSB   L2→L3 disag.  → Prop. hist Croston/TSB
Rare        → Hist. mean    Reconcile     → MinT-Shrink Hist. mean
Dead/New    → Cold-start    Validate      → Sum check   Cold-start
```

---

## 2. Segmentación por Criterio Syntetos-Boylan

```python
# src/ml/segmentation.py

import polars as pl
import numpy as np

def classify_demand_pattern(
    series: np.ndarray,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
    min_active_regular: int = 13
) -> dict:
    """
    Classify a time series using the Syntetos-Boylan criterion.

    ADI = Average Demand Interval (mean weeks between non-zero demands)
    CV² = (StdDev of non-zero demands / Mean of non-zero demands)²

    Returns:
        quadrant:  'smooth' | 'intermittent' | 'erratic' | 'lumpy'
        segment:   'regular' | 'intermittent' | 'rare' | 'dead'
        adi:       float
        cv2:       float
        active_weeks: int
    """
    non_zero = series[series > 0]
    active_weeks = len(non_zero)
    total_weeks = len(series)

    if active_weeks == 0:
        return {"quadrant": "dead", "segment": "dead",
                "adi": np.inf, "cv2": 0.0, "active_weeks": 0}

    if active_weeks < 4:
        return {"quadrant": "rare", "segment": "rare",
                "adi": total_weeks / active_weeks, "cv2": 0.0,
                "active_weeks": active_weeks}

    # Compute ADI: average inter-demand interval
    nonzero_indices = np.where(series > 0)[0]
    if len(nonzero_indices) > 1:
        intervals = np.diff(nonzero_indices)
        adi = float(np.mean(intervals))
    else:
        adi = float(total_weeks)

    # Compute CV²: squared coefficient of variation of demand sizes
    mean_d = float(np.mean(non_zero))
    std_d  = float(np.std(non_zero, ddof=1)) if len(non_zero) > 1 else 0.0
    cv2    = (std_d / mean_d) ** 2 if mean_d > 0 else 0.0

    # Syntetos-Boylan quadrant
    if adi < adi_threshold and cv2 < cv2_threshold:
        quadrant = "smooth"
        segment  = "regular"
    elif adi >= adi_threshold and cv2 < cv2_threshold:
        quadrant = "intermittent"
        segment  = "intermittent"
    elif adi < adi_threshold and cv2 >= cv2_threshold:
        quadrant = "erratic"
        segment  = "intermittent"  # Use intermittent models for erratic too
    else:
        quadrant = "lumpy"
        segment  = "intermittent"  # TSB handles lumpy better than Croston

    # Override: if very few active weeks, downgrade
    if active_weeks < min_active_regular and segment == "regular":
        segment = "intermittent"

    return {
        "quadrant": quadrant,
        "segment": segment,
        "adi": round(adi, 3),
        "cv2": round(cv2, 3),
        "active_weeks": active_weeks
    }


def build_series_registry(
    feature_store: pl.DataFrame,
    params: dict
) -> pl.DataFrame:
    """Classify all series and persist as series_registry.parquet"""

    # Use last 52 weeks as evaluation window
    eval_window = (
        feature_store
        .sort("yearweek")
        .group_by(["channel","material"])
        .tail(52)
    )

    records = []
    for (channel, material), grp in eval_window.group_by(["channel","material"]):
        series = grp.sort("yearweek")["sell_in"].to_numpy()
        result = classify_demand_pattern(
            series,
            adi_threshold=params["segmentation"]["adi_threshold"],
            cv2_threshold=params["segmentation"]["cv2_threshold"],
            min_active_regular=params["segmentation"]["min_active_weeks_regular"],
        )
        records.append({
            "channel":         channel,
            "material":        material,
            "product_family":  grp["product_family"][0],
            "segment":         result["segment"],
            "quadrant":        result["quadrant"],
            "adi":             result["adi"],
            "cv2":             result["cv2"],
            "active_weeks_52w": result["active_weeks"],
            "first_active_week": grp.filter(pl.col("sell_in") > 0)["yearweek"].min(),
            "last_active_week":  grp.filter(pl.col("sell_in") > 0)["yearweek"].max(),
        })

    return pl.DataFrame(records)
```

---

## 3. Forecast Pipeline Completo

```
┌─────────────────────────────────────────────────────────────────────────┐
│               FORECAST PIPELINE — FLUJO COMPLETO                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  feature_store.parquet                                                   │
│       │                                                                  │
│       ├──────────────────────────────────────────────────────────┐      │
│       │                                                           │      │
│       ▼                                                           ▼      │
│  REGULAR series (~739)                                    INTERMITTENT   │
│  segment: regular                                         (~1,601)       │
│       │                                                           │      │
│       ▼                                                           ▼      │
│  MLForecast + LightGBM                                StatsForecast     │
│  (global, cross-series)                               Croston/TSB/IMAPA │
│       │                                                           │      │
│       ▼                                                           │      │
│  Aggregate to L2 level                                            │      │
│  (Family × Channel)                                               │      │
│       │                                                           │      │
│       ▼                                                           │      │
│  MinT-Shrink Reconciliation ◄─────────────────────────────────────┘      │
│  (all levels simultaneously)                                             │
│       │                                                                  │
│       ├──── L0 (Total)                                                   │
│       ├──── L1 (Family)                                                  │
│       ├──── L2 (Family × Channel)   ← reconciled                        │
│       └──── L3 (SKU × Channel)      ← disaggregated + reconciled        │
│                │                                                         │
│                ▼                                                         │
│         RARE + DEAD series                                               │
│         Historical mean / Cold-start                                     │
│         (appended to L3, not reconciled)                                 │
│                │                                                         │
│                ▼                                                         │
│         Inventory-Aware Adjustment                                       │
│         (DOS-based suppression/boost)                                    │
│                │                                                         │
│                ▼                                                         │
│         forecast_output.parquet                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Métricas de Evaluación Estandarizadas

### Por Qué sMAPE, No MAPE

MAPE = `|actual - forecast| / actual` es indefinido cuando `actual = 0`.  
Con 50% de semanas con cero ventas, MAPE produce infinitos o divisiones por cero.

```python
# src/ml/metrics.py

import numpy as np
import polars as pl

def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    Defined even when actual=0.
    Range: [0%, 200%]. Perfect=0%.
    """
    denom = (np.abs(actual) + np.abs(forecast)) / 2
    mask = denom > 0
    return float(np.mean(
        np.abs(actual[mask] - forecast[mask]) / denom[mask]
    ) * 100)

def mase(actual: np.ndarray, forecast: np.ndarray,
         seasonal_period: int = 52) -> float:
    """
    Mean Absolute Scaled Error vs seasonal naive.
    MASE < 1.0 = beats seasonal naive.
    """
    naive = actual[:-seasonal_period]  # same week last year
    naive_errors = np.abs(actual[seasonal_period:] - naive)
    mae_model = np.mean(np.abs(actual - forecast[:len(actual)]))
    return float(mae_model / (np.mean(naive_errors) + 1e-6))

def bias_pct(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Signed percentage bias. Positive = over-forecast.
    """
    mean_actual = np.mean(actual)
    if mean_actual < 0.01:
        return 0.0
    return float((np.mean(forecast) - mean_actual) / mean_actual * 100)

def hit_rate(actual: np.ndarray, forecast: np.ndarray,
             threshold: float = 0.25) -> float:
    """
    Percentage of forecasts within ±threshold of actual.
    """
    mask = actual > 0
    if mask.sum() == 0:
        return 0.0
    within = np.abs(actual[mask] - forecast[mask]) / actual[mask] <= threshold
    return float(within.mean() * 100)

METRICS_REGISTRY = {
    "smape":     smape,
    "mase":      mase,
    "bias_pct":  bias_pct,
    "hit_rate":  hit_rate,
}

METRIC_TARGETS = {
    "regular":     {"smape": 20.0, "bias_pct": 8.0, "hit_rate": 65.0},
    "intermittent":{"smape": 35.0, "bias_pct": 15.0},
    "rare":        {"smape": 55.0},
    "overall":     {"smape": 28.0, "bias_pct": 10.0},
}
```

---

## 5. Walk-Forward Cross-Validation

```
PROTOCOLO CV — EXPANDING WINDOW

Semanas disponibles: W01-2023 → W33-2025 (137 semanas)

Fold 1:  Train [W01-2023 → W48-2024]  |  Val [W49-2024 → W52-2024]
Fold 2:  Train [W01-2023 → W04-2025]  |  Val [W05-2025 → W08-2025]
Fold 3:  Train [W01-2023 → W08-2025]  |  Val [W09-2025 → W12-2025]
Fold 4:  Train [W01-2023 → W12-2025]  |  Val [W13-2025 → W16-2025]
Fold 5:  Train [W01-2023 → W16-2025]  |  Val [W17-2025 → W20-2025]
Fold 6:  Train [W01-2023 → W20-2025]  |  Val [W21-2025 → W24-2025]
Fold 7:  Train [W01-2023 → W24-2025]  |  Val [W25-2025 → W28-2025]
Fold 8:  Train [W01-2023 → W28-2025]  |  Val [W29-2025 → W33-2025]

Horizon evaluado: H+1, H+2, H+4 en cada fold
Step size: 4 semanas
Mínimo training: 52 semanas (W01-2023 siempre incluido)
```

---

## 6. Baselines Obligatorios

Todo modelo ML debe superar estos baselines antes de ir a producción:

| Baseline | Implementación | sMAPE Esperado |
|---|---|---|
| **Naive Seasonal** | `F(t) = Actual(t-52)` | ~35% |
| **Moving Average 4** | Media de últimas 4 semanas | ~45% |
| **Seasonal Naive + Drift** | Naive + tendencia lineal YoY | ~28% |
| **Historical Mean** | Media de toda la historia | ~55% |

**Criterio de promoción:** LightGBM debe superar `Seasonal Naive + Drift` por ≥ 15% de reducción relativa de sMAPE en el segmento Regular.

---

## 7. Horizon Decay Expected Profile

```
DEGRADACIÓN ESPERADA DE ACCURACY POR HORIZONTE

H+1:  sMAPE ~15%   (muy cercano, alta predictibilidad)
H+2:  sMAPE ~17%
H+4:  sMAPE ~20%   (objetivo para Regular)
H+8:  sMAPE ~25%
H+13: sMAPE ~28%
H+19: sMAPE ~32%   (horizonte máximo; máxima incertidumbre)

Si H+4 supera 25% → modelo necesita revisión
Si H+1 supera 22% → datos tienen problemas no resueltos
```

---

*AI-DLC Traceability ID: FCST-STRAT-ITER3-001 | Version: 3.0*
