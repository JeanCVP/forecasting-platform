# Drift Detection — v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** DRIFT-ITER3-001

---

## 1. Tipos de Drift Monitoreados

```
TAXONOMÍA DE DRIFT EN ESTE SISTEMA

┌─────────────────────────────────────────────────────────────────────┐
│  TIPO 1: DATA DRIFT (Covariate Shift)                               │
│  Las distribuciones de las features de entrada cambian              │
│  Ejemplo: days_of_supply promedio sube de 30d a 65d                 │
│  Detección: PSI (Population Stability Index)                        │
│  Acción: Warning → investigar; luego retrain si persiste            │
├─────────────────────────────────────────────────────────────────────┤
│  TIPO 2: CONCEPT DRIFT (Target Drift)                               │
│  La relación entre features y sell_in cambia                        │
│  Ejemplo: mismo DOS=30d solía predecir 100 units; ahora solo 60     │
│  Detección: Degradación de MAPE en ventana rolling                  │
│  Acción: Retrain inmediato                                           │
├─────────────────────────────────────────────────────────────────────┤
│  TIPO 3: LABEL DRIFT (Target Distribution Shift)                    │
│  La distribución del sell_in en sí misma cambia                     │
│  Ejemplo: portfolio se contrae → media de sell_in baja 30%          │
│  Detección: PSI en distribución del target                          │
│  Acción: Contextual; puede ser cambio de negocio normal             │
├─────────────────────────────────────────────────────────────────────┤
│  TIPO 4: SCHEMA DRIFT                                               │
│  El esquema del CSV fuente cambia                                   │
│  Ejemplo: nueva columna, cambio de nombre, tipo diferente           │
│  Detección: Great Expectations Gate-0                               │
│  Acción: HALT pipeline + alerta crítica                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. PSI — Population Stability Index

### Fundamento

$$PSI = \sum_{i=1}^{n} (P_{current,i} - P_{reference,i}) \cdot \ln\left(\frac{P_{current,i}}{P_{reference,i}}\right)$$

| PSI | Interpretación | Acción |
|---|---|---|
| < 0.10 | Sin cambio significativo | 🟢 Monitor normal |
| 0.10 – 0.20 | Cambio moderado | 🟡 Investigar |
| > 0.20 | Cambio significativo | 🔴 Retrain requerido |

### Implementación

```python
# src/monitoring/psi.py

import numpy as np
import polars as pl
from typing import Optional

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index between reference and current distributions.

    Args:
        reference: Array of reference period values
        current:   Array of current period values
        n_bins:    Number of equal-width bins
        eps:       Small value to avoid log(0)

    Returns:
        PSI scalar
    """
    # Remove NaN
    reference = reference[~np.isnan(reference)]
    current   = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Build bins from reference distribution
    min_val = np.percentile(reference, 1)
    max_val = np.percentile(reference, 99)
    bins = np.linspace(min_val, max_val, n_bins + 1)

    # Compute histograms
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current,   bins=bins)

    # Normalize to proportions
    ref_pct = (ref_hist + eps) / (ref_hist.sum() + eps * n_bins)
    cur_pct = (cur_hist + eps) / (cur_hist.sum() + eps * n_bins)

    # PSI formula
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def compute_psi_all_features(
    feature_store: pl.DataFrame,
    reference_weeks: list[str],
    current_weeks: list[str],
    monitored_features: list[str],
) -> pl.DataFrame:
    """
    Compute PSI for all monitored features.
    Returns DataFrame with one row per feature.
    """
    ref_df = feature_store.filter(pl.col("yearweek").is_in(reference_weeks))
    cur_df = feature_store.filter(pl.col("yearweek").is_in(current_weeks))

    records = []
    for feat in monitored_features:
        if feat not in feature_store.columns:
            continue

        ref_vals = ref_df[feat].drop_nulls().to_numpy()
        cur_vals = cur_df[feat].drop_nulls().to_numpy()

        psi = compute_psi(ref_vals, cur_vals)

        records.append({
            "feature":         feat,
            "psi":             psi,
            "status":          "green" if psi < 0.10 else
                               "yellow" if psi < 0.20 else "red",
            "ref_mean":        float(np.mean(ref_vals)) if len(ref_vals) > 0 else None,
            "cur_mean":        float(np.mean(cur_vals)) if len(cur_vals) > 0 else None,
            "ref_p50":         float(np.median(ref_vals)) if len(ref_vals) > 0 else None,
            "cur_p50":         float(np.median(cur_vals)) if len(cur_vals) > 0 else None,
            "mean_shift_pct":  float(
                (np.mean(cur_vals) - np.mean(ref_vals)) /
                (abs(np.mean(ref_vals)) + 1e-6) * 100
            ) if len(ref_vals) > 0 and len(cur_vals) > 0 else None,
            "ref_n":           len(ref_vals),
            "cur_n":           len(cur_vals),
        })

    return pl.DataFrame(records)
```

---

## 3. Features Monitoreadas

```python
# src/monitoring/config.py

MONITORED_FEATURES_PRIMARY = [
    # Señales de demanda
    "sell_in_lag_1",
    "sell_in_ma4",
    "sell_in_ma13",
    "sales_ma4",
    "sales_ma13",
    # Señales de inventario
    "days_of_supply",
    "inv_lag_1",
    "inv_delta_1",
    # Ratios de comportamiento
    "sell_through_rate_4w",
    "prob_nonzero_4w",
    "prob_nonzero_13w",
    # Target distribution
    "sell_in",  # Para label drift
]

MONITORED_FEATURES_SECONDARY = [
    "inv_momentum",
    "replenishment_gap",
    "yoy_sell_in_ratio",
    "sku_age_weeks",
]

REFERENCE_WINDOW_WEEKS = 26  # Usar últimas 26 semanas del training como referencia
CURRENT_WINDOW_WEEKS   = 4   # Comparar contra las últimas 4 semanas
```

---

## 4. Concept Drift — MAPE Rolling

```python
# src/monitoring/concept_drift.py

import polars as pl
import numpy as np

def detect_concept_drift(
    accuracy_log: pl.DataFrame,
    current_week: str,
    rolling_window: int = 4,
    baseline_window: int = 8,
    threshold_delta_pp: float = 5.0,
) -> dict:
    """
    Detect concept drift via sustained MAPE degradation.

    Args:
        accuracy_log:       Historical accuracy log
        current_week:       Current week for analysis
        rolling_window:     Weeks to compute rolling MAPE
        baseline_window:    Weeks for baseline reference
        threshold_delta_pp: PP increase that triggers drift flag

    Returns:
        drift report dict
    """
    log = accuracy_log.sort("yearweek")

    # Baseline: first N weeks after first training
    baseline = log.head(baseline_window)["smape_overall"].to_numpy()
    baseline_smape = float(np.mean(baseline))

    # Recent: last rolling_window weeks
    recent = log.tail(rolling_window)["smape_overall"].to_numpy()
    recent_smape = float(np.mean(recent))

    delta_pp = recent_smape - baseline_smape

    # Linear trend over recent window (positive slope = worsening)
    if len(recent) >= 2:
        trend_slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
    else:
        trend_slope = 0.0

    drift_detected = delta_pp > threshold_delta_pp

    return {
        "concept_drift_detected": drift_detected,
        "baseline_smape":         round(baseline_smape, 2),
        "recent_smape":           round(recent_smape, 2),
        "delta_pp":               round(delta_pp, 2),
        "trend_slope_per_week":   round(trend_slope, 3),
        "status":                 "red" if drift_detected else
                                  "yellow" if delta_pp > threshold_delta_pp * 0.6 else
                                  "green",
        "weeks_analyzed":         len(log),
        "current_week":           current_week,
    }
```

---

## 5. Schema Drift — Great Expectations

```python
# src/monitoring/schema_drift.py

def detect_schema_drift(
    new_file_path: str,
    year: int,
    expected_schema: dict,
) -> dict:
    """
    Detect changes in the CSV schema vs. expected contract.
    Triggered at every ingestion.
    """
    import polars as pl

    try:
        df = pl.read_csv(new_file_path, n_rows=5)
    except Exception as e:
        return {"drift_detected": True, "type": "READ_ERROR",
                "detail": str(e), "severity": "CRITICAL"}

    drifts = []

    # Column existence
    expected_fixed = {"Channel", "Material Description", "Category"}
    missing = expected_fixed - set(df.columns)
    if missing:
        drifts.append({"type": "MISSING_COLUMNS", "cols": list(missing),
                       "severity": "CRITICAL"})

    # Column count
    week_cols = [c for c in df.columns if c.startswith(str(year))]
    if len(week_cols) != 52:
        drifts.append({"type": "WRONG_WEEK_COUNT",
                       "expected": 52, "found": len(week_cols),
                       "severity": "HIGH"})

    # Category values
    cats = set(pl.read_csv(new_file_path)["Category"].unique().to_list())
    expected_cats = {"Sell-in", "Cust. Sales", "Channel Inv."}
    extra = cats - expected_cats
    missing_cats = expected_cats - cats
    if extra or missing_cats:
        drifts.append({"type": "CATEGORY_DRIFT",
                       "extra": list(extra), "missing": list(missing_cats),
                       "severity": "CRITICAL"})

    return {
        "drift_detected": len(drifts) > 0,
        "n_drifts":        len(drifts),
        "drifts":          drifts,
        "severity":        max((d["severity"] for d in drifts),
                              key=lambda s: ["LOW","MEDIUM","HIGH","CRITICAL"].index(s))
                           if drifts else "NONE",
    }
```

---

## 6. Drift Log Schema

```sql
-- data/gold/drift_log.parquet

CREATE TABLE drift_log (
    yearweek              CHAR(6),
    computed_at           TIMESTAMP,
    drift_type            VARCHAR(20),    -- 'data', 'concept', 'label', 'schema'
    feature               VARCHAR(100),   -- feature name or 'overall'
    psi                   FLOAT,          -- for data drift
    delta_pp              FLOAT,          -- for concept drift
    status                VARCHAR(10),    -- 'green', 'yellow', 'red'
    ref_mean              FLOAT,
    cur_mean              FLOAT,
    mean_shift_pct        FLOAT,
    action_triggered      VARCHAR(30),    -- 'none', 'warning', 'retrain'
    PRIMARY KEY (yearweek, drift_type, feature)
);
```

---

## 7. Umbrales de Acción

| Drift Type | Metric | Green | Yellow | Red (Action) |
|---|---|---|---|---|
| Data (feature) | PSI | < 0.10 | 0.10–0.20 | > 0.20 → WARNING |
| Data (feature) | PSI | — | — | > 0.30 → RETRAIN |
| Concept | MAPE delta | < 2pp | 2–5pp | > 5pp → RETRAIN |
| Label | PSI sell_in | < 0.15 | 0.15–0.30 | > 0.30 → INVESTIGATE |
| Schema | Any column missing | — | — | → HALT PIPELINE |

---

## 8. Drift Dashboard (Streamlit)

```python
# dashboards/pages/drift_monitor.py

import streamlit as st
import polars as pl

def render_drift_dashboard():
    st.header("📉 Feature & Concept Drift Monitor")

    drift_log = pl.read_parquet("data/gold/drift_log.parquet")
    latest_week = drift_log["yearweek"].max()

    current = drift_log.filter(
        (pl.col("yearweek") == latest_week) &
        (pl.col("drift_type") == "data")
    ).sort("psi", descending=True)

    # Summary KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Green Features",
                (current["status"] == "green").sum())
    col2.metric("🟡 Yellow Features",
                (current["status"] == "yellow").sum(),
                delta_color="inverse")
    col3.metric("🔴 Red Features",
                (current["status"] == "red").sum(),
                delta_color="inverse")

    # PSI table
    st.subheader(f"Feature PSI — Week {latest_week}")
    styled = current.select([
        "feature", "psi", "status",
        "ref_mean", "cur_mean", "mean_shift_pct"
    ])
    st.dataframe(styled, use_container_width=True)

    # Concept drift trend
    concept = drift_log.filter(pl.col("drift_type") == "concept").sort("yearweek")
    if len(concept) > 0:
        st.subheader("Concept Drift — MAPE Rolling Trend")
        import plotly.express as px
        fig = px.line(
            concept.to_pandas(),
            x="yearweek", y="delta_pp",
            title="MAPE Delta vs Baseline (pp)",
            labels={"delta_pp": "MAPE Delta (pp)", "yearweek": "Week"}
        )
        fig.add_hline(y=5, line_dash="dash", line_color="red",
                      annotation_text="Retrain threshold")
        fig.add_hline(y=2, line_dash="dash", line_color="orange",
                      annotation_text="Warning threshold")
        st.plotly_chart(fig, use_container_width=True)
```

---

*AI-DLC Traceability ID: DRIFT-ITER3-001 | Version: 3.0*
