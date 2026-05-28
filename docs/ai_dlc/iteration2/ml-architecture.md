# ML Architecture
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ML SYSTEM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │  Gold Layer  │───▶│ Feature Store│───▶│    Series Segmentation  │   │
│  │  (DuckDB)    │    │  (Parquet)   │    │    Classifier           │   │
│  └─────────────┘    └──────────────┘    └────────────┬────────────┘   │
│                                                       │                 │
│                       ┌───────────────────────────────┤                │
│                       │                               │                │
│              ┌────────▼──────┐              ┌─────────▼──────────┐    │
│              │  SEGMENT A     │              │  SEGMENT B/C/D     │    │
│              │  Regular       │              │  Intermittent/     │    │
│              │  (~739 series) │              │  Rare / Dead       │    │
│              └────────┬──────┘              └─────────┬──────────┘    │
│                       │                               │                │
│              ┌────────▼──────┐              ┌─────────▼──────────┐    │
│              │ MLForecast     │              │ StatsForecast      │    │
│              │ + LightGBM     │              │ Croston/TSB/IMAPA  │    │
│              │ Global Model   │              │ + Historical Mean  │    │
│              └────────┬──────┘              └─────────┬──────────┘    │
│                       │                               │                │
│              ┌────────▼───────────────────────────────▼───────────┐   │
│              │           HIERARCHICAL RECONCILIATION               │   │
│              │         (MinT-Shrink via StatsForecast)             │   │
│              └─────────────────────────┬──────────────────────────┘   │
│                                        │                               │
│              ┌─────────────────────────▼──────────────────────────┐   │
│              │              FORECAST STORE (Gold Layer)            │   │
│              │    (Channel × Material × Week × Model × Version)   │   │
│              └─────────────────────────┬──────────────────────────┘   │
│                                        │                               │
│              ┌─────────────┬───────────┴─────────────┐               │
│              │             │                           │               │
│     ┌────────▼───┐  ┌──────▼──────┐          ┌────────▼───────┐      │
│     │ Streamlit   │  │  MLflow     │          │  Monitoring    │      │
│     │ Dashboard   │  │  Registry   │          │  (Drift/MAPE)  │      │
│     └────────────┘  └─────────────┘          └────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Series Segmentation Classifier

**Purpose:** Route each series to the appropriate model family before training.

**Input:** Gold layer (Channel × Material × Sell-in time series)

**Logic:**
```python
def classify_series(series: pd.Series, min_obs_regular=13, min_obs_intermittent=4) -> str:
    """
    Classify a Sell-in time series into demand segment.
    Returns: 'regular' | 'intermittent' | 'rare' | 'dead'
    """
    active_weeks = (series > 0).sum()
    
    if active_weeks >= min_obs_regular:
        return 'regular'
    elif active_weeks >= min_obs_intermittent:
        return 'intermittent'
    elif active_weeks >= 1:
        return 'rare'
    else:
        return 'dead'
```

**Segment Sizes (estimated, 2024 basis):**
| Segment | Criteria | Est. Series | Strategy |
|---|---|---|---|
| Regular | ≥ 13 active weeks | ~739 (6%) | LightGBM global model |
| Intermittent | 4–12 active weeks | ~1,601 (13%) | Croston / TSB / IMAPA |
| Rare | 1–3 active weeks | ~3,796 (31%) | Historical mean + family analog |
| Dead | 0 active weeks | ~6,209 (50%) | Zero forecast / cold-start logic |

**Segmentation is recalculated before every training run** using the most recent 52 weeks as the evaluation window.

---

### 2.2 Global LightGBM Model (Regular Segment)

**Framework:** MLForecast 0.x with LightGBM backend

**Key Design Choices:**

```python
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from lightgbm import LGBMRegressor

model = MLForecast(
    models={
        'lgbm': LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_data_in_leaf=20,       # prevents overfit on sparse data
            lambda_l1=0.1,             # L1 regularization
            lambda_l2=0.1,             # L2 regularization
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1
        )
    },
    freq='W',
    lags=[1, 2, 4, 13, 26, 52],
    lag_transforms={
        1: [(rolling_mean, 4), (rolling_mean, 13), (rolling_std, 4)],
        52: [(rolling_mean, 4)],        # YoY same-period average
    },
    date_features=['week', 'month', 'quarter'],
    target_transforms=[Differences([52])],  # Remove annual seasonality
)
```

**Cross-Validation Strategy:**
- Walk-forward expanding window
- Minimum 52 weeks training
- 8 validation folds × 4-week step
- Horizon evaluated: H+1, H+4, H+8, H+19

**Leakage Prevention:**
- All lag features computed with strict `shift(1)` minimum
- Inventory features use `inv_lag_1` (t−1 only)
- No future information in any feature
- CI test: Pearson correlation between any feature and residual at t must be < 0.05

---

### 2.3 Intermittent Demand Models (StatsForecast)

**Framework:** StatsForecast (Nixtla)

```python
from statsforecast import StatsForecast
from statsforecast.models import (
    CrostonOptimized,
    TSB,         # Teunter-Syntetos-Babai
    IMAPA,       # Intermittent Multiple Aggregation Prediction Algorithm
    HistoricAverage
)

sf_intermittent = StatsForecast(
    models=[
        CrostonOptimized(),
        TSB(alpha_d=0.3, alpha_p=0.1),
        IMAPA(),
    ],
    freq='W',
    n_jobs=-1
)
```

**Model Selection per Series:** Best model by MAPE on 4-fold cross-validation. If all models produce identical output (constant zero), fallback to `HistoricAverage`.

---

### 2.4 Hierarchical Reconciliation

**Framework:** StatsForecast HierarchicalReconciliation

**Hierarchy Definition:**
```
Level 0: Total (all channels, all products)
Level 1: Product Family (MOBILE, LED TV, QLED TV, ...)
Level 2: Product Family × Channel
Level 3: Material (SKU) × Channel     ← base level forecast
```

**Reconciliation Method:** MinT-Shrink (Minimum Trace with shrinkage estimator)
- Shrinkage handles ill-conditioned covariance matrices (common with many series)
- Guarantees additive consistency at all levels
- Produces coherent probabilistic intervals

```python
from statsforecast.models import AutoARIMA
from hierarchicalforecast.methods import MinTrace
from hierarchicalforecast.core import HierarchicalReconciliation

hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method='mint_shrink')])
```

**Validation:** Sum of Level 3 forecasts must equal Level 0 forecast ± 0.01 (floating point tolerance).

---

### 2.5 Cold-Start Logic (Dead / New SKUs)

**Problem:** SKUs with zero history (new launches, newly listed at a channel) need a forecast.

**Strategy:**
```
1. Find "analog SKUs" — same Product Family, similar model generation, similar size
2. Use analog's demand pattern scaled by:
   - Channel historical volume ratio (how big is this channel for this family?)
   - SKU price tier relative to analog (if available)
3. If no analog exists: use family-level per-channel average
4. Apply exponential ramp-up curve (first 4 weeks discounted)
```

```python
def cold_start_forecast(
    new_material: str,
    channel: str,
    product_family: str,
    gold_df: pd.DataFrame,
    weeks: int = 19
) -> pd.Series:
    # Find analog: same family, most recent model, same channel
    analogs = gold_df[
        (gold_df['product_family'] == product_family) &
        (gold_df['Channel'] == channel) &
        (gold_df['sku_age_weeks'] > 8) &   # established SKUs only
        (gold_df['sku_age_weeks'] < 104)    # not too old
    ]
    if analogs.empty:
        # Fallback to family average for this channel
        return family_channel_average(product_family, channel, gold_df, weeks)
    
    # Take median of top-3 analogs by recency
    analog_forecast = analogs.nlargest(3, 'cust_sales_ma4')['sell_in'].median()
    ramp = np.array([0.3, 0.6, 0.8, 0.9] + [1.0] * (weeks - 4))[:weeks]
    return pd.Series(analog_forecast * ramp)
```

---

### 2.6 Probabilistic Forecasting

**Point forecast:** Primary output (mean prediction)

**Prediction Intervals:** Computed via:
- LightGBM: quantile regression at q=[0.1, 0.25, 0.5, 0.75, 0.9]
- Croston/TSB: bootstrap simulation (500 samples)

**Output format:**
```
(channel, material, yearweek) → {
    p10, p25, p50, p75, p90, mean
}
```

---

## 3. Model Registry

Models are registered in MLflow with:

| Attribute | Value |
|---|---|
| Stage | Staging → Production → Archived |
| Aliases | `champion`, `challenger` |
| Tags | `segment`, `training_date`, `mape_regular`, `mape_intermittent` |
| Artifacts | model pickle, feature importance, CV results, validation plots |

**Promotion Criteria (Staging → Production):**
- MAPE (Regular) < 25%
- MAPE (Intermittent) < 40%
- Bias < ±10%
- No feature leakage detected
- Reconciliation residual < 0.01

---

## 4. Forecast Output Schema

```sql
CREATE TABLE forecast_output (
    channel              VARCHAR(20),
    material             VARCHAR(200),
    yearweek             CHAR(6),
    forecast_date        DATE,           -- when forecast was generated
    model_name           VARCHAR(50),    -- 'lgbm_global', 'croston', etc.
    model_version        VARCHAR(20),    -- MLflow run ID
    segment              VARCHAR(20),    -- 'regular', 'intermittent', etc.
    horizon_weeks        INT,            -- 1, 4, 8, 19
    sell_in_p10          FLOAT,
    sell_in_p25          FLOAT,
    sell_in_p50          FLOAT,          -- primary point forecast
    sell_in_p75          FLOAT,
    sell_in_p90          FLOAT,
    sell_in_mean         FLOAT,
    is_reconciled        BOOLEAN,
    is_cold_start        BOOLEAN,
    PRIMARY KEY (channel, material, yearweek, forecast_date, model_name)
);
```

---

## 5. The 10 Validation Checkpoints

### V1 — Hierarchical Leakage Check
```python
# Ensure no future sell-in info leaks into features
assert all(lag >= 1 for lag in model_config['lags']), "Lag must be >= 1"
assert 'sell_in_lag_0' not in feature_names, "t=0 lag is leakage"
```

### V2 — Inventory Temporal Semantics
```python
# Inventory at t reflects stock BEFORE week t transactions
# Feature must use inv(t-1) not inv(t)
assert 'inv_lag_1' in features, "Must use lagged inventory"
assert 'channel_inv' not in direct_features, "Current inventory is leakage"
```

### V3 — Sparse Series Handling
```python
# Validate Croston is applied only to intermittent, not regular
for series_id in intermittent_series:
    active_weeks = (series > 0).sum()
    assert 4 <= active_weeks < 13, f"Series {series_id} misclassified"
```

### V4 — Truncated Value Mitigation
```python
# Flag all 2025 values == 999 as censored
df_2025['is_censored'] = (df_2025['value'] == 999)
# Exclude censored values from loss computation
assert model.loss_function != 'mse_with_censored', "Use censored-aware loss"
```

### V5 — Duplicate Aggregation Logic
```python
# Validate no duplicates exist in Silver layer
assert df_silver.duplicated(['channel','material','category','yearweek']).sum() == 0
```

### V6 — Forecast Granularity Strategy
```python
# Verify middle-out hierarchy is intact
assert set(df_hierarchy['level'].unique()) == {'sku_channel', 'family_channel', 'family', 'total'}
```

### V7 — Cross-Series Learning Approach
```python
# Verify global model trained on ALL segments, not per-series
assert model.n_series_trained > 1, "Model must be global"
assert model.n_series_trained == len(regular_series_ids)
```

### V8 — Cold-Start Handling
```python
# Every dead/new series must have a non-null forecast
dead_series_forecast = forecast_df[forecast_df['segment'].isin(['dead','new'])]
assert dead_series_forecast['sell_in_p50'].notna().all(), "Missing cold-start forecast"
```

### V9 — Inventory-Aware Forecasting
```python
# Days of Supply is included as feature
assert 'days_of_supply' in model.feature_names_
# High-DOS series should produce lower sell-in predictions (validate direction)
high_dos = test_df[test_df['days_of_supply'] > 60]
assert high_dos['predicted'].mean() < overall_mean_prediction * 0.8
```

### V10 — Promotion Blindness Limitation
```python
# Document promotion events not in training data
promotion_weeks = [202217, 202316, 202417]  # placeholder — needs business input
for wk in promotion_weeks:
    week_error = evaluation_df[evaluation_df['yearweek']==wk]['mape'].mean()
    if week_error > 50:
        logger.warning(f"Promotion blindness detected at week {wk}: MAPE={week_error:.1f}%")
```

---

## 6. Technology Stack Summary

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| ML Orchestration | MLForecast | 0.13+ | Global model training, lag features |
| Primary Learner | LightGBM | 4.x | Gradient boosting for Regular segment |
| Secondary Learner | CatBoost | 1.2+ | Challenger model / categorical features |
| Statistical Models | StatsForecast | 1.6+ | Baselines, Croston, IMAPA, TSB |
| Reconciliation | HierarchicalForecast | 0.3+ | MinT-Shrink reconciliation |
| Data Processing | Polars | 0.20+ | Fast DataFrame ops on large wide tables |
| Query Engine | DuckDB | 0.10+ | SQL on Parquet |
| Experiment Tracking | MLflow | 2.x | Model registry, metrics, artifacts |
| Orchestration | Prefect | 2.x | Pipeline scheduling and monitoring |
| Versioning | DVC | 3.x | Dataset and model versioning |
| Serving | Python scripts + Parquet | — | Batch inference, no real-time needed |

---

*AI-DLC Traceability ID: ML-ARCH-ITER2-001 | Version: 2.0*
