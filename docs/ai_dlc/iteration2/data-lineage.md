# Data Lineage
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## 1. End-to-End Data Flow

```
SOURCE SYSTEM (ERP/BI)
      │
      │  Weekly CSV export (manual trigger)
      │  Files: 2023.csv, 2024.csv, 2025.csv
      │
      ▼
[GATE: Schema Validation]
  ✓ Column count = 55
  ✓ Column names match contract
  ✓ Encoding = UTF-8
  ✗ FAIL → Alert + block pipeline
      │
      ▼
BRONZE: ingest.py
  Transformation:
  ├── pd.melt() → wide to long format
  ├── pd.to_numeric() → fix 2025 string dtypes
  ├── str.strip() → clean whitespace
  ├── add ingested_at timestamp
  └── save as Parquet (append mode per year)
  
  Output: bronze/sell_data_{year}.parquet
  Schema: (channel, material, category, yearweek, value, ingested_at)
      │
      ▼
[GATE: Bronze DQ Check]
  ✓ Row count within expected range
  ✓ No nulls in key columns
  ✓ yearweek format = 6 digits
  ✗ FAIL → Alert + halt
      │
      ▼
SILVER: clean.py
  Transformation:
  ├── CONCAT all years
  ├── GROUPBY (channel, material, category, yearweek)
  │   → SUM(value)                    [DEDUP: aggregation]
  ├── Channel Inv. < 0 → replace with 0  [NEGATIVE FIX]
  ├── Add is_censored = (value == 999 AND year == 2025)
  ├── Parse product_family from material[0]
  ├── Parse model_code from material[1]
  ├── Derive ISO date from yearweek
  └── Build dim_channel, dim_material, dim_calendar
  
  Output: silver/timeseries_clean.parquet
  Schema: (channel, material, category, yearweek, date, year, week,
           value, is_censored, inv_is_negative, product_family,
           model_code, sku_size)
      │
      ▼
[GATE: Silver DQ Check]
  ✓ 0 duplicate (channel, material, category, yearweek) tuples
  ✓ 0 negative channel_inv values
  ✓ Row count = unique series × 52 weeks × 3 years (approx)
  ✓ is_censored rate < 0.1%
  ✗ FAIL → Alert + halt
      │
      ▼
GOLD: feature_engineering.py
  Transformation:
  ├── Pivot category → columns (sell_in, cust_sales, channel_inv)
  ├── Compute 13 groups of features (see feature-store-design.md)
  ├── Classify series (regular/intermittent/rare/dead)
  ├── Compute seasonal indices
  └── Add cross-series aggregate features
  
  Output: gold/feature_store.parquet
  Schema: 105 columns × ~1.15M rows
      │
      ▼
[GATE: Gold Feature Check]
  ✓ No NaN in lag_1 features for series with >1 week of history
  ✓ days_of_supply ∈ [0, 500] (no negatives, no extreme outliers)
  ✓ sell_through_rate_4w ∈ [0, 10] (10× as upper bound)
  ✓ Leakage test: 0 forbidden direct features
  ✗ FAIL → Alert + halt
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
TRAINING SPLIT                         INFERENCE SET
  train_flow.py                          forecast_runner.py
  ├── Filter: yearweek < cutoff          ├── Input: feature_store rows
  ├── Filter: is_censored == False           where yearweek >= W34-2025
  ├── Segment routing                    ├── Load champion model from MLflow
  └── Save training_set.parquet         └── Generate probabilistic forecasts
      │                                      │
      ▼                                      ▼
MLflow Experiment                      gold/forecast_output.parquet
  ├── Log params, metrics               Schema: (channel, material,
  ├── Log feature importance               yearweek, p10, p25, p50,
  ├── Log CV results                       p75, p90, segment, model_version)
  └── Register model in Registry           │
                                           ▼
                                   [GATE: Forecast Sanity]
                                     ✓ No negative p50 values
                                     ✓ p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90
                                     ✓ Sum(forecasts) within 3σ of historical
                                     ✗ FAIL → Alert; serve last valid forecast
                                           │
                                           ▼
                                   BI SERVING LAYER
                                   Streamlit / Power BI read
                                   forecast_output.parquet + feature_store.parquet
```

---

## 2. Transformation Audit Table

| Step | Input | Output | Transformation | Reversible? |
|---|---|---|---|---|
| Ingest 2025 | `2025.csv` | `bronze/sell_data_2025.parquet` | melt + to_numeric + strip | Yes (raw preserved) |
| Ingest 2023/24 | `202{3,4}.csv` | `bronze/sell_data_{y}.parquet` | melt | Yes |
| Deduplicate | Bronze union | Silver clean | groupby+sum | No (but Bronze preserved) |
| Neg. Inv. Fix | Silver | Silver | clip(0) | No (flagged in `inv_is_negative`) |
| Feature Eng. | Silver clean | Gold feature_store | lag/rolling/ratio | Yes (recomputable) |
| Segmentation | Gold | series_registry | rule-based | Yes (recomputable) |
| Training split | Gold | training_set | filter | Yes |
| Inference | Gold + model | forecast_output | ML prediction | Partially (model pinned) |

---

## 3. Data Provenance per Column

| Column | Origin Layer | Source Column | Transformation |
|---|---|---|---|
| `channel` | Bronze | `Channel` | Identity |
| `material` | Bronze | `Material Description` | Identity |
| `yearweek` | Bronze | Column headers (YYYYWW) | Melt key |
| `value` | Bronze | Numeric weekly cell | to_numeric |
| `date` | Silver | `yearweek` | `pd.to_datetime(YYYYWW + '1', format='%Y%W%w')` |
| `product_family` | Silver | `material` | `material.split(',')[0].strip()` |
| `is_censored` | Silver | `value`, `year` | `value == 999 AND year == 2025` |
| `sell_in` | Gold | `value` WHERE category='Sell-in' | Pivot |
| `sell_in_lag_1` | Gold | `sell_in` | `shift(1)` per series |
| `days_of_supply` | Gold | `inv_lag_1`, `sales_ma4` | `inv / (sales/7)` |
| `segment` | Gold | `sell_in` history | Active week count rule |
| `p50_forecast` | Forecast | `feature_store` | LightGBM/Croston inference |

---

## 4. DVC Pipeline Graph

```
data/raw/2023.csv ──────┐
data/raw/2024.csv ──────┤──▶ [ingest] ──▶ data/bronze/*.parquet
data/raw/2025.csv ──────┘
                                │
src/ingestion/ingest.py ────────┘
                                │
                                ▼
data/bronze/*.parquet ──────────┐
src/transformation/clean.py ────┤──▶ [clean] ──▶ data/silver/*.parquet
                                │
                                ▼
data/silver/*.parquet ──────────┐
src/transformation/features.py  ┤──▶ [features] ──▶ data/gold/feature_store.parquet
data/silver/dim_calendar.parquet┘
params.yaml (feature config)
                                │
                                ▼
data/gold/feature_store.parquet ┐
src/ml/train_global.py ─────────┤──▶ [train] ──▶ mlruns/ (MLflow)
params.yaml (model config) ─────┘                 models/champion.pkl
                                │
                                ▼
data/gold/feature_store.parquet ┐
models/champion.pkl ────────────┤──▶ [forecast] ──▶ data/gold/forecast_output.parquet
                                │
                                ▼
data/gold/forecast_output.parquet ──▶ [dashboards] ──▶ Streamlit / Power BI
```

---

## 5. Lineage Tags (MLflow)

Every MLflow run logs:
```python
mlflow.set_tags({
    'feature_store_hash': dvc.api.get_url('data/gold/feature_store.parquet'),
    'silver_hash': dvc.api.get_url('data/silver/timeseries_clean.parquet'),
    'bronze_2023_hash': dvc.api.get_url('data/bronze/sell_data_2023.parquet'),
    'bronze_2024_hash': dvc.api.get_url('data/bronze/sell_data_2024.parquet'),
    'bronze_2025_hash': dvc.api.get_url('data/bronze/sell_data_2025.parquet'),
    'training_cutoff_week': params['training_cutoff_week'],
    'training_rows': str(len(training_df)),
    'n_series_trained': str(n_series),
    'segment_distribution': json.dumps(segment_counts),
})
```

This ensures every model artifact is traceable back to its exact input data state.

---

*AI-DLC Traceability ID: LINEAGE-ITER2-001 | Version: 2.0*
