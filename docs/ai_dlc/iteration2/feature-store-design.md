# Feature Store Design
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## 1. Feature Store Architecture

```
FEATURE STORE (data/gold/feature_store.parquet)
─────────────────────────────────────────────────────────────────
Primary Key: (channel, material, yearweek)
Grain: One row per Channel × Material × Week (all categories pivoted)
Source: Silver timeseries_clean + dim_calendar
─────────────────────────────────────────────────────────────────
```

### Why a flat parquet feature store (not a dedicated tool)?
At ~8,400 series × 137 weeks = ~1.15M rows, a flat Parquet file queried via DuckDB is sufficient for training throughput. Feast/Tecton are over-engineered for this scale and budget. The pattern remains: compute features once, store, version, reuse across all model training runs.

---

## 2. Feature Catalog

### GROUP 1: Identifiers (not model inputs — for joining/filtering)
| Feature | Type | Description |
|---|---|---|
| `channel` | str | Customer identifier |
| `material` | str | SKU identifier |
| `yearweek` | str (YYYYWW) | ISO year + week |
| `date` | date | Monday of that ISO week |
| `year` | int | Calendar year |
| `week` | int | ISO week number (1–52) |
| `product_family` | str | Parsed from Material[0] |
| `model_code` | str | Parsed from Material[1] |
| `sku_size` | str | Parsed from Material[2] |
| `channel_id` | int | Label-encoded channel |
| `material_id` | int | Label-encoded material |
| `series_id` | str | Hash(channel + material) |
| `segment` | str | regular/intermittent/rare/dead |

---

### GROUP 2: Target Variables
| Feature | Type | Description | Leakage Risk |
|---|---|---|---|
| `sell_in` | float | Weekly Sell-in units (target) | — (target) |
| `cust_sales` | float | Weekly Customer Sales units | — (secondary target) |
| `channel_inv` | float | Weekly Channel Inventory (stock) | — (not used as target) |
| `is_censored` | bool | True if value == 999 (2025 cap) | — |
| `inv_is_negative` | bool | True if channel_inv < 0 (error flag) | — |

---

### GROUP 3: Lag Features — Sell-in
*All computed with strict shift(n) — no leakage possible.*

| Feature | Lag | Description |
|---|---|---|
| `sell_in_lag_1` | t−1 | Last week's Sell-in |
| `sell_in_lag_2` | t−2 | 2 weeks ago |
| `sell_in_lag_4` | t−4 | 4 weeks ago (~1 month) |
| `sell_in_lag_8` | t−8 | 8 weeks ago (~2 months) |
| `sell_in_lag_13` | t−13 | Quarterly lag |
| `sell_in_lag_26` | t−26 | Semi-annual lag |
| `sell_in_lag_52` | t−52 | Same week prior year (YoY) |
| `sell_in_lag_51` | t−51 | YoY −1 week |
| `sell_in_lag_53` | t−53 | YoY +1 week |

*YoY triplet (51/52/53) captures year-over-year with ±1 week tolerance for calendar alignment.*

---

### GROUP 4: Lag Features — Cust. Sales (Demand Pull Signal)
| Feature | Lag | Description |
|---|---|---|
| `sales_lag_1` | t−1 | Last week's POS |
| `sales_lag_2` | t−2 | 2 weeks ago |
| `sales_lag_4` | t−4 | Monthly demand |
| `sales_lag_13` | t−13 | Quarterly demand |
| `sales_lag_52` | t−52 | Same week prior year demand |

---

### GROUP 5: Lag Features — Channel Inventory
*Critical: inventory lags must ONLY use t−1 or older to prevent leakage.*

| Feature | Lag | Description |
|---|---|---|
| `inv_lag_1` | t−1 | Stock at end of last week |
| `inv_lag_2` | t−2 | Stock 2 weeks ago |
| `inv_lag_4` | t−4 | Stock 4 weeks ago |

---

### GROUP 6: Rolling Window Features
*All windows compute over strictly lagged window (shift 1 before rolling).*

| Feature | Window | Base | Description |
|---|---|---|---|
| `sell_in_ma4` | 4w | sell_in | 4-week average Sell-in |
| `sell_in_ma13` | 13w | sell_in | 13-week average Sell-in |
| `sell_in_ma26` | 26w | sell_in | 26-week average Sell-in |
| `sell_in_std4` | 4w | sell_in | 4-week Sell-in std dev |
| `sell_in_std13` | 13w | sell_in | 13-week Sell-in std dev |
| `sell_in_cv4` | 4w | sell_in | Coefficient of Variation (std/mean) |
| `sales_ma4` | 4w | cust_sales | 4-week average Sales |
| `sales_ma13` | 13w | cust_sales | 13-week average Sales |
| `sales_ma26` | 26w | cust_sales | 26-week average Sales |
| `sales_std4` | 4w | cust_sales | 4-week Sales std dev |
| `inv_ma4` | 4w | channel_inv | 4-week average Inventory |
| `inv_ma13` | 13w | channel_inv | 13-week average Inventory |
| `inv_std4` | 4w | channel_inv | 4-week Inventory std dev |

---

### GROUP 7: Inventory Ratio Features
| Feature | Formula | Description |
|---|---|---|
| `days_of_supply` | `inv_lag_1 / max(sales_ma4/7, 0.01)` | Days of stock coverage |
| `weeks_of_supply` | `inv_lag_1 / max(sales_ma4, 0.01)` | Weeks of stock coverage |
| `sell_through_rate_4w` | `sales_ma4 / max(sell_in_ma4, 0.01)` | 4-week sell-through ratio |
| `sell_through_rate_13w` | `sales_ma13 / max(sell_in_ma13, 0.01)` | 13-week sell-through ratio |
| `inv_vs_sales_ratio` | `inv_lag_1 / max(sales_lag_1, 0.01)` | Stock-to-sales ratio |
| `inv_delta_1` | `inv_lag_1 − inv_lag_2` | Inventory velocity (1w) |
| `inv_delta_4` | `inv_lag_1 − inv_lag_4` | Inventory velocity (4w) |
| `inv_momentum` | `inv_ma4 − inv_ma13` | Short vs. medium trend |
| `replenishment_gap` | `sell_in_lag_1 − sales_lag_1` | Net flow last week |
| `cumulative_gap_4w` | `SUM(sell_in - sales)(t−1:t−4)` | 4-week net inventory build |
| `inv_overstock_flag` | `days_of_supply > 60` (0/1) | Overstock indicator |
| `inv_stockout_flag` | `days_of_supply < 14` (0/1) | Stockout risk indicator |

---

### GROUP 8: Temporal / Seasonal Features
| Feature | Formula | Description |
|---|---|---|
| `week_num` | ISO week (1–52) | Raw week number |
| `month` | Date month (1–12) | Calendar month |
| `quarter` | Date quarter (1–4) | Fiscal quarter |
| `week_sin` | `sin(2π × week / 52)` | Cyclic annual encoding |
| `week_cos` | `cos(2π × week / 52)` | Cyclic annual encoding |
| `month_sin` | `sin(2π × month / 12)` | Cyclic monthly encoding |
| `month_cos` | `cos(2π × month / 12)` | Cyclic monthly encoding |
| `is_q4` | `quarter == 4` (0/1) | Holiday quarter flag |
| `is_jan` | `month == 1` (0/1) | Post-holiday trough |
| `weeks_since_epoch` | Global week index from W1-2023 | Linear trend |
| `year_normalized` | `(year − 2023) / 2` | Scaled year for linear trend |

---

### GROUP 9: Colombian Calendar Features
| Feature | Date(s) | Description |
|---|---|---|
| `is_mothers_day_week` | W18–W19 (2nd Sunday May) | Mother's Day peak |
| `is_fathers_day_week` | W26 (3rd Sunday June) | Father's Day |
| `is_black_friday_week` | W47–W48 | Black Friday / Cyber Monday |
| `is_christmas_week` | W50–W52 | Christmas gifting |
| `is_new_year_restock` | W01–W03 | Channel restock period |
| `is_back_to_school` | W30–W34 | Back to school (Colombia: Aug) |
| `is_dia_sin_iva` | TBD (business input) | Tax-free shopping days |
| `weeks_to_black_friday` | min(0, W48 − week) | Lead-up to Black Friday |
| `weeks_after_black_friday` | max(0, week − W48) | Post-Black Friday trough |

---

### GROUP 10: YoY Seasonal Index Features
| Feature | Formula | Description |
|---|---|---|
| `yoy_sell_in_ratio` | `sell_in_lag_52 / (sell_in_lag_104 + ε)` | Sell-in growth rate YoY |
| `yoy_sales_ratio` | `sales_lag_52 / (sales_lag_104 + ε)` | Sales growth rate YoY |
| `seasonal_index_sell_in` | `mean(sell_in, week=W, year=prior) / global_mean` | Week-specific historical index |
| `seasonal_index_sales` | `mean(sales, week=W, year=prior) / global_mean` | Week-specific index |

---

### GROUP 11: Zero-Inflation Features (critical for intermittent demand)
| Feature | Formula | Description |
|---|---|---|
| `prob_nonzero_4w` | `(sell_in_lag_1:4 > 0).mean()` | % non-zero in last 4 weeks |
| `prob_nonzero_13w` | `(sell_in_lag_1:13 > 0).mean()` | % non-zero in last 13 weeks |
| `prob_nonzero_52w` | `(sell_in_lag_1:52 > 0).mean()` | Annual activity rate |
| `inter_demand_interval` | Mean weeks between non-zero events | Croston ADI metric |
| `demand_size_mean` | Mean Sell-in when > 0 | Average demand size |
| `demand_size_cv` | CV of Sell-in when > 0 | Demand size variability |
| `weeks_since_last_nonzero` | t − last week with sell_in > 0 | Recency of last order |
| `weeks_since_last_sales` | t − last week with sales > 0 | Demand recency |

---

### GROUP 12: Cross-Series Aggregate Features
| Feature | Formula | Description |
|---|---|---|
| `family_total_sell_in_4w` | SUM(sell_in_ma4) for same family, all channels | Market-level demand |
| `family_channel_share` | sell_in_ma4 / family_total_sell_in_4w | Channel's share of family |
| `channel_total_sell_in_4w` | SUM(sell_in_ma4) for same channel, all SKUs | Channel health |
| `sku_channel_share` | sell_in_ma4 / channel_total_sell_in_4w | SKU importance |

---

### GROUP 13: SKU Lifecycle Features
| Feature | Formula | Description |
|---|---|---|
| `sku_age_weeks` | weeks since first non-zero value | Product maturity |
| `sku_is_new` | `sku_age_weeks <= 8` (0/1) | New product flag |
| `sku_is_mature` | `sku_age_weeks > 26` (0/1) | Established product |
| `sku_is_retiring` | No sales in last 8 weeks AND age > 26 (0/1) | End-of-life signal |
| `channel_tenure_weeks` | weeks since channel first appeared in data | Channel age |

---

## 3. Feature Computation Pipeline

```python
# src/transformation/feature_engineering.py

import polars as pl
import numpy as np

def compute_all_features(silver_df: pl.DataFrame) -> pl.DataFrame:
    """
    Master feature engineering pipeline.
    Input: silver timeseries_clean (long format, all categories)
    Output: gold feature_store (wide format with all features)
    """
    
    # Step 1: Pivot categories to columns
    df = silver_df.pivot(
        index=['channel', 'material', 'yearweek', 'date', 'year', 'week'],
        on='category',
        values='value'
    ).rename({
        'Sell-in': 'sell_in',
        'Cust. Sales': 'cust_sales',
        'Channel Inv.': 'channel_inv'
    })
    
    # Step 2: Sort by series and time
    df = df.sort(['channel', 'material', 'date'])
    
    # Step 3: Compute lag features
    key = ['channel', 'material']
    for lag in [1, 2, 4, 8, 13, 26, 52, 51, 53]:
        df = df.with_columns([
            pl.col('sell_in').shift(lag).over(key).alias(f'sell_in_lag_{lag}'),
        ])
    for lag in [1, 2, 4, 13, 52]:
        df = df.with_columns([
            pl.col('cust_sales').shift(lag).over(key).alias(f'sales_lag_{lag}'),
        ])
    for lag in [1, 2, 4]:
        df = df.with_columns([
            pl.col('channel_inv').shift(lag).over(key).alias(f'inv_lag_{lag}'),
        ])
    
    # Step 4: Rolling features (shift 1 before rolling to prevent leakage)
    for window in [4, 13, 26]:
        df = df.with_columns([
            pl.col('sell_in').shift(1).rolling_mean(window).over(key).alias(f'sell_in_ma{window}'),
            pl.col('sell_in').shift(1).rolling_std(window).over(key).alias(f'sell_in_std{window}'),
            pl.col('cust_sales').shift(1).rolling_mean(window).over(key).alias(f'sales_ma{window}'),
            pl.col('channel_inv').shift(1).rolling_mean(window).over(key).alias(f'inv_ma{window}'),
        ])
    
    # Step 5: Inventory ratios
    df = df.with_columns([
        (pl.col('inv_lag_1') / (pl.col('sales_ma4') / 7).clip(lower_bound=0.01)).alias('days_of_supply'),
        (pl.col('sales_ma4') / pl.col('sell_in_ma4').clip(lower_bound=0.01)).alias('sell_through_rate_4w'),
        (pl.col('inv_lag_1') - pl.col('inv_lag_2')).alias('inv_delta_1'),
        (pl.col('sell_in_lag_1') - pl.col('sales_lag_1')).alias('replenishment_gap'),
    ])
    
    # Step 6: Temporal features
    df = df.with_columns([
        (2 * np.pi * pl.col('week') / 52).sin().alias('week_sin'),
        (2 * np.pi * pl.col('week') / 52).cos().alias('week_cos'),
    ])
    
    # Step 7: Add zero-inflation features
    df = df.with_columns([
        (pl.col('sell_in').shift(1) > 0).cast(pl.Float32)
          .rolling_mean(4).over(key).alias('prob_nonzero_4w'),
        (pl.col('sell_in').shift(1) > 0).cast(pl.Float32)
          .rolling_mean(13).over(key).alias('prob_nonzero_13w'),
    ])
    
    return df
```

---

## 4. Feature Store Schema (Parquet)

```
feature_store.parquet
├── Identifiers (13 columns)
├── Target Variables (5 columns)
├── Lag Features — Sell-in (9 columns)
├── Lag Features — Sales (5 columns)
├── Lag Features — Inventory (3 columns)
├── Rolling Features (13 columns)
├── Inventory Ratios (12 columns)
├── Temporal Features (13 columns)
├── Calendar Features (11 columns)
├── YoY Features (4 columns)
├── Zero-Inflation Features (8 columns)
├── Cross-Series Features (4 columns)
└── Lifecycle Features (5 columns)
─────────────────────────────────────
TOTAL: ~105 columns × ~1.15M rows
Estimated size: ~80MB compressed Parquet
```

---

## 5. Leakage Prevention Rules (Enforced in CI)

```python
# tests/test_leakage.py

FORBIDDEN_DIRECT_FEATURES = [
    'sell_in',       # target itself (lag=0)
    'cust_sales',    # same-week sales (available only after week ends)
    'channel_inv',   # current-week inventory incorporates sell_in
]

def test_no_target_leakage(feature_df):
    """No t=0 features of target or co-variates should appear as inputs."""
    model_input_cols = [c for c in feature_df.columns 
                        if not c.endswith('_target') and 'lag_0' not in c]
    for forbidden in FORBIDDEN_DIRECT_FEATURES:
        assert forbidden not in model_input_cols, \
            f"Leakage: {forbidden} used as direct feature (t=0)"

def test_inventory_lag_only(feature_df):
    """Channel inventory must only be used as lagged feature."""
    inv_features = [c for c in feature_df.columns if 'inv' in c and 'lag' not in c 
                    and c not in ['inv_overstock_flag', 'inv_stockout_flag', 
                                   'inv_delta_1', 'inv_momentum']]
    assert len(inv_features) == 0, f"Potential inventory leakage: {inv_features}"
```

---

## 6. Feature Versioning

Feature versions are tracked via DVC:
```yaml
# dvc.yaml
stages:
  feature_engineering:
    cmd: python src/transformation/feature_engineering.py
    deps:
      - data/silver/timeseries_clean.parquet
      - src/transformation/feature_engineering.py
      - data/silver/dim_calendar.parquet
    outs:
      - data/gold/feature_store.parquet
    params:
      - params.yaml:
          - features.lag_windows
          - features.rolling_windows
          - features.calendar_events
```

Every change to feature engineering code that produces a different `feature_store.parquet` hash is automatically versioned. Model runs reference the exact feature store hash used for training.

---

*AI-DLC Traceability ID: FEAT-STORE-ITER2-001 | Version: 2.0*
