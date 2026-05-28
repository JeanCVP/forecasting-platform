# Feature Engineering Plan
**AI-DLC — Initial Assessment | Phase 0**
**Generated:** 2026-05-21

---

## 1. Overview

This document defines the full feature engineering roadmap for the multi-series weekly forecasting pipeline. All features must be computed **after** data preprocessing (duplicate aggregation, type normalization, negative handling).

**Target variable(s):**
- Primary: `Sell-in` (weekly units shipped per Channel × SKU)
- Secondary: `Cust. Sales` (weekly POS demand)
- Derived: `Channel Inv.` (for constraint/validation use)

**Feature computation base:** Long-format table
```
(Channel, Material, Category, Year_Week) → value
```

---

## 2. Preprocessing Pipeline (Pre-Feature Engineering)

```python
# Step 1: Melt wide → long
df_long = df.melt(
    id_vars=['Channel', 'Material Description', 'Category'],
    var_name='yearweek', value_name='value'
)

# Step 2: Parse yearweek
df_long['year'] = df_long['yearweek'].str[:4].astype(int)
df_long['week'] = df_long['yearweek'].str[4:].astype(int)

# Step 3: Create ISO date (Monday of each week)
import pandas as pd
df_long['date'] = pd.to_datetime(
    df_long['year'].astype(str) + df_long['week'].astype(str).str.zfill(2) + '1',
    format='%Y%W%w'
)

# Step 4: Aggregate duplicates
df_clean = df_long.groupby(
    ['Channel', 'Material Description', 'Category', 'yearweek', 'date', 'year', 'week'],
    as_index=False
)['value'].sum()

# Step 5: Pivot categories into columns
df_pivot = df_clean.pivot_table(
    index=['Channel', 'Material Description', 'yearweek', 'date', 'year', 'week'],
    columns='Category', values='value', fill_value=0
).reset_index()
df_pivot.columns.name = None
# Now has: sell_in, cust_sales, channel_inv per row
```

---

## 3. Feature Groups

### 3.1 Temporal Features

| Feature | Formula | Rationale |
|---|---|---|
| `week_of_year` | `date.dt.isocalendar().week` | Annual seasonality encoding |
| `month` | `date.dt.month` | Monthly patterns |
| `quarter` | `date.dt.quarter` | Quarterly business cycles |
| `week_sin` | `sin(2π × week / 52)` | Cyclic encoding of annual season |
| `week_cos` | `cos(2π × week / 52)` | Cyclic encoding (pair with sin) |
| `month_sin` | `sin(2π × month / 12)` | Cyclic monthly encoding |
| `month_cos` | `cos(2π × month / 12)` | Cyclic monthly encoding |
| `year` | Raw year (2023, 2024, 2025) | Year-level trend |
| `year_normalized` | `(year − 2023) / 2` | Scaled year for regression |
| `is_q4` | `quarter == 4` | Holiday/end-of-year indicator |
| `weeks_since_start` | Global week index from W1 2023 | Global trend capture |

### Colombian Holiday & Commercial Event Features

| Feature | Definition | Dates (approximate) |
|---|---|---|
| `is_dia_sin_iva` | Tax-free shopping days (0/1) | Varies by government decree |
| `is_mothers_day_week` | Week containing Mother's Day | W18–W19 (2nd Sunday May) |
| `is_black_friday_week` | Black Friday / Cyber Monday week | W47–W48 |
| `is_christmas_season` | Weeks W46–W52 | High gifting demand |
| `is_new_year_restock` | Weeks W01–W04 | Channel restocking |
| `is_easter_week` | Holy Week (variable March/April) | Retail trough |
| `weeks_to_next_event` | Distance to nearest commercial event | Lead-time demand signal |
| `weeks_since_last_event` | Recency of prior event | Post-event trough signal |

---

### 3.2 Lag Features (Sell-in)

| Feature | Formula | Rationale |
|---|---|---|
| `sell_in_lag_1` | Sell-in(t−1) | Most recent shipment |
| `sell_in_lag_2` | Sell-in(t−2) | 2-week memory |
| `sell_in_lag_4` | Sell-in(t−4) | Monthly cycle |
| `sell_in_lag_13` | Sell-in(t−13) | Quarterly cycle |
| `sell_in_lag_26` | Sell-in(t−26) | Semi-annual |
| `sell_in_lag_52` | Sell-in(t−52) | Same week prior year |
| `sell_in_lag_53` | Sell-in(t−53) | One-year-minus-one-week |
| `sell_in_lag_51` | Sell-in(t−51) | One-year-plus-one-week |

### Lag Features (Cust. Sales — demand signal for Sell-in forecasting)

| Feature | Formula | Rationale |
|---|---|---|
| `sales_lag_1` | Cust. Sales(t−1) | Demand pull |
| `sales_lag_2` | Cust. Sales(t−2) | Short-term demand memory |
| `sales_lag_4` | Cust. Sales(t−4) | Monthly demand pattern |
| `sales_lag_52` | Cust. Sales(t−52) | YoY demand comparison |

### Lag Features (Channel Inv.)

| Feature | Formula | Rationale |
|---|---|---|
| `inv_lag_1` | Channel Inv.(t−1) | Prior stock level |
| `inv_lag_2` | Channel Inv.(t−2) | 2-week inventory memory |
| `inv_lag_4` | Channel Inv.(t−4) | Monthly stock cycle |

---

### 3.3 Rolling Window Features

| Feature | Formula | Window | Rationale |
|---|---|---|---|
| `sell_in_ma4` | Mean Sell-in(t−1:t−4) | 4 weeks | Short-term trend |
| `sell_in_ma13` | Mean Sell-in(t−1:t−13) | 13 weeks | Quarterly average |
| `sell_in_ma26` | Mean Sell-in(t−1:t−26) | 26 weeks | Semi-annual baseline |
| `sell_in_std4` | StdDev Sell-in(t−1:t−4) | 4 weeks | Recent volatility |
| `sell_in_std13` | StdDev Sell-in(t−1:t−13) | 13 weeks | Medium volatility |
| `sales_ma4` | Mean Cust. Sales(t−1:t−4) | 4 weeks | Demand trend |
| `sales_ma13` | Mean Cust. Sales(t−1:t−13) | 13 weeks | Demand baseline |
| `inv_ma4` | Mean Channel Inv.(t−1:t−4) | 4 weeks | Inventory trend |
| `inv_ma13` | Mean Channel Inv.(t−1:t−13) | 13 weeks | Inventory baseline |

> **Important:** All rolling features must use **strict lookback** (exclude current week t) to prevent data leakage.

---

### 3.4 Inventory Ratio & Supply Chain Features

| Feature | Formula | Rationale |
|---|---|---|
| `days_of_supply` | `inv_lag_1 / max(sales_ma4/7, 0.01)` | Weeks of coverage |
| `weeks_of_supply` | `inv_lag_1 / max(sales_ma4, 0.01)` | Inventory health index |
| `sell_through_rate` | `sales_ma4 / max(sell_in_ma4, 0.01)` | Channel efficiency |
| `inv_vs_sales_ratio` | `inv_lag_1 / max(sales_lag_1, 0.01)` | Overstock indicator |
| `inv_delta` | `inv_lag_1 − inv_lag_2` | Inventory velocity |
| `inv_momentum` | `inv_ma4 − inv_ma13` | Stock build/depletion trend |
| `replenishment_rate` | `sell_in_ma4 / max(sales_ma4, 0.01)` | Replenishment vs. demand balance |
| `inventory_gap` | `sell_in_lag_1 − sales_lag_1` | Weekly net flow |
| `cumulative_gap_4w` | SUM(Sell-in − Sales)(t−1:t−4) | 4-week inventory build |

---

### 3.5 Seasonality Features

| Feature | Formula | Rationale |
|---|---|---|
| `sell_in_yoy_ratio` | `sell_in_lag_52 / (sell_in_lag_104 + ε)` | Year-over-year growth rate |
| `sales_yoy_ratio` | `sales_lag_52 / (sales_lag_104 + ε)` | Demand growth signal |
| `seasonal_index_week` | `mean(Sell-in, week=W, all years) / overall_mean` | Week-specific seasonal index |
| `seasonal_index_month` | `mean(Sell-in, month=M, all years) / overall_mean` | Month-level index |
| `sell_in_vs_seasonal` | `sell_in_lag_1 / seasonal_index_week` | Deseasonalized value |

---

### 3.6 SKU / Channel Identity Features

| Feature | Formula / Source | Rationale |
|---|---|---|
| `product_family` | Parsed from Material Description[0] | Product category |
| `model_code` | Parsed from Material Description[1] | Model identifier |
| `sku_size` | Parsed from Material Description[2] | Size/screen/capacity |
| `channel_id` | Encoded CUSTOMER_N → integer | Channel identity |
| `sku_id` | Label-encoded Material Description | SKU identity |
| `channel_sku_id` | Hash(Channel + Material) | Unique series ID |

### SKU Lifecycle Features

| Feature | Formula | Rationale |
|---|---|---|
| `sku_age_weeks` | Weeks since first non-zero value | Product maturity |
| `sku_is_new` | sku_age_weeks ≤ 8 (0/1) | Cold-start flag |
| `sku_is_retiring` | Last non-zero was >8 weeks ago (0/1) | End-of-life flag |
| `channel_tenure` | Weeks since channel first appeared | Channel maturity |

### Cross-Series Features (at aggregated level)

| Feature | Formula | Rationale |
|---|---|---|
| `family_total_sell_in` | SUM Sell-in across all channels for same product family | Market-level demand |
| `family_sell_in_share` | Series Sell-in / family_total_sell_in | Channel market share |
| `channel_total_sell_in` | SUM Sell-in across all SKUs for same channel | Channel health |
| `channel_sell_in_share` | Series Sell-in / channel_total_sell_in | SKU importance in channel |

---

### 3.7 Zero-Inflation Features (for Intermittent Demand)

| Feature | Formula | Rationale |
|---|---|---|
| `prob_nonzero_4w` | Count(Sell-in > 0 in t−1:t−4) / 4 | Recent activity rate |
| `prob_nonzero_13w` | Count(Sell-in > 0 in t−1:t−13) / 13 | Medium-term activity |
| `weeks_since_last_nonzero` | t − last week where Sell-in > 0 | Inter-demand interval |
| `inter_demand_interval_mean` | Mean weeks between non-zero events | Croston λ analog |
| `demand_size_when_active` | Mean Sell-in in non-zero weeks | Average demand size |

---

## 4. Feature Engineering Pipeline Pseudocode

```python
def engineer_features(df_long_clean, target_col='Sell-in', cutoff_week=None):
    """
    df_long_clean: long-format DataFrame with columns:
        [Channel, Material, yearweek, date, year, week, Sell-in, Cust. Sales, Channel Inv.]
    cutoff_week: if set, only use data before this week for feature computation (prevents leakage)
    """
    df = df_long_clean.sort_values(['Channel', 'Material', 'date'])
    key = ['Channel', 'Material']
    
    # --- Lag features ---
    for lag in [1, 2, 4, 13, 26, 52, 51, 53]:
        df[f'sell_in_lag_{lag}'] = df.groupby(key)['Sell-in'].shift(lag)
        df[f'sales_lag_{lag}'] = df.groupby(key)['Cust. Sales'].shift(lag)
    
    for lag in [1, 2, 4]:
        df[f'inv_lag_{lag}'] = df.groupby(key)['Channel Inv.'].shift(lag)
    
    # --- Rolling features ---
    for window in [4, 13, 26]:
        df[f'sell_in_ma{window}'] = (
            df.groupby(key)['Sell-in']
              .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f'sell_in_std{window}'] = (
            df.groupby(key)['Sell-in']
              .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std().fillna(0))
        )
        df[f'sales_ma{window}'] = (
            df.groupby(key)['Cust. Sales']
              .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    
    # --- Inventory ratios ---
    df['days_of_supply'] = df['inv_lag_1'] / (df['sales_ma4'] / 7).clip(lower=0.01)
    df['sell_through_rate'] = df['sales_ma4'] / df['sell_in_ma4'].clip(lower=0.01)
    df['inv_delta'] = df['inv_lag_1'] - df['inv_lag_2']
    
    # --- Temporal features ---
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_q4'] = (df['quarter'] == 4).astype(int)
    
    # --- Zero-inflation features ---
    df['prob_nonzero_4w'] = (
        df.groupby(key)['Sell-in']
          .transform(lambda x: (x.shift(1) > 0).rolling(4, min_periods=1).mean())
    )
    df['weeks_since_last_nonzero'] = (
        df.groupby(key)['Sell-in']
          .transform(lambda x: x.shift(1).apply(lambda v: 0 if v > 0 else np.nan)
                                .fillna(method='ffill').fillna(0))
    )
    
    return df
```

---

## 5. Feature Importance Priorities (Expected)

Based on domain knowledge and data characteristics:

| Priority | Feature Group | Expected Importance |
|---|---|---|
| 1 | Lag Sell-in (1, 4, 52) | Highest — autoregressive signal |
| 2 | Lag Cust. Sales (1, 4, 52) | High — demand pull |
| 3 | Seasonal features (week_sin/cos, yoy_ratio) | High — annual pattern |
| 4 | Inventory ratio (days_of_supply, inv_lag_1) | Medium-High — replenishment trigger |
| 5 | Rolling MA (4w, 13w) | Medium — trend |
| 6 | Holiday/event flags | Medium — spike prediction |
| 7 | Zero-inflation features | Medium — intermittent demand |
| 8 | SKU/Channel identity | Medium — series-specific bias |
| 9 | Cross-series aggregates | Low-Medium — market context |

---

## 6. Feature Store Schema (Target)

```sql
CREATE TABLE feature_store (
    channel           VARCHAR(20),
    material          VARCHAR(200),
    yearweek          CHAR(6),          -- e.g., '202401'
    date              DATE,
    -- Target
    sell_in           FLOAT,
    cust_sales        FLOAT,
    channel_inv       FLOAT,
    -- Lag features (auto-generated)
    sell_in_lag_1     FLOAT,
    sell_in_lag_4     FLOAT,
    sell_in_lag_52    FLOAT,
    sales_lag_1       FLOAT,
    sales_lag_4       FLOAT,
    sales_lag_52      FLOAT,
    inv_lag_1         FLOAT,
    -- Rolling features
    sell_in_ma4       FLOAT,
    sell_in_ma13      FLOAT,
    sell_in_std4      FLOAT,
    sales_ma4         FLOAT,
    inv_ma4           FLOAT,
    -- Inventory ratios
    days_of_supply    FLOAT,
    sell_through_rate FLOAT,
    inv_delta         FLOAT,
    -- Temporal
    week              INT,
    month             INT,
    quarter           INT,
    week_sin          FLOAT,
    week_cos          FLOAT,
    is_q4             INT,
    is_black_friday   INT,
    is_mothers_day    INT,
    -- Zero-inflation
    prob_nonzero_4w   FLOAT,
    weeks_since_nonzero INT,
    -- Identity
    product_family    VARCHAR(50),
    model_code        VARCHAR(50),
    channel_id        INT,
    sku_age_weeks     INT,
    PRIMARY KEY (channel, material, yearweek)
);
```

---

*Document Version: 1.0 | AI-DLC Traceability ID: ASSESSMENT-2026-001-FE*
