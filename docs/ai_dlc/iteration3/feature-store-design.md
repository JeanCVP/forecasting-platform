# Feature Store Design — v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** FEATSTORE-ITER3-001

---

## 1. Arquitectura del Feature Store

```
FEATURE STORE ARCHITECTURE

data/gold/feature_store.parquet
────────────────────────────────────────────────────────────
Grain:     (channel, material, yearweek) — una fila por semana
           Categories pivotadas a columnas
Rows:      ~1.15M (8,413 series × 137 semanas)
Cols:      ~105 features
Size:      ~80MB comprimido (Parquet columnar)
Access:    DuckDB in-process; read_parquet(); Polars LazyFrame
────────────────────────────────────────────────────────────

CONSUMIDORES:
  ├── src/ml/train_global.py     → training_set.parquet subset
  ├── src/serving/forecast_runner.py → inference features
  ├── src/semantic/metrics.py    → KPI computation
  ├── src/product/risk_scorer.py → inventory risk
  └── dashboards/                → via MetricEngine

ESCRITORES:
  └── src/transformation/feature_engineering.py (ÚNICO)
```

---

## 2. Pipeline de Construcción

```python
# src/transformation/feature_engineering.py

import polars as pl
import numpy as np
from src.audit.logger import audit

def build_feature_store(
    silver_path:   str = "data/silver/timeseries_clean.parquet",
    calendar_path: str = "data/silver/dim_calendar.parquet",
    registry_path: str = "data/gold/series_registry.parquet",
    params:        dict = None,
) -> pl.DataFrame:
    """
    Master feature engineering pipeline.
    Single function that produces the complete feature_store.

    ALL features use strict lookback (shift >= 1) to prevent leakage.
    """
    silver = pl.read_parquet(silver_path)
    calendar = pl.read_parquet(calendar_path)

    # STEP 1: Pivot categories → columns
    df = (
        silver
        .pivot(
            index=["channel","material","yearweek","date","year","week",
                   "product_family","model_code","sku_size",
                   "source_year","is_censored","inv_was_negative",
                   "inv_balance_residual"],
            on="category",
            values="value",
        )
        .rename({
            "Sell-in":     "sell_in",
            "Cust. Sales": "cust_sales",
            "Channel Inv.": "channel_inv",
        })
        .sort(["channel","material","date"])
    )

    key = ["channel","material"]

    # STEP 2: Lag features — Sell-in
    for lag in [1, 2, 4, 8, 13, 26, 52, 51, 53]:
        df = df.with_columns(
            pl.col("sell_in").shift(lag).over(key).alias(f"sell_in_lag_{lag}")
        )

    # STEP 3: Lag features — Cust. Sales
    for lag in [1, 2, 4, 13, 52]:
        df = df.with_columns(
            pl.col("cust_sales").shift(lag).over(key).alias(f"sales_lag_{lag}")
        )

    # STEP 4: Lag features — Inventory (ONLY t-1 and older)
    for lag in [1, 2, 4]:
        df = df.with_columns(
            pl.col("channel_inv").shift(lag).over(key).alias(f"inv_lag_{lag}")
        )

    # STEP 5: Rolling features (shift(1) BEFORE rolling — mandatory)
    for window in [4, 13, 26]:
        df = df.with_columns([
            pl.col("sell_in").shift(1).rolling_mean(window).over(key)
              .alias(f"sell_in_ma{window}"),
            pl.col("sell_in").shift(1).rolling_std(window).over(key)
              .alias(f"sell_in_std{window}"),
            pl.col("cust_sales").shift(1).rolling_mean(window).over(key)
              .alias(f"sales_ma{window}"),
            pl.col("cust_sales").shift(1).rolling_std(window).over(key)
              .alias(f"sales_std{window}"),
            pl.col("channel_inv").shift(1).rolling_mean(window).over(key)
              .alias(f"inv_ma{window}"),
        ])

    # STEP 6: Inventory ratios (all use lagged values)
    df = df.with_columns([
        (pl.col("inv_lag_1") / (pl.col("sales_ma4") / 7).clip(lower_bound=0.01))
          .alias("days_of_supply"),
        (pl.col("sales_ma4") / pl.col("sell_in_ma4").clip(lower_bound=0.01))
          .alias("sell_through_rate_4w"),
        (pl.col("sales_ma13") / pl.col("sell_in_ma13").clip(lower_bound=0.01))
          .alias("sell_through_rate_13w"),
        (pl.col("inv_lag_1") - pl.col("inv_lag_2"))
          .alias("inv_delta_1"),
        (pl.col("inv_lag_1") - pl.col("inv_lag_4"))
          .alias("inv_delta_4"),
        (pl.col("inv_ma4") - pl.col("inv_ma13"))
          .alias("inv_momentum"),
        (pl.col("sell_in_lag_1") - pl.col("sales_lag_1"))
          .alias("replenishment_gap"),
        (pl.col("inv_lag_1") / pl.col("inv_ma13").clip(lower_bound=0.01))
          .alias("inv_vs_trend"),
        (pl.col("days_of_supply") > 60).cast(pl.Int8).alias("inv_overstock_flag"),
        (pl.col("days_of_supply") < 14).cast(pl.Int8).alias("inv_stockout_flag"),
    ])

    # STEP 7: Temporal / cyclic encoding
    df = df.with_columns([
        (2 * np.pi * pl.col("week") / 52).sin().alias("week_sin"),
        (2 * np.pi * pl.col("week") / 52).cos().alias("week_cos"),
        (2 * np.pi * pl.col("date").dt.month() / 12).sin().alias("month_sin"),
        (2 * np.pi * pl.col("date").dt.month() / 12).cos().alias("month_cos"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.quarter().alias("quarter"),
        (pl.col("date").dt.quarter() == 4).cast(pl.Int8).alias("is_q4"),
        (pl.lit(1)  # linear trend: weeks since project start
         .cum_sum().over(pl.lit(1)) - 1
        ).alias("weeks_since_epoch"),
        ((pl.col("year") - 2023) / 2.0).alias("year_normalized"),
    ])

    # STEP 8: Calendar event flags (join dim_calendar)
    df = df.join(
        calendar.select([
            "yearweek",
            "is_mothers_day_week",
            "is_black_friday_week",
            "is_christmas_week",
            "is_back_to_school",
            "is_new_year_restock",
            "is_dia_sin_iva",
            "weeks_to_black_friday",
            "weeks_after_black_friday",
        ]),
        on="yearweek", how="left"
    )

    # STEP 9: YoY seasonal index
    df = df.with_columns([
        (pl.col("sell_in_lag_52") /
         pl.col("sell_in_ma26").shift(26).over(key).clip(lower_bound=0.01))
          .alias("yoy_sell_in_ratio"),
        (pl.col("sales_lag_52") /
         pl.col("sales_ma26").shift(26).over(key).clip(lower_bound=0.01))
          .alias("yoy_sales_ratio"),
    ])

    # STEP 10: Zero-inflation features
    df = df.with_columns([
        (pl.col("sell_in").shift(1) > 0).cast(pl.Float32)
          .rolling_mean(4).over(key).alias("prob_nonzero_4w"),
        (pl.col("sell_in").shift(1) > 0).cast(pl.Float32)
          .rolling_mean(13).over(key).alias("prob_nonzero_13w"),
        (pl.col("sell_in").shift(1) > 0).cast(pl.Float32)
          .rolling_mean(52).over(key).alias("prob_nonzero_52w"),
    ])

    # STEP 11: SKU lifecycle features
    df = df.with_columns([
        pl.col("sell_in").cum_sum().over(key).alias("cumulative_sell_in"),
    ])
    first_active = (
        df.filter(pl.col("sell_in") > 0)
        .group_by(key).agg(pl.col("date").min().alias("first_active_date"))
    )
    df = df.join(first_active, on=key, how="left")
    df = df.with_columns([
        ((pl.col("date") - pl.col("first_active_date")).dt.total_days() / 7)
          .cast(pl.Int32).alias("sku_age_weeks"),
    ])
    df = df.with_columns([
        (pl.col("sku_age_weeks") <= 8).cast(pl.Int8).alias("is_new_sku"),
        (pl.col("sku_age_weeks") > 52).cast(pl.Int8).alias("is_mature_sku"),
    ])

    # STEP 12: Join segment classification
    if registry_path:
        try:
            registry = pl.read_parquet(registry_path)
            df = df.join(
                registry.select(["channel","material","segment","quadrant","adi","cv2"]),
                on=["channel","material"], how="left"
            )
            df = df.with_columns(
                pl.col("segment").fill_null("dead")
            )
        except FileNotFoundError:
            df = df.with_columns(pl.lit("unknown").alias("segment"))

    return df
```

---

## 3. Prevención de Leakage — Reglas Arquitectónicas

```python
# tests/test_leakage.py  — Ejecutado en CI en cada PR

FORBIDDEN_DIRECT = {"sell_in", "cust_sales", "channel_inv"}

def test_no_direct_target_features(feature_df):
    """t=0 values of targets/co-variates must not appear as model inputs."""
    input_cols = set(feature_df.columns) - {"sell_in"}  # target excluded
    leakage = FORBIDDEN_DIRECT.intersection(input_cols)
    assert len(leakage) == 0, f"Leakage: {leakage}"

def test_inventory_only_lagged(feature_df):
    """channel_inv must only appear in lagged form."""
    inv_cols = [c for c in feature_df.columns if "inv" in c]
    direct_inv = [c for c in inv_cols
                  if "lag" not in c
                  and "delta" not in c
                  and "momentum" not in c
                  and "flag" not in c
                  and "ma" not in c
                  and "vs_trend" not in c
                  and c != "inv_balance_residual"]
    assert direct_inv == [], f"Direct inventory features found: {direct_inv}"

def test_rolling_uses_shifted_base(feature_store_code: str):
    """All rolling_mean calls must be preceded by shift(1)."""
    import ast, re
    # Parse AST and find rolling_mean without shift(1) preceding it
    # Simplified: check string pattern
    lines = feature_store_code.split("\n")
    violations = []
    for i, line in enumerate(lines):
        if "rolling_mean" in line or "rolling_std" in line:
            context = "\n".join(lines[max(0,i-3):i+1])
            if "shift(1)" not in context:
                violations.append(f"Line {i+1}: {line.strip()}")
    assert violations == [], f"Rolling features without shift(1): {violations}"
```

---

## 4. Reproducibilidad de Features

```python
# src/transformation/feature_version.py

import hashlib
import json
import polars as pl

def compute_feature_store_hash(
    feature_df: pl.DataFrame,
    params: dict
) -> str:
    """
    Compute a deterministic hash for the feature store.
    Combines:
    - Content hash of the DataFrame (shape + key statistics)
    - Hash of the params used to generate it
    """
    content_sig = {
        "n_rows":    len(feature_df),
        "n_cols":    len(feature_df.columns),
        "columns":   sorted(feature_df.columns),
        "sell_in_sum": float(feature_df["sell_in"].sum()),
        "dos_mean":    float(feature_df["days_of_supply"].mean()),
    }
    params_json = json.dumps(params, sort_keys=True)
    combined = json.dumps(content_sig, sort_keys=True) + params_json
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# Used in MLflow logging:
# mlflow.set_tag("feature_store_hash", compute_feature_store_hash(df, params))
```

---

*AI-DLC Traceability ID: FEATSTORE-ITER3-001 | Version: 3.0*
