# Business KPIs
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## 1. KPI Framework

KPIs are organized in three tiers:

```
TIER 1 — STRATEGIC (C-Suite, monthly)
  Revenue proxy, market efficiency, overall supply chain health

TIER 2 — OPERATIONAL (Demand Planners, Sales Managers, weekly)
  Channel-level performance, inventory health, forecast accuracy

TIER 3 — ANALYTICAL (ML Team, data scientists, weekly)
  Model performance, data quality, pipeline health
```

---

## 2. Tier 1 — Strategic KPIs

### KPI-S01: Total Sell-in Volume (Units)
```
Definition:  SUM of all weekly Sell-in values for the period
Granularity: Total → by Year → by Product Family
Formula:     SUM(sell_in) WHERE yearweek IN period_weeks
Unit:        Units
Benchmark:   YoY growth target (business-defined)
Alert:       If YTD is >10% below prior-year same period
```

### KPI-S02: Total Customer Sales (POS Units)
```
Definition:  SUM of all weekly Customer Sales (sell-out) for the period
Granularity: Total → by Channel → by Product Family
Formula:     SUM(cust_sales) WHERE yearweek IN period_weeks
Unit:        Units
Benchmark:   Sell-in volume (excess sell-in vs. POS = inventory build)
Alert:       If gap(Sell-in, POS) > 15% sustained over 4 weeks
```

### KPI-S03: Portfolio Sell-through Rate
```
Definition:  Proportion of sell-in volume that reaches end consumers
Formula:     SUM(cust_sales_ytd) / SUM(sell_in_ytd) × 100
Unit:        Percentage
Benchmark:   > 75% = healthy; 60–75% = watch; < 60% = critical
Alert:       If < 65% for any 4-week rolling period
Note:        A rate > 100% means channel is depleting inventory
```

### KPI-S04: Average Days of Supply (Portfolio)
```
Definition:  Average stock coverage across all active SKU-Channel pairs
Formula:     MEAN(channel_inv / max(cust_sales_ma4/7, 0.01))
Unit:        Days
Target zone: 14–45 days
Alert:       Portfolio avg < 14d (stockout risk) or > 60d (overstock risk)
```

### KPI-S05: Channel Inventory Level
```
Definition:  Total units held across all channels
Formula:     SUM(channel_inv) WHERE yearweek = current_week
Unit:        Units
Benchmark:   Prior year same week ± 20%
Alert:       If > 2× prior year same week (potential excess build)
```

---

## 3. Tier 2 — Operational KPIs

### KPI-O01: Days of Supply (DOS) — per Channel × SKU
```
Formula:     inv_lag_1 / max(sales_ma4 / 7, 0.01)
Unit:        Days
Thresholds:
  🔴 Critical Stockout:  DOS < 7 days
  🟠 Warning Stockout:   DOS 7–14 days
  ✅ Healthy:            DOS 14–45 days
  🟡 Warning Overstock:  DOS 45–60 days
  🔴 Critical Overstock: DOS > 60 days
Dashboard:   DB-03 (Channel Health), DB-06 (Inventory Risk Radar)
```

### KPI-O02: Sell-through Rate — per Channel × Family (4-week)
```
Formula:     sales_ma4 / max(sell_in_ma4, 0.01)
Unit:        Ratio (0–∞; > 1 means depleting stock)
Healthy:     0.75–1.10
Watch:       < 0.60 (slow selling) or > 1.20 (depleting fast)
Dashboard:   DB-03, DB-04
```

### KPI-O03: Inventory Velocity
```
Formula:     inv_delta_1 = inv_lag_1 - inv_lag_2
Unit:        Units per week (+ = building, − = depleting)
Purpose:     Early warning signal before DOS crosses thresholds
Alert:       If velocity consistently positive for 6+ weeks (overstock trajectory)
Dashboard:   DB-06
```

### KPI-O04: Channel Replenishment Rate
```
Formula:     sell_in_ma4 / max(sales_ma4, 0.01)
Unit:        Ratio
Healthy:     0.90–1.10 (replenishing at demand rate)
Alert:       < 0.70 (under-replenishing → stockout risk)
              > 1.40 (over-replenishing → overstock risk)
Dashboard:   DB-03
```

### KPI-O05: Active SKU Count
```
Formula:     COUNT(DISTINCT material) WHERE sell_in_ma4 > 0 AND yearweek = current_week
Unit:        Count
Purpose:     Portfolio health; declining = SKU rationalization or stockouts
Alert:       If drops > 10% week-over-week
Dashboard:   DB-01, DB-04
```

### KPI-O06: SKU Channel Reach
```
Formula:     COUNT(DISTINCT channel) WHERE sell_in_ma4 > 0 per material
Unit:        Channels per SKU
Purpose:     Distribution breadth; narrow reach = vulnerability
Tiers:       > 20 channels = wide; 5–20 = medium; < 5 = narrow
Dashboard:   DB-04
```

### KPI-O07: Overstock Risk Volume (Units at Risk)
```
Formula:     SUM(channel_inv - target_inv) WHERE DOS > 60d
             target_inv = cust_sales_ma4 × 45/7   (45-day target)
Unit:        Units
Purpose:     Quantifies excess inventory to liquidate
Dashboard:   DB-06
```

### KPI-O08: Stockout Risk Count
```
Formula:     COUNT(series) WHERE DOS < 14d AND sell_in_ma4 > 0
Unit:        Series count (SKU-Channel pairs)
Alert:       If > 5% of active series at stockout risk
Dashboard:   DB-06
```

---

## 4. Tier 3 — Analytical KPIs

### KPI-A01: Forecast sMAPE by Segment
```
Formula:     200 × SUM(|actual - forecast|) / SUM(|actual| + |forecast|)
Segments:    Regular (target < 20%), Intermittent (< 35%), Rare (< 55%)
Frequency:   Weekly (vs. H+1 actuals), Monthly (vs. H+4, H+8, H+19)
Dashboard:   DB-05
```

### KPI-A02: Forecast Bias
```
Formula:     MEAN(forecast - actual) / MEAN(actual) × 100
Unit:        Percentage (+ = over-forecast, − = under-forecast)
Target:      ±8% for Regular, ±15% for Intermittent
Dashboard:   DB-05
```

### KPI-A03: Forecast Hit Rate
```
Formula:     COUNT(|actual - forecast|/actual < 0.25) / COUNT(total) × 100
Unit:        Percentage (within 25% accuracy)
Target:      > 65% for Regular segment
Dashboard:   DB-05
```

### KPI-A04: MASE (Model vs. Naive)
```
Formula:     MAE_model / MAE_seasonal_naive
Unit:        Ratio (< 1 = beats naive)
Target:      < 0.85 (15% better than naive)
Dashboard:   DB-05 (internal view)
```

### KPI-A05: Data Quality Score
```
Formula:     100 - (critical_violations × 20) - (warning_violations × 5)
Unit:        Score 0–100
Target:      ≥ 85 before training, ≥ 90 in production
Dashboard:   DB-05 (internal view)
```

### KPI-A06: Feature Drift Score
```
Formula:     MAX(PSI) across monitored features
Unit:        PSI score (< 0.1 green, 0.1–0.2 yellow, > 0.2 red)
Dashboard:   DB-05 (internal view)
```

### KPI-A07: Forecast Coverage
```
Formula:     COUNT(series with forecast) / COUNT(active series) × 100
Unit:        Percentage
Target:      100% of active series (segment ≠ 'dead') must have forecast
Alert:       Any coverage < 100% halts dashboard refresh
Dashboard:   DB-05
```

---

## 5. KPI Calculation Reference

```python
# src/product/kpi_engine.py

import polars as pl

class KPIEngine:
    def __init__(self, feature_store: pl.DataFrame, forecast_output: pl.DataFrame):
        self.fs = feature_store
        self.fc = forecast_output
    
    def dos(self, week: str = None) -> pl.DataFrame:
        """Days of Supply per series."""
        df = self.fs
        if week:
            df = df.filter(pl.col("yearweek") == week)
        return df.select([
            "channel", "material", "product_family", "yearweek",
            pl.col("days_of_supply"),
            pl.when(pl.col("days_of_supply") < 7).then(pl.lit("critical_stockout"))
              .when(pl.col("days_of_supply") < 14).then(pl.lit("warning_stockout"))
              .when(pl.col("days_of_supply") < 45).then(pl.lit("healthy"))
              .when(pl.col("days_of_supply") < 60).then(pl.lit("warning_overstock"))
              .otherwise(pl.lit("critical_overstock"))
              .alias("dos_status")
        ])
    
    def sell_through_rate(self, weeks: int = 4) -> pl.DataFrame:
        return self.fs.select([
            "channel", "material", "yearweek",
            (pl.col("sales_ma4") / (pl.col("sell_in_ma4").clip(lower_bound=0.01)))
              .alias("sell_through_rate_4w")
        ])
    
    def overstock_risk_volume(self, dos_threshold: float = 60.0) -> float:
        """Total excess units across all overstock series."""
        target_inv = self.fs["sales_ma4"] * (45.0 / 7.0)
        excess = (self.fs["inv_lag_1"] - target_inv).clip(lower_bound=0)
        overstock_mask = self.fs["days_of_supply"] > dos_threshold
        return float(excess.filter(overstock_mask).sum())
    
    def portfolio_sell_through_ytd(self, ytd_weeks: list[str]) -> float:
        df = self.fs.filter(pl.col("yearweek").is_in(ytd_weeks))
        total_sales = df["cust_sales"].sum()
        total_sell_in = df["sell_in"].sum()
        return float(total_sales / (total_sell_in + 1e-6)) * 100
```

---

## 6. KPI Thresholds Summary Card

| KPI | Green ✅ | Yellow 🟡 | Red 🔴 |
|---|---|---|---|
| Days of Supply | 14–45 days | 7–14 or 45–60 | < 7 or > 60 |
| Sell-through Rate | 75–110% | 60–75% or >120% | < 60% or > 150% |
| Portfolio Sell-through YTD | > 75% | 60–75% | < 60% |
| Replenishment Rate | 0.9–1.1 | 0.7–0.9 or 1.1–1.4 | < 0.7 or > 1.4 |
| Forecast sMAPE (Regular) | < 20% | 20–30% | > 30% |
| Forecast sMAPE (Intermittent) | < 35% | 35–50% | > 50% |
| Forecast Bias | ±8% | ±8–15% | > ±15% |
| Data Quality Score | > 90 | 80–90 | < 80 |
| Feature PSI (max) | < 0.10 | 0.10–0.20 | > 0.20 |
| Forecast Coverage | 100% | 95–99% | < 95% |
| Stockout Risk (% active) | < 3% | 3–8% | > 8% |
| Overstock Risk (% active) | < 10% | 10–20% | > 20% |

---

*AI-DLC Traceability ID: BKPI-ITER2-001 | Version: 2.0*
