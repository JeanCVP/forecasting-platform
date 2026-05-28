# Inventory Risk Framework
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## 1. Framework Overview

The Inventory Risk Framework produces a **weekly risk score per SKU-Channel series**, classifying each into risk categories and prioritizing remediation actions for supply chain and sales teams.

```
RISK FRAMEWORK COMPONENTS
─────────────────────────────────────────────────────────────────
1. RISK SCORING      — composite score per series (0–100)
2. RISK CLASSIFICATION — 5-level categorical label
3. RISK DRIVERS      — which factors contribute most to score
4. TREND DETECTION   — is risk increasing or decreasing?
5. REMEDIATION ACTIONS — recommended action per risk type
6. IMPACT ESTIMATION — financial/operational impact of risk
─────────────────────────────────────────────────────────────────
```

---

## 2. Risk Classification Tiers

| Tier | Label | DOS Range | Color | Action |
|---|---|---|---|---|
| 5 | CRITICAL OVERSTOCK | > 90 days | 🔴 Dark Red | Immediate markdown / liquidation |
| 4 | HIGH OVERSTOCK | 60–90 days | 🟠 Orange | Suppress next orders; promotional push |
| 3 | HEALTHY | 14–60 days | ✅ Green | No action required |
| 2 | LOW STOCK WARNING | 7–14 days | 🟡 Yellow | Expedite next order |
| 1 | CRITICAL STOCKOUT | < 7 days | 🔴 Red | Emergency replenishment |

---

## 3. Composite Risk Score Model

The risk score (0–100) combines multiple signals:

```python
# src/product/inventory_risk_scorer.py

import polars as pl
import numpy as np

def compute_risk_score(series_row: dict) -> dict:
    """
    Compute composite inventory risk score for one SKU-Channel series.
    
    Returns: {
        'risk_score': float (0-100, higher = more risk),
        'risk_tier': str,
        'risk_type': str ('overstock' | 'stockout' | 'healthy'),
        'drivers': dict,
    }
    """
    
    dos = series_row.get("days_of_supply", 30)
    inv_velocity = series_row.get("inv_delta_1", 0)       # + = building inventory
    sell_through = series_row.get("sell_through_rate_4w", 0.8)
    prob_nonzero = series_row.get("prob_nonzero_13w", 0.5) # demand regularity
    sales_trend = series_row.get("sales_ma4", 1) / max(series_row.get("sales_ma13", 1), 0.01)
    
    # ─── OVERSTOCK RISK SCORE ────────────────────────────────────
    overstock_score = 0
    overstock_drivers = {}
    
    # Component 1: DOS level (0–40 points)
    if dos > 90:
        dos_score = 40
    elif dos > 60:
        dos_score = 20 + (dos - 60) / 30 * 20
    else:
        dos_score = max(0, (dos - 45) / 15 * 20)
    overstock_score += dos_score
    overstock_drivers["dos_component"] = dos_score
    
    # Component 2: Inventory building velocity (0–25 points)
    if inv_velocity > 0:
        velocity_score = min(inv_velocity / max(series_row.get("sales_ma4", 1), 0.01) * 25, 25)
    else:
        velocity_score = 0
    overstock_score += velocity_score
    overstock_drivers["velocity_component"] = velocity_score
    
    # Component 3: Low sell-through (0–20 points)
    if sell_through < 0.5:
        st_score = 20
    elif sell_through < 0.75:
        st_score = (0.75 - sell_through) / 0.25 * 20
    else:
        st_score = 0
    overstock_score += st_score
    overstock_drivers["sell_through_component"] = st_score
    
    # Component 4: Declining demand trend (0–15 points)
    if sales_trend < 0.7:
        trend_score = 15
    elif sales_trend < 0.9:
        trend_score = (0.9 - sales_trend) / 0.2 * 15
    else:
        trend_score = 0
    overstock_score += trend_score
    overstock_drivers["demand_trend_component"] = trend_score
    
    # ─── STOCKOUT RISK SCORE ─────────────────────────────────────
    stockout_score = 0
    stockout_drivers = {}
    
    # Component 1: Low DOS (0–40 points)
    if dos < 7:
        dos_low_score = 40
    elif dos < 14:
        dos_low_score = 20 + (14 - dos) / 7 * 20
    else:
        dos_low_score = 0
    stockout_score += dos_low_score
    stockout_drivers["dos_component"] = dos_low_score
    
    # Component 2: Inventory depletion velocity (0–30 points)
    if inv_velocity < 0:
        depletion_score = min(abs(inv_velocity) / max(series_row.get("sales_ma4", 1), 0.01) * 30, 30)
    else:
        depletion_score = 0
    stockout_score += depletion_score
    stockout_drivers["velocity_component"] = depletion_score
    
    # Component 3: High demand regularity + high sell-through (0–20 points)
    if prob_nonzero > 0.7 and sell_through > 1.0:
        demand_score = 20
    elif sell_through > 0.9:
        demand_score = (sell_through - 0.9) / 0.1 * 10
    else:
        demand_score = 0
    stockout_score += demand_score
    stockout_drivers["demand_pressure_component"] = demand_score
    
    # Component 4: Demand acceleration (0–10 points)
    if sales_trend > 1.3:
        accel_score = 10
    elif sales_trend > 1.1:
        accel_score = (sales_trend - 1.1) / 0.2 * 10
    else:
        accel_score = 0
    stockout_score += accel_score
    stockout_drivers["acceleration_component"] = accel_score
    
    # ─── DETERMINE DOMINANT RISK ──────────────────────────────────
    if overstock_score > stockout_score:
        risk_score = overstock_score
        risk_type = "overstock"
        drivers = overstock_drivers
    elif stockout_score > 5:
        risk_score = stockout_score
        risk_type = "stockout"
        drivers = stockout_drivers
    else:
        risk_score = max(overstock_score, stockout_score)
        risk_type = "healthy"
        drivers = {}
    
    # ─── CLASSIFY TIER ───────────────────────────────────────────
    if risk_type == "overstock":
        if dos > 90:     tier = "CRITICAL_OVERSTOCK"
        elif dos > 60:   tier = "HIGH_OVERSTOCK"
        else:            tier = "HEALTHY"
    elif risk_type == "stockout":
        if dos < 7:      tier = "CRITICAL_STOCKOUT"
        elif dos < 14:   tier = "LOW_STOCK_WARNING"
        else:            tier = "HEALTHY"
    else:
        tier = "HEALTHY"
    
    return {
        "risk_score": min(round(risk_score, 1), 100),
        "risk_tier": tier,
        "risk_type": risk_type,
        "drivers": drivers,
        "primary_driver": max(drivers, key=drivers.get) if drivers else "none",
    }
```

---

## 4. Risk Trend Detection

```python
def compute_risk_trend(
    series_risk_history: pl.DataFrame,
    window_weeks: int = 4
) -> str:
    """
    Determine if risk is improving, stable, or worsening over recent weeks.
    
    Returns: 'improving' | 'stable' | 'worsening' | 'critical_worsening'
    """
    recent_scores = series_risk_history.sort("yearweek").tail(window_weeks)["risk_score"].to_numpy()
    
    if len(recent_scores) < 2:
        return "stable"
    
    # Linear trend
    slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
    
    if slope > 5:
        return "critical_worsening"
    elif slope > 2:
        return "worsening"
    elif slope < -2:
        return "improving"
    else:
        return "stable"
```

---

## 5. Recommended Actions by Risk Tier

| Risk Tier | Recommended Action | Urgency | Owner |
|---|---|---|---|
| CRITICAL_OVERSTOCK (DOS > 90d) | Suspend sell-in orders; initiate markdown / trade return negotiation | Immediate | Sales + Finance |
| HIGH_OVERSTOCK (DOS 60–90d) | Reduce next order by 50%; schedule channel promotional push | This week | Sales Manager |
| LOW_STOCK_WARNING (DOS 7–14d) | Expedite next sell-in order; alert logistics | This week | Demand Planner |
| CRITICAL_STOCKOUT (DOS < 7d) | Emergency replenishment; check adjacent channel stock for transfer | Today | Supply Chain |
| HEALTHY (DOS 14–60d) | No action; monitor weekly | — | — |

---

## 6. Impact Estimation

```python
def estimate_risk_impact(series_row: dict, unit_cost: float = 100.0) -> dict:
    """
    Estimate financial and operational impact of inventory risk.
    Note: unit_cost is a placeholder; actual costs from ERP if available.
    """
    
    dos = series_row["days_of_supply"]
    inv = series_row["inv_lag_1"]
    sales_rate_weekly = max(series_row["sales_ma4"], 0.01)
    
    impact = {}
    
    # Overstock impact
    if dos > 60:
        target_inv = sales_rate_weekly * (45.0 / 7.0)
        excess_units = max(inv - target_inv, 0)
        impact["excess_units"] = excess_units
        impact["estimated_carrying_cost_weekly"] = excess_units * unit_cost * 0.003  # 0.3% weekly
        impact["weeks_to_healthy_dos"] = (dos - 45) / 7  # weeks at current rate
        impact["impact_type"] = "overstock"
        impact["severity_label"] = "🔴 Critical" if dos > 90 else "🟠 High"
    
    # Stockout impact
    elif dos < 14:
        weeks_to_stockout = dos / 7.0
        lost_sales_units = sales_rate_weekly * max(0, 2 - weeks_to_stockout)  # est. 2 weeks lost
        impact["weeks_to_stockout"] = weeks_to_stockout
        impact["estimated_lost_sales_units"] = lost_sales_units
        impact["impact_type"] = "stockout"
        impact["severity_label"] = "🔴 Critical" if dos < 7 else "🟡 Warning"
    
    else:
        impact["impact_type"] = "healthy"
        impact["severity_label"] = "✅ Healthy"
    
    return impact
```

---

## 7. Risk Output Schema

```sql
-- data/gold/inventory_risk.parquet
CREATE TABLE inventory_risk (
    channel             VARCHAR(20),
    material            VARCHAR(200),
    yearweek            CHAR(6),
    product_family      VARCHAR(50),
    
    -- Core metrics
    days_of_supply      FLOAT,
    inv_lag_1           FLOAT,
    sales_ma4           FLOAT,
    inv_delta_1         FLOAT,
    sell_through_rate   FLOAT,
    
    -- Risk scoring
    risk_score          FLOAT,          -- 0–100
    risk_tier           VARCHAR(30),    -- CRITICAL_OVERSTOCK, HIGH_OVERSTOCK, HEALTHY, LOW_STOCK_WARNING, CRITICAL_STOCKOUT
    risk_type           VARCHAR(20),    -- overstock, stockout, healthy
    primary_driver      VARCHAR(50),
    risk_trend          VARCHAR(25),    -- improving, stable, worsening, critical_worsening
    
    -- Impact
    excess_units        FLOAT,
    weeks_to_stockout   FLOAT,
    
    -- Metadata
    computed_at         TIMESTAMP,
    
    PRIMARY KEY (channel, material, yearweek)
);
```

---

## 8. Risk Aggregations for Dashboard

```python
# Precomputed aggregations refreshed weekly

def compute_risk_summary(risk_df: pl.DataFrame, current_week: str) -> dict:
    """Risk portfolio summary for Executive dashboard."""
    
    week_df = risk_df.filter(pl.col("yearweek") == current_week)
    
    return {
        "total_active_series":      len(week_df),
        "critical_overstock_count": len(week_df.filter(pl.col("risk_tier") == "CRITICAL_OVERSTOCK")),
        "high_overstock_count":     len(week_df.filter(pl.col("risk_tier") == "HIGH_OVERSTOCK")),
        "healthy_count":            len(week_df.filter(pl.col("risk_tier") == "HEALTHY")),
        "low_stock_count":          len(week_df.filter(pl.col("risk_tier") == "LOW_STOCK_WARNING")),
        "critical_stockout_count":  len(week_df.filter(pl.col("risk_tier") == "CRITICAL_STOCKOUT")),
        "overstock_excess_units":   week_df["excess_units"].sum(),
        "pct_at_risk":              (
            len(week_df.filter(pl.col("risk_tier").is_in([
                "CRITICAL_OVERSTOCK", "HIGH_OVERSTOCK", 
                "LOW_STOCK_WARNING", "CRITICAL_STOCKOUT"
            ]))) / len(week_df) * 100
        ),
        "worsening_trend_count":    len(week_df.filter(
            pl.col("risk_trend").is_in(["worsening", "critical_worsening"])
        )),
    }
```

---

## 9. Risk Score Calibration

The composite risk score is calibrated so:

| Score Range | Expected Tier | % of Series (typical) |
|---|---|---|
| 0–10 | HEALTHY | ~65% |
| 11–25 | HEALTHY (watch) | ~12% |
| 26–40 | HIGH_OVERSTOCK or LOW_STOCK | ~10% |
| 41–60 | HIGH_OVERSTOCK | ~8% |
| 61–80 | CRITICAL_OVERSTOCK | ~3% |
| 81–100 | CRITICAL_OVERSTOCK or CRITICAL_STOCKOUT | ~2% |

Calibration is reviewed quarterly against business outcomes (actual markdowns, actual stockouts).

---

*AI-DLC Traceability ID: INV-RISK-ITER2-001 | Version: 2.0*
