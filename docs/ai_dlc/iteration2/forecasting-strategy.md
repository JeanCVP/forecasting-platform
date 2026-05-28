# Forecasting Strategy
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## 1. Strategy Overview

The forecasting strategy is **segmented, hierarchical, and global-model-first**:

```
STRATEGY PILLARS
─────────────────────────────────────────────
1. SEGMENT first  — classify series by density
2. HIERARCHY      — forecast at family level, disaggregate
3. GLOBAL MODEL   — cross-series learning > per-series
4. RECONCILE      — guarantee additive consistency
5. VALIDATE       — walk-forward CV, not random split
─────────────────────────────────────────────
```

---

## 2. Forecast Targets

| Target | Why Forecast It | Model |
|---|---|---|
| **Sell-in** (primary) | Manufacturer's replenishment decision | LightGBM global / Croston |
| **Cust. Sales** (secondary) | Demand signal used as feature in Sell-in model | LightGBM global |
| **Channel Inv.** (derived) | Not directly forecast — derived from: `Inv(t) = Inv(t-1) + Sell-in(t) − Sales(t)` | Identity |

**Important:** Channel Inventory is NOT forecasted by an independent model to avoid error accumulation. It is computed deterministically from the Sell-in and Sales forecasts. This maintains inventory balance equation consistency.

---

## 3. Hierarchy Structure

```
LEVEL 0: TOTAL
    └── LEVEL 1: PRODUCT FAMILY
            ├── MOBILE
            ├── LED TV
            ├── QLED TV
            ├── TABLET
            ├── MON
            └── ...
            └── LEVEL 2: FAMILY × CHANNEL   ← FORECAST NODE
                    ├── MOBILE × CUSTOMER2
                    ├── MOBILE × CUSTOMER3
                    └── ...
                    └── LEVEL 3: SKU × CHANNEL   ← DISAGGREGATION TARGET
                            ├── SM-A057M × CUSTOMER2
                            ├── SM-A057M × CUSTOMER3
                            └── ...
```

**Middle-out approach:**
- Direct forecast at **Level 2 (Family × Channel)** ~800 series
- Disaggregate down to Level 3 using **historical proportion method**
- Aggregate up to Level 0/1 via summation
- Reconcile all levels using MinT-Shrink

**Why not bottom-up (Level 3)?** Level 3 has 94% sparsity in Sell-in — too sparse for reliable direct forecasting.

**Why not top-down (Level 0)?** Loses channel and family variation needed for operational decisions.

---

## 4. Model Strategy by Segment

### SEGMENT A — Regular (≥13 active weeks)

**Algorithm:** LightGBM via MLForecast
**Training data:** All years combined (2023–2025 W33), with 2025 censored values excluded
**Feature engineering:** Full lag + rolling + inventory + seasonal + calendar features

**Hyperparameter Search:**
```yaml
# params.yaml
lgbm:
  n_estimators: [200, 500, 1000]
  learning_rate: [0.01, 0.05, 0.1]
  max_depth: [4, 6, 8]
  num_leaves: [15, 31, 63]
  min_data_in_leaf: [10, 20, 50]
  lambda_l1: [0, 0.1, 0.5]
  feature_fraction: [0.7, 0.8, 0.9]
```

**Evaluation metric:** sMAPE (symmetric MAPE — handles zero actuals)
**CV:** Walk-forward, 8 folds, 4-week step, 52-week min training window

**Expected MAPE:** 15–25% on Regular segment

---

### SEGMENT B — Intermittent (4–12 active weeks)

**Algorithm:** CrostonOptimized + TSB ensemble via StatsForecast
**Model selection:** Best by sMAPE on hold-out cross-validation

**Croston method:**
- Splits series into: demand size when non-zero + inter-demand interval
- Forecasts each separately; combines for final point forecast
- `CrostonOptimized`: auto-estimates smoothing parameters α

**TSB (Teunter-Syntetos-Babai):**
- Updates demand probability (p) and size (q) separately
- Better than Croston for series where demand probability changes over time
- Better suited for lifecycle dynamics (new/retiring SKUs)

**IMAPA:**
- Aggregates series at multiple temporal levels (2-period, 4-period buckets)
- Reduces variance via multi-scale averaging
- Best for series with very irregular demand size

**Expected sMAPE:** 30–45%

---

### SEGMENT C — Rare (1–3 active weeks)

**Algorithm:** Historical mean + product family analog
**Logic:**
```python
forecast = max(
    series_historical_mean,       # last seen demand level
    family_channel_mean * 0.5     # family context, conservatively halved
)
```

**Confidence:** Low. These forecasts are flagged as `low_confidence=True` in output.
**Expected sMAPE:** 50–70% (inherently unpredictable)

---

### SEGMENT D — Dead (0 active weeks)

**Sub-cases:**
| Sub-case | Definition | Forecast |
|---|---|---|
| D1: Discontinued | No sales for >26 weeks AND SKU marked old | Zero forecast |
| D2: Not yet listed | New SKU at this channel, no history | Cold-start analog |
| D3: Seasonal inactive | Sales exist in other seasons only | Seasonal pattern imputation |

**D3 detection:** If `series.sum(W27-W52) > 0` but `series.sum(W01-W26) == 0`, classify as seasonal inactive (not truly dead).

---

## 5. Cross-Validation Protocol

**Method:** Walk-Forward Validation (Expanding Window)

```
Training Window 1:  [W01-2023 ... W52-2023] → Validate [W01-2024 ... W04-2024]
Training Window 2:  [W01-2023 ... W04-2024] → Validate [W05-2024 ... W08-2024]
...
Training Window 8:  [W01-2023 ... W28-2024] → Validate [W29-2024 ... W32-2024]
```

**Fold details:**
- 8 folds × 4-week forecast horizon = 32 weeks of validation data
- Minimum training window: 52 weeks
- Step size: 4 weeks (no overlapping validation periods)

**Why expanding window?** Matches production behavior where model always trains on all available data.

**Why NOT k-fold?** Random splits cause temporal leakage — future observations would appear in training data.

---

## 6. Evaluation Metrics

| Metric | Formula | Use Case |
|---|---|---|
| **sMAPE** | `200 × |A-F| / (|A| + |F|)` | Primary — handles zero actuals |
| **MAPE** | `|A-F|/A × 100` | Secondary — used when A > 0 |
| **RMSE** | `sqrt(mean((A-F)²))` | Error magnitude in units |
| **MAE** | `mean(|A-F|)` | Interpretable in units |
| **Bias** | `mean(F-A)/mean(A) × 100` | Systematic over/under-forecast |
| **MASE** | `MAE / MAE_naive` | Scale-independent comparison vs. naive |
| **Hit Rate** | `count(|A-F|/A < 0.25)` | % within 25% accuracy |

**Threshold targets:**
| Segment | sMAPE Target | Bias Limit |
|---|---|---|
| Regular | < 20% | ±8% |
| Intermittent | < 35% | ±15% |
| Rare | < 55% | ±25% |
| Overall (weighted) | < 28% | ±10% |

---

## 7. Baseline Models (Benchmarks)

All ML models must beat these baselines to justify complexity:

| Baseline | Description | Expected sMAPE |
|---|---|---|
| **Naive Seasonal** | `F(t) = A(t−52)` — same week last year | ~35% |
| **Moving Average 4** | Mean of last 4 weeks | ~45% |
| **Historical Mean** | Mean of all historical values | ~55% |
| **Seasonal Naive + Growth** | Last year × (current year total / last year total) | ~28% |

**The LightGBM model must beat Seasonal Naive + Growth by ≥ 15% sMAPE reduction** to be considered production-worthy.

---

## 8. Promotion Blindness Mitigation

**Current limitation:** No promotional calendar available. The model will fail during major events.

**Interim strategy:**
1. **Detect outlier weeks** using residual analysis: weeks where actual > forecast × 2 are flagged as potential promotional weeks
2. **Build a retroactive event calendar** from residual spikes
3. **Override workflow:** Demand planners can input expected lift % for upcoming events in the Demand Planning Workbench
4. **Post-event learning:** After event passes, residuals feed back into model as correction signal

**Long-term fix:** Obtain formal promotional calendar from Commercial team and add as binary features.

---

## 9. Forecast Refresh Cadence

| Type | Frequency | Trigger | Horizon |
|---|---|---|---|
| Rolling forecast | Weekly | New data arrives (Monday) | W+1 to W+19 |
| Model retraining | Monthly | Scheduled OR drift alert | Full retrain |
| Hyperparameter search | Quarterly | Scheduled | Full HPO run |
| Hierarchy restructure | Annually | New SKU catalog | Rebuild segments |

---

## 10. Inventory-Aware Forecast Adjustment

After generating the base Sell-in forecast, apply inventory constraint logic:

```python
def inventory_aware_adjustment(
    sell_in_forecast: float,
    days_of_supply: float,
    sales_forecast: float,
    max_dos_threshold: float = 60.0,
    min_dos_threshold: float = 14.0
) -> float:
    """
    Adjust Sell-in forecast based on current inventory health.
    
    High inventory → reduce sell-in recommendation
    Low inventory → increase sell-in recommendation
    """
    if days_of_supply > max_dos_threshold:
        # Overstock: suppress sell-in
        adjustment = max(0.0, 1.0 - (days_of_supply - max_dos_threshold) / 30.0)
        return sell_in_forecast * adjustment
    
    elif days_of_supply < min_dos_threshold:
        # Near stockout: boost sell-in
        boost = 1.0 + (min_dos_threshold - days_of_supply) / min_dos_threshold
        return sell_in_forecast * min(boost, 2.5)  # cap at 2.5× boost
    
    return sell_in_forecast  # Healthy zone: no adjustment
```

**Note:** This is an adjustment layer on top of the ML forecast, not a replacement. The raw ML forecast is preserved in the output for comparison.

---

*AI-DLC Traceability ID: FCST-STRAT-ITER2-001 | Version: 2.0*
