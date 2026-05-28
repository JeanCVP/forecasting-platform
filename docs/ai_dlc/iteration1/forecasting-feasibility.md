# Forecasting Feasibility Assessment
**AI-DLC — Initial Assessment | Phase 0**
**Generated:** 2026-05-21

---

## 1. Executive Verdict

| Dimension | Assessment | Score |
|---|---|---|
| Historical data volume | ✅ Sufficient (2.6 years) | 75/100 |
| Series density | ⚠️ Highly sparse (50% dead series) | 35/100 |
| Temporal granularity | ✅ Weekly is optimal for consumer electronics | 90/100 |
| Seasonality signals | ✅ Detectable peaks | 75/100 |
| Portfolio stability | ⚠️ High churn requires cold-start handling | 50/100 |
| Data quality readiness | 🔴 Requires significant cleaning | 40/100 |
| **Overall Feasibility** | **FEASIBLE WITH PREPROCESSING** | **61/100** |

**Verdict:** Forecasting is viable but requires mandatory preprocessing, series segmentation, and a multi-model strategy. A single-model approach applied naively to raw data will produce poor results.

---

## 2. Historical Data Sufficiency

### Available History
- **Full years:** 2023 (W01–W52), 2024 (W01–W52)
- **Partial year:** 2025 (W01–W33, i.e., through ~mid-August 2025)
- **Total weeks per series:** Up to 137 data points (W1 2023 → W33 2025)

### Minimum Viable History by Model Type

| Model Family | Min Weeks Required | Available | Verdict |
|---|---|---|---|
| Statistical (ARIMA, ETS) | 52 | 137 | ✅ Sufficient |
| Holt-Winters (seasonal) | 104 | 137 | ✅ Borderline |
| ML regression (LightGBM, XGBoost) | 26 | 137 | ✅ Sufficient |
| Deep Learning (LSTM, TFT) | 200+ | 137 | ⚠️ Marginal |
| Prophet | 52 | 137 | ✅ Sufficient |
| Croston / TSB (intermittent) | 20 | 137 | ✅ Sufficient |

### Caveat: Effective History per Series
Due to extreme sparsity, the **effective history** for most individual SKU-Channel series is far shorter:
- 50% of series: 0 effective data points (never sold)
- 31% of series: 1–4 weeks of data
- 13% of series: 5–12 weeks
- 6% of series: 13+ weeks (viable for individual modeling)

**Only ~739 series** (~6%) have sufficient individual history for standard time-series models. All others require aggregation or global model approaches.

---

## 3. Optimal Forecasting Granularity

### Recommended: Weekly
| Factor | Assessment |
|---|---|
| Raw data granularity | Weekly ✅ (no upsampling needed) |
| Business decision cycle | Weekly replenishment orders ✅ |
| Seasonality signal period | 52-week annual cycle ✅ |
| Computational feasibility | Manageable at weekly level ✅ |

### Alternative Granularities (Trade-offs)
| Granularity | Pros | Cons |
|---|---|---|
| **Daily** | More granular demand signal | Not available in raw data |
| **Weekly** ✅ (recommended) | Matches data; reduces sparsity | Still sparse at SKU-Channel level |
| **Monthly** | Reduces sparsity significantly | Loses weekly seasonality |
| **Bi-weekly** | Compromise | Non-standard; harder to interpret |

---

## 4. Forecast Horizon Assessment

### Target Horizon: W34–W52 2025 (19 weeks remaining)

| Metric | Value |
|---|---|
| Forecast horizon | 19 weeks |
| Historical lookback (recommended) | 26–52 weeks |
| Horizon-to-history ratio | 19/52 = 0.37 (acceptable) |
| Peak periods in horizon | W39 (pre-season), W48 (Black Friday prep) |
| Risk of concept drift | Medium (new models may have entered market) |

A 19-week horizon is manageable for most model types. Quality will degrade beyond W44 (~10 weeks out) for individual sparse series.

---

## 5. Seasonality Analysis

### Detected Seasonal Patterns

**Annual (52-week) cycle confirmed:**
- Consistent low-demand trough following major promotional events
- End-of-year acceleration (W44–W52)
- Mid-year secondary peak (W22–W26)
- Post-New Year restock (W01–W04)

**Weekly patterns within month:** Not directly observable (data is already weekly-aggregated)

### Seasonality by Product Family (Hypothesis)

| Product Family | Primary Season | Secondary Season |
|---|---|---|
| Mobile | Back-to-school (W30–W35) | Holiday (W46–W52) |
| TV | World Cup / FIFA events | Black Friday (W48) |
| Air Conditioning | Summer (W10–W22 in tropics) | — |
| Home Appliances | Year-round / gifting peaks | Mother's Day (W17–W18) |
| Tablets | Holiday (W46–W52) | Back-to-school |

**Colombian-specific factors to consider:**
- No standard Northern Hemisphere summer peak (tropical climate)
- Mother's Day (2nd Sunday May ≈ W18–W19) is major retail event
- Día sin IVA (tax-free shopping days) — critical demand spikes, date varies
- Black Friday adopted in Colombia since ~2019

---

## 6. Series Segmentation Strategy

### Recommended 4-Segment Approach

| Segment | Criteria | Count (est.) | Model Strategy |
|---|---|---|---|
| **Regular** | >12 active Sell-in weeks in 2024 | ~739 series | Individual ARIMA / Prophet / LightGBM |
| **Intermittent** | 4–12 active weeks | ~1,601 series | Croston / TSB / Zero-inflated models |
| **Rare** | 1–3 active weeks | ~3,796 series | Bootstrapped historical mean; category-level similarity |
| **Dead / New** | 0 active weeks in 2024 | ~6,209 series | Cold-start (product family analogs / global model) |

### Aggregation Hierarchy for Sparsity Reduction

```
Level 0: Channel × SKU × Category (most sparse — ~25K series)
Level 1: Channel × Product Family × Category (medium — ~thousands)
Level 2: Product Family × Category (aggregate — ~dozens)
Level 3: Category only (all-channel total — 3 series, most stable)
```

**Recommended ML approach:** Forecast at Level 1–2, then disaggregate using historical proportions (top-down or middle-out).

---

## 7. Candidate Predictive Variables

### Endogenous (within the dataset)

| Variable | Description | Usefulness |
|---|---|---|
| Lagged Sell-in (t−1 to t−4) | Prior shipments predict future orders | ⭐⭐⭐⭐⭐ |
| Lagged Cust. Sales (t−1 to t−4) | Demand pull signal | ⭐⭐⭐⭐⭐ |
| Channel Inventory level | High inv. → lower Sell-in | ⭐⭐⭐⭐ |
| Days of Supply (DOS) | Inv. health indicator | ⭐⭐⭐⭐ |
| Sell-through Rate | Sales / Sell-in ratio | ⭐⭐⭐⭐ |
| Rolling average (4-week, 13-week) | Trend smoothing | ⭐⭐⭐⭐ |
| Year-over-Year week comparison | Seasonality signal | ⭐⭐⭐ |
| Inventory velocity | Change in Inv. level | ⭐⭐⭐ |

### Exogenous (external — not currently in dataset)

| Variable | Availability | Usefulness |
|---|---|---|
| Colombian CPI / economic indicators | Public | ⭐⭐⭐ |
| Día sin IVA event dates | Known calendar | ⭐⭐⭐⭐⭐ |
| Mother's Day / holidays | Known calendar | ⭐⭐⭐⭐ |
| Exchange rate USD/COP | Public | ⭐⭐⭐ |
| Competitor promotional activity | Proprietary | ⭐⭐⭐ |
| Company's own promotional calendar | Proprietary | ⭐⭐⭐⭐⭐ |

---

## 8. ML Risk Register

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| Extreme sparsity causing model failure | 🔴 High | 🔴 Certain | Series segmentation; aggregation |
| Duplicate rows inflating training data | 🔴 High | 🔴 Certain | Mandatory pre-aggregation |
| 2025 data cap (999) biasing recent history | 🟠 Medium | 🟠 High | Flag & exclude capped values |
| Cold-start for new SKUs | 🟠 Medium | 🟠 High | Product family similarity; global model |
| Customer churn breaking history | 🟡 Low | 🟠 High | Model only stable channels |
| Seasonality regime shift 2023→2024 | 🟡 Low | 🟡 Medium | Year-specific seasonal indices |
| Inventory balance inconsistencies in features | 🟡 Low | 🟡 Medium | Use raw values; avoid derived inv. balance |
| Overfitting on small per-series datasets | 🔴 High | 🔴 Certain | Global model + cross-series learning |

---

## 9. Recommended Model Architecture

### Phase 1 (Baseline — 4 weeks)
- Aggregate duplicates, fix types, segment series
- Naive seasonal baseline (same week prior year)
- Simple moving average baseline

### Phase 2 (Statistical — 6 weeks)
- Prophet for Regular segment (interpretable seasonality)
- Croston / TSB for Intermittent segment
- Historical mean for Rare/Dead segments

### Phase 3 (ML — 8 weeks)
- LightGBM global model trained across all series with rich lag features
- One model per Category (Sell-in, Sales, Inventory modeled separately)
- Hierarchical reconciliation (top-down proportional disaggregation)

### Phase 4 (Advanced — ongoing)
- Temporal Fusion Transformer (if data volume justifies)
- Multi-task learning: Sell-in + Sales jointly
- Exogenous variable integration

---

*Document Version: 1.0 | AI-DLC Traceability ID: ASSESSMENT-2026-001-FF*
