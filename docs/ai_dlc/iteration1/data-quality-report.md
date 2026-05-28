# Data Quality Report
**AI-DLC — Initial Assessment | Phase 0**
**Generated:** 2026-05-21
**Classification:** CRITICAL FINDINGS INCLUDED

---

## Executive Summary

| Severity | Count | Description |
|---|---|---|
| 🔴 CRITICAL | 3 | Issues that directly compromise model integrity |
| 🟠 HIGH | 3 | Issues that will degrade forecast accuracy |
| 🟡 MEDIUM | 3 | Issues requiring preprocessing decisions |
| 🟢 LOW | 2 | Cosmetic or informational |

**Overall Data Quality Score: 61 / 100**
The dataset requires significant preprocessing before any ML pipeline can be safely executed.

---

## 1. 🔴 CRITICAL: Duplicate Primary Keys with Divergent Values

### Description
Records with identical `(Channel, Material Description, Category)` composite keys exist across all three years — but unlike simple duplicates, these rows contain **different values in different weeks**. This means data is split across multiple rows for the same logical series.

| Year | Duplicate Records | Rows Involved | Duplicate Groups (w/ different values) |
|---|---|---|---|
| 2023 | 6,390 extra rows | 11,298 | 3,617 / 4,908 (73.7%) |
| 2024 | 6,561 extra rows | 11,694 | — |
| 2025 | 4,008 extra rows | 7,287 | — |

**Example:**
`CUSTOMER6 | FWM,SELACOL,CO,22 | Sell-in` in 2023 has 8 rows, each with non-overlapping non-zero values spread across different weeks — as if the same series was split into 8 fragments.

### Root Cause Hypothesis
Data extracted from an ERP/BI system where a single SKU maps to multiple internal IDs (e.g., warehouse locations, sub-channels, cost centers) that were not aggregated before export.

### Impact
- **Severe for forecasting:** A model trained on fragments will severely underestimate actual demand
- **Aggregation required:** Must SUM all rows per `(Channel, Material, Category, Week)` before use

### ✅ Recommended Mitigation
```python
# Aggregate all duplicates by summing weekly values
df_clean = df.groupby(['Channel', 'Material Description', 'Category'])[week_cols].sum().reset_index()
```

---

## 2. 🔴 CRITICAL: 2025 Weekly Columns Stored as String Type

### Description
Weeks 202501 through 202533 in `2025.csv` have `dtype = object` (string), while the same column range in 2023 and 2024 are `int64`. Values appear as `"0.0"`, `"25.0"`, etc. — float representations serialized as strings.

| File | Weeks 01–33 dtype | Weeks 34–52 dtype |
|---|---|---|
| 2023.csv | `int64` | `int64` |
| 2024.csv | `int64` | `int64` |
| 2025.csv | `object` (string) ❌ | `float64` ✅ |

### Root Cause Hypothesis
The 2025 data was exported from a different system version or mixed Excel/CSV pipeline that cast numeric columns to strings before weeks 34+ were appended as true numeric float.

### Impact
- Any direct arithmetic on 2025 weeks will silently fail or produce wrong results
- Pandas `sum()`, `mean()`, comparisons will behave incorrectly on string columns
- Type inconsistency across years breaks unified multi-year concatenation

### ✅ Recommended Mitigation
```python
week_cols_2025 = [c for c in df25.columns if c.startswith('2025')]
df25[week_cols_2025] = df25[week_cols_2025].apply(pd.to_numeric, errors='coerce')
```

---

## 3. 🔴 CRITICAL: 2025 Values Capped at 999 (Data Truncation)

### Description
In 2025.csv, the observed maximum value across all categories is **999**, compared to 27,518 (Sell-in), 17,159 (Cust. Sales), and 27,544 (Channel Inv.) in 2024. This is a ~97% reduction in the observable ceiling.

| Category | 2023 Max | 2024 Max | 2025 Max |
|---|---|---|---|
| Sell-in | 16,999 | 27,518 | **999** |
| Cust. Sales | 4,479 | 17,159 | **999** |
| Channel Inv. | 43,484 | 27,544 | **999** |

8 cells hold exactly `999.0`, and 2 cells hold exactly `-500.0`, indicating hard clamps.

### Root Cause Hypothesis
The 2025 export system or BI layer applied a numeric clip (possibly a display formatting limit, API response limit, or data anonymization rule).

### Impact
- High-volume SKUs and channels will have **systematically underreported** 2025 data
- Training on 2025 data as-is will bias models toward lower demand estimates
- Channel Inv. values are most affected (stock levels often exceed 999)

### ✅ Recommended Mitigation
- Obtain uncapped 2025 data extract from source system
- As interim: flag all `== 999` cells as suspicious; exclude from loss computation or treat as censored observations
- Do not use 2025 data as ground truth for high-volume SKUs until resolved

---

## 4. 🟠 HIGH: Negative Values Across All Categories

### Description
Negative weekly values exist in all three categories and all three years. While returns/adjustments are a valid business concept for Sell-in and Cust. Sales, negative Channel Inventory is physically impossible.

| Year | Category | Negative Cells | Minimum Value |
|---|---|---|---|
| 2023 | Sell-in | 2,496 | −2,300 |
| 2023 | Cust. Sales | 2,190 | −1,377 |
| 2023 | Channel Inv. | 2,584 | −262 |
| 2024 | Sell-in | 1,540 | −1,000 |
| 2024 | Cust. Sales | 1,654 | −555 |
| 2024 | Channel Inv. | 804 | −523 |
| 2025 | Sell-in | 911 | −200 |
| 2025 | Cust. Sales | 788 | −500 |
| 2025 | Channel Inv. | 450 | −371 |

Notable spike: W43 2024 Sell-in has a single value of `−1,000` which is extreme relative to typical negatives of ≤ −30.

### ✅ Recommended Mitigation
- **Sell-in & Cust. Sales negatives:** Accept as valid (return corrections); clip at 0 for demand forecasting, or model separately as a returns signal
- **Channel Inv. negatives:** Flag as data errors; replace with 0 or interpolate from adjacent weeks
- Investigate extreme outlier (−1,000 in W43 2024) before training

---

## 5. 🟠 HIGH: Extreme Outliers in Sell-in

### Description
Sell-in shows extreme right-tail values inconsistent with typical weekly volumes:

| Year | p99 Sell-in | Max Sell-in | Ratio Max/p99 |
|---|---|---|---|
| 2023 | 50 | 16,999 | 340× |
| 2024 | 100 | 27,518 | 275× |

A single week's Sell-in of 27,518 units is 275× higher than the 99th percentile. These could represent:
- Bulk promotional/promotional buys
- Data entry errors (wrong unit of measure)
- Aggregation of multiple sub-channels in one row

### ✅ Recommended Mitigation
- Investigate top-20 outlier rows for business context
- Apply Winsorization at p99.9 or use robust scalers for model training
- Log-transform Sell-in before modeling

---

## 6. 🟠 HIGH: Extreme Sparsity — 50% of SKU-Channel Pairs Have Zero Sell-in Entire Year

### Description
Among the 12,345 unique SKU-Channel combinations in 2024:
- **6,209 (50.3%)** had zero Sell-in for the entire year
- **3,796 (30.8%)** had only 1–4 active weeks
- Only **739 (6.0%)** had more than 12 active weeks

### Impact
- Standard time-series models (ARIMA, Holt-Winters) cannot be applied to sparse series
- Sparse SKUs require intermittent demand models (Croston, TSB) or zero-inflated approaches
- Feature engineering must handle zero-inflation explicitly

### ✅ Recommended Mitigation
- Classify series into segments: **Regular** (>12 active weeks), **Intermittent** (4–12), **Rare** (<4), **Dead** (0)
- Apply different model families per segment
- Consider SKU-Channel aggregation to reduce sparsity

---

## 7. 🟡 MEDIUM: SKU Portfolio Volatility

### Description
The SKU portfolio contracted significantly over 3 years:

| Transition | Dropped | Added | Net Change |
|---|---|---|---|
| 2023 → 2024 | −696 SKUs | +437 new | −259 (−15.7%) |
| 2024 → 2025 | −239 SKUs | +12 new | −227 (−16.4%) |

696 SKUs disappeared in a single year (2023→2024). New SKUs have no historical data, making cold-start forecasting necessary.

### ✅ Recommended Mitigation
- Build SKU lifecycle metadata (launch date, discontinuation date)
- Use product family / product line hierarchy for cold-start imputation
- Filter discontinued SKUs from training if < 3 months of history

---

## 8. 🟡 MEDIUM: Customer Churn Across Years

### Description
17 customers present in 2023 are absent in 2024; 7 more disappear by 2025. Only 68 customers are stable across all three years.

### ✅ Recommended Mitigation
- Model only stable customers for initial forecast delivery
- Handle churned customers as series termination events
- New customers (C93–C98) require cold-start handling

---

## 9. 🟡 MEDIUM: Inventory Balance Equation Not Always Satisfied

### Description
The theoretical balance: `Inv(t) = Inv(t-1) + Sell-in(t) − Sales(t)` should hold. Preliminary spot checks show deviations, which may indicate:
- Returns and adjustments not captured separately
- Inventory adjustments (write-offs, stock corrections)
- Reporting lag between sell-in booking and inventory update

### ✅ Recommended Mitigation
- Compute inventory balance residual as a derived quality metric
- Large residuals = candidate features (channel adjustment activity)
- Do not attempt to enforce the equation artificially

---

## 10. 🟢 LOW: 2025 Weeks 34–52 Contain All-Zero Placeholder Columns

### Description
Weeks 202534–202552 exist in the schema but contain 0.0 for every row. These are forecast target weeks, not historical data.

### ✅ Recommended Mitigation
- Exclude from historical feature computation
- Use as output columns for the forecasting task

---

## 11. 🟢 LOW: Trailing Whitespace in 2025 String Columns

### Description
2025 string week values include trailing space in some entries (e.g., `"0.0 "`, `"25.0 "`). Observed in a subset of rows.

### ✅ Recommended Mitigation
```python
df25[week_cols] = df25[week_cols].apply(lambda x: x.str.strip() if x.dtype == object else x)
```

---

## 12. Data Quality Metrics Summary

| Dimension | Score | Notes |
|---|---|---|
| Completeness | 95/100 | No nulls; 50% zero rows are valid sparse data, not missing |
| Uniqueness | 55/100 | 15–18% of rows are fragmented duplicates requiring aggregation |
| Consistency | 50/100 | Type mismatch in 2025, value capping, negative inventory |
| Accuracy | 65/100 | Extreme outliers, capping, negative inv. compromise accuracy |
| Timeliness | 85/100 | 2025 data current to W33; 2 full prior years available |
| **Overall** | **61/100** | Significant preprocessing required before ML use |

---

*Document Version: 1.0 | AI-DLC Traceability ID: ASSESSMENT-2026-001-DQ*
