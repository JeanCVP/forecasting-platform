# Data Dictionary
**AI-DLC — Initial Assessment | Phase 0**
**Generated:** 2026-05-21

---

## 1. Files Covered

| File | Year | Rows | Cols |
|---|---|---|---|
| `2023.csv` | 2023 | 43,338 | 55 |
| `2024.csv` | 2024 | 37,035 | 55 |
| `2025.csv` | 2025 | 25,239 | 55 |

---

## 2. Dimension Columns (Columns 1–3)

### `Channel`
| Attribute | Value |
|---|---|
| **Type** | String (categorical) |
| **Role** | Primary dimension — identifies the retail/distribution customer |
| **Format** | `CUSTOMER{N}` where N is a sequential integer |
| **Cardinality** | 92 (2023) · 80 (2024) · 74 (2025) |
| **Nulls** | 0 |
| **Notes** | Values are anonymized. Represents trade partners (retailers, distributors, chains). Not a single end-consumer. Customers appear/disappear across years, indicating channel churn. |

---

### `Material Description`
| Attribute | Value |
|---|---|
| **Type** | String (high-cardinality categorical) |
| **Role** | Primary dimension — identifies the product SKU |
| **Format** | Comma-delimited composite: `{Product Type},{Model},{Geography/Specs},...` |
| **Cardinality** | 1,645 (2023) · 1,386 (2024) · 1,159 (2025) |
| **Nulls** | 0 |
| **Example Values** | `MOBILE,SM-A057M,COLOMBIA,LIGHT BLUE,64GB` / `QLED TV,QN55Q60DAK,55,COLOMBIA,QWK30/Q55` / `MWO(COMMON),1.1,120V 60HZ,REAL STAINLESS` |
| **Parsed Components** | `[0]` Product family · `[1]` Model code · `[2]` Size or variant · `[3]` Market (COLOMBIA) · `[4+]` Color/config/variant |
| **Notes** | Field is a natural language composite, not a single code. Parsing requires splitting on commas. Same physical product may have slightly different descriptions across years if labelling changes. |

---

### `Category`
| Attribute | Value |
|---|---|
| **Type** | String (low-cardinality categorical) |
| **Role** | Metric type differentiator — determines what the weekly numbers represent |
| **Cardinality** | 3 (fixed) |
| **Values** | `Sell-in` · `Cust. Sales` · `Channel Inv.` |
| **Nulls** | 0 |
| **Distribution** | Perfectly equal — exactly 1/3 of rows per year per value |

#### Category Definitions

| Value | Alias | Business Meaning | Nature |
|---|---|---|---|
| `Sell-in` | Shipment / Replenishment | Units shipped **from manufacturer to the retailer/distributor** (channel). Represents manufacturer revenue recognition event. | **Flow** (weekly delta) |
| `Cust. Sales` | Sell-out / POS | Units sold **from retailer to end consumers** at point of sale. Reflects true market demand. | **Flow** (weekly delta) |
| `Channel Inv.` | Stock on Hand | Units currently **held in retailer/distributor inventory**. Should equal prior inventory + Sell-in − Cust. Sales (with adjustment). | **Stock** (snapshot / level) |

> **Critical business identity:** `Channel Inv.(t) ≈ Channel Inv.(t-1) + Sell-in(t) − Cust. Sales(t)`. Deviations indicate returns, write-offs, adjustments, or data quality issues.

---

## 3. Time-Series Columns (Columns 4–55)

### Column Naming Convention
```
{YYYY}{WW}
  YYYY = 4-digit year (2023, 2024, 2025)
  WW   = 2-digit ISO week number (01 through 52)

Examples: 202301 = Week 1 of 2023
          202452 = Week 52 of 2024
          202533 = Week 33 of 2025 (last populated week)
```

### Data Types by Year
| File | Weeks 01–33 | Weeks 34–52 |
|---|---|---|
| 2023.csv | `int64` | `int64` |
| 2024.csv | `int64` | `int64` |
| 2025.csv | `str` (float-like: `"25.0"`) | `float64` (all = 0.0) |

> **2025 anomaly:** Weeks 01–33 stored as string with decimal format. This is a lossless encoding — all values convert cleanly to numeric. Weeks 34–52 are future forecast placeholders (all zeros).

### Value Semantics by Category

| Category | Unit | Expected Range | Negatives Meaning |
|---|---|---|---|
| `Sell-in` | Units shipped | 0 – ~27,500 | Returns to manufacturer / order cancellations |
| `Cust. Sales` | Units sold at POS | 0 – ~17,000 | Returns from consumer / corrections |
| `Channel Inv.` | Units in stock | 0 – ~43,000 | Data anomaly — inventory cannot be negative in physical reality |

### Observed Value Ranges

| Year | Category | Min | Max | p99 | Mean (non-zero) | % Zero |
|---|---|---|---|---|---|---|
| 2023 | Sell-in | −2,300 | 16,999 | 50 | — | ~94% |
| 2023 | Cust. Sales | −1,377 | 4,479 | 83 | — | ~79% |
| 2023 | Channel Inv. | −262 | 43,484 | 869 | — | — |
| 2024 | Sell-in | −1,000 | 27,518 | 100 | — | ~94% |
| 2024 | Cust. Sales | −555 | 17,159 | 107 | — | ~79% |
| 2024 | Channel Inv. | −523 | 27,544 | 860 | — | — |
| 2025 | Sell-in | −200 | 999 | 80 | — | — |
| 2025 | Cust. Sales | −500 | 999 | 93 | — | — |
| 2025 | Channel Inv. | −371 | 999 | 405 | — | — |

> **2025 cap at 999:** The 2025 max is 999 across all categories, versus 27,500+ in prior years. This suggests a data export cap or system limitation on the 2025 source. **8 cells exactly = 999** and **2 cells exactly = −500** in 2025, confirming boundary clamping.

---

## 4. Composite Primary Key

The logical primary key for a unique time series is:

```
(Channel, Material Description, Category, Year) → [52 weekly values]
```

**Note on duplicates:** True duplicates on this 4-tuple exist (6,390 in 2023, 6,561 in 2024, 4,008 in 2025). Analysis confirms these are **not identical rows** — they have different values in different weeks, suggesting split records for the same SKU-Channel-Category (possible sub-location split or data pipeline error). See Data Quality Report for details.

---

*Document Version: 1.0 | AI-DLC Traceability ID: ASSESSMENT-2026-001-DD*
