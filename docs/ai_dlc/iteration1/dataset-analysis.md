# Dataset Analysis Report
**AI-DLC — Initial Assessment | Phase 0**
**Generated:** 2026-05-21
**Analyst Roles:** Senior Data Engineer · Senior ML Engineer · AI-DLC Technical Analyst · Enterprise Data Architect

---

## 1. Dataset Overview

| Attribute | 2023.csv | 2024.csv | 2025.csv |
|---|---|---|---|
| **Rows** | 43,338 | 37,035 | 25,239 |
| **Columns** | 55 | 55 | 55 |
| **Dimension cols** | 3 | 3 | 3 |
| **Time-series cols** | 52 (weeks) | 52 (weeks) | 52 (weeks) |
| **Channels (customers)** | 92 | 80 | 74 |
| **Unique Materials (SKUs)** | 1,645 | 1,386 | 1,159 |
| **Categories** | 3 | 3 | 3 |
| **Total nulls** | 0 | 0 | 0 |
| **File encoding** | UTF-8 | UTF-8 | UTF-8 |

---

## 2. Schema Structure

All three files share an identical logical schema:

```
Channel            — Customer/retailer identifier (anonymized as CUSTOMER_N)
Material Description — Product SKU description (comma-delimited fields)
Category           — Metric type: Sell-in | Cust. Sales | Channel Inv.
{YYYY}{WW}         — Weekly quantity (integer or float), weeks 01–52
```

Each row represents one **Channel × Material × Category** time series for one calendar year in wide (pivoted) format.

---

## 3. Data Types

### 2023 & 2024 — Clean Types
| Column Group | Type | Count |
|---|---|---|
| Channel, Material Description, Category | String | 3 |
| Weekly values (YYYY01–YYYY52) | int64 | 52 |

### 2025 — ⚠️ Mixed Types (CRITICAL)
| Column Group | Type | Count | Notes |
|---|---|---|---|
| Channel, Material Description, Category | String | 3 | |
| Weeks 01–33 (202501–202533) | String | 33 | Values look like floats: `"0.0"`, `"25.0"` |
| Weeks 34–52 (202534–202552) | float64 | 19 | Numeric, all zero (future) |

> **Issue:** Weeks 1–33 in 2025 are stored as string dtype with trailing `.0` notation — likely an upstream export artifact (e.g., Excel float formatting applied before CSV save). All values are valid floats with no non-numeric strings detected. Safe to `pd.to_numeric()` convert.

---

## 4. Volume & Cardinality

### Rows per File Decomposed
Each file has exactly **3× the number of unique (Channel × Material)** combinations, because every combination appears once per Category:

| Year | (Channel × Material) combos | × 3 Categories | = Rows |
|---|---|---|---|
| 2023 | ~14,446 | × 3 | 43,338 |
| 2024 | ~12,345 | × 3 | 37,035 |
| 2025 | ~8,413 | × 3 | 25,239 |

### Category Distribution (perfectly balanced)
Each category always has exactly 1/3 of total rows per year — the dataset is structurally symmetric across categories.

---

## 5. Temporal Coverage

| Year | Weeks Present | Weeks with Data | Data Cut-off |
|---|---|---|---|
| 2023 | W01–W52 | W01–W52 | Full year |
| 2024 | W01–W52 | W01–W52 | Full year |
| 2025 | W01–W52 | W01–W33 | Week 33 (Aug 2025) |

2025 weeks W34–W52 are pre-allocated in the schema but contain all zeros — they represent the forecast horizon (future periods to be predicted).

**Total historical weekly observations available:** ~2.6 years × 52 weeks = ~135 data points per SKU-Channel-Category series (when series is active).

---

## 6. Granularity

- **Temporal granularity:** Weekly (ISO calendar weeks)
- **Business granularity:** Channel (customer) × Material (SKU) × Category (metric type)
- **Finest addressable unit:** One week of one metric for one SKU at one customer

---

## 7. Channel (Customer) Evolution

| Transition | Dropped | Added | Stable |
|---|---|---|---|
| 2023 → 2024 | 17 customers | 5 new (C93–C97) | 75 |
| 2024 → 2025 | 7 customers | 1 new (C98) | 73 |
| **Active all 3 years** | — | — | **68** |

Customer churn rate: ~20% across years. New customer codes appearing in 2024–2025 suggest business expansion or new channel partners.

---

## 8. Material (SKU) Evolution

| Transition | Dropped SKUs | New SKUs | Stable |
|---|---|---|---|
| 2023 → 2024 | 696 | 437 | 949 |
| 2024 → 2025 | 239 | 12 | 1,147 |
| **Present all 3 years** | — | — | **726** |

SKU portfolio contracted significantly from 2023 to 2024 (−696 discontinued), with moderate new introductions. The 2024→2025 transition shows stabilization with only 12 truly new SKUs, suggesting portfolio rationalization is complete.

---

## 9. Product Categories Identified

Based on `Material Description` prefix analysis (2025 portfolio):

| Category | SKU Count | % of Portfolio |
|---|---|---|
| MOBILE | 407 | 35.1% |
| LED TV | 125 | 10.8% |
| QLED TV | 122 | 10.5% |
| TABLET | 69 | 6.0% |
| MON (Monitor) | 54 | 4.7% |
| AV RECEIVER | 43 | 3.7% |
| DVM | 29 | 2.5% |
| RTF (Refrigerator) | 20 | 1.7% |
| LFD | 20 | 1.7% |
| Others | 270 | 23.3% |

**Business Context:** This appears to be a major consumer electronics manufacturer (TVs, mobile, appliances, AV) operating in Colombia.

---

## 10. Total Volume Summary

| Year | Sell-in | Cust. Sales | Channel Inv. |
|---|---|---|---|
| 2023 | 3,210,417 | 3,489,256 | 31,085,204 |
| 2024 | 3,441,891 | 3,449,154 | 26,838,537 |
| 2025 (W01–W33) | 1,405,602 | 1,840,427 | 7,979,829 |

Channel Inventory is approximately 8–9× higher than weekly sales — representing on-hand stock rather than flows, as expected.

---

## 11. Sparsity Profile

The dataset is **highly sparse**:

| Metric | Sell-in 2024 | Cust. Sales 2024 |
|---|---|---|
| Mean row sparsity (% zero weeks) | 94.1% | 78.7% |
| Rows with >90% zero weeks | 10,293 / 12,345 | 6,520 / 12,345 |
| SKU-Channel pairs with 0 active weeks | 6,209 / 12,345 (50%) | — |
| SKU-Channel pairs with 1–4 active weeks | 3,796 / 12,345 (31%) | — |
| SKU-Channel pairs with >12 active weeks | 739 / 12,345 (6%) | — |

Only ~6% of SKU-Channel pairs have rich enough time series for individual-level ML forecasting. The vast majority require aggregation or global model approaches.

---

*Document Version: 1.0 | AI-DLC Traceability ID: ASSESSMENT-2026-001-DA*
