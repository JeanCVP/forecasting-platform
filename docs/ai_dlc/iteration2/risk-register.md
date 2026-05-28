# Risk Register
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## Risk Matrix Legend

| Severity | Probability | Risk Score |
|---|---|---|
| 🔴 Critical (4) | 🔴 Almost Certain (4) | 13–16: STOP |
| 🟠 High (3) | 🟠 Likely (3) | 9–12: ESCALATE |
| 🟡 Medium (2) | 🟡 Possible (2) | 5–8: MANAGE |
| 🟢 Low (1) | 🟢 Unlikely (1) | 1–4: MONITOR |

---

## Active Risks

### DATA RISKS

#### RISK-D01 — 2025 Data Export Cap at 999
| Field | Value |
|---|---|
| **Category** | Data Quality |
| **Description** | 2025 CSV has all weekly values hard-capped at 999 (vs. 27,500+ in 2024). High-volume SKUs are systematically truncated. |
| **Severity** | 🔴 Critical (4) |
| **Probability** | 🔴 Almost Certain (4) |
| **Risk Score** | **16 — STOP** |
| **Impact** | Models trained on 2025 data will produce biased low-demand forecasts for high-volume SKUs. Inventory risk scoring will be incorrect. |
| **Owner** | Data Platform Engineer |
| **Mitigation** | (1) Request uncapped extract from source ERP immediately. (2) In interim: flag all `== 999` cells as censored; exclude from loss function computation. (3) Use 2023–2024 data as primary training signal; weight 2025 lower. |
| **Residual Risk** | 🟠 High if uncapped data not obtained |
| **Status** | 🔴 OPEN — escalated to business owner |

---

#### RISK-D02 — Duplicate Fragmented Rows Inflating Training Data
| Field | Value |
|---|---|
| **Category** | Data Quality / Pipeline |
| **Description** | 15–18% of rows are fragments of the same logical series split across multiple rows with different values per week. Training on raw data severely underestimates actual demand. |
| **Severity** | 🔴 Critical (4) |
| **Probability** | 🔴 Almost Certain (4) |
| **Risk Score** | **16 — STOP** |
| **Impact** | All model outputs invalid if not addressed. |
| **Owner** | Data Platform Engineer |
| **Mitigation** | Mandatory `groupby(['Channel','Material','Category','week']).sum()` aggregation in Bronze→Silver pipeline. Enforced as data contract. Validated by row-count assertion. |
| **Residual Risk** | 🟢 Low — fully mitigated by pipeline design |
| **Status** | 🟡 MITIGATED IN DESIGN — pending pipeline build |

---

#### RISK-D03 — 2025 String Dtype Breaking Numeric Operations
| Field | Value |
|---|---|
| **Category** | Data Quality |
| **Description** | 2025 weeks 01–33 stored as `object` dtype with float-like strings (`"25.0 "`). Silent failures in aggregation and feature computation. |
| **Severity** | 🟠 High (3) |
| **Probability** | 🔴 Almost Certain (4) |
| **Risk Score** | **12 — ESCALATE** |
| **Mitigation** | `pd.to_numeric()` + `strip()` in Bronze ingestion layer. Schema validation at ingestion gate. |
| **Residual Risk** | 🟢 Low |
| **Status** | 🟡 MITIGATED IN DESIGN |

---

#### RISK-D04 — High Portfolio Churn Breaking Historical Continuity
| Field | Value |
|---|---|
| **Category** | ML / Data |
| **Description** | 42% of SKUs replaced annually. New SKUs have zero history; 696 SKUs disappeared in 2023→2024 transition. |
| **Severity** | 🟠 High (3) |
| **Probability** | 🟠 Likely (3) |
| **Risk Score** | **9 — ESCALATE** |
| **Impact** | Cold-start problem for ~40% of active portfolio each year. No product lifecycle metadata available. |
| **Mitigation** | (1) Product family hierarchy for cold-start imputation. (2) Global model cross-series learning. (3) SKU lifecycle features (`sku_age_weeks`, `is_new`). |
| **Residual Risk** | 🟡 Medium |
| **Status** | 🟡 MANAGED IN ML DESIGN |

---

#### RISK-D05 — No Promotional Calendar Available
| Field | Value |
|---|---|
| **Category** | Feature Gap |
| **Description** | Día sin IVA, company promotions, and channel-specific events are not in the dataset. These cause demand spikes that models cannot predict. |
| **Severity** | 🟠 High (3) |
| **Probability** | 🟠 Likely (3) |
| **Risk Score** | **9 — ESCALATE** |
| **Impact** | Models will systematically under-forecast during promotional weeks, over-forecast post-promotion. MAPE spikes during events. |
| **Mitigation** | (1) Obtain promotional calendar from commercial team. (2) Interim: detect promotional weeks via statistical outlier detection on residuals. (3) Add known Colombian holiday dates. |
| **Residual Risk** | 🟠 High until calendar integrated |
| **Status** | 🔴 OPEN — pending business input |

---

### ML RISKS

#### RISK-ML01 — Extreme Sparsity Causing Global Model Failure on Rare Series
| Field | Value |
|---|---|
| **Category** | ML |
| **Description** | 50% of SKU-Channel pairs have zero Sell-in in 2024. Standard models produce nonsensical forecasts. |
| **Severity** | 🟠 High (3) |
| **Probability** | 🔴 Almost Certain (4) |
| **Risk Score** | **12 — ESCALATE** |
| **Mitigation** | Series segmentation: Regular / Intermittent / Rare / Dead. Different model families per segment. Croston/TSB for intermittent; historical mean for rare. |
| **Residual Risk** | 🟡 Medium (inherent uncertainty in intermittent series) |
| **Status** | 🟡 MANAGED IN ML DESIGN |

---

#### RISK-ML02 — Target Leakage via Inventory Features
| Field | Value |
|---|---|
| **Category** | ML / Validity |
| **Description** | Channel Inventory at time t is computed partly FROM Sell-in at time t. Using `channel_inv(t)` as a feature when predicting `sell_in(t)` causes circular leakage. |
| **Severity** | 🔴 Critical (4) |
| **Probability** | 🟠 Likely (3) |
| **Risk Score** | **12 — ESCALATE** |
| **Mitigation** | All inventory features use strictly lagged values: `inv_lag_1` = Inventory(t−1). Enforced in feature store schema. Leakage test in CI pipeline (correlation of feature with target residuals). |
| **Residual Risk** | 🟢 Low if strictly enforced |
| **Status** | 🟡 MITIGATED IN DESIGN |

---

#### RISK-ML03 — Hierarchical Forecast Inconsistency
| Field | Value |
|---|---|
| **Category** | ML |
| **Description** | Top-down, bottom-up, and middle-out forecasts may produce contradictory totals. Business will distrust forecasts that don't sum correctly. |
| **Severity** | 🟡 Medium (2) |
| **Probability** | 🟠 Likely (3) |
| **Risk Score** | **6 — MANAGE** |
| **Mitigation** | Use `HierarchicalReconciliation` from MLForecast/StatsForecast. Min-trace (MinT) or OLS reconciliation to guarantee additive consistency. |
| **Residual Risk** | 🟢 Low |
| **Status** | 🟡 MANAGED IN ML DESIGN |

---

#### RISK-ML04 — Concept Drift After SKU Refresh
| Field | Value |
|---|---|
| **Category** | MLOps |
| **Description** | Annual model refresh introduces new demand patterns not present in training data. Model degrades silently without monitoring. |
| **Severity** | 🟠 High (3) |
| **Probability** | 🟠 Likely (3) |
| **Risk Score** | **9 — ESCALATE** |
| **Mitigation** | Weekly MAPE monitoring per segment. PSI drift detection on feature distributions. Automated retraining trigger when MAPE ≥ threshold. |
| **Residual Risk** | 🟡 Medium |
| **Status** | 🟡 MANAGED IN MLOPS DESIGN |

---

#### RISK-ML05 — Overfit on Small Regular Segment
| Field | Value |
|---|---|
| **Category** | ML |
| **Description** | Only 739 series qualify as "Regular" (>12 active weeks). Global model may overfit to these and generalize poorly to sparse series. |
| **Severity** | 🟡 Medium (2) |
| **Probability** | 🟡 Possible (2) |
| **Risk Score** | **4 — MONITOR** |
| **Mitigation** | Cross-validation with walk-forward splits. Regularization tuning (LightGBM: `min_data_in_leaf`, `lambda_l1`). Separate model per segment. |
| **Residual Risk** | 🟢 Low |
| **Status** | 🟡 MANAGED |

---

### PLATFORM RISKS

#### RISK-P01 — Data Pipeline Fragility on CSV Format Changes
| Field | Value |
|---|---|
| **Category** | Platform |
| **Description** | 2025 demonstrated unexpected dtype changes vs. prior years. Future extracts may change column names, add/remove weeks, or change encoding. |
| **Severity** | 🟠 High (3) |
| **Probability** | 🟠 Likely (3) |
| **Risk Score** | **9 — ESCALATE** |
| **Mitigation** | Data contracts (Great Expectations) with schema validation at ingestion. Fail-fast on schema violations with alerting. |
| **Residual Risk** | 🟡 Medium |
| **Status** | 🟡 MANAGED IN PLATFORM DESIGN |

---

#### RISK-P02 — DVC Dataset Versioning Complexity
| Field | Value |
|---|---|
| **Category** | MLOps |
| **Description** | As new annual CSVs arrive and historical data is corrected, maintaining reproducible dataset versions becomes complex. |
| **Severity** | 🟡 Medium (2) |
| **Probability** | 🟡 Possible (2) |
| **Risk Score** | **4 — MONITOR** |
| **Mitigation** | DVC with content-addressed hashing. Immutable Bronze layer. Explicit dataset versioning protocol. |
| **Residual Risk** | 🟢 Low |
| **Status** | 🟡 MANAGED IN PLATFORM DESIGN |

---

### BUSINESS RISKS

#### RISK-B01 — Stakeholder Expectation Mismatch on Forecast Accuracy
| Field | Value |
|---|---|
| **Category** | Business |
| **Description** | Business may expect >95% accuracy; realistic MAPE for intermittent consumer electronics demand is 25–40%. |
| **Severity** | 🟠 High (3) |
| **Probability** | 🟠 Likely (3) |
| **Risk Score** | **9 — ESCALATE** |
| **Mitigation** | Establish accuracy benchmarks by segment in kick-off. Present naive baseline first. Frame accuracy improvement as journey, not a point. |
| **Residual Risk** | 🟡 Medium |
| **Status** | 🔴 OPEN — stakeholder alignment needed |

---

#### RISK-B02 — Promotion Blindness Leading to Visible Forecast Failures
| Field | Value |
|---|---|
| **Category** | Business |
| **Description** | Without promotional calendar, the model will visibly fail during major Colombian retail events. This can undermine trust in the entire system. |
| **Severity** | 🟠 High (3) |
| **Probability** | 🔴 Almost Certain (4) |
| **Risk Score** | **12 — ESCALATE** |
| **Mitigation** | (1) Block forecast from being used during known event weeks initially. (2) Override workflow in dashboard. (3) Integrate calendar ASAP. |
| **Residual Risk** | 🟠 High until addressed |
| **Status** | 🔴 OPEN |

---

## Risk Heat Map

```
Probability
   Almost  | D03      | D01,D02   | ML01     | B02      |
   Certain |          | ML02      |          |          |
   --------+----------+-----------+----------+----------+
   Likely  |          | D04,D05   | P01,ML04 | B01      |
           |          | ML03      |          |          |
   --------+----------+-----------+----------+----------+
   Possible|          |           | D05      | B01      |
           |          |           |          |          |
   --------+----------+-----------+----------+----------+
   Unlikely| ML05,P02 |           |          |          |
           +----------+-----------+----------+----------+
              Low       Medium      High      Critical
                           Severity
```

---

*AI-DLC Traceability ID: RISK-ITER2-001 | Version: 2.0*
