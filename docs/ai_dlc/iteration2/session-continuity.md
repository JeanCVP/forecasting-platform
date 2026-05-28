# Session Continuity Document
**Project:** AI-DLC-FORECAST-COL-001
**Purpose:** Enable any new AI-DLC session or team member to resume work without context loss
**Last Updated:** 2026-05-21 | Iteration 2

---

## Quick Resume Checklist

When starting a new AI-DLC session on this project, verify:

- [ ] Read `aidlc-state.md` — current phase and blockers
- [ ] Read `risk-register.md` — active 🔴 risks
- [ ] Read `decision-log.md` — locked decisions (do not re-debate)
- [ ] Check `dataset-analysis.md` (Phase 0) — data facts
- [ ] Check `data-quality-report.md` (Phase 0) — critical issues
- [ ] Confirm 2025 uncapped data status with business owner

---

## Project Narrative (1-Page Summary)

We are building an enterprise-grade **weekly demand forecasting system** for a consumer electronics manufacturer operating in Colombia. The system will forecast weekly Sell-in (units shipped to retail partners) across ~8,400 product-channel combinations.

**The data** consists of three years of weekly CSV extracts (2023, 2024, 2025) containing three metrics per product-channel pair: Sell-in, Customer Sales (POS), and Channel Inventory. The data has 3 critical quality issues (cap, type mismatch, duplicate fragmentation) that are handled in the Bronze→Silver pipeline.

**The ML approach** uses a segmented global forecasting strategy: the ~8,400 series are classified into Regular (6%), Intermittent (13%), Rare (31%), and Dead (50%). A LightGBM global model via MLForecast handles Regular and Intermittent segments at the Product Family × Channel hierarchy level; Croston/TSB handles intermittent series; historical mean handles rare series. Forecasts are reconciled hierarchically using MinT-Shrink to guarantee additive consistency.

**The platform** is a local-first lakehouse (DuckDB + Parquet, Bronze/Silver/Gold layers), orchestrated by Prefect, tracked by MLflow, versioned by DVC.

**The product** is 6 operational dashboards: Executive KPIs, Demand Planning Workbench, Channel Health Monitor, SKU Performance Tracker, Forecast Accuracy, and Inventory Risk Radar. Prototype in Streamlit, production in Power BI.

**Current blocker:** 2025 data is capped at 999 — needs uncapped source extract before training on 2025 data.

---

## What Has Been Decided (Do Not Revisit Without Explicit Re-opener)

| Topic | Decision |
|---|---|
| Data store | DuckDB + Parquet (lakehouse) |
| ML framework | MLForecast + LightGBM + StatsForecast |
| Orchestration | Prefect 2.x |
| Experiment tracking | MLflow |
| BI prototype | Streamlit |
| BI production | Power BI |
| Temporal granularity | Weekly |
| Hierarchy strategy | Middle-out (Family × Channel) |
| Duplicate handling | SUM aggregation in Silver |
| Negative Inv. handling | Replace with 0 in Silver |
| Negative Sell-in handling | Retain in Silver; clip in Gold features |

---

## Open Questions Requiring Business Input

| # | Question | Urgency | Owner |
|---|---|---|---|
| OQ-1 | Can we get uncapped 2025 data extract? | 🔴 Critical | Data Owner |
| OQ-2 | Promotional calendar: Which weeks had Día sin IVA / major promos? | 🟠 High | Commercial Team |
| OQ-3 | Are "CUSTOMER" codes stable year-over-year? (Could CUSTOMER5 2023 ≠ CUSTOMER5 2024?) | 🟠 High | Data Owner |
| OQ-4 | What is the channel type taxonomy? (Online / Chain / Distributor) | 🟡 Medium | Sales Team |
| OQ-5 | Is there a product hierarchy master file? (Brand / Line / Family / Model) | 🟡 Medium | Category Management |
| OQ-6 | Are returns modeled differently from negative orders in the source system? | 🟡 Medium | Data Owner |

---

## File Map

```
ai-dlc-iter2/
├── core/
│   ├── aidlc-state.md          ← Project state and phase tracker
│   ├── risk-register.md        ← All risks with scores and mitigations
│   ├── decision-log.md         ← Technical decisions (locked)
│   └── session-continuity.md   ← THIS FILE
├── architecture/
│   ├── ml-architecture.md      ← Full ML system design
│   ├── lakehouse-architecture.md ← Data platform design
│   ├── forecasting-strategy.md ← Model strategy per segment
│   ├── feature-store-design.md ← Feature definitions and pipeline
│   └── data-lineage.md         ← Data flow and transformation map
├── data-platform/
│   ├── bronze-silver-gold.md   ← Layer definitions and transformations
│   ├── data-contracts.md       ← Schema contracts and validation rules
│   ├── dataset-versioning.md   ← DVC strategy and versioning protocol
│   └── data-validation-framework.md ← Great Expectations rules
├── mlops/
│   ├── mlops-architecture.md   ← MLOps system design
│   ├── experiment-tracking.md  ← MLflow setup and conventions
│   ├── monitoring-strategy.md  ← Drift detection and alerting
│   └── retraining-strategy.md  ← Automated retraining logic
└── product/
    ├── dashboard-specifications.md ← Full dashboard specs
    ├── business-kpis.md            ← KPI definitions and thresholds
    └── inventory-risk-framework.md ← Risk scoring model
```

---

## Key Numbers to Remember

| Metric | Value |
|---|---|
| Total active series (2025) | ~8,413 (Channel × Material) |
| Total with 3 categories | ~25,239 rows |
| Regular segment (forecasted individually) | ~739 series (6%) |
| Intermittent segment | ~1,601 series (13%) |
| Stable customers (all 3 years) | 68 |
| Stable SKUs (all 3 years) | 726 |
| Forecast horizon | 19 weeks (W34–W52 2025) |
| Historical lookback | 137 weeks (W1 2023 – W33 2025) |
| Target MAPE (Regular segment) | < 20% |
| Target MAPE (Intermittent segment) | < 35% |
| Data quality score (Phase 0) | 61/100 |
| Target data quality score (Silver layer) | ≥ 85/100 |

---

## Iteration 2 Deliverables Checklist

### Core AI-DLC State
- [x] aidlc-state.md
- [x] risk-register.md
- [x] decision-log.md
- [x] session-continuity.md

### Architecture
- [x] ml-architecture.md
- [x] lakehouse-architecture.md
- [x] forecasting-strategy.md
- [x] feature-store-design.md
- [x] data-lineage.md

### Data Platform
- [x] bronze-silver-gold.md
- [x] data-contracts.md
- [x] dataset-versioning.md
- [x] data-validation-framework.md

### MLOps
- [x] mlops-architecture.md
- [x] experiment-tracking.md
- [x] monitoring-strategy.md
- [x] retraining-strategy.md

### Product / BI
- [x] dashboard-specifications.md
- [x] business-kpis.md
- [x] inventory-risk-framework.md

---

*AI-DLC Traceability ID: CONTINUITY-ITER2-001 | Version: 2.0*
