# AI-DLC State Document
**Project:** Enterprise Demand Forecasting & Inventory Planning System
**Iteration:** 2 — Architecture & Foundation
**State Date:** 2026-05-21
**Classification:** INTERNAL — CONFIDENTIAL

---

## 1. Project Identity

| Attribute | Value |
|---|---|
| **Project Code** | AI-DLC-FORECAST-COL-001 |
| **Domain** | Consumer Electronics · Supply Chain · Demand Planning |
| **Geography** | Colombia |
| **Business Owner** | Commercial Director / VP Supply Chain |
| **Technical Owner** | Principal ML Architect |
| **Current Phase** | Phase 0 → Phase 1 Transition |
| **Target Go-Live** | Phase 1: 12 weeks from kickoff |

---

## 2. Phase History

| Phase | Name | Status | Key Output |
|---|---|---|---|
| **Phase 0** | Dataset Assessment & Feasibility | ✅ COMPLETE | 7 assessment documents; DQ score 61/100 |
| **Phase 1** | Architecture & Foundation | 🔄 IN PROGRESS | This document set (17 documents) |
| **Phase 2** | Data Platform Build | ⬜ PLANNED | Bronze/Silver/Gold lakehouse operational |
| **Phase 3** | ML Baseline | ⬜ PLANNED | Naive + Statistical models in production |
| **Phase 4** | ML Advanced | ⬜ PLANNED | LightGBM global model; TFT evaluation |
| **Phase 5** | MLOps Production | ⬜ PLANNED | Full monitoring, retraining, drift detection |
| **Phase 6** | BI & Product | ⬜ PLANNED | 6 dashboards live; stakeholder adoption |

---

## 3. Current State Snapshot

### What We Know (Confirmed from Phase 0)
- **Data:** 3 years weekly data (2023–2025 W33), 105K+ rows, wide format
- **Entities:** Channel × Material × Category = ~8,413 active series (2025)
- **Product universe:** Consumer electronics (Mobile 35%, TV 21%, others)
- **Market:** Colombia; identified seasonality around events (Black Friday, Mother's Day, Día sin IVA)
- **Forecast target:** Sell-in W34–W52 2025 (19-week horizon)
- **Data quality score:** 61/100 — preprocessing mandatory

### Critical Blockers Identified
| Blocker | Severity | Owner | Status |
|---|---|---|---|
| 2025 data capped at 999 | 🔴 Critical | Data Platform | ⬜ Unresolved — source extract needed |
| Duplicate fragmented rows | 🔴 Critical | Data Platform | ⬜ Mitigated in pipeline design (aggregation) |
| 2025 string dtypes | 🔴 Critical | Data Platform | ⬜ Mitigated in pipeline design (cast) |
| 50% dead series | 🟠 High | ML Team | ⬜ Addressed in segmentation strategy |
| No promotional calendar | 🟠 High | Business | ⬜ Pending business input |

### Assumptions Locked
1. The three CSV files represent a single consistent business entity (one manufacturer, one country)
2. `Channel` values are anonymized retail/distribution partners
3. `Material Description` comma-delimited fields are: [0]=Family, [1]=Model, [2]=Size/Variant, [3]=Geography, [4+]=Config
4. ISO calendar weeks are used (week 1 = first week with Thursday in new year)
5. Negative values in Sell-in and Cust. Sales represent returns/adjustments (valid business events)
6. Negative Channel Inventory values are data errors (physically impossible)
7. 2025 W34–W52 zero values represent the forecast target, not historical data

---

## 4. Scope Boundaries

### IN SCOPE
- Weekly Sell-in forecasting (primary target)
- Weekly Cust. Sales forecasting (secondary; used as feature and KPI)
- Channel Inventory tracking and risk scoring
- Hierarchical forecasting (SKU-Channel → Product Family → Total)
- Intermittent demand handling
- MLOps: experiment tracking, model registry, monitoring, retraining
- 6 operational dashboards

### OUT OF SCOPE (this iteration)
- Daily forecasting
- Financial forecasting (revenue, margin)
- New market expansion (non-Colombia)
- Promotional scenario planning (blocked by missing promo calendar)
- Real-time streaming ingestion
- Mobile app

### DEFERRED
- Deep learning (TFT, NBEATS) — evaluate after LightGBM baseline
- External demand signals (macroeconomic, weather)
- Price elasticity modeling

---

## 5. Active Decisions

| Decision ID | Topic | Decision | Date | Status |
|---|---|---|---|---|
| DEC-001 | Data store | DuckDB + Parquet (lakehouse) vs. cloud warehouse | Pending | ⬜ |
| DEC-002 | ML framework | MLForecast + LightGBM as primary | ✅ Locked | 2026-05-21 |
| DEC-003 | Orchestration | Prefect 2.x | ✅ Locked | 2026-05-21 |
| DEC-004 | Experiment tracking | MLflow | ✅ Locked | 2026-05-21 |
| DEC-005 | BI layer | Streamlit (prototype) → Power BI (production) | ✅ Locked | 2026-05-21 |
| DEC-006 | Granularity | Weekly forecasting as primary | ✅ Locked | 2026-05-21 |
| DEC-007 | Hierarchy strategy | Middle-out (family-level forecast, disaggregated) | Pending | ⬜ |

---

## 6. AI-DLC Quality Gates

| Gate | Checkpoint | Criteria | Phase |
|---|---|---|---|
| QG-01 | Data Platform Ready | Bronze/Silver/Gold pipelines passing; DQ score ≥ 80/100 | Phase 2 |
| QG-02 | Baseline Model Live | Naive seasonal MAPE computed; CI pipeline green | Phase 3 |
| QG-03 | ML Model Qualified | LightGBM MAPE < 25% on Regular segment | Phase 4 |
| QG-04 | MLOps Operational | Drift alerts firing; retraining pipeline tested | Phase 5 |
| QG-05 | BI Adoption | ≥ 3 stakeholder personas using dashboards weekly | Phase 6 |

---

## 7. Team & Responsibilities

| Role | Responsibilities |
|---|---|
| Principal ML Architect | ML strategy, model design, hierarchy logic |
| Senior Data Platform Engineer | Lakehouse, pipelines, DVC, data contracts |
| MLOps Architect | MLflow, Prefect, monitoring, retraining |
| AI-DLC Constructor | Documentation, state management, cross-cutting decisions |

---

*AI-DLC Traceability ID: AIDLC-STATE-ITER2-001 | Version: 2.0*
