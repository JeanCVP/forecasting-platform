# Decision Log
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## Decision Record Format

Each decision captures: context, options considered, decision made, rationale, tradeoffs accepted, and reversal criteria.

---

## DEC-001 — Primary Data Store Architecture

| Field | Value |
|---|---|
| **ID** | DEC-001 |
| **Date** | 2026-05-21 |
| **Status** | ✅ LOCKED |
| **Deciders** | Principal ML Architect + Data Platform Engineer |

**Context:** We need a data store that supports: multi-year weekly time series, ~100K rows/year, analytical queries, versioning, portability, and low operational overhead (no managed cloud infrastructure budget confirmed).

**Options Considered:**
1. PostgreSQL — relational, familiar, but poor for time-series analytics and columnar scans
2. Cloud Data Warehouse (BigQuery / Snowflake) — managed, scalable, but cost/cloud dependency
3. **DuckDB + Parquet on local/S3** — analytical, columnar, zero-dependency, embedded
4. ClickHouse — fast analytics, but heavier operational footprint

**Decision:** DuckDB + Parquet files organized in lakehouse layers (Bronze/Silver/Gold)

**Rationale:**
- DuckDB processes all 3 CSV files in milliseconds with full SQL support
- Parquet provides columnar compression (~5× smaller than CSV)
- Zero operational overhead — runs embedded in Python
- DVC integrates natively with file-based storage
- Can migrate to BigQuery/Snowflake in Phase 5+ by changing connector only

**Tradeoffs Accepted:**
- No multi-user concurrent writes (acceptable: batch pipeline, single writer)
- No built-in BI connector (mitigated: Streamlit reads DuckDB directly)

**Reversal Criteria:** If data volume exceeds 10GB or concurrent query users > 5, evaluate migration to MotherDuck or BigQuery.

---

## DEC-002 — Primary ML Framework

| Field | Value |
|---|---|
| **ID** | DEC-002 |
| **Date** | 2026-05-21 |
| **Status** | ✅ LOCKED |

**Context:** Need a framework that handles: global forecasting across thousands of series, intermittent demand, lag/rolling features, hierarchical reconciliation, fast training.

**Options Considered:**
1. Pure statsmodels (ARIMA per series) — no cross-series learning; too slow for 8K series
2. Prophet per series — good interpretability, poor scalability, no intermittent handling
3. **MLForecast + LightGBM** — global model, ML-native features, fast, handles sparsity
4. Nixtla NeuralForecast (NBEATS/TFT) — state-of-art but requires ~200+ obs per series; most series fail
5. GluonTS — complex setup, heavyweight

**Decision:** MLForecast as orchestration layer with LightGBM as primary learner; StatsForecast for statistical baselines and Croston/TSB for intermittent segment.

**Rationale:**
- MLForecast natively handles wide→long conversion, lag feature engineering, cross-validation
- LightGBM: fast, handles sparsity via leaf-wise splits, built-in feature importance
- StatsForecast: Croston, ADIDA, IMAPA for intermittent demand (segment-2)
- Unified API across model families
- Active community; Nixtla stack = consistent ecosystem

**Tradeoffs Accepted:**
- ML models are less interpretable than pure statistical for individual series
- LightGBM requires careful leakage prevention (enforced in feature store)

**Reversal Criteria:** If LightGBM MAPE > 30% on Regular segment after tuning, evaluate TFT/NBEATS for that segment only.

---

## DEC-003 — Orchestration Engine

| Field | Value |
|---|---|
| **ID** | DEC-003 |
| **Date** | 2026-05-21 |
| **Status** | ✅ LOCKED |

**Decision:** Prefect 2.x (self-hosted Prefect Server or Prefect Cloud free tier)

**Rationale:** Python-native, decorator-based flows, built-in retries/notifications, excellent DVC/MLflow integration, lighter than Airflow for this scale.

**Tradeoffs:** Less mature than Airflow for enterprise; Prefect Cloud free tier has flow run limits.

---

## DEC-004 — Experiment Tracking

| Field | Value |
|---|---|
| **ID** | DEC-004 |
| **Date** | 2026-05-21 |
| **Status** | ✅ LOCKED |

**Decision:** MLflow (self-hosted, SQLite backend initially; PostgreSQL for production)

**Rationale:** Industry standard, native LightGBM/sklearn logging, model registry, artifact store, zero additional cost.

---

## DEC-005 — BI Layer Strategy

| Field | Value |
|---|---|
| **ID** | DEC-005 |
| **Date** | 2026-05-21 |
| **Status** | ✅ LOCKED |

**Decision:** Two-phase BI:
- **Phase 1 (Prototype):** Streamlit — Python-native, fast to build, reads DuckDB directly
- **Phase 2 (Production):** Power BI connected to Gold layer Parquet/DuckDB

**Rationale:** Streamlit allows ML team to prototype dashboards without BI specialists. Power BI for enterprise stakeholder adoption.

---

## DEC-006 — Temporal Granularity

| Field | Value |
|---|---|
| **ID** | DEC-006 |
| **Date** | 2026-05-21 |
| **Status** | ✅ LOCKED |

**Decision:** Weekly forecasting as primary granularity. No daily disaggregation in this phase.

**Rationale:** Data is natively weekly; business decisions (replenishment orders) are weekly; weekly reduces sparsity vs. daily.

---

## DEC-007 — Hierarchical Forecasting Strategy

| Field | Value |
|---|---|
| **ID** | DEC-007 |
| **Date** | 2026-05-21 |
| **Status** | ✅ LOCKED |

**Decision:** Middle-out strategy — forecast at Product Family × Channel level, reconcile downward to SKU-Channel and upward to Total.

**Rationale:**
- SKU-Channel level too sparse for direct accurate forecasting
- Total level loses operational detail
- Product Family × Channel = ~500–800 series with sufficient density
- MLForecast + HierarchicalReconciliation (MinT-Shrink) for reconciliation

**Tradeoffs:** Family-level forecasts may miss individual SKU launch dynamics. Mitigated by cold-start analog logic.

---

## DEC-008 — Duplicate Aggregation Strategy

| Field | Value |
|---|---|
| **ID** | DEC-008 |
| **Date** | 2026-05-21 |
| **Status** | ✅ LOCKED |

**Decision:** SUM aggregation for all duplicate `(Channel, Material, Category, Week)` tuples in Silver layer.

**Rationale:** Fragments represent sub-channel/location splits of the same logical SKU-Channel series. Summing recovers the true demand signal. Mean would undercount; max would produce outliers.

**Validation:** Post-aggregation row count must equal `unique(Channel) × unique(Material) × 3 categories × 52 weeks`.

---

## DEC-009 — Negative Value Handling

| Field | Value |
|---|---|
| **ID** | DEC-009 |
| **Date** | 2026-05-21 |
| **Status** | ✅ LOCKED |

**Decision:**
- Sell-in negatives: **retain as-is** in Silver (valid return signals); **clip at 0** in Gold training features only
- Cust. Sales negatives: same as Sell-in
- Channel Inv. negatives: **replace with 0** in Silver (physically impossible); log count for audit

**Rationale:** Preserving negatives in Silver maintains data lineage integrity. Clipping in Gold protects ML features.

---

*AI-DLC Traceability ID: DECISION-ITER2-001 | Version: 2.0*
