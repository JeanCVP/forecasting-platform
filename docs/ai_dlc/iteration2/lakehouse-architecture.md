# Lakehouse Architecture
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## 1. Architecture Philosophy

The data platform follows a **medallion lakehouse architecture** (Bronze / Silver / Gold) on a local-first stack designed for zero cloud dependency in Phase 1, with a clear upgrade path to cloud storage.

**Core Principles:**
- **Immutability:** Bronze layer is append-only and never modified
- **Reproducibility:** All transformations are code-defined and versioned via DVC
- **Observability:** Every layer transition logs row counts, DQ metrics, and timestamps
- **Portability:** Parquet format enables migration to any cloud object store

---

## 2. System Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         LAKEHOUSE ARCHITECTURE                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SOURCE ZONE                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  data/raw/                                                            │ │
│  │  ├── 2023.csv   (43,338 rows × 55 cols)                              │ │
│  │  ├── 2024.csv   (37,035 rows × 55 cols)                              │ │
│  │  └── 2025.csv   (25,239 rows × 55 cols)                              │ │
│  │  [Tracked by DVC — immutable after ingestion]                        │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│            │                                                               │
│            │  [Prefect: ingest_flow]                                       │
│            ▼                                                               │
│  BRONZE LAYER — Raw Ingestion                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  data/bronze/                                                         │ │
│  │  ├── sell_data_2023.parquet   ← CSV melted to long format            │ │
│  │  ├── sell_data_2024.parquet   ← dtype cast; no business logic        │ │
│  │  └── sell_data_2025.parquet   ← string→float fix applied             │ │
│  │                                                                       │ │
│  │  Schema: (channel, material, category, yearweek, value, ingested_at) │ │
│  │  [Append-only. DQ checks: schema, nulls, row count]                  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│            │                                                               │
│            │  [Prefect: clean_flow]                                        │
│            ▼                                                               │
│  SILVER LAYER — Cleaned & Conformed                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  data/silver/                                                         │ │
│  │  ├── timeseries_clean.parquet   ← Aggregated, deduplicated           │ │
│  │  ├── dim_channel.parquet        ← Channel dimension                  │ │
│  │  ├── dim_material.parquet       ← Material/SKU dimension             │ │
│  │  └── dim_calendar.parquet       ← Calendar with events               │ │
│  │                                                                       │ │
│  │  Transformations applied:                                             │ │
│  │  • Duplicate aggregation (SUM by key)                                │ │
│  │  • Negative inv. → 0                                                 │ │
│  │  • ISO date column added                                              │ │
│  │  • Product family parsed                                              │ │
│  │  • is_censored flag (2025 == 999)                                    │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│            │                                                               │
│            │  [Prefect: feature_flow]                                      │
│            ▼                                                               │
│  GOLD LAYER — Feature-Rich Training & Serving                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  data/gold/                                                           │ │
│  │  ├── feature_store.parquet     ← All features computed               │ │
│  │  ├── training_set.parquet      ← Filtered for ML training            │ │
│  │  ├── forecast_output.parquet   ← Model predictions                   │ │
│  │  └── series_registry.parquet   ← Segment classification              │ │
│  │                                                                       │ │
│  │  Features added:                                                      │ │
│  │  • Lag features (1,2,4,13,26,52)                                     │ │
│  │  • Rolling means and stdevs (4w, 13w, 26w)                           │ │
│  │  • Days of Supply, Sell-through Rate                                 │ │
│  │  • Seasonal sin/cos encoding                                         │ │
│  │  • Colombian calendar flags                                           │ │
│  │  • Series segment label                                               │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│            │                                                               │
│    ┌───────┴────────┐                                                      │
│    ▼                ▼                                                      │
│  ┌──────────┐  ┌─────────────────────────────────────────────────────┐    │
│  │ MLflow   │  │ BI / Serving                                         │    │
│  │ Registry │  │ Streamlit / Power BI read from gold/*.parquet        │    │
│  └──────────┘  └─────────────────────────────────────────────────────┘    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
project-root/
│
├── data/
│   ├── raw/                     # Source CSVs (immutable, DVC-tracked)
│   │   ├── 2023.csv
│   │   ├── 2024.csv
│   │   └── 2025.csv
│   │
│   ├── bronze/                  # Raw parquet (DVC-tracked)
│   │   ├── sell_data_2023.parquet
│   │   ├── sell_data_2024.parquet
│   │   └── sell_data_2025.parquet
│   │
│   ├── silver/                  # Cleaned parquet (DVC-tracked)
│   │   ├── timeseries_clean.parquet
│   │   ├── dim_channel.parquet
│   │   ├── dim_material.parquet
│   │   └── dim_calendar.parquet
│   │
│   └── gold/                    # Feature-rich parquet (DVC-tracked)
│       ├── feature_store.parquet
│       ├── training_set.parquet
│       ├── forecast_output.parquet
│       └── series_registry.parquet
│
├── src/
│   ├── ingestion/
│   │   ├── ingest.py            # Bronze pipeline
│   │   └── schema_validators.py
│   ├── transformation/
│   │   ├── clean.py             # Silver pipeline
│   │   ├── feature_engineering.py  # Gold pipeline
│   │   └── calendar_builder.py
│   ├── ml/
│   │   ├── segmentation.py
│   │   ├── train_global.py
│   │   ├── train_intermittent.py
│   │   ├── reconcile.py
│   │   └── cold_start.py
│   ├── monitoring/
│   │   ├── drift_detector.py
│   │   └── accuracy_tracker.py
│   └── serving/
│       └── forecast_runner.py
│
├── flows/                       # Prefect flows
│   ├── ingest_flow.py
│   ├── clean_flow.py
│   ├── feature_flow.py
│   ├── train_flow.py
│   ├── forecast_flow.py
│   └── monitor_flow.py
│
├── dashboards/                  # Streamlit apps
│   ├── executive_kpis.py
│   ├── demand_planning.py
│   ├── channel_health.py
│   └── components/
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_leakage.py          # Critical: leakage detection tests
│   └── test_reconciliation.py
│
├── mlruns/                      # MLflow tracking (gitignored)
├── .dvc/                        # DVC metadata
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Model hyperparameters
├── requirements.txt
└── Makefile                     # make ingest / make train / make forecast
```

---

## 4. DuckDB Integration

DuckDB serves as the in-process analytical engine for all Parquet queries:

```python
import duckdb

con = duckdb.connect()

# Register gold layer
con.execute("CREATE VIEW feature_store AS SELECT * FROM 'data/gold/feature_store.parquet'")
con.execute("CREATE VIEW forecast_output AS SELECT * FROM 'data/gold/forecast_output.parquet'")

# Query: latest sell-in by product family
result = con.execute("""
    SELECT
        product_family,
        SUM(sell_in_lag_1) as last_week_sell_in,
        AVG(days_of_supply) as avg_dos
    FROM feature_store
    WHERE yearweek = (SELECT MAX(yearweek) FROM feature_store)
      AND segment != 'dead'
    GROUP BY product_family
    ORDER BY last_week_sell_in DESC
""").df()
```

**Performance characteristics (estimated):**
- Full feature_store scan (8K series × 137 weeks): < 2 seconds
- Join across Bronze/Silver/Gold: < 5 seconds
- Dashboard query (aggregated): < 500ms

---

## 5. Data Flow Timing

| Step | Trigger | Expected Duration | SLA |
|---|---|---|---|
| Source CSV arrival | Manual (weekly export from ERP) | — | Monday 08:00 |
| Bronze ingestion | Prefect trigger on file arrival | 2 min | Monday 09:00 |
| Silver transformation | After bronze complete | 5 min | Monday 09:15 |
| Gold feature engineering | After silver complete | 15 min | Monday 09:30 |
| ML inference (forecast) | After gold complete | 30 min | Monday 10:00 |
| Dashboard refresh | After inference complete | 5 min | Monday 10:15 |
| Monitoring checks | After dashboard refresh | 10 min | Monday 10:30 |
| Weekly report | Automated summary email | 5 min | Monday 11:00 |

---

## 6. Cloud Migration Path

| Phase | Storage | Compute | BI |
|---|---|---|---|
| Phase 1 (Local) | Local disk + Parquet | Python local | Streamlit |
| Phase 2 (Cloud-lite) | S3/GCS bucket + Parquet | Same Python, S3 paths | Streamlit on EC2/Cloud Run |
| Phase 3 (Enterprise) | S3 + Delta Lake | Spark or DuckDB MotherDuck | Power BI + DuckDB connector |

**Migration cost:** Changing `data/` path prefix from `data/gold/` to `s3://bucket/gold/` requires updating 2 config lines. DuckDB reads S3 Parquet natively.

---

*AI-DLC Traceability ID: LAKE-ARCH-ITER2-001 | Version: 2.0*
