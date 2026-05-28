# Data Contracts — Especificación v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3 | **AI-DLC Traceability ID:** CONTRACTS-ITER3-001

---

## Contrato C01 — Fuente → Bronze

```yaml
contract_id: C01-SOURCE-TO-BRONZE
version: "3.0"
parties:
  producer: "ERP/BI Export System"
  consumer: "Bronze Ingestion Pipeline"

schema_requirements:
  fixed_columns:
    - name: "Channel"
      type: string
      nullable: false
      regex: "^CUSTOMER\\d+$"
    - name: "Material Description"
      type: string
      nullable: false
      min_length: 5
      max_length: 250
    - name: "Category"
      type: string
      nullable: false
      allowed_values: ["Sell-in", "Cust. Sales", "Channel Inv."]
  weekly_columns:
    pattern: "^\\d{6}$"
    count: 52
    type: numeric_or_numeric_string
    range: [-10000, 50000]

volume:
  rows_min: 10000
  rows_max: 60000
  rows_multiple_of: 3  # always 3 categories per channel×material

quality_slas:
  null_rate_key_cols: 0.0
  encoding: "UTF-8"
  delimiter: ","

on_violation:
  action: HALT_PIPELINE
  notify: ["data_platform_team@company.com"]
  log_to: "data/audit/audit_log.parquet"
```

---

## Contrato C02 — Bronze → Silver

```yaml
contract_id: C02-BRONZE-TO-SILVER

transformations_mandatory:
  - id: T01
    name: dedup_aggregation
    description: "groupby (channel,material,category,yearweek) + SUM(value)"
    validation: "len(silver) < len(bronze)"
    post_check: "0 duplicates on PK"

  - id: T02
    name: negative_inv_fix
    description: "Channel Inv. < 0 → 0; flag inv_was_negative=True"
    validation: "0 negative values in Channel Inv. rows post-transform"

  - id: T03
    name: censored_flag
    description: "value==999 AND source_year==2025 → is_censored=True"
    validation: "all 2025 values == 999 are flagged"

  - id: T04
    name: iso_date_derivation
    description: "date = Monday of ISO week from yearweek string"
    validation: "date.dt.weekday() == 0 for all rows"

  - id: T05
    name: product_family_parse
    description: "material.split(',')[0].strip() → product_family"
    validation: "product_family is not null for >99% of rows"

silver_pk_uniqueness:
  columns: ["channel","material","category","yearweek"]
  allowed_violations: 0
  action_on_violation: HALT

silver_inv_sign:
  condition: "category=='Channel Inv.' AND value < 0"
  allowed_violations: 0
  action_on_violation: HALT
```

---

## Contrato C03 — Silver → Gold (Features)

```yaml
contract_id: C03-SILVER-TO-GOLD

leakage_prohibition:
  forbidden_direct_features:
    - "sell_in"         # t=0 del target
    - "cust_sales"      # t=0 del co-variado
    - "channel_inv"     # t=0 del inventario (circular con sell_in)
  enforcement: HALT_ON_DETECTION
  test_file: "tests/test_leakage.py::test_no_target_leakage"

inventory_temporal_semantics:
  rule: "channel_inv features MUST use lag >= 1"
  correct_usage: ["inv_lag_1", "inv_lag_2", "inv_ma4"]
  forbidden_usage: ["channel_inv (direct t=0)"]

feature_ranges:
  days_of_supply: [0, 1000]
  sell_through_rate_4w: [0, 20]
  prob_nonzero_4w: [0, 1]
  week_sin: [-1, 1]
  week_cos: [-1, 1]

quantile_monotonicity:
  condition: "p10 <= p25 <= p50 <= p75 <= p90"
  applies_to: "forecast_output.parquet"
  action_on_violation: HALT

forecast_coverage:
  condition: "all active series have p50 forecast"
  active_definition: "segment != 'dead' OR is_new_sku=True"
  action_on_violation: HALT
```

---

## Contrato C04 — Gold → BI/Dashboards

```yaml
contract_id: C04-GOLD-TO-BI

staleness_sla:
  max_hours_since_refresh: 168  # 1 semana
  action_on_breach: SHOW_STALENESS_BANNER

metric_consistency:
  rule: "all dashboards use src/semantic/metrics.py"
  forbidden: "inline KPI calculations in dashboard code"
  test: "grep -r 'days_of_supply' dashboards/ | grep -v 'MetricEngine'"

data_completeness:
  forecast_coverage: 100%  # all active series
  historical_coverage: "2023W01 through current_week"
```

---

# Data Lineage — Mapa End-to-End
**AI-DLC Traceability ID:** LINEAGE-ITER3-001

---

```
LINAJE COMPLETO: De ERP a Dashboard

ERP Sistema
  └── Export manual semanal → data/raw/{year}.csv
      ├── [AUDIT: CSV_RECEIVED]
      │
      ├── Gate-0: Great Expectations schema validation
      │   ├── PASS → continúa
      │   └── FAIL → [AUDIT: CONTRACT_VIOLATION] → HALT + alert
      │
      └── Bronze Pipeline (ingest.py)
          ├── melt: wide → long
          ├── dtype fix: str→float (2025)
          ├── strip whitespace
          ├── add: source_file, ingested_at
          └── [AUDIT: BRONZE_COMPLETE] → data/bronze/sell_data_{year}.parquet
              │
              ├── Gate-1: Row count, nulls, yearweek format
              │
              └── Silver Pipeline (clean.py)
                  ├── union 3 años
                  ├── T01: groupby+SUM (dedup)
                  ├── T02: neg inv → 0
                  ├── T03: is_censored flag
                  ├── T04: ISO date
                  ├── T05: product_family parse
                  ├── T06: inv_balance_residual
                  └── [AUDIT: SILVER_COMPLETE] → data/silver/timeseries_clean.parquet
                      │
                      ├── Gate-2: Uniqueness, neg inv, category balance
                      │
                      ├── dim_channel.parquet ─────────────────────────┐
                      ├── dim_material.parquet ────────────────────────┤
                      └── dim_calendar.parquet ────────────────────────┤
                          │                                             │
                          └── Gold Pipeline (feature_engineering.py)   │
                              ├── pivot: categories → columns          │
                              ├── lag features (shift(n))              │
                              ├── rolling features (shift(1)+rolling)  │
                              ├── inventory ratios                     │
                              ├── seasonal encoding                    │
                              ├── calendar flags (join dim_calendar) ──┘
                              ├── ADI/CV² segmentation
                              └── [AUDIT: GOLD_COMPLETE] → data/gold/feature_store.parquet
                                  │
                                  ├── Gate-3: Leakage, feature ranges, coverage
                                  │
                                  ├──── training_set.parquet
                                  │         │
                                  │         └── MLflow Training Run
                                  │               ├── [tag: feature_store_hash]
                                  │               ├── [tag: silver_hash]
                                  │               ├── CV metrics logged
                                  │               └── Model registered
                                  │                     │
                                  │               Champion Model
                                  │                     │
                                  └──── Inference ◄─────┘
                                            │
                                            └── forecast_output.parquet
                                                    │
                                                ┌───┴─────────────────────┐
                                                ▼                         ▼
                                          Streamlit App            Power BI
                                          (via MetricEngine)       (via Parquet connector)
```

---

# Validation Framework — 12 Checkpoints
**AI-DLC Traceability ID:** VALID-ITER3-001

---

## Checkpoint Map

| Gate | Checkpoint | Capa | Severidad | Acción |
|---|---|---|---|---|
| G0 | V01: Schema CSV | Source→Bronze | CRITICAL | HALT |
| G0 | V02: Encoding | Source→Bronze | CRITICAL | HALT |
| G1 | V03: Row count range | Bronze | HIGH | WARNING |
| G1 | V04: Null in keys | Bronze | CRITICAL | HALT |
| G2 | V05: Zero duplicates PK | Silver | CRITICAL | HALT |
| G2 | V06: Zero negative inv | Silver | CRITICAL | HALT |
| G2 | V07: Category balance | Silver | HIGH | WARNING |
| G3 | V08: No t=0 leakage | Gold | CRITICAL | HALT |
| G3 | V09: Inv temporal semantics | Gold | CRITICAL | HALT |
| G3 | V10: Feature ranges | Gold | HIGH | WARNING+clip |
| G3 | V11: Quantile monotonicity | Gold forecast | CRITICAL | HALT |
| G3 | V12: Forecast coverage 100% | Gold forecast | CRITICAL | HALT |

---

## Implementación Great Expectations

```python
# src/validation/gates.py

import great_expectations as gx
import polars as pl

class ValidationGates:
    
    def gate_0_source(self, filepath: str, year: int) -> bool:
        """Validate raw CSV before Bronze ingestion."""
        df = pl.read_csv(filepath, n_rows=10)
        
        checks = [
            # V01: Schema
            set(["Channel","Material Description","Category"]).issubset(df.columns),
            # V02: Category values
            df["Category"].unique().to_list() == ["Sell-in","Cust. Sales","Channel Inv."],
        ]
        
        df_full = pl.read_csv(filepath)
        week_cols = [c for c in df_full.columns if c.startswith(str(year))]
        checks.append(len(week_cols) == 52)
        checks.append(df_full["Channel"].null_count() == 0)
        
        return all(checks)
    
    def gate_2_silver(self, silver_path: str) -> dict:
        """Validate Silver layer. Returns result dict."""
        df = pl.read_parquet(silver_path)
        results = {}
        
        # V05: Zero duplicates
        dup_count = df.is_duplicated(
            subset=["channel","material","category","yearweek"]
        ).sum()
        results["V05_no_duplicates"] = {
            "passed": dup_count == 0,
            "detail": f"{dup_count} duplicates",
            "severity": "CRITICAL"
        }
        
        # V06: Zero negative inventory
        neg_inv = df.filter(
            (pl.col("category") == "Channel Inv.") & (pl.col("value") < 0)
        ).height
        results["V06_no_neg_inv"] = {
            "passed": neg_inv == 0,
            "detail": f"{neg_inv} negative inventory rows",
            "severity": "CRITICAL"
        }
        
        # V07: Category balance
        cat_counts = df.group_by(["channel","material"]).agg(
            pl.col("category").n_unique().alias("n_cats")
        )
        unbalanced = cat_counts.filter(pl.col("n_cats") != 3).height
        results["V07_category_balance"] = {
            "passed": unbalanced == 0,
            "detail": f"{unbalanced} series missing a category",
            "severity": "HIGH"
        }
        
        return results
    
    def gate_3_leakage(self, feature_df: pl.DataFrame) -> dict:
        """V08 + V09: Leakage and temporal semantics checks."""
        results = {}
        
        # V08: No t=0 target features
        forbidden = ["sell_in", "cust_sales", "channel_inv"]
        model_features = [c for c in feature_df.columns 
                          if not c.endswith("_target") and "lag_0" not in c]
        leakage_found = [f for f in forbidden if f in model_features]
        results["V08_no_target_leakage"] = {
            "passed": len(leakage_found) == 0,
            "detail": f"Leakage features: {leakage_found}",
            "severity": "CRITICAL"
        }
        
        # V09: Inventory uses only lagged values
        inv_direct = [c for c in model_features 
                      if "inv" in c and "lag" not in c
                      and c not in ["inv_delta_1","inv_momentum","inv_overstock_flag","inv_stockout_flag"]]
        results["V09_inv_temporal"] = {
            "passed": len(inv_direct) == 0,
            "detail": f"Direct inventory features: {inv_direct}",
            "severity": "CRITICAL"
        }
        
        return results
    
    def compute_dq_score(self, all_results: dict) -> int:
        """Aggregate all gate results into a 0-100 DQ score."""
        score = 100
        for check_name, result in all_results.items():
            if not result["passed"]:
                if result["severity"] == "CRITICAL":
                    score -= 20
                elif result["severity"] == "HIGH":
                    score -= 5
                else:
                    score -= 2
        return max(0, score)
```

---

# Semantic Layer Definition
**AI-DLC Traceability ID:** SEMLAYER-ITER3-001

---

## DuckDB Views Centralizadas

```sql
-- src/semantic/views.sql
-- Cargadas al iniciar cualquier sesión DuckDB

-- VIEW: Inventario actual con DOS
CREATE OR REPLACE VIEW v_inventory_health AS
SELECT
    fs.channel,
    fs.material,
    fs.product_family,
    fs.yearweek,
    fs.inv_lag_1            AS current_inv_units,
    fs.sales_ma4            AS avg_weekly_sales,
    fs.days_of_supply,
    CASE
        WHEN fs.days_of_supply < 7   THEN 'critical_stockout'
        WHEN fs.days_of_supply < 14  THEN 'warning_stockout'
        WHEN fs.days_of_supply < 45  THEN 'healthy'
        WHEN fs.days_of_supply < 60  THEN 'warning_overstock'
        ELSE 'critical_overstock'
    END AS dos_status,
    fs.sell_through_rate_4w,
    fs.segment
FROM read_parquet('data/gold/feature_store.parquet') fs
WHERE fs.yearweek = (SELECT MAX(yearweek) FROM read_parquet('data/gold/feature_store.parquet'));

-- VIEW: Sell-through YTD
CREATE OR REPLACE VIEW v_sell_through_ytd AS
SELECT
    channel,
    product_family,
    SUM(cust_sales)  AS total_sales_ytd,
    SUM(sell_in)     AS total_sell_in_ytd,
    SUM(cust_sales) * 100.0 / NULLIF(SUM(sell_in), 0) AS sell_through_pct
FROM read_parquet('data/gold/feature_store.parquet')
WHERE year = YEAR(CURRENT_DATE)
GROUP BY channel, product_family;

-- VIEW: Forecast vs Actual
CREATE OR REPLACE VIEW v_forecast_accuracy AS
SELECT
    f.channel,
    f.material,
    f.yearweek,
    f.sell_in_p50     AS forecast_p50,
    a.sell_in         AS actual,
    ABS(f.sell_in_p50 - a.sell_in) / NULLIF((ABS(f.sell_in_p50) + ABS(a.sell_in)) / 2, 0) * 200
                       AS smape,
    f.segment,
    f.model_version
FROM read_parquet('data/gold/forecast_output.parquet') f
JOIN read_parquet('data/gold/feature_store.parquet') a
  ON f.channel = a.channel
  AND f.material = a.material
  AND f.yearweek = a.yearweek;

-- VIEW: Risk portfolio summary
CREATE OR REPLACE VIEW v_risk_summary AS
SELECT
    yearweek,
    COUNT(*)                                                AS total_series,
    SUM(CASE WHEN risk_tier = 'critical_overstock'  THEN 1 ELSE 0 END) AS critical_overstock,
    SUM(CASE WHEN risk_tier = 'warning_overstock'   THEN 1 ELSE 0 END) AS high_overstock,
    SUM(CASE WHEN risk_tier = 'healthy'             THEN 1 ELSE 0 END) AS healthy,
    SUM(CASE WHEN risk_tier = 'warning_stockout'    THEN 1 ELSE 0 END) AS low_stock,
    SUM(CASE WHEN risk_tier = 'critical_stockout'   THEN 1 ELSE 0 END) AS critical_stockout,
    SUM(excess_units)                                       AS total_excess_units,
    AVG(days_of_supply)                                     AS avg_dos
FROM read_parquet('data/gold/inventory_risk.parquet')
GROUP BY yearweek;
```

---

*AI-DLC Traceability IDs: CONTRACTS-ITER3-001 | LINEAGE-ITER3-001 | VALID-ITER3-001 | SEMLAYER-ITER3-001*
