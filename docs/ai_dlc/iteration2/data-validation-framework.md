# Data Validation Framework
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## 1. Validation Architecture

```
VALIDATION GATES
─────────────────────────────────────────────────────────────
Gate 0: Source CSV          [Great Expectations]
Gate 1: Bronze Layer        [Schema + row count assertions]
Gate 2: Silver Layer        [Business rules + dedup check]
Gate 3: Gold Layer          [Leakage + feature range checks]
Gate 4: Forecast Output     [Sanity + monotonicity checks]
Gate 5: Dashboard Data      [Completeness + staleness checks]
─────────────────────────────────────────────────────────────
All gates: HALT on CRITICAL failure, WARN + continue on WARNING
```

---

## 2. Gate 0 — Source CSV Validation

```python
# Great Expectations rules for raw CSV

GATE_0_RULES = {
    "column_existence": [
        "Channel", "Material Description", "Category"
    ],
    "column_types": {
        "Channel": "string",
        "Material Description": "string",
        "Category": "string",
    },
    "value_sets": {
        "Category": ["Sell-in", "Cust. Sales", "Channel Inv."]
    },
    "regex_patterns": {
        "Channel": r"^CUSTOMER\d+$"
    },
    "row_count_range": (10_000, 60_000),
    "weekly_value_range": (-10_000, 50_000),
    "no_nulls_in": ["Channel", "Material Description", "Category"],
}
```

---

## 3. Gate 2 — Silver Business Rules

```python
def validate_silver(df: pl.DataFrame) -> ValidationResult:
    results = []
    
    # Rule S01: No duplicate primary keys
    dups = df.is_duplicated(subset=['channel','material','category','yearweek']).sum()
    results.append(ValidationCheck(
        name="S01_no_duplicates",
        severity="CRITICAL",
        passed=(dups == 0),
        detail=f"{dups} duplicate rows found"
    ))
    
    # Rule S02: No negative channel inventory
    neg_inv = df.filter(
        (pl.col('category') == 'Channel Inv.') & (pl.col('value') < 0)
    ).height
    results.append(ValidationCheck(
        name="S02_no_negative_inv",
        severity="CRITICAL",
        passed=(neg_inv == 0),
        detail=f"{neg_inv} negative inventory values"
    ))
    
    # Rule S03: Category balance (each channel×material has all 3 categories)
    cat_counts = df.group_by(['channel','material']).agg(
        pl.col('category').n_unique().alias('n_cats')
    )
    unbalanced = cat_counts.filter(pl.col('n_cats') != 3).height
    results.append(ValidationCheck(
        name="S03_category_balance",
        severity="WARNING",
        passed=(unbalanced == 0),
        detail=f"{unbalanced} series missing a category"
    ))
    
    # Rule S04: Censored rate acceptable
    censored_rate = df['is_censored'].mean()
    results.append(ValidationCheck(
        name="S04_censored_rate",
        severity="WARNING",
        passed=(censored_rate < 0.02),
        detail=f"Censored rate = {censored_rate:.3%}"
    ))
    
    return ValidationResult(checks=results)
```

---

## 4. Gate 3 — Gold Leakage & Feature Validation

```python
# Leakage detection (CI-enforced)
FORBIDDEN_DIRECT_COLS = ['sell_in', 'cust_sales', 'channel_inv']

def validate_gold_features(feature_df: pl.DataFrame) -> ValidationResult:
    results = []
    
    model_input_cols = get_model_input_columns(feature_df)
    
    # G01: No direct target leakage
    leakage = [c for c in FORBIDDEN_DIRECT_COLS if c in model_input_cols]
    results.append(ValidationCheck("G01_no_leakage", "CRITICAL", 
                                    passed=len(leakage)==0, detail=str(leakage)))
    
    # G02: days_of_supply in valid range
    dos = feature_df['days_of_supply']
    dos_ok = (dos >= 0).all() and (dos <= 1000).all()
    results.append(ValidationCheck("G02_dos_range", "WARNING",
                                    passed=dos_ok.item()))
    
    # G03: No NaN in lag_1 for series with history
    old_series = feature_df.filter(pl.col('sku_age_weeks') > 1)
    lag1_nulls = old_series['sell_in_lag_1'].null_count()
    results.append(ValidationCheck("G03_lag1_not_null", "WARNING",
                                    passed=(lag1_nulls==0), 
                                    detail=f"{lag1_nulls} nulls in lag_1"))
    
    # G04: Quantile monotonicity (forecast output)
    if 'sell_in_p10' in feature_df.columns:
        mono = (feature_df['sell_in_p10'] <= feature_df['sell_in_p50']).all()
        results.append(ValidationCheck("G04_quantile_mono", "CRITICAL",
                                        passed=mono.item()))
    
    return ValidationResult(checks=results)
```

---

## 5. Validation Reporting

Every pipeline run generates a validation report saved as JSON:

```json
{
  "run_id": "prefect-run-2026-05-21-0900",
  "timestamp": "2026-05-21T09:00:00Z",
  "gates": {
    "gate_0_source": {"passed": true, "checks": 12, "warnings": 0, "criticals": 0},
    "gate_1_bronze": {"passed": true, "checks": 6, "warnings": 0, "criticals": 0},
    "gate_2_silver": {"passed": true, "checks": 4, "warnings": 1, "criticals": 0},
    "gate_3_gold": {"passed": true, "checks": 5, "warnings": 0, "criticals": 0},
    "gate_4_forecast": {"passed": true, "checks": 4, "warnings": 0, "criticals": 0}
  },
  "overall": "PASS",
  "data_quality_score": 94
}
```

The `data_quality_score` is computed as:
```
score = 100 - (criticals × 20) - (warnings × 5)
```

Target: score ≥ 85 before model training is allowed to proceed.

---

## 6. Automated Data Quality Score Evolution

| Metric | Phase 0 (Raw) | Phase 2 Target (Silver) | Phase 3 Target (Gold) |
|---|---|---|---|
| DQ Score | 61/100 | ≥ 85/100 | ≥ 90/100 |
| Completeness | 95 | 98 | 99 |
| Uniqueness | 55 | 99 | 100 |
| Consistency | 50 | 90 | 95 |
| Accuracy | 65 | 85 | 90 |

---

*AI-DLC Traceability ID: VALIDATION-ITER2-001 | Version: 2.0*
