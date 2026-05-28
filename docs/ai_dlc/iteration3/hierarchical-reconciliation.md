# Hierarchical Reconciliation — MinT-Shrink
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** RECONCILE-ITER3-001

---

## 1. El Problema de Coherencia

Sin reconciliación, los forecasts de diferentes niveles son incoherentes:

```
Sin reconciliación:
  L0 (Total)          → 50,000 unidades
  L1 MOBILE           → 22,000 unidades
  L1 LED TV           → 19,000 unidades
  L1 QLED TV          → 11,000 unidades
  SUM(L1)             → 52,000 ≠ 50,000  ← INCONSISTENCIA

Con MinT-Shrink:
  Todos los niveles ajustados simultáneamente
  SUM(L3) == L2 == L1 == L0 (garantizado)
```

---

## 2. MinT-Shrink: Fundamento Matemático

### Formulación

Dada la jerarquía de $n$ series base (L3) y sus aggregados:

$$\tilde{Y} = SP\hat{Y}$$

Donde:
- $\hat{Y}$: vector de forecasts base (todos los niveles, incoherentes)
- $S$: matriz de sumación ($m \times n$, donde $m$ = total series, $n$ = series base)
- $P$: matriz de proyección que minimiza la traza de la covarianza del error
- $\tilde{Y}$: forecasts reconciliados (coherentes)

### MinT Solución

$$P = (S^T W^{-1} S)^{-1} S^T W^{-1}$$

Donde $W$ es la matriz de covarianza de errores de forecast.

### Por Qué Shrink

Con miles de series y alta sparsity, $W$ es mal condicionada (casi singular).  
MinT-Shrink aplica estimador James-Stein al estimador de $W$:

$$\hat{W}_{shrink} = (1 - \rho) \hat{W}_{sample} + \rho \cdot \hat{W}_{diag}$$

donde $\rho$ se optimiza automáticamente. Esto regulariza la matriz y la hace invertible.

---

## 3. Implementación con HierarchicalForecast

```python
# src/forecasting/reconcile.py

import polars as pl
import pandas as pd
import numpy as np
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace
from hierarchicalforecast.utils import aggregate

def reconcile_forecasts(
    Y_hat_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    hierarchy_spec: dict
) -> pd.DataFrame:
    """
    Reconcile base forecasts using MinT-Shrink.

    Args:
        Y_hat_df: Base forecasts DataFrame with columns
                  [unique_id, ds, y_hat, y_hat_lo_10, ...]
        Y_df:     Historical actuals DataFrame
                  [unique_id, ds, y]
        hierarchy_spec: mapping from unique_id to [L0, L1, L2, L3]

    Returns:
        Reconciled forecasts (same shape as Y_hat_df)
    """

    # Step 1: Build summing matrix S from hierarchy_spec
    S_df, tags = aggregate(Y_df, spec=hierarchy_spec)

    # Step 2: HierarchicalReconciliation
    hrec = HierarchicalReconciliation(
        reconcilers=[
            MinTrace(method='mint_shrink')
        ]
    )

    # Step 3: Reconcile
    reconciled_df = hrec.reconcile(
        Y_hat_df=Y_hat_df,
        Y_df=Y_df,
        S=S_df,
        tags=tags,
    )

    return reconciled_df


def validate_reconciliation(
    reconciled_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tolerance: float = 0.01
) -> dict:
    """
    Verify that reconciled forecasts sum correctly at all levels.

    Returns validation result dict with passed flag.
    """
    # Extract L0 (Total) forecast
    l0 = reconciled_df[reconciled_df["unique_id"] == "TOTAL"]["y_hat"].values

    # Extract L3 (bottom level) forecasts and sum
    l3_ids = [uid for uid in reconciled_df["unique_id"].unique()
              if "__" in uid and uid.count("__") >= 1]  # heuristic: SKU__CHANNEL
    l3_sum = (
        reconciled_df[reconciled_df["unique_id"].isin(l3_ids)]
        .groupby("ds")["y_hat"].sum()
        .values
    )

    residuals = np.abs(l0 - l3_sum)
    max_residual = float(residuals.max())
    pct_error = float(residuals.max() / (np.abs(l0).max() + 1e-6) * 100)

    return {
        "passed":         max_residual <= tolerance,
        "max_residual":   max_residual,
        "pct_error":      pct_error,
        "n_forecast_points": len(l0),
        "tolerance":      tolerance,
        "message": (
            f"PASS: max residual = {max_residual:.4f}"
            if max_residual <= tolerance else
            f"FAIL: max residual = {max_residual:.4f} > {tolerance}"
        )
    }
```

---

## 4. Manejo de Series Sparse en la Reconciliación

Series con muchos ceros generan problemas en MinT porque su varianza estimada es casi cero.

```python
def filter_for_reconciliation(
    series_registry: pl.DataFrame,
    min_active_weeks: int = 4
) -> tuple[list, list]:
    """
    Split series into:
    - reconcilable: sufficient data for MinT
    - bypass: too sparse, assign cold-start directly

    Returns (reconcilable_ids, bypass_ids)
    """
    reconcilable = series_registry.filter(
        pl.col("active_weeks_52w") >= min_active_weeks
    )["channel"].zip_with(
        series_registry.filter(
            pl.col("active_weeks_52w") >= min_active_weeks
        )["material"]
    )

    bypass = series_registry.filter(
        pl.col("active_weeks_52w") < min_active_weeks
    )

    reconcilable_ids = [
        f"{ch}__{mat}" for ch, mat in
        zip(
            series_registry.filter(pl.col("active_weeks_52w") >= min_active_weeks)["channel"].to_list(),
            series_registry.filter(pl.col("active_weeks_52w") >= min_active_weeks)["material"].to_list(),
        )
    ]
    bypass_ids = [
        f"{ch}__{mat}" for ch, mat in
        zip(bypass["channel"].to_list(), bypass["material"].to_list())
    ]

    return reconcilable_ids, bypass_ids
```

---

## 5. Reconciliación Probabilística

Para intervalos de predicción coherentes:

```python
# Reconciliar cada cuantil independientemente
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

reconciled_by_quantile = {}
for q in QUANTILES:
    q_col = f"y_hat_q{int(q*100)}"
    Y_hat_q = Y_hat_df.rename(columns={"y_hat": q_col})[
        ["unique_id","ds",q_col]
    ].rename(columns={q_col: "y_hat"})

    reconciled_q = reconcile_forecasts(Y_hat_q, Y_df, hierarchy_spec)
    reconciled_by_quantile[q] = reconciled_q

# Garantizar monotonía después de reconciliación
# p10 <= p25 <= p50 <= p75 <= p90
def enforce_quantile_monotonicity(reconciled_by_q: dict) -> dict:
    qs = sorted(reconciled_by_q.keys())
    for i in range(1, len(qs)):
        q_prev = qs[i-1]
        q_curr = qs[i]
        df_prev = reconciled_by_q[q_prev]
        df_curr = reconciled_by_q[q_curr]
        # Clip lower bound to previous quantile
        df_curr["y_hat"] = np.maximum(
            df_curr["y_hat"].values,
            df_prev["y_hat"].values
        )
        reconciled_by_q[q_curr] = df_curr
    return reconciled_by_q
```

---

## 6. Gate de Validación Post-Reconciliación

```python
# tests/test_reconciliation.py

def test_hierarchical_consistency(reconciled_df, S_df, tolerance=0.01):
    result = validate_reconciliation(reconciled_df, S_df, tolerance)
    assert result["passed"], (
        f"Hierarchical inconsistency detected!\n"
        f"L0 total: {result['l0_total']:.2f}\n"
        f"L3 sum:   {result['l3_sum']:.2f}\n"
        f"Residual: {result['max_residual']:.4f}\n"
        f"Pct err:  {result['pct_error']:.4f}%"
    )

def test_quantile_monotonicity(reconciled_df):
    q_cols = ["sell_in_p10","sell_in_p25","sell_in_p50","sell_in_p75","sell_in_p90"]
    for i in range(len(q_cols)-1):
        violations = (
            reconciled_df[q_cols[i]] > reconciled_df[q_cols[i+1]]
        ).sum()
        assert violations == 0, (
            f"Quantile monotonicity violated: "
            f"{q_cols[i]} > {q_cols[i+1]} in {violations} rows"
        )

def test_no_negative_p50(reconciled_df):
    neg = (reconciled_df["sell_in_p50"] < 0).sum()
    assert neg == 0, f"{neg} negative p50 forecasts found"
```

---

*AI-DLC Traceability ID: RECONCILE-ITER3-001 | Version: 3.0*
