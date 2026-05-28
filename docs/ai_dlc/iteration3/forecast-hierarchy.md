# Forecast Hierarchy — Especificación Formal
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** HIERARCHY-ITER3-001

---

## 1. Jerarquía de 4 Niveles

```
NIVEL 0 — TOTAL
│
│  Identificador: "TOTAL"
│  Series: 1
│  Descripción: Suma de todo el portfolio, todos los canales
│
├── NIVEL 1 — FAMILIA DE PRODUCTO
│   │
│   │  Identificador: product_family
│   │  Series: ~12 (MOBILE, LED TV, QLED TV, TABLET, MON, AV RECEIVER, ...)
│   │  Descripción: Categoría de producto agregada
│   │
│   ├── NIVEL 2 — FAMILIA × CANAL  ← NODO DE FORECAST DIRECTO
│   │   │
│   │   │  Identificador: (product_family, channel)
│   │   │  Series: ~500–800 series activas
│   │   │  Descripción: Demanda de una familia de producto en un canal específico
│   │   │  DENSIDAD: Suficiente para ML global → este es el nivel donde se hace el forecast
│   │   │
│   │   └── NIVEL 3 — SKU × CANAL  ← OBJETIVO DE DISAGREGACIÓN
│   │
│   │       Identificador: (material, channel)
│   │       Series: ~8,413 pares activos
│   │       Descripción: Forecast individual por SKU y canal
│   │       DENSIDAD: 94% sparse → se obtiene por disagregación, NO forecast directo
│
```

### Por Qué Esta Estructura

| Nivel | Series | Densidad | Enfoque |
|---|---|---|---|
| L0: Total | 1 | 100% | Sanity check / reconciliación |
| L1: Familia | ~12 | ~95% | Trend de categoría |
| **L2: Familia×Canal** | **~800** | **~70%** | **FORECAST DIRECTO** |
| L3: SKU×Canal | ~8,413 | ~6% | Disagregación proporcional |

---

## 2. Matriz de Sumación S

La matriz S define cómo los niveles inferiores se agregan hacia arriba:

```python
# src/forecasting/hierarchy.py

import polars as pl
import numpy as np
from hierarchicalforecast.utils import aggregate

def build_hierarchy_spec(series_registry: pl.DataFrame) -> dict:
    """
    Build the S (summing) matrix specification.
    
    Returns hierarchy_spec dict compatible with HierarchicalForecast.
    """
    
    # Cada serie de L3 (SKU×Canal) tiene una sola ruta hacia L0
    # L3 → L2: misma family+channel
    # L2 → L1: misma family
    # L1 → L0: always TOTAL
    
    hierarchy_spec = {
        # "unique_id": ["L1_id", "L2_id", "L3_id"]
        # Se construye dinámicamente desde series_registry
    }
    
    for row in series_registry.iter_rows(named=True):
        uid = f"{row['channel']}__{row['material']}"   # L3 unique_id
        l2_id = f"{row['product_family']}__{row['channel']}"
        l1_id = row['product_family']
        l0_id = "TOTAL"
        
        hierarchy_spec[uid] = [l0_id, l1_id, l2_id, uid]
    
    return hierarchy_spec


def build_S_matrix(
    Y_df: pl.DataFrame,
    hierarchy_spec: dict
) -> tuple[np.ndarray, list[str]]:
    """
    Build S matrix where S[i,j] = 1 if series j belongs to aggregate i.
    Uses HierarchicalForecast's aggregate utility.
    """
    # aggregate() returns (S_df, tags)
    # S_df: summing matrix as DataFrame
    # tags: dict mapping level to series IDs
    
    S_df, tags = aggregate(Y_df, spec=hierarchy_spec)
    return S_df, tags
```

---

## 3. Middle-Out Strategy

```
PASO 1: Forecast directo en L2 (Familia × Canal)
         → ~800 series con suficiente densidad
         → MLForecast + LightGBM global model
         → 19 semanas forward

PASO 2: Suma hacia L0/L1 (Bottom-up parcial desde L2)
         → L1 = SUM(L2 de misma familia)
         → L0 = SUM(todos los L1)

PASO 3: Disagregación hacia L3 (Top-down desde L2)
         → Proporciones históricas: p(sku|family×channel)
         → L3_forecast = L2_forecast × historical_proportion

PASO 4: Reconciliación MinT-Shrink
         → Garantiza consistencia aditiva en todos los niveles
         → Ajusta todos los niveles simultáneamente

PASO 5: Validación de coherencia
         → SUM(L3) == L0 ± 0.01
         → HALT si violación
```

---

## 4. Proporción Histórica para Disagregación

```python
def compute_historical_proportions(
    training_df: pl.DataFrame,
    window_weeks: int = 26
) -> pl.DataFrame:
    """
    Compute the historical proportion of each SKU within its Family×Channel.
    Used for L2 → L3 disaggregation.
    
    proportion(sku, channel, week) = 
        sell_in(sku, channel, t-window:t) / sell_in(family, channel, t-window:t)
    """
    # Aggregate L2 (family×channel level)
    l2 = training_df.group_by(
        ["product_family", "channel", "yearweek"]
    ).agg(pl.col("sell_in").sum().alias("family_channel_sell_in"))
    
    # Join back to L3
    df_with_l2 = training_df.join(
        l2, on=["product_family", "channel", "yearweek"], how="left"
    )
    
    # Compute proportion
    df_with_l2 = df_with_l2.with_columns([
        (pl.col("sell_in") / 
         pl.col("family_channel_sell_in").clip(lower_bound=0.001)
        ).alias("sku_proportion_raw")
    ])
    
    # Average proportion over window
    df_with_l2 = df_with_l2.sort(["channel","material","yearweek"])
    proportions = df_with_l2.group_by(["channel","material","product_family"]).agg([
        pl.col("sku_proportion_raw").tail(window_weeks).mean()
          .alias("historical_proportion")
    ])
    
    # Normalize so proportions sum to 1.0 per Family×Channel
    l2_sum = proportions.group_by(["product_family","channel"]).agg(
        pl.col("historical_proportion").sum().alias("prop_sum")
    )
    proportions = proportions.join(l2_sum, on=["product_family","channel"])
    proportions = proportions.with_columns([
        (pl.col("historical_proportion") / 
         pl.col("prop_sum").clip(lower_bound=0.001)
        ).alias("normalized_proportion")
    ])
    
    return proportions


def disaggregate_l2_to_l3(
    l2_forecasts: pl.DataFrame,
    proportions: pl.DataFrame
) -> pl.DataFrame:
    """
    Distribute L2 (Family×Channel) forecasts to L3 (SKU×Channel).
    """
    # Join proportions to all SKUs in the family×channel
    result = proportions.join(
        l2_forecasts,
        on=["product_family", "channel"],
        how="left"
    )
    
    for q in ["p10","p25","p50","p75","p90"]:
        result = result.with_columns([
            (pl.col(f"l2_{q}") * pl.col("normalized_proportion"))
              .alias(f"sell_in_{q}")
        ])
    
    return result
```

---

## 5. Cold-Start en la Jerarquía

Para SKUs nuevos (sin history), la disagregación falla porque no tienen proporción histórica.

```python
def cold_start_proportion(
    new_material: str,
    channel: str,
    product_family: str,
    series_registry: pl.DataFrame,
    proportions: pl.DataFrame
) -> float:
    """
    Assign a starting proportion for a new SKU in its Family×Channel.
    
    Strategy:
    1. Find analogous SKUs (same family, same channel, similar age profile)
    2. Use their median proportion as the cold-start proportion
    3. Ramp-up: discount 70% in weeks 1-4, 90% in weeks 5-8, 100% from week 9
    """
    family_channel_props = proportions.filter(
        (pl.col("product_family") == product_family) &
        (pl.col("channel") == channel) &
        (pl.col("historical_proportion") > 0)
    )
    
    if len(family_channel_props) == 0:
        # No analogs in this family×channel → assign equal share
        n_active = series_registry.filter(
            (pl.col("product_family") == product_family) &
            (pl.col("channel") == channel) &
            (pl.col("segment") != "dead")
        ).height
        return 1.0 / max(n_active + 1, 1)
    
    # Use lower quartile of analogs (conservative estimate for new SKU)
    return float(family_channel_props["historical_proportion"].quantile(0.25))
```

---

## 6. Validación de Consistencia Jerárquica

```python
def validate_hierarchy_consistency(
    reconciled_forecasts: pl.DataFrame,
    tolerance: float = 0.01
) -> dict:
    """
    Post-reconciliation: verify L3 sums equal L0.
    """
    l3_total = reconciled_forecasts.filter(
        pl.col("level") == "L3"
    )["sell_in_p50"].sum()
    
    l0_total = reconciled_forecasts.filter(
        pl.col("level") == "L0"
    )["sell_in_p50"].sum()
    
    residual = abs(l3_total - l0_total)
    
    return {
        "l3_sum": float(l3_total),
        "l0_total": float(l0_total),
        "residual": float(residual),
        "is_consistent": residual <= tolerance,
        "pct_error": float(residual / max(l0_total, 0.001) * 100)
    }
```

---

*AI-DLC Traceability ID: HIERARCHY-ITER3-001 | Version: 3.0*
