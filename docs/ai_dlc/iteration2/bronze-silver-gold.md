# Bronze / Silver / Gold — Especificación v3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** BSG-ITER3-001

---

## 1. Arquitectura de Capas

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEDALLION LAKEHOUSE ARCHITECTURE v3                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SOURCE                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  data/raw/           — CSVs originales (INMUTABLES, DVC tracked) │   │
│  │  2023.csv · 2024.csv · 2025.csv                                  │   │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │ Gate-0: Schema + encoding validation   │
│                                 ▼                                        │
│  BRONZE — Ingesta Fiel                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  data/bronze/                                                    │    │
│  │  sell_data_2023.parquet  — wide→long, dtype fix, audit cols      │    │
│  │  sell_data_2024.parquet                                          │    │
│  │  sell_data_2025.parquet  — string→float fix, is_censored flag    │    │
│  │                                                                  │    │
│  │  PRINCIPIO: 1 row Bronze = 1 cell del CSV original              │    │
│  │  NUNCA modificar contenido de negocio aquí                       │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │ Gate-1: Row count, nulls, ranges       │
│                                 ▼                                        │
│  SILVER — Limpieza y Conformación                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  data/silver/                                                    │    │
│  │  timeseries_clean.parquet  — ÚNICO origen de verdad             │    │
│  │  dim_channel.parquet       — dimensión canal                    │    │
│  │  dim_material.parquet      — dimensión SKU                      │    │
│  │  dim_calendar.parquet      — dimensión calendario               │    │
│  │                                                                  │    │
│  │  TRANSFORMACIONES APLICADAS:                                     │    │
│  │  ✓ Union 3 años             ✓ Dedup por SUM                     │    │
│  │  ✓ Inv negativo → 0         ✓ is_censored flag                  │    │
│  │  ✓ ISO date derivado        ✓ product_family parseado           │    │
│  │  ✓ inventory_balance_residual calculado                         │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │ Gate-2: Uniqueness, business rules     │
│                                 ▼                                        │
│  GOLD — Feature Store + Serving                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  data/gold/                                                      │    │
│  │  feature_store.parquet   — 105 columnas × ~1.15M rows           │    │
│  │  training_set.parquet    — subset filtrado para ML              │    │
│  │  series_registry.parquet — segmentación ADI/CV²                 │    │
│  │  forecast_output.parquet — predicciones + intervalos            │    │
│  │  inventory_risk.parquet  — scores de riesgo semanales           │    │
│  │  accuracy_log.parquet    — MAPE histórico por semana            │    │
│  │  drift_log.parquet       — PSI histórico por feature            │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │ Gate-3: Leakage, feature ranges        │
│                                 ▼                                        │
│  AUDIT                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  data/audit/                                                     │    │
│  │  audit_log.parquet  — append-only trail de todos los eventos    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  SEMANTIC VIEWS (DuckDB in-memory)                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  views: v_dos · v_sell_through · v_portfolio_kpis · v_forecast  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Bronze Layer — Especificación Detallada

### Propósito
Preservar los datos fuente con la mínima transformación estructural necesaria para ser almacenables en Parquet. **Cero lógica de negocio.**

### Transformaciones Permitidas en Bronze

| Transformación | Tipo | Justificación |
|---|---|---|
| Wide → Long (melt) | Estructural | Parquet requiere formato largo |
| String → Float (2025 cols) | Tipo | Corrección de bug del sistema fuente |
| Strip whitespace | Limpieza superficial | Artefacto de exportación CSV |
| Rename: snake_case | Convención | Estandarización de nombres |
| Agregar: `source_file`, `ingested_at` | Auditoría | Trazabilidad |

### Transformaciones PROHIBIDAS en Bronze

| Transformación | Motivo de Prohibición |
|---|---|
| Agregar duplicados | Lógica de negocio — pertenece a Silver |
| Reemplazar negativos | Lógica de negocio — pertenece a Silver |
| Calcular `product_family` | Lógica de negocio — pertenece a Silver |
| Filtrar filas | Modifica la representación fiel del fuente |
| Imputar nulos | Cambia el significado de los datos |

### Schema Bronze

```python
BRONZE_SCHEMA = {
    "channel":       pl.Utf8,
    "material":      pl.Utf8,
    "category":      pl.Utf8,
    "yearweek":      pl.Utf8,      # YYYYWW string
    "value":         pl.Float64,
    "source_file":   pl.Utf8,
    "source_year":   pl.Int32,
    "ingested_at":   pl.Utf8,      # ISO 8601 UTC
}

# Invariantes post-ingesta:
# 1. No hay nulos en channel, material, category, yearweek
# 2. yearweek coincide con re.match(r'^\d{6}$')
# 3. Exactamente 52 yearweek values por year
# 4. source_year en {2023, 2024, 2025}
```

---

## 3. Silver Layer — Especificación Detallada

### Propósito
Producir la única fuente de verdad limpia y conformada para todos los consumidores posteriores. Aplicar todas las correcciones de calidad de datos documentadas en el data quality report.

### Pipeline Silver Completo

```python
# src/transformation/clean.py

import polars as pl
from src.audit.logger import audit

def build_silver() -> pl.DataFrame:
    
    # PASO 1: Union de todos los años Bronze
    bronze_files = [
        "data/bronze/sell_data_2023.parquet",
        "data/bronze/sell_data_2024.parquet",
        "data/bronze/sell_data_2025.parquet",
    ]
    df = pl.concat([pl.read_parquet(f) for f in bronze_files])
    rows_before_dedup = len(df)
    
    # PASO 2: DEDUPLICACIÓN CRÍTICA — SUM aggregation
    # Razón: fragmentación sub-canal en ERP genera múltiples rows por serie lógica
    df = df.group_by(["channel","material","category","yearweek"]).agg([
        pl.col("value").sum(),
        pl.col("source_year").first(),
        pl.col("ingested_at").max(),
    ])
    rows_after_dedup = len(df)
    audit.log_pipeline_step("TRANSFORMATION", "DEDUP_AGGREGATION",
        "bronze/*", "silver/timeseries_clean.parquet",
        rows_before_dedup, rows_after_dedup,
        params_json=f'{{"rows_removed": {rows_before_dedup - rows_after_dedup}}}')
    
    # PASO 3: FLAG censored (2025 values capped at 999)
    df = df.with_columns([
        ((pl.col("value") == 999) & (pl.col("source_year") == 2025))
            .alias("is_censored")
    ])
    
    # PASO 4: CORRECCIÓN inventario negativo
    # Inventario negativo es físicamente imposible → reemplazar con 0
    # Preservar flag para auditoría
    df = df.with_columns([
        ((pl.col("category") == "Channel Inv.") & (pl.col("value") < 0))
            .alias("inv_was_negative"),
    ])
    df = df.with_columns([
        pl.when(
            (pl.col("category") == "Channel Inv.") & (pl.col("value") < 0)
        ).then(pl.lit(0.0)).otherwise(pl.col("value")).alias("value")
    ])
    
    # PASO 5: Derivar ISO date desde yearweek
    df = df.with_columns([
        pl.col("yearweek").str.slice(0, 4).cast(pl.Int32).alias("year"),
        pl.col("yearweek").str.slice(4, 2).cast(pl.Int32).alias("week"),
    ])
    # Lunes de la semana ISO: 
    # date = jan_4th_of_year + (week-1)*7 days - weekday(jan_4th)
    df = df.with_columns([
        (
            pl.date(pl.col("year"), 1, 4)
            .dt.offset_by(
                ((pl.col("week") - 1) * 7 - 
                 pl.date(pl.col("year"), 1, 4).dt.weekday()
                ).cast(pl.Utf8) + "d"
            )
        ).alias("date")
    ])
    
    # PASO 6: Parse product_family, model_code, sku_size
    df = df.with_columns([
        pl.col("material").str.splitn(",", 5)
          .list.get(0).str.strip_chars().alias("product_family"),
        pl.col("material").str.splitn(",", 5)
          .list.get(1).str.strip_chars().alias("model_code"),
        pl.col("material").str.splitn(",", 5)
          .list.get(2).str.strip_chars().alias("sku_size"),
    ])
    
    # PASO 7: Calcular inventory balance residual (señal de ajustes)
    # Se calcula en un join posterior al pivotear categorías
    df = _add_inventory_residual(df)
    
    return df


def _add_inventory_residual(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula: residual = inv(t) - [inv(t-1) + sell_in(t) - sales(t)]
    Series con residual sistemáticamente != 0 tienen ajustes no capturados.
    """
    # Pivot categories
    pivot = df.pivot(
        index=["channel","material","yearweek","date","year","week"],
        on="category", values="value"
    ).rename({
        "Sell-in": "sell_in_temp",
        "Cust. Sales": "sales_temp",
        "Channel Inv.": "inv_temp"
    })
    
    # Lag inv_temp por serie
    pivot = pivot.sort(["channel","material","date"])
    pivot = pivot.with_columns([
        pl.col("inv_temp").shift(1).over(["channel","material"])
          .alias("inv_prev_temp")
    ])
    pivot = pivot.with_columns([
        (pl.col("inv_temp") - 
         (pl.col("inv_prev_temp").fill_null(pl.col("inv_temp")) + 
          pl.col("sell_in_temp").fill_null(0) - 
          pl.col("sales_temp").fill_null(0))
        ).alias("inv_balance_residual")
    ])
    
    # Join residual back
    residual_df = pivot.select(["channel","material","yearweek","inv_balance_residual"])
    df = df.join(residual_df, on=["channel","material","yearweek"], how="left")
    return df
```

### Schema Silver

```python
SILVER_SCHEMA = {
    "channel":                pl.Utf8,
    "material":               pl.Utf8,
    "category":               pl.Utf8,
    "yearweek":               pl.Utf8,
    "date":                   pl.Date,
    "year":                   pl.Int32,
    "week":                   pl.Int32,
    "value":                  pl.Float64,    # Cleaned value
    "is_censored":            pl.Boolean,
    "inv_was_negative":       pl.Boolean,
    "product_family":         pl.Utf8,
    "model_code":             pl.Utf8,
    "sku_size":               pl.Utf8,
    "source_year":            pl.Int32,
    "ingested_at":            pl.Utf8,
    "inv_balance_residual":   pl.Float64,   # Solo para category=Channel Inv.
}

# Invariantes post-Silver:
# 1. CERO duplicados en (channel, material, category, yearweek)
# 2. CERO valores negativos en Channel Inv. rows
# 3. len(Silver) <= len(Bronze) — dedup solo puede reducir
# 4. Exactamente 3 categorías por (channel, material, yearweek)
```

---

## 4. Gold Layer — Especificación Detallada

### Sub-tablas y sus Propósitos

| Archivo | Propósito | Filas Est. | Columnas |
|---|---|---|---|
| `feature_store.parquet` | ML features + targets | ~1.15M | 105 |
| `training_set.parquet` | Subset para training | ~400K | 105 |
| `series_registry.parquet` | Clasificación ADI/CV² | ~8,413 | 15 |
| `forecast_output.parquet` | Predicciones semanales | ~160K/año | 18 |
| `inventory_risk.parquet` | Scores de riesgo | ~8,413/semana | 20 |
| `accuracy_log.parquet` | Historial MAPE | ~52/año | 12 |
| `drift_log.parquet` | Historial PSI | ~52×8/año | 8 |

### Filtros para training_set

```python
def build_training_set(feature_store: pl.DataFrame) -> pl.DataFrame:
    return feature_store.filter(
        # Solo datos históricos (excluir placeholders futuros)
        (pl.col("yearweek") <= "202533") &
        # Excluir valores truncados de 2025
        (~pl.col("is_censored")) &
        # Solo Sell-in (target primario)
        (pl.col("sell_in").is_not_null()) &
        # Suficiente historia para lags anuales
        (pl.col("sell_in_lag_52").is_not_null()) &
        # Excluir series completamente muertas (nunca tuvieron demanda)
        (pl.col("sell_in_ma26") > 0)
    )
```

---

## 5. Matriz de Responsabilidades de Capas

| Aspecto | Raw/Bronze | Silver | Gold | Audit |
|---|---|---|---|---|
| Modificable post-write | ❌ Nunca | ❌ Solo rebuild | ✅ Rebuild semanal | ❌ Nunca |
| Lógica de negocio | ❌ | ✅ | ✅ | ❌ |
| Features ML | ❌ | ❌ | ✅ | ❌ |
| Duplicados | Presentes | Eliminados | N/A | N/A |
| Neg. inventario | Presentes | Corregidos | Heredado | N/A |
| ISO dates | ❌ | ✅ | ✅ | ✅ |
| DVC tracked | ✅ | ✅ | ✅ | ✅ |
| Consumido por BI | ❌ | Indirecto | ✅ | Queries |

---

*AI-DLC Traceability ID: BSG-ITER3-001 | Version: 3.0*
