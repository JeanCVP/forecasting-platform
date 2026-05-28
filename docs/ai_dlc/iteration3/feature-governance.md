# Feature Governance
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** FEATGOV-ITER3-001

---

## 1. Principios de Gobierno de Features

| Principio | Descripción |
|---|---|
| **Single Writer** | Solo `feature_engineering.py` puede escribir el feature store |
| **Immutability** | Un feature store generado para un `training_cutoff` dado no se modifica |
| **Reproducibility** | Mismo código + mismos datos + mismos params = mismo hash de features |
| **Leakage-Free** | Todo cambio a features pasa por `tests/test_leakage.py` en CI |
| **Documented First** | Ningún feature va a producción sin estar en `feature-catalog.md` |
| **Semantic Alignment** | Features de KPI deben alinearse con definiciones de `semantic-governance.md` |

---

## 2. Feature Ownership

| Grupo de Features | Owner Técnico | Owner de Negocio | Revisión |
|---|---|---|---|
| Lags de Sell-in/Sales | ML Platform | Supply Chain | Quarterly |
| Rolling MAs | ML Platform | Supply Chain | Quarterly |
| Inventory ratios (DOS, etc.) | ML Platform | Supply Chain | Monthly |
| Calendar events | ML Platform | Commercial | Before each season |
| Lifecycle features (SKU age) | ML Platform | Product | Quarterly |
| Cross-series aggregates | ML Platform | Commercial | Quarterly |
| Zero-inflation features | ML Platform | ML Platform | Quarterly |

---

## 3. Feature Lifecycle

```
ESTADOS DE UN FEATURE

  PROPUESTO        → Nuevo feature sugerido; pendiente evaluación
  EN DESARROLLO    → Implementado en rama de desarrollo
  STAGING          → En feature store staging; evaluación de impacto
  PRODUCCIÓN       → En feature_store.parquet activo
  DEPRECADO        → Marcado para remoción; no usar en nuevos modelos
  RETIRADO         → Eliminado del feature store
```

### Proceso de Adición de Feature

```
1. Crear entry en feature-catalog.md con:
   - nombre, fórmula, tipo, rango válido, dueño
2. Implementar en feature_engineering.py en rama dev
3. Pasar tests/test_leakage.py
4. Pasar tests/test_feature_ranges.py
5. Medir impacto en CV sMAPE (debe mejorar ≥ 0.5% o ser neutro)
6. PR con aprobación del ML Architect
7. Merge → feature entra a producción en próximo pipeline run
```

### Proceso de Deprecación

```
1. Marcar en feature-catalog.md como DEPRECATED con fecha
2. Verificar que ningún modelo en Production stage usa el feature
3. Añadir WARNING en feature_engineering.py si el feature se sigue computando
4. Después de 2 meses sin uso → remover del pipeline
5. Actualizar feature-catalog.md a RETIRED
```

---

## 4. SLAs de Features

| Dimensión | SLA |
|---|---|
| Disponibilidad | Feature store disponible ≤ 4h después de llegada de nuevos datos |
| Latencia de actualización | Silver completado → Gold features ≤ 30 min |
| Correctitud | 0 features con leakage detectado (CI gate) |
| Cobertura | 100% de series activas tienen features computadas |
| Hash consistency | Feature hash registrado en todo MLflow run |

---

## 5. Reglas Específicas de Features Críticos

### Regla F-01: days_of_supply
```
CORRECTO:   inv_lag_1 / max(sales_ma4 / 7, 0.01)
INCORRECTO: channel_inv / cust_sales  ← usa valores de t=0
INCORRECTO: inv_lag_1 / sales_lag_1   ← una semana es demasiado volátil

Razón: sales_ma4 es más estable que sales_lag_1 como proxy de
       velocidad de demanda. sales_lag_1 puede ser 0 en semanas
       sin ventas, causando DOS = ∞.
```

### Regla F-02: sell_through_rate
```
CORRECTO:   sales_ma4 / max(sell_in_ma4, 0.01)
INCORRECTO: cust_sales / sell_in  ← valores puntuales muy volátiles

Razón: Las medias móviles suavizan la alta volatilidad semanal.
```

### Regla F-03: rolling features
```
CORRECTO:   sell_in.shift(1).rolling_mean(4).over(key)
INCORRECTO: sell_in.rolling_mean(4).over(key)  ← incluye semana actual

Razón: Sin shift(1), la ventana rolling incluye el valor de la
       semana que se está prediciendo → leakage temporal.
```

### Regla F-04: YoY features
```
Usar triplete (lag_51, lag_52, lag_53) cuando sea posible.
Razón: El calendario ISO no es exactamente anual (53 semanas
       algunos años). El triplete captura ±1 semana de tolerancia.
```

---

## 6. Testing Framework

```python
# tests/test_features.py

import polars as pl
import pytest
from src.transformation.feature_engineering import build_feature_store

@pytest.fixture(scope="module")
def feature_df():
    return build_feature_store()

def test_no_nulls_in_key_identifiers(feature_df):
    for col in ["channel","material","yearweek","segment"]:
        nulls = feature_df[col].null_count()
        assert nulls == 0, f"{col} has {nulls} nulls"

def test_dos_non_negative(feature_df):
    neg = (feature_df["days_of_supply"] < 0).sum()
    assert neg == 0, f"{neg} negative DOS values"

def test_sell_through_reasonable(feature_df):
    extreme = (feature_df["sell_through_rate_4w"] > 20).sum()
    assert extreme < 100, f"{extreme} extreme sell-through values (>20x)"

def test_cyclic_features_range(feature_df):
    for col in ["week_sin","week_cos","month_sin","month_cos"]:
        assert float(feature_df[col].min()) >= -1.01
        assert float(feature_df[col].max()) <= 1.01

def test_prob_nonzero_in_01(feature_df):
    for col in ["prob_nonzero_4w","prob_nonzero_13w","prob_nonzero_52w"]:
        if col in feature_df.columns:
            assert float(feature_df[col].min()) >= 0.0
            assert float(feature_df[col].max()) <= 1.0

def test_feature_count(feature_df):
    # Should have at least 80 features
    assert len(feature_df.columns) >= 80, (
        f"Feature count {len(feature_df.columns)} below minimum 80"
    )

def test_series_coverage(feature_df, series_registry):
    """All active series must have features."""
    active_ids = set(
        series_registry.filter(pl.col("segment") != "dead")
        .select(pl.col("channel") + "__" + pl.col("material"))
        ["channel"].to_list()
    )
    feature_ids = set(
        (feature_df["channel"] + "__" + feature_df["material"]).unique().to_list()
    )
    missing = active_ids - feature_ids
    assert len(missing) == 0, f"{len(missing)} active series missing from feature store"
```

---

*AI-DLC Traceability ID: FEATGOV-ITER3-001 | Version: 3.0*
