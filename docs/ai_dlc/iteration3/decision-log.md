# Decision Log — Iteración 3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3 — Data Platform & Governance
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** DEC-ITER3-001

---

> Las decisiones DEC-001 a DEC-009 están bloqueadas desde Iter2.
> Este documento registra únicamente DEC-010 en adelante.

---

## DEC-010 — Modelo de Gobierno: Data Mesh Lite

| Campo | Valor |
|---|---|
| **ID** | DEC-010 |
| **Fecha** | 2026-05-25 |
| **Status** | ✅ LOCKED |
| **Deciders** | Principal Data Architect + AI-DLC Constructor |

**Contexto:** El proyecto maneja datos de múltiples dominios funcionales (Supply Chain, Sales, Finance, Product) pero con un equipo técnico pequeño. Un Data Mesh completo requiere infraestructura distribuida por dominio que no está justificada.

**Opciones:**
1. Data Monolith centralizado — simple, pero sin ownership claro por dominio
2. **Data Mesh Lite** — ownership por dominio funcional, infraestructura centralizada
3. Data Mesh completo — demasiado overhead para el tamaño del equipo

**Decisión:** Data Mesh Lite — dominios funcionales con ownership formal, sobre infraestructura técnica centralizada (DuckDB + Parquet único).

**Dominios definidos:**
| Dominio | Data Owner | Datasets Owned |
|---|---|---|
| Supply Chain | VP Supply Chain | channel_inv, sell_in, replenishment |
| Commercial | Commercial Director | cust_sales, channel_kpis, promotions |
| Product | Category Manager | material_catalog, product_hierarchy |
| ML Platform | ML Architect | feature_store, forecast_output, model_registry |

**Tradeoff aceptado:** Ownership es contractual, no técnico. No hay data planes separadas.

---

## DEC-011 — Semantic Layer: DuckDB Views + Python Module

| Campo | Valor |
|---|---|
| **ID** | DEC-011 |
| **Fecha** | 2026-05-25 |
| **Status** | ✅ LOCKED |

**Contexto:** Sin una capa semántica, cada dashboard calcula KPIs de forma independiente → inconsistencia de métricas → pérdida de confianza.

**Opciones:**
1. dbt sobre DuckDB — añade una capa de transformación SQL versionada
2. **DuckDB Views + Python module centralizado** — más liviano, mismo beneficio de consistencia
3. Cube.js / Metriql — over-engineering para este escala

**Decisión:** `src/semantic/metrics.py` define todas las métricas como funciones Python. DuckDB views materializan las más usadas. Ningún dashboard calcula KPIs inline — todos importan desde el semantic module.

**Tradeoff:** Menor poder expresivo que dbt, pero cero overhead de infraestructura adicional.

---

## DEC-012 — Feature Reproducibility: Hash-Based Versioning

| Campo | Valor |
|---|---|
| **ID** | DEC-012 |
| **Fecha** | 2026-05-25 |
| **Status** | ✅ LOCKED |

**Decisión:** Cada `feature_store.parquet` tiene un SHA-256 hash registrado en MLflow como tag `feature_store_hash`. Antes de inference, el sistema compara el hash del feature_store actual contra el hash almacenado en el modelo. Si difieren → WARNING + log. Si la diferencia es en columnas críticas (lags, ratios) → HALT.

**Implementación:**
```python
import hashlib, dvc.api

def get_feature_store_hash() -> str:
    """Returns content hash of feature_store.parquet"""
    return dvc.api.get_url("data/gold/feature_store.parquet")[:16]
```

---

## DEC-013 — Audit Trail: Append-Only Parquet Log

| Campo | Valor |
|---|---|
| **ID** | DEC-013 |
| **Fecha** | 2026-05-25 |
| **Status** | ✅ LOCKED |

**Decisión:** Un archivo `data/audit/audit_log.parquet` recibe appends de todos los eventos auditables: ingestions, transformations, training runs, forecast generations, overrides, alerts.

**Schema:**
```
event_id, event_type, timestamp, actor, input_hash, output_hash, 
params_json, status, error_message, duration_seconds
```

**Principio:** Append-only (nunca se modifica). Versionado por DVC. Sirve como trail forense para cualquier pregunta de "¿qué modelo generó este forecast?" o "¿quién modificó este dato?".

---

## DEC-014 — Método de Reconciliación: MinT-Shrink

| Campo | Valor |
|---|---|
| **ID** | DEC-014 |
| **Fecha** | 2026-05-25 |
| **Status** | ✅ LOCKED |

**Decisión:** MinT-Shrink (Minimum Trace con estimador shrinkage) como método de reconciliación jerárquica.

**Justificación técnica:**
- MinT-OLS: óptimo en teoría pero matriz de covarianza singular con muchas series sparse
- MinT-Shrink: regulariza la matriz con un estimador de shrinkage → estable numéricamente
- BU (Bottom-Up): ignora los forecasts de niveles superiores, los cuales son más precisos en nuestro caso
- TD (Top-Down): descarta variabilidad por canal y familia

**Validación requerida:** Sum(level3) == level0 ± 0.01 post-reconciliación.

---

## DEC-015 — Criterio Clasificación Intermitente: Syntetos-Boylan

| Campo | Valor |
|---|---|
| **ID** | DEC-015 |
| **Fecha** | 2026-05-25 |
| **Status** | ✅ LOCKED |

**Decisión:** Usar el criterio Syntetos-Boylan para clasificar series en el cuadrante de demanda:
- ADI (Average Demand Interval) = promedio de semanas entre demandas no-cero
- CV² = coeficiente de variación al cuadrado del tamaño de demanda

```
ADI < 1.32  y  CV² < 0.49  → Suave (Regular, usar MA/SES)
ADI ≥ 1.32  y  CV² < 0.49  → Intermitente (usar Croston)
ADI < 1.32  y  CV² ≥ 0.49  → Errático (usar SES con ajuste)
ADI ≥ 1.32  y  CV² ≥ 0.49  → Lumpy (usar TSB/IMAPA)
```

Este criterio es más preciso que solo contar semanas activas.

---

*AI-DLC Traceability ID: DEC-ITER3-001 | Version: 3.0*
