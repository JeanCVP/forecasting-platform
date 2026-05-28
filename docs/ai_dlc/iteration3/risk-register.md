# Risk Register — Iteración 3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3 — Data Platform & Governance
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** RISK-ITER3-001

---

## Escala de Valoración

| Severidad | Probabilidad | Score = S × P |
|---|---|---|
| 🔴 Crítico (4) | 🔴 Casi Seguro (4) | 13–16: DETENER |
| 🟠 Alto (3) | 🟠 Probable (3) | 9–12: ESCALAR |
| 🟡 Medio (2) | 🟡 Posible (2) | 5–8: GESTIONAR |
| 🟢 Bajo (1) | 🟢 Improbable (1) | 1–4: MONITOREAR |

---

## BLOQUE A — RIESGOS DE DATOS (Heredados + Actualizados)

### RISK-D01 — 2025 Export Cap en 999 *(ESCALADO)*
| Campo | Valor |
|---|---|
| **Score** | **16 — DETENER** |
| **Status** | 🔴 ABIERTO — sin resolución desde Iter2 |
| **Severidad** | Crítica (4) |
| **Probabilidad** | Casi Seguro (4) |
| **Impacto Iter3** | Silver pipeline diseñado con flag `is_censored`; Gold training_set excluye estas celdas. Sin extract corregido, el 33% del año 2025 es inutilizable para high-volume SKUs. |
| **Acción Inmediata** | Bloquear uso de 2025 en training hasta confirmación. Usar 2023–2024 como años base. Aplicar `is_censored=True` flag en todo pipeline. |
| **Escalación** | → VP Supply Chain + Equipo ERP. SLA: resolución en ≤ 2 semanas. |
| **Mitigación Interim** | Entrenar con 2023–2024 únicamente. Evaluar si 2025 W01–W33 sin cap para low-volume SKUs (DOS < 100) es confiable. |

---

### RISK-D02 — Duplicados Fragmentados (Mitigado en Diseño)
| Campo | Valor |
|---|---|
| **Score** | **8 — GESTIONAR** (bajó de 16 por mitigación en pipeline) |
| **Status** | 🟡 MITIGADO EN DISEÑO — pendiente validación en implementación |
| **Mitigación** | groupby+SUM en Bronze→Silver. Gate de validación: 0 duplicados en Silver. |
| **Riesgo Residual** | Si el groupby SUM introduce errores en series donde la fragmentación tiene semántica distinta (e.g., diferentes almacenes con independencia de inventario), el agregado SUM puede sumar stocks que no son fungibles. |
| **Acción** | Confirmar con business owner si CUSTOMER_X tiene un único warehouse por SKU o múltiples. |

---

### RISK-D03 — String Dtypes en 2025 (Mitigado)
| Campo | Valor |
|---|---|
| **Score** | **4 — MONITOREAR** |
| **Status** | ✅ MITIGADO — `pd.to_numeric()` + schema gate en Bronze |
| **Riesgo Residual** | Futuras extracciones pueden introducir nuevos problemas de tipo |
| **Control** | Great Expectations schema validation en Gate-0 |

---

### RISK-D04 — Churn de Portfolio (42% anual)
| Campo | Valor |
|---|---|
| **Score** | **9 — ESCALAR** |
| **Status** | 🟠 ACTIVO — afecta cold-start strategy |
| **Impacto Iter3** | Sin master de productos, la jerarquía no puede asignar nuevas SKUs a familias correctamente. El `product_family` se infiere de `material[0]`, pero SKUs nuevas con naming inconsistente quedarán en familia "UNKNOWN". |
| **Mitigación** | Reglas de normalización de familia en Silver. Familia "UNKNOWN" recibe cold-start conservador (mean del total). Solicitar master de productos al equipo de Category Management. |

---

### RISK-D05 — Ausencia de Calendario Promocional
| Campo | Valor |
|---|---|
| **Score** | **12 — ESCALAR** |
| **Status** | 🔴 ABIERTO — ningún progreso desde Iter2 |
| **Impacto Cuantificado** | En semanas de Día sin IVA, la demanda puede ser 3–8× la media semanal. Sin este feature, el modelo sub-forecasta estas semanas con error MAPE de 200–600%. |
| **Plan de Mitigación** | (1) Detección retroactiva: semanas donde actual > 3× media histórica → flag `is_outlier_week`. (2) Override manual en Demand Planning Workbench. (3) Solicitar calendario formal con prioridad ALTA. |

---

## BLOQUE B — RIESGOS DE GOBERNANZA (Nuevos en Iter3)

### RISK-G01 — Definiciones de Métricas Inconsistentes entre Dashboards
| Campo | Valor |
|---|---|
| **Score** | **9 — ESCALAR** |
| **Status** | 🟠 NUEVO en Iter3 |
| **Descripción** | Sin una capa semántica central, "Sell-through Rate" puede calcularse de 3 formas distintas en 3 dashboards diferentes, generando inconsistencia entre equipos. |
| **Impacto** | Pérdida de confianza del negocio cuando dos reportes muestran números distintos para el mismo KPI. |
| **Mitigación** | `semantic-layer-definition.md` + DuckDB views centralizadas. Todos los dashboards consumen las mismas views, no el feature_store directamente. |
| **Control** | CI test: todos los dashboards importan desde `src/semantic/metrics.py`, nunca calculan KPIs inline. |

---

### RISK-G02 — Ausencia de Data Ownership Formal
| Campo | Valor |
|---|---|
| **Score** | **6 — GESTIONAR** |
| **Status** | 🟡 NUEVO en Iter3 |
| **Descripción** | Sin asignación formal de data owners, los problemas de calidad de datos no tienen responsable claro. Las correcciones de datos se demoran indefinidamente. |
| **Mitigación** | `semantic-governance.md` define owners por dominio. Escalation path documentado en `audit.md`. |

---

### RISK-G03 — Feature Reproducibility Failure
| Campo | Valor |
|---|---|
| **Score** | **9 — ESCALAR** |
| **Status** | 🟠 NUEVO en Iter3 |
| **Descripción** | Si el código de feature engineering cambia sin que el modelo sea reentrenado, el modelo en producción recibe features con distribuciones diferentes a las de entrenamiento. Silent failure. |
| **Impacto** | Forecasts incorrectos sin ninguna alerta visible. |
| **Mitigación** | Feature hash vinculado a cada run de MLflow. Gate en inference: hash del feature_store actual debe coincidir con hash en metadata del modelo. Si no coincide → HALT + alerta. |

---

### RISK-G04 — Temporal Leakage en Features de Inventario
| Campo | Valor |
|---|---|
| **Score** | **12 — ESCALAR** |
| **Status** | 🟠 ACTIVO — riesgo sistémico de diseño |
| **Descripción** | `channel_inv` en semana t incorpora `sell_in` de la misma semana t. Usar `channel_inv(t)` como feature para predecir `sell_in(t)` es leakage circular. |
| **Especificidad** | Esta es la forma más peligrosa de leakage porque: (1) el modelo aprende una identidad contable, no un patrón real; (2) en producción, el inventario real de t aún no existe cuando predices; (3) mejora artificialmente las métricas CV. |
| **Mitigación** | Regla arquitectónica hard: `channel_inv` solo se usa como `inv_lag_1` (t-1) o mayor. Test de leakage en CI que detecta features de t=0 del target. |

---

### RISK-G05 — Drift sin Detección → Forecast Stale
| Campo | Valor |
|---|---|
| **Score** | **9 — ESCALAR** |
| **Status** | 🟠 ACTIVO |
| **Descripción** | El modelo puede degradarse silenciosamente durante semanas sin que ningún stakeholder lo note si no hay monitoreo automático. |
| **Mitigación** | `drift-detection.md` + `monitoring-strategy.md`. PSI semanal en 8 features críticas. MAPE rolling 4 semanas con threshold alert. |

---

## BLOQUE C — RIESGOS ML (Actualizados Iter3)

### RISK-ML01 — Sparsity Extrema Invalida Métricas Estándar
| Campo | Valor |
|---|---|
| **Score** | **12 — ESCALAR** |
| **Status** | 🟠 ACTIVO |
| **Descripción** | MAPE estándar es indefinido cuando actual=0 (división por cero). El 50% de las series tienen actual=0 la mayoría de las semanas. Usar MAPE naively producirá resultados nonsense o infinitos. |
| **Mitigación** | sMAPE (simétrico) para todas las métricas. MASE como métrica secundaria. Filtrar actual=0 para MAPE tradicional. Documentar en `experiment-tracking.md`. |

---

### RISK-ML02 — Reconciliación Jerárquica Falla con Series Muy Sparse
| Campo | Valor |
|---|---|
| **Score** | **8 — GESTIONAR** |
| **Status** | 🟡 ACTIVO |
| **Descripción** | MinT requiere estimar la matriz de covarianza de errores. Con 8,400 series, esta matriz es 8400×8400 y muchas series tienen varianza cero (todas predicciones = 0). Matriz singular → falla numérica. |
| **Mitigación** | MinT-Shrink usa estimador shrinkage que regulariza la matriz. Alternatively: reconciliar solo series con active_weeks > 4. Series dead → zero forecast sin reconciliación. |

---

## BLOQUE D — RIESGOS DE PLATAFORMA (Actualizados Iter3)

### RISK-P01 — DuckDB In-Process Limitations en Producción
| Campo | Valor |
|---|---|
| **Score** | **6 — GESTIONAR** |
| **Status** | 🟡 ACTIVO |
| **Descripción** | DuckDB es in-process y no soporta múltiples escritores concurrentes. En producción con Prefect multi-task, escrituras paralelas al mismo Parquet pueden corromperse. |
| **Mitigación** | Cada task escribe a archivo temporal único. Merge al final del flow en single writer. Usar MotherDuck o DuckDB file locking si escala a multi-proceso. |

---

## Resumen del Heat Map

```
              IMPROBABLE  POSIBLE   PROBABLE  CASI SEGURO
CRÍTICO    |            |  G04     |          | D01        |
ALTO       |            |  ML02    | D02,D05  | ML01,G01   |
           |            |          | G03,G05  | RISK-P01   |
MEDIO      |            | D03      | D04,G02  |            |
BAJO       |            |          |          |            |
```

---

## Registro de Riesgos Cerrados

| ID | Descripción | Fecha Cierre | Cierre |
|---|---|---|---|
| RISK-D03 | String dtypes 2025 | 2026-05-25 | Mitigado por schema gate Bronze |

---

*AI-DLC Traceability ID: RISK-ITER3-001 | Version: 3.0*
