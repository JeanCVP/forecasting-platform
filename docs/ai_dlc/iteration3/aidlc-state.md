# AI-DLC State Document — Iteración 3
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3 — Data Platform, Governance & MLOps Architecture
**State Date:** 2026-05-25
**Classification:** INTERNAL — CONFIDENTIAL
**AI-DLC Traceability ID:** AIDLC-STATE-ITER3-001

---

## 1. Continuidad desde Iteraciones Previas

### Iteración 0 (Assessment)
- ✅ Dataset profiling: 105K filas, 3 años, 3 categorías métricas
- ✅ DQ Score inicial: 61/100 — 3 issues críticos identificados
- ✅ Feasibility: VIABLE con preprocessing obligatorio
- ✅ Feature engineering plan: 105 features definidas

### Iteración 1–2 (Architecture Foundation)
- ✅ Stack tecnológico locked: DuckDB + Polars + MLForecast + LightGBM + Prefect + MLflow + DVC
- ✅ Lakehouse medallion: Bronze/Silver/Gold diseñado
- ✅ ML strategy: Middle-out jerárquico, 4 segmentos de series
- ✅ 20 documentos arquitectónicos entregados
- ⚠️ BLOQUEANTE ACTIVO: 2025 data cap en 999 — pendiente extract sin truncar
- ⚠️ BLOQUEANTE ACTIVO: Calendario promocional ausente

### Iteración 3 (Este documento)
- 🔄 Data Platform & Governance Architecture completa
- 🔄 Forecast hierarchy formal con reconciliación
- 🔄 Feature store governance y catalog
- 🔄 MLOps operational con drift detection
- 🔄 Semantic layer y audit trail

---

## 2. Estado de Fase Actualizado

| Fase | Nombre | Status | Output Esperado |
|---|---|---|---|
| **Fase 0** | Dataset Assessment | ✅ COMPLETE | 7 docs assessment |
| **Fase 1** | Architecture Foundation | ✅ COMPLETE | 20 docs arquitectura |
| **Fase 2** | **Data Platform & Governance** | 🔄 IN PROGRESS | **25 docs (esta iteración)** |
| **Fase 3** | Data Platform Build | ⬜ NEXT | Pipelines operacionales en código |
| **Fase 4** | ML Baseline & Validation | ⬜ PLANNED | Modelos baseline + CV framework |
| **Fase 5** | ML Advanced Production | ⬜ PLANNED | LightGBM global champion |
| **Fase 6** | MLOps Full Operations | ⬜ PLANNED | Monitoring + retraining automatizado |
| **Fase 7** | BI & Stakeholder Adoption | ⬜ PLANNED | 6 dashboards live |

---

## 3. Decisiones Arquitectónicas Confirmadas (No Reabrir)

| ID | Decisión | Estado |
|---|---|---|
| DEC-001 | DuckDB + Parquet como data store | 🔒 LOCKED |
| DEC-002 | MLForecast + LightGBM como framework primario | 🔒 LOCKED |
| DEC-003 | Prefect 2.x como orquestador | 🔒 LOCKED |
| DEC-004 | MLflow como experiment tracker | 🔒 LOCKED |
| DEC-005 | Streamlit (prototipo) → Power BI (producción) | 🔒 LOCKED |
| DEC-006 | Weekly como granularidad temporal primaria | 🔒 LOCKED |
| DEC-007 | Middle-out jerárquico (Family × Channel) | 🔒 LOCKED |
| DEC-008 | SUM aggregation para duplicados | 🔒 LOCKED |
| DEC-009 | Negative inv → 0 en Silver; flag preservado | 🔒 LOCKED |

---

## 4. Nuevas Decisiones — Iteración 3

| ID | Tema | Decisión | Rationale |
|---|---|---|---|
| DEC-010 | Governance model | Data Mesh lite — dominios funcionales sin infraestructura distribuida | Balance entre rigor y overhead operacional |
| DEC-011 | Semantic layer | dbt-style definitions sobre DuckDB views | Consistencia de métricas entre dashboards y modelos |
| DEC-012 | Feature reproducibility | Hash-based feature versioning + DVC | Garantiza trazabilidad modelo→features→datos |
| DEC-013 | Audit trail | Append-only audit log en Parquet | Sin base de datos adicional; compatible con DVC |
| DEC-014 | Reconciliation method | MinT-Shrink como método principal | Mejor que OLS para matrices mal condicionadas |
| DEC-015 | Intermittent threshold | ADI > 1.32 ó CV² > 0.49 → intermittente | Criterio Syntetos-Boylan confirmado |

---

## 5. Blockers Críticos — Estado Actualizado

| Blocker | Severidad | Semanas Bloqueante | Impacto si No Resuelto |
|---|---|---|---|
| 2025 data cap (999) | 🔴 CRÍTICO | ∞ | Training 2025 inválido para high-volume SKUs |
| Calendario promocional | 🟠 ALTO | ∞ | MAPE spikes en semanas de eventos; distrust del modelo |
| Master de productos ausente | 🟡 MEDIO | 4 sem | Hierarchy incompleta; cold-start degradado |
| Taxonomía de canales | 🟡 MEDIO | 4 sem | Segmentación de canales imposible |

---

## 6. Métricas de Calidad Objetivo por Fase

| Fase | DQ Score | MAPE Regular | MAPE Intermitente | Coverage |
|---|---|---|---|---|
| Fase 0 (actual raw) | 61/100 | — | — | — |
| Fase 3 (Silver) | ≥ 85/100 | — | — | — |
| Fase 4 (Baseline) | ≥ 90/100 | ≤ 35% (naive) | ≤ 50% | 100% |
| Fase 5 (ML) | ≥ 92/100 | ≤ 20% | ≤ 35% | 100% |
| Fase 6 (Production) | ≥ 95/100 | ≤ 18% | ≤ 32% | 100% |

---

## 7. Mapa de Documentos — Iteración 3

```
ai-dlc-iter3/
├── governance/
│   ├── aidlc-state.md              ← ESTE DOCUMENTO
│   ├── risk-register.md            ← Riesgos actualizados Iter3
│   ├── decision-log.md             ← Decisiones DEC-010 a DEC-015
│   ├── audit.md                    ← Framework de auditoría y trazabilidad
│   └── semantic-governance.md      ← Gobierno de métricas y definiciones
├── data-platform/
│   ├── bronze-silver-gold.md       ← Especificación detallada capas (v3)
│   ├── data-contracts.md           ← Contratos formales con GE rules
│   ├── dataset-versioning.md       ← DVC pipeline completo
│   ├── data-lineage.md             ← Mapa de linaje end-to-end
│   ├── validation-framework.md     ← Framework GE + 12 checkpoints
│   └── semantic-layer-definition.md ← Vistas DuckDB + definiciones métricas
├── forecasting/
│   ├── forecasting-strategy.md     ← Estrategia completa (v3 ampliada)
│   ├── forecast-hierarchy.md       ← Jerarquía formal 4 niveles
│   ├── hierarchical-reconciliation.md ← MinT-Shrink implementación
│   ├── intermittent-demand-strategy.md ← Croston/TSB/IMAPA detallado
│   └── inventory-aware-forecasting.md ← Integración inventario en forecast
├── feature-store/
│   ├── feature-store-design.md     ← Diseño completo (v3 ampliado)
│   ├── feature-governance.md       ← Ownership, lifecycle, SLAs
│   └── feature-catalog.md          ← Catálogo completo 105+ features
└── mlops/
    ├── mlops-architecture.md       ← Arquitectura MLOps end-to-end
    ├── experiment-tracking.md      ← MLflow conventions completas
    ├── retraining-strategy.md      ← Triggers y protocolo (v3)
    ├── drift-detection.md          ← PSI + concept drift detallado
    └── monitoring-strategy.md      ← Observability stack completo
```

---

*AI-DLC Traceability ID: AIDLC-STATE-ITER3-001 | Version: 3.0 | Continúa desde ITER2*
