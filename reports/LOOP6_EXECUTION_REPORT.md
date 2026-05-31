# AI-DLC CONSTRUCTION LOOP 6 — EXECUTION REPORT
**Walk-Forward CV con Stage 2 Segmentado + Intervalos Q10/Q90 en Dashboard**
`Generated: 2026-05-31 17:49 UTC | Status: ✅ ALL STAGES COMPLETE`

---

## Executive Summary

Loop 6 valida el modelo Loop 5 con walk-forward CV de 5 folds usando un **Stage 2 segmentado**
(regular demand vs sparse demand) y genera los datos de intervalos de confianza Q10/Q90
para el dashboard Streamlit.

| Métrica | Loop 5 (fold 5) | Loop 6 CV avg | Objetivo | Estado |
|---|---|---|---|---|
| active_WAPE | 67.3403 | **72.8447** | < 80 | ✅ |
| MASE | 0.4585 | **0.4952** | < 0.55 | ✅ |
| bias% | -57.1 | **-59.3326** | reducir | — |
| demand_F1 | 0.803 | **0.7748** | > 0.35 | ✅ |
| Classifier AUC | 0.9631 | **0.9644** | — | — |

---

## 1. Walk-Forward CV — 5 Folds (Segmented Stage 2)

| Fold | Train | Val | AUC | Seg (Reg/Spa) | active_WAPE | MASE | demand_F1 |
|---|---|---|---|---|---|---|---|
| 1 | ≤202352 | 202401–202413 | 0.9418 | 10814 / 20039 | 79.2067 | 0.6154 | 0.5801 |
| 2 | ≤202413 | 202414–202426 | 0.9712 | 13803 / 25389 | 75.475 | 0.4744 | 0.7774 |
| 3 | ≤202426 | 202427–202439 | 0.9719 | 16586 / 30791 | 73.6628 | 0.4739 | 0.8541 |
| 4 | ≤202439 | 202440–202452 | 0.9733 | 19636 / 35960 | 68.9721 | 0.4573 | 0.8596 |
| 5 | ≤202452 | 202501–202513 | 0.964 | 22517 / 40574 | 66.9069 | 0.4548 | 0.8026 |
| **AVG** | | | **0.9644** | | **72.8447** | **0.4952** | **0.7748** |

---

## 2. Stage 2 Segmentado

El modelo Stage 2 (regresor de cantidad) se divide en dos modelos según `demand_rate_12`:

| Segmento | Criterio | Tratamiento |
|---|---|---|
| **regular** | demand_rate_12 > 0.10 | Huber + log_target=True (Optuna params) |
| **sparse** | demand_rate_12 ≤ 0.10 | Huber + log_target=True (mismo config) |
| **fallback** | segmento sin datos | usa el modelo disponible del otro segmento |

---

## 3. Intervalos de Confianza Q10/Q90 — Dashboard

- **Datos generados:** `data/forecasts/loop6_ci_predictions.parquet`
- **Fuente:** fold 5 del CV (val W01–W13·2025, 21 features)
- **Dashboard:** banda verde Q10–Q90 visible en pestaña "Proyección de Demanda"
- **KPIs nuevos:** active_WAPE y MASE del CV directo en el dashboard

---

## 4. MLflow

- **Experimento:** `ai_dlc_loop6_cv_segmented`
- **Run ID:** `a673a6dcf8c14c15ae4520b3a24ab2ae`

---

## 5. Loop 7 Priorities

```
MODELADO
  ☐ Optuna conjunto Stage 1 + Stage 2 (actualmente sólo Stage 2 fue tunado)
  ☐ Bias correction multiplicativa: ratio mean_actual / mean_predicted por segmento
  ☐ Features de canal cruzado con retraso de 1 período (leading indicator)
  ☐ Modelo dedicado para SKUs de alta frecuencia (demand_rate_12 > 0.3)

INFRAESTRUCTURA
  ☐ Forecast recursivo: predecir W34·2025→W52·2026 con lag features actualizados
  ☐ Alertas de inventario usando Q90 como escenario pesimista de cobertura
  ☐ Modelo registry formal: champion vs challenger en MLflow

DASHBOARD
  ☐ Mostrar banda CI en el forecast futuro (actualmente sólo en val W01–W13·2025)
  ☐ Selector de nivel de confianza (70% / 80% / 90%) en UI
```

---

*AI-DLC Loop 6 — Segmented Stage 2 + Q10/Q90 Dashboard Integration — Complete.*
