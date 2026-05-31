# AI-DLC CONSTRUCTION LOOP 5 — EXECUTION REPORT
**Log-Transform + Optuna + Isotonic Calibration + Quantile Intervals**
`Generated: 2026-05-31 17:41 UTC | Status: ✅ ALL STAGES COMPLETE`

---

## Executive Summary

Loop 5 ataca el bias de -87.5% de Loop 4 con log-transform del target en Stage 2
y optimización de hiperparámetros con Optuna (30 trials).
Añade calibración isotónica al clasificador e intervalos de confianza Q10/Q90.

| Métrica | Loop 4 CV | Loop 5 | Objetivo | Estado |
|---|---|---|---|---|
| active_WAPE | 93.7601 | **67.3403** | < 80 | ✅ |
| MASE | 0.6169 | **0.4585** | < 0.60 | ✅ |
| bias% | -87.5 | **-57.1073** | reducir | ✅ |
| demand_F1 | 0.832 | **0.8026** | > 0.35 | ✅ |

---

## 1. Optuna Hyperparameter Tuning (Stage 2)

- **Trials:** 30 | **Best active_WAPE:** 65.8938
- **Sampler:** TPE (Tree-structured Parzen Estimator), seed=42

### Best hyperparameters

| Parámetro | Valor |
|---|---|
| objective | huber |
| log_target | ✅ True |
| num_leaves | 154 |
| learning_rate | 0.14839058155279436 |
| min_child_samples | 19 |
| feature_fraction | 0.5014767193713563 |
| bagging_fraction | 0.7352211639093376 |

### Top 5 trials

| # | active_WAPE | objective | log_target | num_leaves | lr |
|---|---|---|---|---|---|
| 13 | 65.8938 | huber | ✅ | 154 | 0.1484 |
| 24 | 76.6877 | huber | ✅ | 198 | 0.0456 |
| 25 | 76.7226 | huber | ✅ | 190 | 0.0461 |
| 14 | 76.8691 | huber | ✅ | 138 | 0.1496 |
| 18 | 76.8743 | huber | ✅ | 146 | 0.0674 |

---

## 2. Loop 5 Features Nuevas

| Feature | Descripción | Impacto esperado |
|---|---|---|
| `log_lag_1` | log(1+lag_1) | Comprime outliers de cantidad |
| `log_lag_4` | log(1+lag_4) | Escala 4 semanas atrás |
| `log_rolling_mean_12` | log(1+mean_12w) | Referencia de escala log |
| `cv_12` | std_12 / mean_12 (coef. variación) | Señal de volatilidad |
| `channel_nonzero_avg` | Avg qty no-cero por canal | Escala típica del canal |

**Total features Loop 5:** 21 (Base 11 + Loop3 5 + Loop5 5)

---

## 3. Stage 1 — Clasificador con Calibración Isotónica

| Métrica | Valor |
|---|---|
| AUC | 0.9631 |
| F1 @ opt_threshold | 0.8026 |
| Opt. threshold (Youden's J) | 0.0506 |
| Calibración | ✅ Isotonic Regression |
| Best iter | 4 |

---

## 4. Stage 2 — Regresor con Log-Transform

| Aspecto | Loop 4 | Loop 5 |
|---|---|---|
| Objetivo LGB | regression_l1 | huber |
| Log-transform target | ❌ No | ✅ Sí |
| num_leaves | 63 | 154 |
| min_child_samples | 10 | 19 |
| Non-zero train rows | 63,091 | 63,091 |
| Best iter | — | 22 |

**Top features Stage 2 (gain):**
  - rolling_std_12: 68265
  - channel_nonzero_avg: 50125
  - inventory_days_of_supply: 36624
  - log_rolling_mean_12: 15521
  - rolling_mean_12: 9587

---

## 5. Métricas de Validación (W01–W13·2025)

| Métrica | Loop 4 | Loop 5 | Δ |
|---|---|---|---|
| sMAPE | 10.75 | 12.2179 | — |
| WAPE | 96.11 | 68.9058 | — |
| **active_WAPE** | **93.76** | **67.3403** | **+28.2%** |
| MASE | 0.617 | 0.4585 | ▼ mejor |
| bias% | -87.51 | -57.1073 | ▼ menos sesgo |
| demand_F1 | 0.832 | 0.8026 | — |
| pinball_50 | — | 3.4161 | — |
| Q10 pinball | — | 1.0099 | ← nuevo |
| Q90 pinball | — | 2.5133 | ← nuevo |

---

## 6. MLflow

- **Experimento:** `ai_dlc_loop5_enhanced` — run: `d08c3e45977b4fa0a19a2f07589ff867`
- **Modelos guardados:**
  - `data/models/loop5_stage1_classifier.txt`
  - `data/models/loop5_stage1_calibrator.pkl`
  - `data/models/loop5_stage2_regressor.txt`
  - `data/models/loop5_stage2_q10.txt`
  - `data/models/loop5_stage2_q90.txt`

---

## 7. Loop 6 Priorities

```
DASHBOARD
  ☐ Añadir intervalos de confianza Q10/Q90 al forecast del dashboard
  ☐ Mostrar bias% y active_WAPE como KPIs en pestaña de Proyección

MODELADO
  ☐ Walk-forward CV con modelo Loop 5 (5 folds) para confirmar active_WAPE CV
  ☐ Features de precio/promo si están disponibles
  ☐ Modelo por segmento (SKUs de alta frecuencia vs intermitentes puros)

INFRAESTRUCTURA
  ☐ Model registry: Loop 5 como champion, Seasonal Naïve como baseline
  ☐ Weekly batch inference pipeline con Q10/Q50/Q90
  ☐ Alertas de inventario usando Q90 como cobertura conservadora
```

---

*AI-DLC Loop 5 — Log-Transform + Optuna + Isotonic Calibration + Quantile Intervals — Complete.*
