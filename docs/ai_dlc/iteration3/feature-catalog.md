# Feature Catalog — Catálogo Completo
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** FEATCAT-ITER3-001

---

## Convención del Catálogo

| Campo | Descripción |
|---|---|
| **ID** | Identificador único F-XXX |
| **Nombre** | Nombre exacto en feature_store.parquet |
| **Grupo** | Categoría del feature |
| **Tipo** | float / int / bool / categorical |
| **Fórmula** | Definición computacional exacta |
| **Rango válido** | [min, max] esperado |
| **Status** | PRODUCCIÓN / STAGING / DEPRECADO |
| **Leakage risk** | NONE / LOW / MEDIUM / HIGH |
| **Owner** | Equipo responsable |

---

## GRUPO 0 — Identificadores (No son model inputs)

| ID | Nombre | Tipo | Descripción | Status |
|---|---|---|---|---|
| F-000 | channel | str | Código de canal (CUSTOMER_N) | PRODUCCIÓN |
| F-001 | material | str | Descripción SKU | PRODUCCIÓN |
| F-002 | yearweek | str | YYYYWW | PRODUCCIÓN |
| F-003 | date | date | Lunes de la semana ISO | PRODUCCIÓN |
| F-004 | year | int | Año calendario | PRODUCCIÓN |
| F-005 | week | int | Semana ISO (1-52) | PRODUCCIÓN |
| F-006 | product_family | str | Familia parseada de material[0] | PRODUCCIÓN |
| F-007 | model_code | str | Código modelo de material[1] | PRODUCCIÓN |
| F-008 | sku_size | str | Tamaño/variante de material[2] | PRODUCCIÓN |
| F-009 | segment | str | regular/intermittent/rare/dead | PRODUCCIÓN |
| F-010 | quadrant | str | smooth/erratic/intermittent/lumpy | PRODUCCIÓN |
| F-011 | is_censored | bool | True si value==999 en 2025 | PRODUCCIÓN |

---

## GRUPO 1 — Variables Target (No como model inputs)

| ID | Nombre | Fórmula | Rango | Leakage |
|---|---|---|---|---|
| F-020 | sell_in | valor de category='Sell-in' | [-10K, 50K] | TARGET |
| F-021 | cust_sales | valor de category='Cust. Sales' | [-5K, 20K] | FORBIDDEN |
| F-022 | channel_inv | valor de category='Channel Inv.' | [0, 100K] | FORBIDDEN (t=0) |

---

## GRUPO 2 — Lags de Sell-in

| ID | Nombre | Fórmula | Rango | Leakage |
|---|---|---|---|---|
| F-030 | sell_in_lag_1 | sell_in.shift(1).over(key) | [-10K, 50K] | NONE |
| F-031 | sell_in_lag_2 | sell_in.shift(2).over(key) | [-10K, 50K] | NONE |
| F-032 | sell_in_lag_4 | sell_in.shift(4).over(key) | [-10K, 50K] | NONE |
| F-033 | sell_in_lag_8 | sell_in.shift(8).over(key) | [-10K, 50K] | NONE |
| F-034 | sell_in_lag_13 | sell_in.shift(13).over(key) | [-10K, 50K] | NONE |
| F-035 | sell_in_lag_26 | sell_in.shift(26).over(key) | [-10K, 50K] | NONE |
| F-036 | sell_in_lag_52 | sell_in.shift(52).over(key) | [-10K, 50K] | NONE |
| F-037 | sell_in_lag_51 | sell_in.shift(51).over(key) | [-10K, 50K] | NONE |
| F-038 | sell_in_lag_53 | sell_in.shift(53).over(key) | [-10K, 50K] | NONE |

---

## GRUPO 3 — Lags de Cust. Sales (Señal de Demand Pull)

| ID | Nombre | Fórmula | Rango | Leakage |
|---|---|---|---|---|
| F-040 | sales_lag_1 | cust_sales.shift(1).over(key) | [-5K, 20K] | NONE |
| F-041 | sales_lag_2 | cust_sales.shift(2).over(key) | [-5K, 20K] | NONE |
| F-042 | sales_lag_4 | cust_sales.shift(4).over(key) | [-5K, 20K] | NONE |
| F-043 | sales_lag_13 | cust_sales.shift(13).over(key) | [-5K, 20K] | NONE |
| F-044 | sales_lag_52 | cust_sales.shift(52).over(key) | [-5K, 20K] | NONE |

---

## GRUPO 4 — Lags de Inventario (⚠️ SOLO LAGGED)

| ID | Nombre | Fórmula | Rango | Leakage |
|---|---|---|---|---|
| F-050 | inv_lag_1 | channel_inv.shift(1).over(key) | [0, 100K] | NONE |
| F-051 | inv_lag_2 | channel_inv.shift(2).over(key) | [0, 100K] | NONE |
| F-052 | inv_lag_4 | channel_inv.shift(4).over(key) | [0, 100K] | NONE |

---

## GRUPO 5 — Rolling Features de Sell-in

| ID | Nombre | Fórmula | Rango | Notas |
|---|---|---|---|---|
| F-060 | sell_in_ma4 | sell_in.shift(1).rolling_mean(4).over(key) | [0, 50K] | Short-term trend |
| F-061 | sell_in_ma13 | sell_in.shift(1).rolling_mean(13).over(key) | [0, 50K] | Quarterly trend |
| F-062 | sell_in_ma26 | sell_in.shift(1).rolling_mean(26).over(key) | [0, 50K] | Semi-annual trend |
| F-063 | sell_in_std4 | sell_in.shift(1).rolling_std(4).over(key) | [0, 20K] | Short volatility |
| F-064 | sell_in_std13 | sell_in.shift(1).rolling_std(13).over(key) | [0, 20K] | Medium volatility |

---

## GRUPO 6 — Rolling Features de Sales

| ID | Nombre | Fórmula | Rango |
|---|---|---|---|
| F-070 | sales_ma4 | cust_sales.shift(1).rolling_mean(4).over(key) | [0, 20K] |
| F-071 | sales_ma13 | cust_sales.shift(1).rolling_mean(13).over(key) | [0, 20K] |
| F-072 | sales_ma26 | cust_sales.shift(1).rolling_mean(26).over(key) | [0, 20K] |
| F-073 | sales_std4 | cust_sales.shift(1).rolling_std(4).over(key) | [0, 10K] |

---

## GRUPO 7 — Rolling Features de Inventario

| ID | Nombre | Fórmula | Rango |
|---|---|---|---|
| F-080 | inv_ma4 | channel_inv.shift(1).rolling_mean(4).over(key) | [0, 100K] |
| F-081 | inv_ma13 | channel_inv.shift(1).rolling_mean(13).over(key) | [0, 100K] |

---

## GRUPO 8 — Inventory Ratios

| ID | Nombre | Fórmula | Rango | Semántica |
|---|---|---|---|---|
| F-090 | days_of_supply | inv_lag_1 / max(sales_ma4/7, 0.01) | [0, 1000] | Días de stock |
| F-091 | weeks_of_supply | inv_lag_1 / max(sales_ma4, 0.01) | [0, 142] | Semanas de stock |
| F-092 | sell_through_rate_4w | sales_ma4 / max(sell_in_ma4, 0.01) | [0, 20] | Eficiencia canal 4w |
| F-093 | sell_through_rate_13w | sales_ma13 / max(sell_in_ma13, 0.01) | [0, 20] | Eficiencia canal 13w |
| F-094 | inv_delta_1 | inv_lag_1 - inv_lag_2 | [-50K, 50K] | Velocidad inv 1w |
| F-095 | inv_delta_4 | inv_lag_1 - inv_lag_4 | [-50K, 50K] | Velocidad inv 4w |
| F-096 | inv_momentum | inv_ma4 - inv_ma13 | [-50K, 50K] | Acumulación vs tendencia |
| F-097 | replenishment_gap | sell_in_lag_1 - sales_lag_1 | [-20K, 20K] | Balance neto semana |
| F-098 | inv_vs_trend | inv_lag_1 / max(inv_ma13, 0.01) | [0, 10] | Stock vs tendencia |
| F-099 | inv_overstock_flag | days_of_supply > 60 | {0, 1} | Flag sobrestock |
| F-100 | inv_stockout_flag | days_of_supply < 14 | {0, 1} | Flag stock bajo |

---

## GRUPO 9 — Features Temporales

| ID | Nombre | Fórmula | Rango |
|---|---|---|---|
| F-110 | week_sin | sin(2π × week/52) | [-1, 1] |
| F-111 | week_cos | cos(2π × week/52) | [-1, 1] |
| F-112 | month_sin | sin(2π × month/12) | [-1, 1] |
| F-113 | month_cos | cos(2π × month/12) | [-1, 1] |
| F-114 | month | date.month | [1, 12] |
| F-115 | quarter | date.quarter | [1, 4] |
| F-116 | is_q4 | quarter == 4 | {0, 1} |
| F-117 | year_normalized | (year - 2023) / 2.0 | [0, 1.5] |
| F-118 | weeks_since_epoch | semanas desde W01-2023 | [0, 200] |

---

## GRUPO 10 — Calendario Colombia

| ID | Nombre | Semanas | Status |
|---|---|---|---|
| F-120 | is_mothers_day_week | W18-W19 | PRODUCCIÓN |
| F-121 | is_black_friday_week | W47-W48 | PRODUCCIÓN |
| F-122 | is_christmas_week | W50-W52 | PRODUCCIÓN |
| F-123 | is_back_to_school | W30-W34 | PRODUCCIÓN |
| F-124 | is_new_year_restock | W01-W03 | PRODUCCIÓN |
| F-125 | is_dia_sin_iva | TBD | ⚠️ PENDIENTE input comercial |
| F-126 | weeks_to_black_friday | max(0, W48 - week) | PRODUCCIÓN |
| F-127 | weeks_after_black_friday | max(0, week - W48) | PRODUCCIÓN |

---

## GRUPO 11 — YoY Seasonal

| ID | Nombre | Fórmula | Rango |
|---|---|---|---|
| F-130 | yoy_sell_in_ratio | sell_in_lag_52 / max(sell_in_ma26.shift(26), 0.01) | [0, 10] |
| F-131 | yoy_sales_ratio | sales_lag_52 / max(sales_ma26.shift(26), 0.01) | [0, 10] |

---

## GRUPO 12 — Zero-Inflation (Demanda Intermitente)

| ID | Nombre | Fórmula | Rango |
|---|---|---|---|
| F-140 | prob_nonzero_4w | (sell_in.shift(1)>0).rolling_mean(4).over(key) | [0, 1] |
| F-141 | prob_nonzero_13w | (sell_in.shift(1)>0).rolling_mean(13).over(key) | [0, 1] |
| F-142 | prob_nonzero_52w | (sell_in.shift(1)>0).rolling_mean(52).over(key) | [0, 1] |

---

## GRUPO 13 — SKU Lifecycle

| ID | Nombre | Fórmula | Rango |
|---|---|---|---|
| F-150 | sku_age_weeks | (date - first_active_date) / 7 | [0, 200] |
| F-151 | is_new_sku | sku_age_weeks <= 8 | {0, 1} |
| F-152 | is_mature_sku | sku_age_weeks > 52 | {0, 1} |
| F-153 | cumulative_sell_in | sell_in.cum_sum().over(key) | [0, ∞] |

---

## Resumen del Catálogo

| Grupo | Features | Status |
|---|---|---|
| G0: Identificadores | 12 | Producción |
| G1: Targets | 3 | Producción (no model inputs) |
| G2: Lags Sell-in | 9 | Producción |
| G3: Lags Sales | 5 | Producción |
| G4: Lags Inventario | 3 | Producción |
| G5: Rolling Sell-in | 5 | Producción |
| G6: Rolling Sales | 4 | Producción |
| G7: Rolling Inventario | 2 | Producción |
| G8: Inventory Ratios | 11 | Producción |
| G9: Temporales | 9 | Producción |
| G10: Calendario | 8 | 7 Prod + 1 Pendiente |
| G11: YoY Seasonal | 2 | Producción |
| G12: Zero-Inflation | 3 | Producción |
| G13: Lifecycle | 4 | Producción |
| **TOTAL** | **80 features** | **79 Prod + 1 Pendiente** |

---

*AI-DLC Traceability ID: FEATCAT-ITER3-001 | Version: 3.0*
