# Inventory-Aware Forecasting
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** INV-AWARE-ITER3-001

---

## 1. El Problema Fundamental

Un modelo puramente estadístico predice sell-in basado solo en patrones históricos de demanda. Pero el sell-in real en consumer electronics no solo depende de la demanda del consumidor — también depende del **nivel de inventario en el canal**.

```
CANAL CON SOBRESTOCK (DOS = 90d):
  Demanda real:     500 unidades/semana
  Modelo naive:     Predice 500 unidades de sell-in
  Realidad:         El canal NO pedirá replenishment hasta que baje el stock
  Forecast correcto: ~50 unidades (replenishment mínimo de emergencia)

CANAL CON STOCK BAJO (DOS = 7d):
  Demanda real:     500 unidades/semana
  Modelo naive:     Predice 500 unidades de sell-in
  Realidad:         El canal hará un pedido urgente MAYOR para reconstruir stock
  Forecast correcto: ~1,500 unidades (demanda + stock rebuild)
```

---

## 2. Arquitectura del Módulo

```
INVENTORY-AWARE FORECASTING — CAPAS

Capa 1: SEÑALES DE INVENTARIO (Features)
  inv_lag_1, days_of_supply, inv_delta_1,
  sell_through_rate_4w, inv_overstock_flag,
  inv_stockout_flag, inv_momentum
         │
         ▼
Capa 2: ML MODEL (LightGBM aprende la relación)
  El modelo aprende implícitamente que DOS > 60 suprime
  el sell-in en las siguientes semanas
         │
         ▼
Capa 3: POST-PROCESSING ADJUSTMENT (Reglas explícitas)
  Ajusta el forecast base con lógica de negocio
  determinística para casos extremos
         │
         ▼
Capa 4: INVENTORY PROJECTION
  Dado el forecast de sell-in y sales, proyecta
  el inventario futuro para validar coherencia
```

---

## 3. Features de Inventario (Capa 1)

### Regla Crítica: Semántica Temporal Correcta

```
⚠️  PROHIBIDO:  usar channel_inv(t) para predecir sell_in(t)
✅  CORRECTO:   usar inv_lag_1 = channel_inv(t-1) para predecir sell_in(t)

Razón: channel_inv(t) = channel_inv(t-1) + sell_in(t) - sales(t)
       channel_inv(t) CONTIENE sell_in(t) → leakage circular
       Al momento de predecir sell_in(t), channel_inv(t) no existe aún.
```

```python
# Definición correcta de features de inventario
INVENTORY_FEATURES = {

    # NIVEL 1 — Estado actual (usando t-1)
    "inv_lag_1":          "Inventario al cierre de la semana anterior (snapshot)",
    "inv_lag_2":          "Inventario hace 2 semanas",
    "inv_lag_4":          "Inventario hace 4 semanas (tendencia mensual)",

    # NIVEL 2 — Ratios derivados (todos usan inv_lag_1)
    "days_of_supply":     "inv_lag_1 / max(sales_ma4/7, 0.01) — días de cobertura",
    "inv_vs_sales":       "inv_lag_1 / max(sales_lag_1, 0.01) — stock vs ventas recientes",
    "weeks_of_supply":    "inv_lag_1 / max(sales_ma4, 0.01) — semanas de cobertura",

    # NIVEL 3 — Velocidad y momentum
    "inv_delta_1":        "inv_lag_1 - inv_lag_2 — velocidad de cambio (+ acumula)",
    "inv_delta_4":        "inv_lag_1 - inv_lag_4 — velocidad mensual",
    "inv_momentum":       "inv_ma4 - inv_ma13 — momentum (+ tendencia a crecer)",

    # NIVEL 4 — Indicadores binarios
    "inv_overstock_flag": "1 si days_of_supply > 60",
    "inv_stockout_flag":  "1 si days_of_supply < 14",
    "inv_healthy_flag":   "1 si 14 <= days_of_supply <= 45",

    # NIVEL 5 — Balance residual (señal de ajustes)
    "inv_balance_residual": "inv(t-1) - [inv(t-2) + sell_in(t-1) - sales(t-1)]",
}
```

---

## 4. Post-Processing Adjustment (Capa 3)

```python
# src/forecasting/inventory_adjustment.py

import numpy as np
import polars as pl

class InventoryAwareAdjuster:
    """
    Applies inventory-state-based adjustments to ML forecast.

    Philosophy:
    - ML model handles normal operating conditions via features
    - This layer handles EXTREME conditions with explicit rules
    - Rules are conservative: never zero out a forecast entirely
    """

    def __init__(
        self,
        dos_critical_overstock:  float = 90.0,
        dos_warning_overstock:   float = 60.0,
        dos_warning_stockout:    float = 14.0,
        dos_critical_stockout:   float = 7.0,
        min_suppression_floor:   float = 0.05,  # Never suppress below 5% of base
        max_boost_cap:           float = 3.0,   # Never boost above 3× base
    ):
        self.dos_crit_over  = dos_critical_overstock
        self.dos_warn_over  = dos_warning_overstock
        self.dos_warn_stock = dos_warning_stockout
        self.dos_crit_stock = dos_critical_stockout
        self.min_floor      = min_suppression_floor
        self.max_boost      = max_boost_cap

    def adjust(
        self,
        base_forecast: float,
        days_of_supply: float,
        inv_momentum: float = 0.0,
        sell_through_rate: float = 0.9,
    ) -> dict:
        """
        Adjust a single week's forecast given inventory state.

        Returns dict with adjusted_forecast and adjustment_reason.
        """
        if days_of_supply is None or np.isnan(days_of_supply):
            return {"adjusted": base_forecast, "reason": "no_inv_data",
                    "multiplier": 1.0}

        # ── OVERSTOCK SUPPRESSION ──────────────────────────────────
        if days_of_supply >= self.dos_crit_over:
            # Critical overstock: near-zero replenishment
            # Channel won't order until stock drops significantly
            suppression = max(
                self.min_floor,
                1.0 - (days_of_supply - self.dos_crit_over) / 60.0
            )
            # Further suppress if inventory is still growing
            if inv_momentum > 0:
                suppression *= 0.7
            return {
                "adjusted": base_forecast * suppression,
                "reason": "critical_overstock_suppression",
                "multiplier": suppression
            }

        elif days_of_supply >= self.dos_warn_over:
            # High overstock: moderate suppression
            suppression = 1.0 - 0.4 * (
                (days_of_supply - self.dos_warn_over) /
                (self.dos_crit_over - self.dos_warn_over)
            )
            return {
                "adjusted": base_forecast * suppression,
                "reason": "warning_overstock_suppression",
                "multiplier": suppression
            }

        # ── STOCKOUT BOOST ────────────────────────────────────────
        elif days_of_supply <= self.dos_crit_stock:
            # Critical stockout: emergency reorder
            # Boost = 2× base + weeks needed to restore to 30-day target
            target_inv = base_forecast * 30.0 / 7.0  # 30-day target
            current_inv = days_of_supply * base_forecast / 7.0
            rebuild     = max(0, target_inv - current_inv)
            boost_factor = min(
                self.max_boost,
                1.0 + rebuild / max(base_forecast, 1.0)
            )
            return {
                "adjusted": base_forecast * boost_factor,
                "reason": "critical_stockout_boost",
                "multiplier": boost_factor
            }

        elif days_of_supply <= self.dos_warn_stock:
            # Low stock: mild boost
            boost_factor = 1.0 + 0.3 * (
                (self.dos_warn_stock - days_of_supply) / self.dos_warn_stock
            )
            return {
                "adjusted": base_forecast * boost_factor,
                "reason": "warning_stockout_boost",
                "multiplier": boost_factor
            }

        # ── HEALTHY ZONE ──────────────────────────────────────────
        return {"adjusted": base_forecast, "reason": "healthy",
                "multiplier": 1.0}


    def adjust_series(
        self,
        forecast_series: np.ndarray,
        dos_series: np.ndarray,
        inv_momentum_series: np.ndarray,
        sell_through_series: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Apply adjustment to a full forecast horizon.
        Returns (adjusted_forecasts, reasons).
        """
        adjusted = np.zeros_like(forecast_series)
        reasons  = []

        for i in range(len(forecast_series)):
            dos = dos_series[i] if i < len(dos_series) else None
            mom = inv_momentum_series[i] if i < len(inv_momentum_series) else 0.0
            str_ = sell_through_series[i] if i < len(sell_through_series) else 0.9

            result = self.adjust(forecast_series[i], dos, mom, str_)
            adjusted[i] = result["adjusted"]
            reasons.append(result["reason"])

        return adjusted, reasons
```

---

## 5. Inventory Projection (Capa 4)

Después de generar el sell-in forecast, proyectamos el inventario futuro para validar coherencia:

```python
def project_inventory(
    current_inv:      float,
    sell_in_forecast: np.ndarray,   # semanas H+1 ... H+n
    sales_forecast:   np.ndarray,   # semanas H+1 ... H+n
) -> np.ndarray:
    """
    Project future channel inventory given sell-in and sales forecasts.

    inv(t) = inv(t-1) + sell_in(t) - sales(t)

    NOTE: This is a projection aid, not a target. Real inventory
    will differ due to returns, write-offs, and system adjustments.
    """
    n = len(sell_in_forecast)
    projected_inv = np.zeros(n)
    inv = current_inv

    for t in range(n):
        inv = inv + sell_in_forecast[t] - sales_forecast[t]
        inv = max(0.0, inv)  # Physical constraint: inventory >= 0
        projected_inv[t] = inv

    return projected_inv


def validate_inventory_projection(
    projected_inv: np.ndarray,
    dos_threshold_overstock: float = 90.0,
    sales_forecast_weekly: np.ndarray = None,
) -> dict:
    """
    Check if the projected inventory stays in a reasonable range.
    Flags weeks where projected DOS would exceed thresholds.
    """
    if sales_forecast_weekly is None or sales_forecast_weekly.mean() == 0:
        return {"warning": "Cannot compute projected DOS without sales forecast"}

    projected_dos = projected_inv / (sales_forecast_weekly / 7.0).clip(min=0.01)

    return {
        "weeks_overstock":     int((projected_dos > dos_threshold_overstock).sum()),
        "max_projected_dos":   float(projected_dos.max()),
        "min_projected_inv":   float(projected_inv.min()),
        "ends_at_inv":         float(projected_inv[-1]),
        "ends_at_dos":         float(projected_dos[-1]),
    }
```

---

## 6. Validación: Inventory Balance Residual como Signal de Calidad

```python
def compute_inv_balance_quality(
    silver_df: pl.DataFrame,
    tolerance: float = 5.0
) -> pl.DataFrame:
    """
    Compute inventory balance residual per series.
    Large residuals indicate unreported adjustments.

    residual = inv(t) - [inv(t-1) + sell_in(t) - sales(t)]
    """
    pivot = silver_df.pivot(
        index=["channel","material","yearweek","date"],
        on="category", values="value"
    ).rename({
        "Sell-in": "sell_in",
        "Cust. Sales": "cust_sales",
        "Channel Inv.": "channel_inv"
    }).sort(["channel","material","date"])

    pivot = pivot.with_columns([
        pl.col("channel_inv").shift(1).over(["channel","material"])
          .alias("inv_prev")
    ])

    pivot = pivot.with_columns([
        (pl.col("channel_inv") -
         (pl.col("inv_prev").fill_null(pl.col("channel_inv")) +
          pl.col("sell_in").fill_null(0) -
          pl.col("cust_sales").fill_null(0))
        ).alias("balance_residual"),
        (pl.col("balance_residual").abs() > tolerance).alias("has_adjustment")
    ])

    summary = pivot.group_by(["channel","material"]).agg([
        pl.col("balance_residual").abs().mean().alias("mean_abs_residual"),
        pl.col("has_adjustment").mean().alias("pct_weeks_with_adjustments"),
        pl.col("balance_residual").abs().max().alias("max_residual"),
    ])

    return summary
```

---

## 7. Reglas de Confiabilidad del Inventario

| Condición | Confiabilidad | Acción |
|---|---|---|
| `mean_abs_residual < 5` | Alta | Usar inventario normalmente como feature |
| `mean_abs_residual 5–50` | Media | Usar con cautela; peso reducido en modelo |
| `mean_abs_residual > 50` | Baja | Flaggear serie; no usar DOS en adjustment |
| `pct_weeks_adjustments > 30%` | Baja | Serie tiene ajustes frecuentes no capturados |
| `channel_inv < 0` (pre-fix) | Cero | Error de sistema; usar silver corregido |

---

*AI-DLC Traceability ID: INV-AWARE-ITER3-001 | Version: 3.0*
