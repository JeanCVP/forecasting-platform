# Semantic Governance
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** SEMGOV-ITER3-001

---

## 1. Problema que Resuelve

Sin gobierno semántico, el mismo KPI puede tener 3 definiciones distintas:
- Dashboard ejecutivo: `sell_through = sales_ytd / sell_in_ytd`
- Dashboard de canal: `sell_through = sales_ma4 / sell_in_ma4`
- Reporte de categoría: `sell_through = sales_last_week / sell_in_last_week`

Resultado: el CFO ve 72%, el Sales Manager ve 81%, el Category Manager ve 68%. Nadie confía en ninguno.

**El gobierno semántico garantiza una única definición por métrica, usada consistentemente en todos los consumidores.**

---

## 2. Modelo de Gobierno

```
JERARQUÍA DE DEFINICIONES
──────────────────────────────────────────────────────────────────
Nivel 1: ENTIDADES       — objetos de negocio (Channel, SKU, Week)
Nivel 2: MÉTRICAS BASE   — agregaciones directas de datos crudos
Nivel 3: MÉTRICAS DERIVADAS — cálculos sobre métricas base
Nivel 4: KPIs            — métricas con thresholds y dimensiones
──────────────────────────────────────────────────────────────────
Todo consumidor (dashboard, modelo, reporte) accede SOLO desde
Nivel 3/4. Nunca desde los datos crudos directamente.
──────────────────────────────────────────────────────────────────
```

---

## 3. Registro de Entidades Semánticas

### Entidad: CHANNEL
```yaml
entity: CHANNEL
description: "Socio comercial que recibe productos del fabricante y los vende a consumidores finales"
identifier: channel (string, formato CUSTOMER_N)
attributes:
  - channel_id:       integer, label-encoded
  - channel_name:     string, anonimizado
  - first_active_week: date, primera semana con Sell-in > 0
  - last_active_week:  date, última semana con cualquier actividad
  - is_active_current: bool, active_weeks en últimas 4 semanas > 0
  - channel_type:      string, PENDIENTE (requiere input business)
  - channel_tier:      string, PENDIENTE (A/B/C por volumen)
owner: Commercial Domain
```

### Entidad: MATERIAL (SKU)
```yaml
entity: MATERIAL
description: "Producto vendible específico con sus atributos de variante"
identifier: material (string, descripción compuesta)
attributes:
  - material_id:       integer, label-encoded
  - product_family:    string, parseable desde material[0]
  - model_code:        string, parseable desde material[1]
  - sku_size:          string, parseable desde material[2]
  - market:            string, siempre 'COLOMBIA' en este dataset
  - sku_age_weeks:     integer, semanas desde primer Sell-in > 0
  - lifecycle_status:  enum(new, growing, mature, declining, discontinued)
owner: Product Domain
```

### Entidad: WEEK
```yaml
entity: WEEK
description: "Semana ISO del calendario comercial"
identifier: yearweek (string, YYYYWW)
attributes:
  - date:              date, lunes de esa semana ISO
  - year:              integer
  - week_num:          integer, 1-52
  - month:             integer, 1-12
  - quarter:           integer, 1-4
  - is_q4:             bool
  - is_mothers_day_week: bool, W18-W19
  - is_black_friday_week: bool, W47-W48
  - is_christmas_season: bool, W50-W52
  - is_back_to_school:   bool, W30-W34
  - is_dia_sin_iva:      bool, PENDIENTE input comercial
owner: ML Platform Domain
```

---

## 4. Registro de Métricas — Nivel 2 (Base)

### METRIC-B01: sell_in_units
```yaml
name: sell_in_units
description: "Unidades enviadas desde fabricante al canal en una semana"
business_definition: "Shipment de producto del fabricante al socio de canal. Evento de reconocimiento de ingreso para el fabricante."
type: FLOW (delta semanal, no acumulable como stock)
formula: SUM(value) WHERE category = 'Sell-in' AND yearweek = W
grain: Channel × Material × Week
unit: Unidades
sign_convention: POSITIVE = envío normal; NEGATIVE = devolución al fabricante
null_meaning: "Semana sin transacción (distinto de cero: sin actividad conocida)"
valid_range: [-10000, 50000]
censored_flag: is_censored (True cuando value==999 en 2025)
source_table: silver.timeseries_clean
source_category: 'Sell-in'
owner: Supply Chain Domain
```

### METRIC-B02: cust_sales_units
```yaml
name: cust_sales_units
description: "Unidades vendidas desde el canal al consumidor final en una semana"
business_definition: "Venta en punto de venta (POS). Representa demanda real del mercado. Señal de sell-out."
type: FLOW (delta semanal)
formula: SUM(value) WHERE category = 'Cust. Sales' AND yearweek = W
grain: Channel × Material × Week
unit: Unidades
sign_convention: POSITIVE = venta; NEGATIVE = devolución de consumidor
null_meaning: "Sin ventas registradas esa semana"
valid_range: [-5000, 20000]
source_table: silver.timeseries_clean
source_category: 'Cust. Sales'
owner: Commercial Domain
```

### METRIC-B03: channel_inv_units
```yaml
name: channel_inv_units
description: "Unidades en inventario del canal al final de una semana"
business_definition: "Stock físico que el canal tiene disponible para venta. Es una fotografía (snapshot) al cierre de semana."
type: STOCK (snapshot, nivel absoluto — NO acumulable entre semanas)
formula: SUM(value) WHERE category = 'Channel Inv.' AND yearweek = W
grain: Channel × Material × Week
unit: Unidades
sign_convention: ALWAYS POSITIVE (negativo = error de datos)
valid_range: [0, 100000]
identity_check: "channel_inv(t) ≈ channel_inv(t-1) + sell_in(t) - cust_sales(t)"
source_table: silver.timeseries_clean
source_category: 'Channel Inv.'
owner: Supply Chain Domain
CRITICAL_NOTE: "Esta métrica es un STOCK. Nunca sumar channel_inv entre semanas consecutivas. Nunca usar channel_inv(t) como feature para predecir sell_in(t) — es leakage temporal."
```

---

## 5. Registro de Métricas — Nivel 3 (Derivadas)

### METRIC-D01: days_of_supply
```yaml
name: days_of_supply
alias: DOS
description: "Días de cobertura de stock dado el ritmo actual de ventas"
formula: "channel_inv_units(t-1) / MAX(avg_cust_sales_4w / 7, 0.01)"
components:
  numerator: channel_inv_units con lag=1 semana (CRÍTICO: no usar t=0 — temporal leakage)
  denominator: promedio de cust_sales_units de las últimas 4 semanas, dividido 7 (escala a días)
  floor: 0.01 para evitar división por cero
unit: Días
interpretation: "Cuántos días puede el canal seguir vendiendo sin recibir nuevo stock"
thresholds:
  critical_stockout: "< 7 días"
  warning_stockout: "7–14 días"
  healthy: "14–45 días"
  warning_overstock: "45–60 días"
  critical_overstock: "> 60 días"
temporal_semantics: "Usa inv_lag_1, no inv corriente — preserva correcta semántica temporal"
owner: Supply Chain Domain
```

### METRIC-D02: sell_through_rate
```yaml
name: sell_through_rate
description: "Proporción del Sell-in que llega al consumidor final"
formula: "avg_cust_sales_4w / MAX(avg_sell_in_4w, 0.01)"
components:
  numerator: media móvil 4 semanas de cust_sales_units (usando shift(1) para no incluir semana actual)
  denominator: media móvil 4 semanas de sell_in_units (shift(1) ídem)
unit: Ratio (0.0 a ∞; > 1.0 = canal depletando stock)
thresholds:
  healthy: "0.75–1.10"
  slow: "< 0.60 (stock acumulándose)"
  depleting: "> 1.20 (riesgo stockout)"
owner: Commercial Domain
```

### METRIC-D03: replenishment_gap
```yaml
name: replenishment_gap
description: "Diferencia neta entre lo enviado y lo vendido en la semana"
formula: "sell_in_units(t-1) - cust_sales_units(t-1)"
unit: Unidades/semana (+ = inventario creciendo, - = inventario disminuyendo)
related_to: "inv_delta_1 = channel_inv(t-1) - channel_inv(t-2) ≈ replenishment_gap si sin ajustes"
owner: Supply Chain Domain
```

### METRIC-D04: inventory_balance_residual
```yaml
name: inventory_balance_residual
description: "Diferencia entre el inventario observado y el teórico por balance de masa"
formula: "channel_inv(t) - [channel_inv(t-1) + sell_in(t) - cust_sales(t)]"
unit: Unidades
interpretation: "Residuo ≠ 0 indica devoluciones, mermas, ajustes de inventario no capturados"
use: "Feature de calidad de datos y señal de actividad de ajuste"
owner: ML Platform Domain
```

---

## 6. Capa Semántica — Implementación

```python
# src/semantic/metrics.py
# FUENTE DE VERDAD para todos los KPIs
# Ningún dashboard debe calcular métricas fuera de este módulo

import polars as pl

class MetricEngine:
    """
    Computes all semantic metrics from the Gold feature store.
    Every dashboard and notebook MUST use this class.
    """
    
    def __init__(self, feature_store_path: str = "data/gold/feature_store.parquet"):
        self.fs = pl.read_parquet(feature_store_path)
    
    # ── NIVEL 2: MÉTRICAS BASE ────────────────────────────────────
    
    def sell_in(self, channel=None, material=None, yearweek=None) -> pl.DataFrame:
        """Sell-in units. Filters optional."""
        df = self.fs
        if channel: df = df.filter(pl.col("channel") == channel)
        if material: df = df.filter(pl.col("material") == material)
        if yearweek: df = df.filter(pl.col("yearweek") == yearweek)
        return df.select(["channel","material","yearweek","sell_in"])
    
    def cust_sales(self, **filters) -> pl.DataFrame:
        return self._filtered("cust_sales", **filters)
    
    def channel_inv(self, **filters) -> pl.DataFrame:
        """Channel inventory. NOTE: Stock metric — never sum across time."""
        return self._filtered("channel_inv", **filters)
    
    # ── NIVEL 3: MÉTRICAS DERIVADAS ──────────────────────────────
    
    def days_of_supply(self, current_week: str) -> pl.DataFrame:
        """
        DOS = inv_lag_1 / (sales_ma4 / 7)
        NOTA: Usa inv_lag_1, no channel_inv corriente.
        """
        return self.fs.filter(pl.col("yearweek") == current_week).select([
            "channel", "material", "product_family", "yearweek",
            pl.col("days_of_supply"),
            pl.when(pl.col("days_of_supply") < 7).then(pl.lit("critical_stockout"))
              .when(pl.col("days_of_supply") < 14).then(pl.lit("warning_stockout"))
              .when(pl.col("days_of_supply") < 45).then(pl.lit("healthy"))
              .when(pl.col("days_of_supply") < 60).then(pl.lit("warning_overstock"))
              .otherwise(pl.lit("critical_overstock"))
              .alias("dos_status")
        ])
    
    def sell_through_rate(self, current_week: str) -> pl.DataFrame:
        """4-week rolling sell-through rate."""
        return self.fs.filter(pl.col("yearweek") == current_week).select([
            "channel", "material", "product_family",
            pl.col("sell_through_rate_4w")
        ])
    
    def portfolio_sell_through_ytd(self, ytd_weeks: list[str]) -> float:
        """Total sales / total sell-in for YTD period."""
        df = self.fs.filter(pl.col("yearweek").is_in(ytd_weeks))
        return float(df["cust_sales"].sum() / max(df["sell_in"].sum(), 1)) * 100
    
    # ── NIVEL 4: KPIs CON ALERTAS ────────────────────────────────
    
    def inventory_health_summary(self, current_week: str) -> dict:
        """Executive inventory health KPIs with alert levels."""
        dos_df = self.days_of_supply(current_week)
        total = len(dos_df)
        
        return {
            "total_active_series":    total,
            "critical_stockout_pct":  dos_df.filter(pl.col("dos_status")=="critical_stockout").height / total * 100,
            "warning_stockout_pct":   dos_df.filter(pl.col("dos_status")=="warning_stockout").height / total * 100,
            "healthy_pct":            dos_df.filter(pl.col("dos_status")=="healthy").height / total * 100,
            "warning_overstock_pct":  dos_df.filter(pl.col("dos_status")=="warning_overstock").height / total * 100,
            "critical_overstock_pct": dos_df.filter(pl.col("dos_status")=="critical_overstock").height / total * 100,
            "avg_dos":                float(dos_df["days_of_supply"].mean()),
        }
    
    # ── HELPERS ──────────────────────────────────────────────────
    
    def _filtered(self, col: str, channel=None, material=None, yearweek=None) -> pl.DataFrame:
        df = self.fs
        if channel: df = df.filter(pl.col("channel") == channel)
        if material: df = df.filter(pl.col("material") == material)
        if yearweek: df = df.filter(pl.col("yearweek") == yearweek)
        return df.select(["channel","material","yearweek", col])
```

---

## 7. Reglas de Gobierno

| Regla | Descripción | Enforcement |
|---|---|---|
| SG-01 | Toda métrica debe estar registrada en este documento antes de usarse en producción | Code review |
| SG-02 | Ningún dashboard calcula KPIs inline — usa `MetricEngine` | Linting test |
| SG-03 | `channel_inv(t)` nunca como feature para predecir `sell_in(t)` | CI leakage test |
| SG-04 | `channel_inv` no se suma entre semanas (es stock, no flujo) | Documentación |
| SG-05 | Cambios a fórmulas de métricas requieren DEC-XXX en decision-log | PRs |
| SG-06 | Nuevas métricas requieren entry en este registro antes del merge | PRs |

---

*AI-DLC Traceability ID: SEMGOV-ITER3-001 | Version: 3.0*
