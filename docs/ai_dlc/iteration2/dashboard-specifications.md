# Dashboard Specifications
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 2
**Last Updated:** 2026-05-21

---

## 1. Dashboard Portfolio

| ID | Name | Tool | Audience | Refresh | Priority |
|---|---|---|---|---|---|
| DB-01 | Executive Supply Chain Command Center | Streamlit / Power BI | C-Suite, Commercial Director | Weekly Mon | P0 |
| DB-02 | Demand Planning Workbench | Streamlit | Demand Planners | Weekly Mon | P0 |
| DB-03 | Channel Health Monitor | Streamlit / Power BI | Sales Managers, KAMs | Weekly Mon | P1 |
| DB-04 | SKU Performance Tracker | Streamlit / Power BI | Category Managers | Weekly Mon | P1 |
| DB-05 | Forecast Accuracy Dashboard | Streamlit | ML / Analytics Team | Weekly Mon | P1 |
| DB-06 | Inventory Risk Radar | Streamlit / Power BI | Supply Chain Managers | Weekly Mon | P2 |

**Data source for all dashboards:** `data/gold/feature_store.parquet` + `data/gold/forecast_output.parquet`

---

## 2. DB-01 — Executive Supply Chain Command Center

### Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  TITLE BAR: "Supply Chain Command Center — W21 2025"   [filters ▼]  │
├───────┬───────┬───────┬───────┬───────┬───────────────────────────  │
│ KPI-1 │ KPI-2 │ KPI-3 │ KPI-4 │ KPI-5 │ KPI-6                      │
│Sell-in│  POS  │Avg DOS│S-thru │Active │Active                       │
│ YTD   │  YTD  │(days) │ Rate  │ SKUs  │Channels                     │
├───────┴───────┴───────┴───────┴───────┴───────────────────────────  │
│                                                                       │
│  CHART 1: Weekly Sell-in + POS Trend (Line + Area)        [full row] │
│  Y-left: Units  |  Y-right: Channel Inventory                        │
│  Series: Sell-in (blue), Cust. Sales (orange), Inv. (grey fill)      │
│                                                                       │
├────────────────────────────┬──────────────────────────────────────── │
│  CHART 2: YoY Comparison   │  CHART 3: Channel DOS Heatmap           │
│  Grouped bars: current     │  Rows: Top 20 channels                  │
│  year vs prior year        │  Cols: Product families                 │
│  by week of year           │  Color: Red/Green/Orange by DOS         │
├────────────────────────────┴──────────────────────────────────────── │
│  CHART 4: Portfolio Matrix (Scatter)                      [full row] │
│  X: Sell-through Rate  |  Y: Sell-in volume  |  Size: Channel Inv.  │
│  Color: Product family  |  Quadrant labels: Stars/CashCows/etc.      │
└──────────────────────────────────────────────────────────────────────┘
```

### KPI Specifications

```python
# dashboards/components/kpis.py

def compute_executive_kpis(
    feature_df: pl.DataFrame,
    forecast_df: pl.DataFrame,
    current_week: str,
    ytd_weeks: list[str]
) -> dict:
    
    sell_in_ytd = feature_df.filter(
        pl.col("yearweek").is_in(ytd_weeks)
    )["sell_in"].sum()
    
    sell_in_ytd_py = feature_df.filter(
        pl.col("yearweek").is_in(prior_year_weeks(ytd_weeks))
    )["sell_in"].sum()
    
    return {
        "sell_in_ytd":      sell_in_ytd,
        "sell_in_yoy_pct":  (sell_in_ytd / sell_in_ytd_py - 1) * 100,
        "pos_ytd":          feature_df.filter(pl.col("yearweek").is_in(ytd_weeks))["cust_sales"].sum(),
        "avg_dos":          feature_df.filter(pl.col("yearweek") == current_week)["days_of_supply"].mean(),
        "sell_through_rate": feature_df["sell_through_rate_4w"].mean() * 100,
        "active_skus":      feature_df.filter(
                               (pl.col("yearweek") == current_week) & 
                               (pl.col("sell_in_ma4") > 0)
                            )["material"].n_unique(),
        "active_channels":  feature_df.filter(
                               (pl.col("yearweek") == current_week) & 
                               (pl.col("sell_in_ma4") > 0)
                            )["channel"].n_unique(),
    }
```

### Filters (sidebar)
- Year/period selector (2023 / 2024 / 2025)
- Channel multi-select (searchable)
- Product family multi-select
- Week range slider (ISO weeks)

---

## 3. DB-02 — Demand Planning Workbench

### Core Feature: Forecast Review + Override

```python
# dashboards/demand_planning.py

import streamlit as st
import polars as pl

def render_demand_workbench():
    st.title("Demand Planning Workbench")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_channel = st.selectbox("Channel", get_active_channels())
    with col2:
        selected_family = st.selectbox("Product Family", get_product_families())
    with col3:
        segment_filter = st.multiselect(
            "Series Segment", 
            ["regular", "intermittent", "rare"],
            default=["regular"]
        )
    
    # Forecast vs Actual chart
    series_df = load_series_data(selected_channel, selected_family, segment_filter)
    
    st.subheader("Forecast vs Actuals + Override")
    
    for idx, row in series_df.iterrows():
        with st.expander(f"{row['material']} — sMAPE: {row['smape']:.1f}%"):
            
            # Chart: historical actuals + forecast with confidence band
            fig = plot_forecast_vs_actual(row)
            st.plotly_chart(fig, use_container_width=True)
            
            # Override input
            col_a, col_b = st.columns(2)
            with col_a:
                override_value = st.number_input(
                    f"Override W34 forecast (model: {row['p50']:.0f})",
                    min_value=0,
                    value=int(row['p50']),
                    key=f"override_{row['series_id']}_w34"
                )
            with col_b:
                override_reason = st.selectbox(
                    "Reason",
                    ["", "Promotion planned", "Stockout risk", "Customer commitment", 
                     "Product launch", "Seasonal adjustment", "Other"],
                    key=f"reason_{row['series_id']}_w34"
                )
            
            if override_value != int(row['p50']) and override_reason:
                if st.button("Apply Override", key=f"btn_{row['series_id']}"):
                    save_override(row['series_id'], 'w34', override_value, override_reason)
                    st.success("Override saved ✓")
    
    # Accuracy summary panel
    st.subheader("Model Accuracy Summary")
    accuracy_df = load_accuracy_log()
    st.dataframe(
        accuracy_df.select(["yearweek", "smape_overall", "smape_regular", 
                            "smape_intermittent", "bias_pct"]).tail(8),
        use_container_width=True
    )
```

### Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  SIDEBAR: Channel | Family | Segment | Error threshold filter        │
├──────────┬────────────────────────────────────────────────────────── │
│ KPI STRIP│ Avg sMAPE | Coverage | # Overrides | Next retrain date   │
├──────────┴────────────────────────────────────────────────────────── │
│  ACCURACY TREND: Rolling sMAPE over last 12 weeks  [line chart]      │
├──────────────────────────────────────────────────────────────────────│
│  SERIES LIST (paginated, sorted by sMAPE desc)                       │
│  Each row: [SKU × Channel] | Segment | sMAPE | Bias | Override Btn  │
│  Expandable: Actual vs Forecast chart + override input               │
├──────────────────────────────────────────────────────────────────────│
│  OVERRIDE LOG (table): Date | User | Series | Old Value | New Value  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. DB-03 — Channel Health Monitor

### Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Channel Selector: [CUSTOMER2 ▼]   Compare with: [None ▼]           │
├────────────┬──────────────┬──────────────┬──────────────────────────  │
│ Total      │ Channel POS  │ Avg DOS      │ SKUs at Risk              │
│ Sell-in(W) │ This Week    │ (all SKUs)   │ Overstocked / Stocked-out │
├────────────┴──────────────┴──────────────┴──────────────────────────  │
│  GAUGE: Inventory Health    │  TREND: Sell-in vs POS (area chart)   │
│  % SKUs in healthy DOS zone │  Shaded area = inventory gap           │
├─────────────────────────────┴──────────────────────────────────────── │
│  SKU TABLE (sortable by DOS):                                         │
│  Material | Family | Sell-in 4w | POS 4w | S-Thru% | DOS | Trend↑↓  │
│  Color-coded rows: Red(DOS<7d), Green(7-45d), Orange(45-60d),        │
│                    Dark Orange(>60d)                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. DB-05 — Forecast Accuracy Dashboard

### Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  MODEL: [champion: lgbm_global_v2.1]    Period: [Last 8 weeks ▼]   │
├────────────┬────────────┬────────────┬────────────┬──────────────── │
│ sMAPE H+1  │ sMAPE H+4  │ sMAPE H+8  │ sMAPE H+19 │ Bias           │
│  14.2%  ✅ │  19.7%  ✅ │  24.1%  🟡 │  31.4%  🟡 │  +1.8%  ✅    │
├────────────┴────────────┴────────────┴────────────┴──────────────── │
│  ACCURACY OVER TIME (Line)    │  ERROR DISTRIBUTION (Histogram)     │
│  X: week | Y: sMAPE           │  X: forecast error% | Y: count     │
│  Series: H+1, H+4, H+8, H+19 │  Reference line at 0               │
├───────────────────────────────┴──────────────────────────────────── │
│  ACCURACY BY SEGMENT (Box)    │  HORIZON DECAY CURVE (Line)        │
│  X: regular/intermittent/rare │  X: H+1 through H+19              │
│  Y: sMAPE distribution        │  Y: sMAPE — shows degradation      │
├───────────────────────────────┴──────────────────────────────────── │
│  TOP 20 WORST PERFORMERS (Table)                                    │
│  Channel | Material | Segment | sMAPE | Bias | [Drill-down]         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. DB-06 — Inventory Risk Radar

### Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Risk Filter: [All ▼]  Family: [All ▼]  Channel: [All ▼]           │
├──────────────┬───────────────┬──────────────┬──────────────────────  │
│ Overstock    │ Stockout Risk │ Healthy      │ Excess Units Est.     │
│ Risk (>60d)  │ (<14d)        │ (14-45d)     │ (above 60d target)    │
│ 187 SKUs 🟡 │ 43 SKUs 🟢   │ 712 SKUs ✅  │ 24,500 units          │
├──────────────┴───────────────┴──────────────┴──────────────────────  │
│  RISK MATRIX (Scatter)                                               │
│  X: Days of Supply | Y: Weekly POS Velocity                         │
│  Size: Inventory units | Color: Family                              │
│  Vertical lines at 14d and 60d thresholds                           │
├────────────────────────────────────────────────────────────────────  │
│  RISK BY CHANNEL (Stacked Bar)    │  TRENDING TO STOCKOUT (Table)  │
│  X: Channel (sorted by risk)     │  Series where DOS dropping      │
│  Stack: Stockout/Healthy/Overstock│  fastest — action required     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 7. Streamlit Application Structure

```python
# dashboards/app.py

import streamlit as st

st.set_page_config(
    page_title="AI-DLC Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = {
    "📊 Executive Overview":     executive_kpis,
    "🔮 Demand Planning":        demand_workbench,
    "🏪 Channel Health":         channel_health,
    "📦 SKU Performance":        sku_performance,
    "🎯 Forecast Accuracy":      forecast_accuracy,
    "⚠️ Inventory Risk":         inventory_risk,
}

with st.sidebar:
    st.image("assets/logo.png", width=120)
    st.markdown("### AI-DLC Forecasting")
    st.caption(f"Data as of: {get_latest_data_week()}")
    st.caption(f"Model: {get_champion_model_name()}")
    
    selected_page = st.radio("Navigation", list(pages.keys()))

pages[selected_page]()
```

---

## 8. Data Refresh Contract (Dashboards)

```python
# All dashboard data loaded from cached Parquet files
# Cache refreshed after each Monday pipeline run

@st.cache_data(ttl=3600)  # 1-hour TTL
def load_feature_store() -> pl.DataFrame:
    return pl.read_parquet("data/gold/feature_store.parquet")

@st.cache_data(ttl=3600)
def load_forecast_output() -> pl.DataFrame:
    return pl.read_parquet("data/gold/forecast_output.parquet")

@st.cache_data(ttl=3600)
def load_accuracy_log() -> pl.DataFrame:
    return pl.read_parquet("data/gold/accuracy_log.parquet")
```

---

*AI-DLC Traceability ID: DASH-SPEC-ITER2-001 | Version: 2.0*
