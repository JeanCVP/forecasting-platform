"""
AI-DLC Demand Forecasting Dashboard
Streamlit app — 4 secciones: Resumen, Tendencias, Forecasts, Riesgo de Inventario
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ─── Configuración ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

SILVER_PATH    = Path("data/silver/silver_dataset.parquet")
FORECASTS_PATH = Path("data/forecasts/forecasts.parquet")
RISK_PATH      = Path("data/forecasts/inventory_risk.parquet")
REPORTS_DIR    = Path("reports")

RISK_COLORS = {
    "CRITICAL": "#d62728",
    "HIGH":     "#ff7f0e",
    "MEDIUM":   "#ffdd57",
    "LOW":      "#2ca02c",
}

# ─── Carga de datos (cacheada) ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_silver() -> pd.DataFrame:
    df = pd.read_parquet(SILVER_PATH)
    df["year_week"] = (
        df["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    )
    return df


@st.cache_data(show_spinner=False)
def load_forecasts() -> pd.DataFrame | None:
    if not FORECASTS_PATH.exists():
        return None
    return pd.read_parquet(FORECASTS_PATH)


@st.cache_data(show_spinner=False)
def load_risk() -> pd.DataFrame | None:
    if not RISK_PATH.exists():
        return None
    return pd.read_parquet(RISK_PATH)


@st.cache_data(show_spinner=False)
def load_report(name: str) -> dict:
    p = REPORTS_DIR / f"{name}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📦 Demand Forecasting")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navegación",
    ["📊 Resumen", "📈 Tendencias Históricas", "🔮 Forecasts", "⚠️ Riesgo de Inventario"],
)
st.sidebar.markdown("---")
st.sidebar.caption("AI-DLC · Colombia · 2025")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Resumen":
    st.title("📊 Resumen del Proyecto")
    st.markdown("Visión general del dataset, calidad de datos y desempeño del modelo.")

    with st.spinner("Cargando datos..."):
        silver = load_silver()
        bench  = load_report("benchmark_report")
        lgbm   = load_report("lgbm_report")
        dq     = load_report("dq_report")
        fc_rep = load_report("forecast_report")

    sellin = silver[silver["Category"] == "Sell-in"]
    n_skus     = sellin[["Channel", "Material Description"]].drop_duplicates().shape[0]
    n_channels = sellin["Channel"].nunique()
    n_weeks    = sellin["year_week"].nunique()
    last_week  = int(sellin["year_week"].max())
    dq_score   = dq.get("dq_metrics", {}).get("quality_score", "—")
    if isinstance(dq_score, float):
        dq_score = f"{dq_score * 100:.0f}/100"

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SKUs activos", f"{n_skus:,}")
    k2.metric("Canales", f"{n_channels:,}")
    k3.metric("Semanas de datos", f"{n_weeks:,}")
    k4.metric("Calidad de datos", dq_score)

    st.markdown("---")

    # Modelo row
    st.subheader("Desempeño del Modelo")
    m1, m2, m3, m4 = st.columns(4)
    best_model = bench.get("best_model", "—")
    best_smape = bench.get("best_smape", "—")
    lgbm_smape = lgbm.get("smape", "—")

    m1.metric("Mejor baseline", best_model.replace("_", " ").title() if best_model != "—" else "—")
    m2.metric("sMAPE baseline", f"{best_smape:.1f}%" if isinstance(best_smape, float) else "—")
    m3.metric("sMAPE LightGBM", f"{lgbm_smape:.1f}%" if isinstance(lgbm_smape, float) else "—")
    if isinstance(best_smape, float) and isinstance(lgbm_smape, float):
        mejora = (best_smape - lgbm_smape) / best_smape * 100
        m4.metric("Mejora vs. baseline", f"{mejora:.0f}%", delta=f"-{best_smape - lgbm_smape:.1f}pp")

    st.markdown("---")

    # Forecast resumen
    if fc_rep:
        st.subheader("Forecasts Generados")
        f1, f2, f3 = st.columns(3)
        f1.metric("SKUs con forecast", f"{fc_rep.get('n_skus_forecast', 0):,}")
        f2.metric("Ventana de forecast",
                  f"W{str(fc_rep.get('forecast_start',''))[-2:]}–W{str(fc_rep.get('forecast_end',''))[-2:]} 2025")
        total = fc_rep.get("total_forecast_lgbm", 0)
        f3.metric("Volumen total estimado", f"{total:,.0f} unidades")
    else:
        st.info("Forecasts aún no generados.")

    st.markdown("---")

    # Comparación modelos
    if bench.get("model_summary"):
        st.subheader("Comparación de Modelos Baseline")
        rows = [
            {"Modelo": k.replace("_", " ").title(), **{m.upper(): round(v, 2)
             for m, v in vals.items() if m in ("smape", "wape", "mase")}}
            for k, vals in bench["model_summary"].items()
        ]
        df_m = pd.DataFrame(rows).sort_values("SMAPE")
        st.dataframe(df_m.reset_index(drop=True), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — TENDENCIAS HISTÓRICAS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Tendencias Históricas":
    st.title("📈 Tendencias Históricas")

    with st.spinner("Cargando datos históricos..."):
        silver = load_silver()

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)
    categories = sorted(silver["Category"].unique())
    sel_cat = col_f1.selectbox("Categoría", categories, index=categories.index("Sell-in") if "Sell-in" in categories else 0)
    channels = ["Todos"] + sorted(silver["Channel"].unique())
    sel_ch = col_f2.selectbox("Canal", channels)
    years = ["Todos"] + sorted(silver["year"].unique().tolist())
    sel_yr = col_f3.selectbox("Año", years)

    df_f = silver[silver["Category"] == sel_cat].copy()
    if sel_ch != "Todos":
        df_f = df_f[df_f["Channel"] == sel_ch]
    if sel_yr != "Todos":
        df_f = df_f[df_f["year"] == sel_yr]

    st.markdown("---")

    # Tendencia semanal
    st.subheader("Volumen semanal")
    weekly = df_f.groupby("year_week")["quantity"].sum().reset_index()
    weekly["year_week_str"] = weekly["year_week"].astype(str)
    chart_weekly = (
        alt.Chart(weekly)
        .mark_line(point=True, strokeWidth=2, color="#1f77b4")
        .encode(
            x=alt.X("year_week_str:O", title="Semana", axis=alt.Axis(labelAngle=-45, labelOverlap="greedy")),
            y=alt.Y("quantity:Q", title="Cantidad"),
            tooltip=["year_week_str:O", alt.Tooltip("quantity:Q", format=",.0f")],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(chart_weekly, use_container_width=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # Top 10 SKUs por volumen
    with col_l:
        st.subheader("Top 10 SKUs por volumen")
        top_skus = (
            df_f.groupby("Material Description")["quantity"]
            .sum()
            .nlargest(10)
            .reset_index()
            .sort_values("quantity")
        )
        top_skus["Material Description"] = top_skus["Material Description"].str[:30]
        chart_top = (
            alt.Chart(top_skus)
            .mark_bar(color="#1f77b4")
            .encode(
                y=alt.Y("Material Description:N", sort="-x", title=""),
                x=alt.X("quantity:Q", title="Cantidad total"),
                tooltip=["Material Description:N", alt.Tooltip("quantity:Q", format=",.0f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_top, use_container_width=True)

    # Distribución por canal
    with col_r:
        st.subheader("Volumen por canal")
        by_channel = (
            df_f.groupby("Channel")["quantity"]
            .sum()
            .nlargest(15)
            .reset_index()
            .sort_values("quantity")
        )
        chart_ch = (
            alt.Chart(by_channel)
            .mark_bar(color="#ff7f0e")
            .encode(
                y=alt.Y("Channel:N", sort="-x", title=""),
                x=alt.X("quantity:Q", title="Cantidad total"),
                tooltip=["Channel:N", alt.Tooltip("quantity:Q", format=",.0f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_ch, use_container_width=True)

    st.markdown("---")

    # Heatmap Canal × Año
    st.subheader("Volumen por Canal y Año")
    heat_data = (
        silver[silver["Category"] == sel_cat]
        .groupby(["Channel", "year"])["quantity"]
        .sum()
        .reset_index()
    )
    top_ch = heat_data.groupby("Channel")["quantity"].sum().nlargest(20).index
    heat_data = heat_data[heat_data["Channel"].isin(top_ch)]
    heat_chart = (
        alt.Chart(heat_data)
        .mark_rect()
        .encode(
            x=alt.X("year:O", title="Año"),
            y=alt.Y("Channel:N", title="Canal", sort="-x"),
            color=alt.Color("quantity:Q", scale=alt.Scale(scheme="blues"), title="Cantidad"),
            tooltip=["Channel:N", "year:O", alt.Tooltip("quantity:Q", format=",.0f")],
        )
        .properties(height=400)
    )
    st.altair_chart(heat_chart, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — FORECASTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecasts":
    st.title("🔮 Forecasts de Demanda")

    forecasts = load_forecasts()
    if forecasts is None:
        st.warning("⏳ Los forecasts aún no han sido generados. Ejecuta: `python -m src.forecasting.forecast_generator`")
        st.stop()

    with st.spinner("Cargando datos históricos..."):
        silver = load_silver()

    fc_rep = load_report("forecast_report")

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("SKUs con forecast", f"{forecasts[['Channel','Material Description']].drop_duplicates().shape[0]:,}")
    k2.metric("Horizonte", f"W{str(fc_rep.get('forecast_start',''))[-2:]}–W{str(fc_rep.get('forecast_end',''))[-2:]} 2025")
    total = forecasts["forecast_lgbm"].sum()
    k3.metric("Volumen total estimado (LightGBM)", f"{total:,.0f} unidades")

    st.markdown("---")

    # Filtro por canal y SKU
    col_f1, col_f2 = st.columns(2)
    channels = ["Todos"] + sorted(forecasts["Channel"].unique())
    sel_ch = col_f1.selectbox("Canal", channels, key="fc_ch")

    fc_filtered = forecasts if sel_ch == "Todos" else forecasts[forecasts["Channel"] == sel_ch]
    mats = ["Todos"] + sorted(fc_filtered["Material Description"].unique())
    sel_mat = col_f2.selectbox("Producto", mats, key="fc_mat")

    if sel_mat != "Todos":
        fc_view = fc_filtered[fc_filtered["Material Description"] == sel_mat]
    else:
        fc_view = fc_filtered

    # Serie histórica + forecast
    st.subheader("Serie histórica y predicción")

    hist_data = silver[silver["Category"] == "Sell-in"].copy()
    if sel_ch != "Todos":
        hist_data = hist_data[hist_data["Channel"] == sel_ch]
    if sel_mat != "Todos":
        hist_data = hist_data[hist_data["Material Description"] == sel_mat]

    hist_agg = (
        hist_data.groupby("year_week")["quantity"]
        .sum().reset_index()
        .rename(columns={"quantity": "value"})
        .assign(tipo="Histórico")
    )
    hist_agg["year_week_str"] = hist_agg["year_week"].astype(str)

    fc_agg = (
        fc_view.groupby("year_week")[["forecast_lgbm", "forecast_naive"]]
        .sum().reset_index()
    )
    fc_lgbm = (
        fc_agg[["year_week", "forecast_lgbm"]]
        .rename(columns={"forecast_lgbm": "value"})
        .assign(tipo="LightGBM")
    )
    fc_naive = (
        fc_agg[["year_week", "forecast_naive"]]
        .rename(columns={"forecast_naive": "value"})
        .assign(tipo="Seasonal Naïve")
    )
    fc_lgbm["year_week_str"]  = fc_lgbm["year_week"].astype(str)
    fc_naive["year_week_str"] = fc_naive["year_week"].astype(str)

    # Últimas 26 semanas históricas + todas las forecast
    hist_recent = hist_agg.tail(26)
    combined = pd.concat([hist_recent, fc_lgbm, fc_naive], ignore_index=True)

    color_scale = alt.Scale(
        domain=["Histórico", "LightGBM", "Seasonal Naïve"],
        range=["#aec7e8", "#1f77b4", "#ff7f0e"],
    )
    chart = (
        alt.Chart(combined)
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X("year_week_str:O", title="Semana", axis=alt.Axis(labelAngle=-45, labelOverlap="greedy")),
            y=alt.Y("value:Q", title="Cantidad"),
            color=alt.Color("tipo:N", scale=color_scale, title="Serie"),
            strokeDash=alt.condition(
                alt.datum.tipo == "Histórico",
                alt.value([1, 0]),
                alt.value([6, 3]),
            ),
            tooltip=["year_week_str:O", "tipo:N", alt.Tooltip("value:Q", format=",.1f")],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # Top 10 SKUs por volumen forecast
    st.subheader("Top 10 SKUs por volumen estimado (LightGBM)")
    top10 = (
        forecasts.groupby(["Channel", "Material Description"])["forecast_lgbm"]
        .sum()
        .nlargest(10)
        .reset_index()
        .sort_values("forecast_lgbm")
    )
    top10["label"] = top10["Channel"].str[:10] + " · " + top10["Material Description"].str[:25]
    chart_top = (
        alt.Chart(top10)
        .mark_bar(color="#1f77b4")
        .encode(
            y=alt.Y("label:N", sort="-x", title=""),
            x=alt.X("forecast_lgbm:Q", title="Unidades estimadas (13 semanas)"),
            tooltip=["Channel:N", "Material Description:N",
                     alt.Tooltip("forecast_lgbm:Q", format=",.0f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart_top, use_container_width=True)

    st.markdown("---")

    # Tabla detalle
    with st.expander("📋 Tabla de forecasts detallada"):
        show_cols = ["Channel", "Material Description", "year_week",
                     "horizon_step", "forecast_lgbm", "forecast_naive"]
        st.dataframe(
            fc_view[show_cols].sort_values(["Channel", "Material Description", "year_week"]),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — RIESGO DE INVENTARIO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Riesgo de Inventario":
    st.title("⚠️ Riesgo de Inventario")
    st.markdown("SKUs donde el inventario actual podría no cubrir la demanda estimada.")

    risk = load_risk()
    if risk is None:
        st.warning("⏳ El scoring de inventario aún no se ha generado. Ejecuta: `python -m src.inventory.risk_scorer`")
        st.stop()

    risk_rep = load_report("inventory_risk_report")

    # KPIs
    dist = risk_rep.get("risk_distribution", {})
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("SKUs analizados", f"{len(risk):,}")
    k2.metric("🔴 CRITICAL", f"{dist.get('CRITICAL', 0):,}")
    k3.metric("🟠 HIGH",     f"{dist.get('HIGH', 0):,}")
    k4.metric("🟡 MEDIUM",   f"{dist.get('MEDIUM', 0):,}")
    k5.metric("🟢 LOW",      f"{dist.get('LOW', 0):,}")

    pct = risk_rep.get("pct_at_risk", 0)
    if pct > 20:
        st.error(f"⚠️  **{pct}%** de los SKUs están en riesgo de quiebre de stock (CRITICAL + HIGH)")
    elif pct > 10:
        st.warning(f"⚠️  **{pct}%** de los SKUs están en riesgo de quiebre de stock (CRITICAL + HIGH)")
    else:
        st.success(f"✅  **{pct}%** de los SKUs en riesgo de quiebre (CRITICAL + HIGH)")

    st.markdown("---")

    col_l, col_r = st.columns([1, 2])

    # Gráfico de distribución
    with col_l:
        st.subheader("Distribución de riesgo")
        dist_df = pd.DataFrame([
            {"Nivel": k, "SKUs": v, "color": RISK_COLORS[k]}
            for k, v in dist.items() if v > 0
        ])
        pie = (
            alt.Chart(dist_df)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta("SKUs:Q"),
                color=alt.Color(
                    "Nivel:N",
                    scale=alt.Scale(
                        domain=list(RISK_COLORS.keys()),
                        range=list(RISK_COLORS.values()),
                    ),
                    legend=alt.Legend(title="Nivel"),
                ),
                tooltip=["Nivel:N", "SKUs:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(pie, use_container_width=True)

    # Distribución de weeks_of_supply
    with col_r:
        st.subheader("Semanas de cobertura por SKU")
        wos = risk[risk["weeks_of_supply"] < 20].copy()  # recortar outliers para el gráfico
        hist_chart = (
            alt.Chart(wos)
            .mark_bar(color="#1f77b4", opacity=0.8)
            .encode(
                x=alt.X("weeks_of_supply:Q", bin=alt.Bin(maxbins=30),
                         title="Semanas de cobertura"),
                y=alt.Y("count():Q", title="Número de SKUs"),
                tooltip=["count():Q"],
            )
            .properties(height=280)
        )
        ref_lines = (
            alt.Chart(pd.DataFrame([
                {"x": 2, "label": "CRITICAL"},
                {"x": 4, "label": "HIGH"},
                {"x": 8, "label": "MEDIUM"},
            ]))
            .mark_rule(strokeDash=[4, 4], strokeWidth=1.5)
            .encode(
                x="x:Q",
                color=alt.Color("label:N",
                                scale=alt.Scale(
                                    domain=["CRITICAL", "HIGH", "MEDIUM"],
                                    range=["#d62728", "#ff7f0e", "#ffdd57"],
                                )),
            )
        )
        st.altair_chart(hist_chart + ref_lines, use_container_width=True)

    st.markdown("---")

    # Filtro por nivel de riesgo
    st.subheader("SKUs en riesgo")
    col_rf1, col_rf2 = st.columns(2)
    levels = ["Todos"] + ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    sel_level = col_rf1.selectbox("Nivel de riesgo", levels)
    sel_ch_r  = col_rf2.selectbox(
        "Canal", ["Todos"] + sorted(risk["Channel"].unique()), key="risk_ch"
    )

    risk_view = risk.copy()
    if sel_level != "Todos":
        risk_view = risk_view[risk_view["risk_level"] == sel_level]
    if sel_ch_r != "Todos":
        risk_view = risk_view[risk_view["Channel"] == sel_ch_r]

    display_cols = [
        "Channel", "Material Description", "risk_level",
        "current_inventory", "forecast_13wk", "weeks_of_supply",
        "coverage_ratio", "stockout_week",
    ]

    def _style_risk(val):
        colors = {"CRITICAL": "#ffd5d5", "HIGH": "#ffe5cc",
                  "MEDIUM": "#fffacc", "LOW": "#d5f5d5"}
        return f"background-color: {colors.get(val, '')}"

    styled = (
        risk_view[display_cols]
        .reset_index(drop=True)
        .style.applymap(_style_risk, subset=["risk_level"])
        .format({
            "current_inventory": "{:,.1f}",
            "forecast_13wk":     "{:,.1f}",
            "weeks_of_supply":   "{:.1f}",
            "coverage_ratio":    "{:.2f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=420)

    # Detalle: semana de quiebre
    critical_rows = risk[
        (risk["risk_level"] == "CRITICAL") & (risk["stockout_week"].notna())
    ].head(5)
    if len(critical_rows) > 0:
        st.markdown("---")
        st.subheader("🚨 Próximos quiebres de stock")
        for _, row in critical_rows.iterrows():
            sw = int(row["stockout_week"])
            yr, wk = sw // 100, sw % 100
            st.error(
                f"**{row['Channel']}** · {row['Material Description'][:40]}  "
                f"→ Quiebre estimado en **W{wk} {yr}**  "
                f"(inventario: {row['current_inventory']:,.0f} | demanda 13s: {row['forecast_13wk']:,.0f})"
            )
