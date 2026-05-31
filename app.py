"""
Dashboard Ejecutivo — Pronóstico de Demanda e Inventario
Diseñado para dirección comercial: lenguaje de negocio, insights claros, acción inmediata.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ─── Configuración ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pronóstico de Demanda",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    footer {visibility: hidden;}

    /* Tarjetas de métricas */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1a56db;
    }
    .kpi-card.red   { border-left-color: #e02424; }
    .kpi-card.green { border-left-color: #0e9f6e; }
    .kpi-card.amber { border-left-color: #c27803; }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 6px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-delta {
        font-size: 0.9rem;
        margin-top: 4px;
        font-weight: 600;
    }
    .delta-up   { color: #0e9f6e; }
    .delta-down { color: #e02424; }

    /* Tarjetas de alerta */
    .alert-card {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 5px solid;
    }
    .alert-critical { background:#fff5f5; border-left-color:#e02424; }
    .alert-high     { background:#fffbf0; border-left-color:#c27803; }

    .alert-title { font-weight: 700; font-size: 0.95rem; color: #111827; }
    .alert-body  { font-size: 0.85rem; color: #4b5563; margin-top: 3px; }

    /* Separador de sección */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
        margin: 8px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e5e7eb;
    }

    /* Insight box */
    .insight-box {
        background: #eff6ff;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #1a56db;
        margin-bottom: 12px;
    }
    .insight-box p { margin: 0; color: #1e40af; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ─── Rutas ────────────────────────────────────────────────────────────────────
SILVER_PATH    = Path("data/silver/silver_dataset.parquet")
FORECASTS_PATH = Path("data/forecasts/forecasts.parquet")
RISK_PATH      = Path("data/forecasts/inventory_risk.parquet")
REPORTS_DIR    = Path("reports")


# ─── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60)
def load_silver():
    df = pd.read_parquet(SILVER_PATH)
    df["year_week"] = df["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    return df

@st.cache_data(show_spinner=False, ttl=60)
def load_forecasts():
    return pd.read_parquet(FORECASTS_PATH) if FORECASTS_PATH.exists() else None

@st.cache_data(show_spinner=False, ttl=60)
def load_risk():
    return pd.read_parquet(RISK_PATH) if RISK_PATH.exists() else None

@st.cache_data(show_spinner=False, ttl=60)
def load_report(name):
    p = REPORTS_DIR / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def _weeks_ago(yw: int, n: int) -> int:
    """Return the year_week that is n weeks before yw."""
    yr, wk = yw // 100, yw % 100
    d = date.fromisocalendar(yr, wk, 1) - timedelta(weeks=n)
    iso = d.isocalendar()
    return iso[0] * 100 + iso[1]


def _fmt_yw(yw) -> str:
    """Format a year_week int as 'Wnn YYYY'."""
    try:
        v = int(yw)
        return f"W{v % 100:02d} {v // 100}"
    except Exception:
        return "—"


def kpi(value, label, color="", delta="", delta_up=True):
    delta_html = ""
    if delta:
        cls = "delta-up" if delta_up else "delta-down"
        arrow = "▲" if delta_up else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="kpi-card {color}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2038/2038854.png", width=60)
    st.markdown("## 📦 Pronóstico de Demanda")
    st.markdown("**Histórico 2023–W33·2025 · Pronóstico W34·2025–W52·2026**")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Resumen Ejecutivo",
        "📈  Comportamiento de Ventas",
        "🔮  Proyección de Demanda",
        "🚨  Alertas de Inventario",
    ], label_visibility="collapsed")
    st.markdown("---")
    if st.button("🔄  Actualizar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Datos reales hasta W33 · 2025 | Pronóstico: W34·2025 → W52·2026")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Resumen Ejecutivo":

    st.markdown("# 🏠 Resumen Ejecutivo")
    st.markdown("Visión general del negocio y principales hallazgos del análisis predictivo.")
    st.markdown("---")

    silver    = load_silver()
    forecasts = load_forecasts()
    risk      = load_risk()
    risk_rep  = load_report("inventory_risk_report")
    fc_rep    = load_report("forecast_report")

    sellin = silver[silver["Category"] == "Sell-in"]

    # ── KPIs principales ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Indicadores Clave</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    n_skus     = sellin[["Channel", "Material Description"]].drop_duplicates().shape[0]
    n_channels = sellin["Channel"].nunique()

    with c1:
        kpi(f"{n_skus:,}", "Productos monitoreados")
    with c2:
        kpi(f"{n_channels:,}", "Clientes activos")

    if forecasts is not None:
        total_fc = forecasts["forecast_naive"].sum()
        with c3:
            kpi(f"{total_fc:,.0f}", "Unidades proyectadas · W34·2025–W52·2026", color="green")
    else:
        with c3:
            kpi("—", "Unidades proyectadas · W34·2025–W52·2026")

    if risk is not None:
        dist = risk_rep.get("risk_distribution", {})
        n_criticos = dist.get("CRITICAL", 0) + dist.get("HIGH", 0)
        pct = risk_rep.get("pct_at_risk", 0)
        color_risk = "red" if pct > 15 else "amber" if pct > 5 else "green"
        with c4:
            kpi(f"{n_criticos:,}", "Productos en alerta de inventario", color=color_risk,
                delta=f"{pct}% del catálogo", delta_up=False)
    else:
        with c4:
            kpi("—", "Productos en alerta de inventario")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Insights principales ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 Lo que debes saber</div>', unsafe_allow_html=True)

    # Volumen histórico reciente (últimas 4 semanas del dataset)
    last_yw    = int(sellin["year_week"].max())
    l4_start   = _weeks_ago(last_yw, 3)
    py_end     = _weeks_ago(last_yw, 52)
    py_start   = _weeks_ago(last_yw, 52 + 3)
    last4w = sellin[sellin["year_week"] >= l4_start]["quantity"].sum()
    prev4w = sellin[(sellin["year_week"] >= py_start) & (sellin["year_week"] <= py_end)]["quantity"].sum()
    trend  = ((last4w - prev4w) / prev4w * 100) if prev4w > 0 else 0

    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        st.markdown(f"""
        <div class="insight-box">
        <p>📦 <strong>Volumen reciente (últimas 4 semanas):</strong> {last4w:,.0f} unidades vendidas.
        {"La demanda muestra tendencia <strong>positiva</strong> frente al mismo período del año anterior." if trend >= 0 else "La demanda muestra una <strong>reducción</strong> frente al mismo período del año anterior."}</p>
        </div>""", unsafe_allow_html=True)

        if forecasts is not None:
            wk_avg = forecasts.groupby("year_week")["forecast_naive"].sum().mean()
            st.markdown(f"""
            <div class="insight-box">
            <p>🔮 <strong>Proyección W34·2025–W52·2026 (71 semanas):</strong> Se estiman en promedio
            <strong>{wk_avg:,.0f} unidades por semana</strong> basado en el comportamiento histórico estacional.</p>
            </div>""", unsafe_allow_html=True)

    with col_ins2:
        if risk is not None:
            dist = risk_rep.get("risk_distribution", {})
            top_crit = risk_rep.get("top_critical_skus", [])
            top1 = top_crit[0] if top_crit else None
            st.markdown(f"""
            <div class="insight-box">
            <p>🚨 <strong>{dist.get('CRITICAL',0):,} productos</strong> agotarán su inventario
            antes de que termine el horizonte de pronóstico. Se requiere reabastecimiento urgente.</p>
            </div>""", unsafe_allow_html=True)

            if top1:
                sw = top1.get("stockout_week")
                sw_str = _fmt_yw(sw) if sw else "próximas semanas"
                st.markdown(f"""
                <div class="insight-box">
                <p>⚠️ <strong>Caso más crítico:</strong> {str(top1.get('Material Description',''))[:45]}
                — inventario estimado agotado en <strong>{sw_str}</strong>.</p>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mini gráfico: histórico + forecast ───────────────────────────────────
    if forecasts is not None:
        st.markdown('<div class="section-header">📉 Tendencia reciente y proyección</div>', unsafe_allow_html=True)

        mini_hist_start = _weeks_ago(last_yw, 15)
        hist_weekly = (
            sellin[sellin["year_week"] >= mini_hist_start]
            .groupby("year_week")["quantity"].sum().reset_index()
            .rename(columns={"quantity": "valor"})
            .assign(tipo="Ventas históricas")
        )
        fc_weekly = (
            forecasts.groupby("year_week")["forecast_naive"].sum().reset_index()
            .rename(columns={"forecast_naive": "valor"})
            .assign(tipo="Proyección")
        )
        combined = pd.concat([hist_weekly, fc_weekly])
        combined["semana"] = combined["year_week"].astype(str)

        color_sc = alt.Scale(domain=["Ventas históricas", "Proyección"],
                             range=["#1a56db", "#0e9f6e"])
        dash_sc  = alt.Scale(domain=["Ventas históricas", "Proyección"],
                             range=[[1, 0], [6, 3]])

        mini = (
            alt.Chart(combined)
            .mark_line(strokeWidth=3, point=alt.OverlayMarkDef(size=50))
            .encode(
                x=alt.X("semana:O", title="Semana",
                         axis=alt.Axis(labelAngle=-45, labelOverlap="greedy")),
                y=alt.Y("valor:Q", title="Unidades"),
                color=alt.Color("tipo:N", scale=color_sc,
                                 legend=alt.Legend(orient="top", title="")),
                strokeDash=alt.StrokeDash("tipo:N", scale=dash_sc),
                tooltip=["semana:O", "tipo:N", alt.Tooltip("valor:Q", format=",.0f", title="Unidades")],
            )
            .properties(height=280)
            .interactive()
        )
        st.altair_chart(mini, width='stretch')
        st.caption("La línea punteada verde representa la proyección W34·2025 → W52·2026 (71 semanas).")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — COMPORTAMIENTO DE VENTAS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Comportamiento de Ventas":
    st.markdown("# 📈 Comportamiento de Ventas")
    st.markdown("Analiza el historial de ventas por cliente, producto y período.")
    st.markdown("---")

    silver = load_silver()
    sellin = silver[silver["Category"] == "Sell-in"].copy()

    # Filtros en línea
    col_f1, col_f2, col_f3 = st.columns(3)
    clientes = ["Todos los clientes"] + sorted(sellin["Channel"].unique())
    sel_cl   = col_f1.selectbox("Cliente", clientes)
    años     = ["Todos los años"] + sorted(sellin["year"].unique().tolist(), reverse=True)
    sel_yr   = col_f2.selectbox("Año", años)
    vista    = col_f3.selectbox("Agrupar por", ["Semana", "Mes (4 semanas)"])

    df = sellin.copy()
    if sel_cl != "Todos los clientes":
        df = df[df["Channel"] == sel_cl]
    if sel_yr != "Todos los años":
        df = df[df["year"] == sel_yr]

    st.markdown("---")

    # KPIs rápidos
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi(f"{df['quantity'].sum():,.0f}", "Unidades totales")
    with k2:
        kpi(f"{df[['Channel','Material Description']].drop_duplicates().shape[0]:,}", "Combinaciones cliente-producto")
    with k3:
        avg_wk = df.groupby("year_week")["quantity"].sum().mean()
        kpi(f"{avg_wk:,.0f}", "Promedio semanal")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tendencia de ventas ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Tendencia de ventas en el tiempo</div>',
                unsafe_allow_html=True)

    if vista == "Semana":
        trend_df = df.groupby("year_week")["quantity"].sum().reset_index()
        trend_df["eje"] = trend_df["year_week"].astype(str)
        x_title = "Semana"
    else:
        df["mes_idx"] = ((df["week_num"] - 1) // 4) + 1
        df["mes_label"] = df["year"].astype(str) + "-M" + df["mes_idx"].astype(str).str.zfill(2)
        trend_df = df.groupby("mes_label")["quantity"].sum().reset_index()
        trend_df["eje"] = trend_df["mes_label"]
        x_title = "Período (4 semanas)"

    trend_chart = (
        alt.Chart(trend_df)
        .mark_area(
            line={"color": "#1a56db", "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#1a56db", offset=0),
                       alt.GradientStop(color="white", offset=1)],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        .encode(
            x=alt.X("eje:O", title=x_title,
                     sort=None,
                     axis=alt.Axis(labelAngle=-45, labelOverlap="greedy")),
            y=alt.Y("quantity:Q", title="Unidades vendidas"),
            tooltip=["eje:O", alt.Tooltip("quantity:Q", format=",.0f", title="Unidades")],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(trend_chart, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # ── Top 10 clientes ───────────────────────────────────────────────────────
    with col_l:
        st.markdown('<div class="section-header">Top 10 clientes por volumen</div>',
                    unsafe_allow_html=True)
        top_cl = (
            df.groupby("Channel")["quantity"].sum()
            .nlargest(10).reset_index().sort_values("quantity")
            .rename(columns={"Channel": "Cliente", "quantity": "Unidades"})
        )
        chart_cl = (
            alt.Chart(top_cl)
            .mark_bar(color="#1a56db", cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("Cliente:N", sort="-x", title=""),
                x=alt.X("Unidades:Q", title="Unidades vendidas"),
                tooltip=["Cliente:N", alt.Tooltip("Unidades:Q", format=",.0f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_cl, width='stretch')

    # ── Top 10 productos ──────────────────────────────────────────────────────
    with col_r:
        st.markdown('<div class="section-header">Top 10 productos por volumen</div>',
                    unsafe_allow_html=True)
        top_pr = (
            df.groupby("Material Description")["quantity"].sum()
            .nlargest(10).reset_index().sort_values("quantity")
            .rename(columns={"Material Description": "Producto", "quantity": "Unidades"})
        )
        top_pr["Producto"] = top_pr["Producto"].str[:35]
        chart_pr = (
            alt.Chart(top_pr)
            .mark_bar(color="#0e9f6e", cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("Producto:N", sort="-x", title=""),
                x=alt.X("Unidades:Q", title="Unidades vendidas"),
                tooltip=["Producto:N", alt.Tooltip("Unidades:Q", format=",.0f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_pr, width='stretch')

    # ── Comparativo anual ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Comparativo anual de ventas</div>',
                unsafe_allow_html=True)

    yoy = (
        sellin.groupby(["year", "week_num"])["quantity"].sum()
        .reset_index()
        .rename(columns={"year": "Año", "week_num": "Semana", "quantity": "Unidades"})
    )
    yoy["Año"] = yoy["Año"].astype(str)

    yoy_chart = (
        alt.Chart(yoy)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Semana:O", title="Semana del año"),
            y=alt.Y("Unidades:Q", title="Unidades vendidas"),
            color=alt.Color("Año:N",
                            scale=alt.Scale(domain=sorted(yoy["Año"].unique().tolist()),
                                            range=["#9ca3af", "#60a5fa", "#1a56db"]),
                            legend=alt.Legend(orient="top", title="Año")),
            tooltip=["Año:N", "Semana:O", alt.Tooltip("Unidades:Q", format=",.0f")],
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(yoy_chart, width='stretch')
    st.caption("Datos históricos 2023–2025. El pronóstico 2026 se visualiza en la página Proyección de Demanda.")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — PROYECCIÓN DE DEMANDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Proyección de Demanda":
    st.markdown("# 🔮 Proyección de Demanda")
    st.markdown("Estimación de ventas desde **W34·2025 hasta W52·2026** — resto de 2025 más todo el 2026 (71 semanas).")
    st.markdown("---")

    forecasts = load_forecasts()
    if forecasts is None:
        st.warning("Los pronósticos aún no están disponibles.")
        st.stop()

    silver = load_silver()
    sellin = silver[silver["Category"] == "Sell-in"]

    # Filtros
    col_f1, col_f2 = st.columns(2)
    clientes = ["Todos los clientes"] + sorted(forecasts["Channel"].unique())
    sel_cl   = col_f1.selectbox("Cliente", clientes, key="fc_cl")
    fc_filt  = forecasts if sel_cl == "Todos los clientes" else forecasts[forecasts["Channel"] == sel_cl]
    productos = ["Todos los productos"] + sorted(fc_filt["Material Description"].unique())
    sel_pr    = col_f2.selectbox("Producto", productos, key="fc_pr")
    if sel_pr != "Todos los productos":
        fc_filt = fc_filt[fc_filt["Material Description"] == sel_pr]

    # KPIs
    total_fc  = fc_filt["forecast_naive"].sum()
    avg_wk_fc = fc_filt.groupby("year_week")["forecast_naive"].sum().mean()

    k1, k2, k3 = st.columns(3)
    with k1: kpi(f"{total_fc:,.0f}", "Unidades proyectadas · 71 semanas", color="green")
    with k2: kpi(f"{avg_wk_fc:,.0f}", "Promedio semanal estimado")
    with k3:
        hist_filt = sellin.copy()
        if sel_cl != "Todos los clientes":
            hist_filt = hist_filt[hist_filt["Channel"] == sel_cl]
        if sel_pr != "Todos los productos":
            hist_filt = hist_filt[hist_filt["Material Description"] == sel_pr]
        last_yw_fc = int(sellin["year_week"].max())
        recent_start = _weeks_ago(last_yw_fc, 8)
        hist_avg = hist_filt[hist_filt["year_week"] >= recent_start].groupby("year_week")["quantity"].sum().mean()
        diff_pct = ((avg_wk_fc - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
        kpi(f"{diff_pct:+.1f}%",
            "vs. promedio reciente (últimas 9 sem. históricas)",
            color="green" if diff_pct >= 0 else "red",
            delta="tendencia al alza" if diff_pct >= 0 else "tendencia a la baja",
            delta_up=diff_pct >= 0)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráfico principal: histórico + proyección ─────────────────────────────
    st.markdown('<div class="section-header">Ventas históricas vs. proyección</div>',
                unsafe_allow_html=True)

    chart_hist_start = _weeks_ago(int(sellin["year_week"].max()), 24)
    hist_data = hist_filt[hist_filt["year_week"] >= chart_hist_start].groupby("year_week")["quantity"].sum().reset_index()
    hist_data = hist_data.rename(columns={"quantity": "valor"}).assign(tipo="Histórico")

    fc_agg = fc_filt.groupby("year_week")["forecast_naive"].sum().reset_index()
    fc_agg = fc_agg.rename(columns={"forecast_naive": "valor"}).assign(tipo="Proyección")

    combined = pd.concat([hist_data, fc_agg])
    combined["semana"] = combined["year_week"].astype(str)

    last_hist_wk = str(int(hist_data["year_week"].max()))

    color_sc = alt.Scale(domain=["Histórico", "Proyección"],
                          range=["#60a5fa", "#0e9f6e"])
    dash_sc  = alt.Scale(domain=["Histórico", "Proyección"],
                          range=[[1, 0], [5, 3]])

    main_chart = (
        alt.Chart(combined)
        .mark_line(strokeWidth=3, point=alt.OverlayMarkDef(size=60))
        .encode(
            x=alt.X("semana:O", sort=None, title="Semana",
                     axis=alt.Axis(labelAngle=-45, labelOverlap="greedy")),
            y=alt.Y("valor:Q", title="Unidades"),
            color=alt.Color("tipo:N", scale=color_sc,
                             legend=alt.Legend(orient="top", title="")),
            strokeDash=alt.StrokeDash("tipo:N", scale=dash_sc),
            tooltip=["semana:O", "tipo:N",
                     alt.Tooltip("valor:Q", format=",.0f", title="Unidades")],
        )
        .properties(height=320)
        .interactive()
    )

    rule = (
        alt.Chart(pd.DataFrame({"semana": [last_hist_wk]}))
        .mark_rule(color="#e02424", strokeDash=[4, 4], strokeWidth=2)
        .encode(x="semana:O")
    )

    st.altair_chart(main_chart + rule, width='stretch')
    st.caption("La línea roja punteada marca el límite entre datos reales y proyección.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top 10 productos con mayor demanda estimada ───────────────────────────
    st.markdown('<div class="section-header">Productos con mayor demanda estimada</div>',
                unsafe_allow_html=True)

    top10 = (
        forecasts.groupby(["Channel", "Material Description"])["forecast_naive"]
        .sum().nlargest(10).reset_index().sort_values("forecast_naive")
        .rename(columns={"forecast_naive": "Unidades proyectadas"})
    )
    top10["Producto"] = top10["Channel"].str[:8] + " · " + top10["Material Description"].str[:30]

    bar_top = (
        alt.Chart(top10)
        .mark_bar(color="#0e9f6e", cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("Producto:N", sort="-x", title=""),
            x=alt.X("Unidades proyectadas:Q", title="Unidades estimadas (W34·2025–W52·2026)"),
            tooltip=["Channel:N", "Material Description:N",
                     alt.Tooltip("Unidades proyectadas:Q", format=",.0f")],
        )
        .properties(height=340)
    )
    st.altair_chart(bar_top, width='stretch')

    # ── Tabla detalle ─────────────────────────────────────────────────────────
    with st.expander("📋 Ver tabla completa de proyecciones por semana"):
        show = (
            fc_filt[["Channel", "Material Description", "year_week", "horizon_step", "forecast_naive"]]
            .rename(columns={
                "Channel": "Cliente",
                "Material Description": "Producto",
                "year_week": "Semana",
                "horizon_step": "Horizonte",
                "forecast_naive": "Unidades proyectadas",
            })
            .sort_values(["Cliente", "Producto", "Semana"])
        )
        st.dataframe(show, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — ALERTAS DE INVENTARIO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨  Alertas de Inventario":
    st.markdown("# 🚨 Alertas de Inventario")
    st.markdown("Productos cuyo inventario actual **no cubre** la demanda proyectada.")
    st.markdown("---")

    risk    = load_risk()
    risk_rep = load_report("inventory_risk_report")
    if risk is None:
        st.warning("El análisis de inventario aún no está disponible.")
        st.stop()

    dist = risk_rep.get("risk_distribution", {})

    # ── Semáforo de riesgo ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Estado general del inventario</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(f"{dist.get('CRITICAL',0):,}", "🔴  Crítico — < 2 semanas", color="red")
    with c2:
        kpi(f"{dist.get('HIGH',0):,}", "🟠  Alerta — 2 a 4 semanas", color="amber")
    with c3:
        kpi(f"{dist.get('MEDIUM',0):,}", "🟡  Revisar — 4 a 8 semanas")
    with c4:
        kpi(f"{dist.get('LOW',0):,}", "🟢  Estable — más de 8 semanas", color="green")

    pct = risk_rep.get("pct_at_risk", 0)
    st.markdown("<br>", unsafe_allow_html=True)
    if pct > 15:
        st.error(f"⚠️ **{pct}%** del catálogo requiere reabastecimiento urgente (Crítico + Alerta).")
    elif pct > 5:
        st.warning(f"⚠️ **{pct}%** del catálogo está en zona de riesgo.")
    else:
        st.success(f"✅ Solo el **{pct}%** del catálogo está en riesgo.")

    st.markdown("---")

    # ── Gráfico de cobertura ──────────────────────────────────────────────────
    col_g1, col_g2 = st.columns([1, 2])

    with col_g1:
        st.markdown('<div class="section-header">Distribución de riesgo</div>',
                    unsafe_allow_html=True)
        pie_df = pd.DataFrame([
            {"Nivel": "🔴 Crítico",  "Productos": dist.get("CRITICAL", 0), "_order": 0},
            {"Nivel": "🟠 Alerta",   "Productos": dist.get("HIGH",     0), "_order": 1},
            {"Nivel": "🟡 Revisar",  "Productos": dist.get("MEDIUM",   0), "_order": 2},
            {"Nivel": "🟢 Estable",  "Productos": dist.get("LOW",      0), "_order": 3},
        ])
        donut = (
            alt.Chart(pie_df)
            .mark_arc(innerRadius=55, outerRadius=90)
            .encode(
                theta=alt.Theta("Productos:Q"),
                color=alt.Color("Nivel:N",
                                scale=alt.Scale(
                                    domain=["🔴 Crítico","🟠 Alerta","🟡 Revisar","🟢 Estable"],
                                    range=["#e02424","#c27803","#e3a008","#0e9f6e"]),
                                legend=alt.Legend(orient="bottom", title="")),
                order=alt.Order("_order:Q"),
                tooltip=["Nivel:N", "Productos:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(donut, width='stretch')

    with col_g2:
        st.markdown('<div class="section-header">Semanas de cobertura por producto</div>',
                    unsafe_allow_html=True)
        wos_plot = risk[risk["weeks_of_supply"] <= 20].copy()
        wos_plot["riesgo"] = wos_plot["weeks_of_supply"].apply(
            lambda x: "Crítico" if x < 2 else ("Alerta" if x < 4 else "Estable")
        )
        hist_wos = (
            alt.Chart(wos_plot)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("weeks_of_supply:Q",
                         bin=alt.Bin(maxbins=25),
                         title="Semanas de cobertura de inventario"),
                y=alt.Y("count():Q", title="Número de productos"),
                color=alt.Color("riesgo:N",
                                scale=alt.Scale(
                                    domain=["Crítico", "Alerta", "Estable"],
                                    range=["#e02424", "#c27803", "#0e9f6e"]),
                                legend=None),
                tooltip=["count():Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(hist_wos, width='stretch')

    st.markdown("---")

    # ── Lista de productos críticos ───────────────────────────────────────────
    st.markdown('<div class="section-header">🔴 Productos críticos — acción inmediata requerida</div>',
                unsafe_allow_html=True)

    col_filt1, col_filt2 = st.columns(2)
    nivel_sel = col_filt1.selectbox(
        "Nivel de riesgo",
        ["Crítico y Alerta", "Solo Crítico", "Solo Alerta", "Todos"],
        key="risk_nivel"
    )
    cliente_sel = col_filt2.selectbox(
        "Cliente",
        ["Todos los clientes"] + sorted(risk["Channel"].unique()),
        key="risk_cl"
    )

    nivel_map = {
        "Crítico y Alerta": ["CRITICAL", "HIGH"],
        "Solo Crítico":     ["CRITICAL"],
        "Solo Alerta":      ["HIGH"],
        "Todos":            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    }
    risk_view = risk[risk["risk_level"].isin(nivel_map[nivel_sel])].copy()
    if cliente_sel != "Todos los clientes":
        risk_view = risk_view[risk_view["Channel"] == cliente_sel]

    # KPIs de resumen
    deficit_total = (risk_view["forecast_total"] - risk_view["current_inventory"]).clip(lower=0).sum()
    crit_count    = (risk_view["risk_level"] == "CRITICAL").sum()
    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        kpi(f"{len(risk_view):,}", "Productos en esta vista")
    with c_s2:
        kpi(f"{deficit_total:,.0f}", "Unidades totales a comprar", color="red")
    with c_s3:
        kpi(f"{crit_count:,}", "Requieren acción inmediata",
            color="red" if crit_count > 0 else "green")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabla de reabastecimiento
    nivel_emoji = {
        "CRITICAL": "🔴 Crítico",
        "HIGH":     "🟠 Alerta",
        "MEDIUM":   "🟡 Revisar",
        "LOW":      "🟢 Estable",
    }

    def fmt_quiebre(sw):
        try:
            return _fmt_yw(sw) if sw and not pd.isna(sw) else "Inminente"
        except Exception:
            return "—"

    tabla = risk_view.copy()
    tabla["Nivel"]            = tabla["risk_level"].map(nivel_emoji)
    tabla["Cliente"]          = tabla["Channel"]
    tabla["Producto"]         = tabla["Material Description"].str[:35]
    tabla["A comprar (uds)"]  = (tabla["forecast_total"] - tabla["current_inventory"]).clip(lower=0).round(0).astype(int)
    tabla["Cobertura (sem)"]  = tabla["weeks_of_supply"].round(1)
    tabla["Quiebre"]          = tabla["stockout_week"].apply(fmt_quiebre)
    tabla = tabla[["Nivel", "Cliente", "Producto", "A comprar (uds)", "Cobertura (sem)", "Quiebre"]]

    def _color_row(row):
        if "Crítico" in str(row["Nivel"]):
            bg = "background-color: #fff5f5"
        elif "Alerta" in str(row["Nivel"]):
            bg = "background-color: #fffbf0"
        else:
            bg = ""
        return [bg] * len(row)

    styled = (
        tabla.style
        .apply(_color_row, axis=1)
        .format({"A comprar (uds)": "{:,.0f}", "Cobertura (sem)": "{:.1f}"})
        .hide(axis="index")
    )
    st.dataframe(styled, width='stretch', height=450)
