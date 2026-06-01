"""
Dashboard Enterprise — Pronóstico de Demanda e Inventario · Samsung Colombia
Hackathon SIC2025 · Análisis predictivo para dirección comercial y operativa.

Modelo: LightGBM Two-Stage (clasificador de demanda + regresor) + Seasonal Naïve.
Cobertura: 17,001 SKUs · 98 clientes · horizonte W34·2025 → W33·2026 (52 semanas).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import theme as T

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pronóstico de Demanda · Samsung",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(T.inject_css(), unsafe_allow_html=True)

# ─── Rutas ────────────────────────────────────────────────────────────────────
SILVER_PATH   = Path("data/silver/silver_dataset.parquet")
SELLIN_PATH   = Path("data/forecasts/sellin_dashboard.parquet")  # pre-filtrado Sell-in
FC_PATH       = Path("data/forecasts/forecasts.parquet")
BLENDED_PATH  = Path("data/forecasts/loop8_blended_forecasts.parquet")
RISK_PATH     = Path("data/forecasts/inventory_risk.parquet")
CHURN_PATH    = Path("data/forecasts/churn_analysis.parquet")
REPORTS_DIR   = Path("reports")


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
def _norm_yw(df: pd.DataFrame) -> pd.DataFrame:
    if "year_week" in df.columns:
        df["year_week"] = (
            df["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
        )
    return df


@st.cache_data(show_spinner=False, ttl=300, max_entries=2)
def load_silver() -> pd.DataFrame:
    """Carga solo Sell-in con dtypes optimizados (~29 MB RAM en vez de 1.3 GB).

    El dashboard solo usa la categoría Sell-in. Se usa el parquet pre-filtrado
    si existe; de lo contrario se filtra el silver completo (más pesado).
    """
    if SELLIN_PATH.exists():
        df = pd.read_parquet(SELLIN_PATH)
    else:
        full = pd.read_parquet(SILVER_PATH, columns=["Channel", "Material Description",
                                                     "Category", "year_week", "quantity"])
        df = full[full["Category"] == "Sell-in"].copy()
        del full
    df = _norm_yw(df)
    # Optimización de memoria: categóricas + float32
    for col in ("Channel", "Material Description"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype("float32")
    if "Category" not in df.columns:
        df["Category"] = "Sell-in"
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_blended() -> pd.DataFrame | None:
    if BLENDED_PATH.exists():
        return _norm_yw(pd.read_parquet(BLENDED_PATH))
    if FC_PATH.exists():
        df = _norm_yw(pd.read_parquet(FC_PATH))
        return df.rename(columns={"forecast_naive": "forecast_q50"})
    return None


@st.cache_data(show_spinner=False, ttl=300)
def load_risk() -> pd.DataFrame | None:
    return pd.read_parquet(RISK_PATH) if RISK_PATH.exists() else None


@st.cache_data(show_spinner=False, ttl=300)
def load_churn() -> pd.DataFrame | None:
    return pd.read_parquet(CHURN_PATH) if CHURN_PATH.exists() else None


@st.cache_data(show_spinner=False, ttl=300)
def load_report(name: str) -> dict:
    p = REPORTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_yw(yw) -> str:
    try:
        v = int(yw)
        return f"W{v % 100:02d}·{v // 100}"
    except Exception:
        return "—"


def weeks_ago(yw: int, n: int) -> int:
    yr, wk = yw // 100, yw % 100
    d = date.fromisocalendar(yr, wk, 1) - timedelta(weeks=n)
    iso = d.isocalendar()
    return iso[0] * 100 + iso[1]


def human(n: float, dec: int = 0) -> str:
    if abs(n) >= 1e6:
        return f"{n/1e6:.{max(dec,1)}f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.{dec}f}K"
    return f"{n:,.0f}"


def base_chart(data, height: int = 300):
    return (
        alt.Chart(data)
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            grid=True, gridColor=T.CHART_GRID, domainColor=T.CHART_GRID,
            labelColor=T.SLATE, titleColor=T.SLATE, labelFontSize=11,
            titleFontSize=12, titleFontWeight="normal",
        )
        .configure_legend(labelColor=T.SLATE, titleColor=T.INK, labelFontSize=11)
    )


def kpi_row(cards: list[str]):
    cols = st.columns(len(cards))
    for col, html in zip(cols, cards):
        col.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATOS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
silver   = load_silver()
blended  = load_blended()
risk     = load_risk()
churn    = load_churn()
cv_rep   = load_report("loop6_cv_report")

sellin = silver[silver["Category"] == "Sell-in"].copy()
LAST_REAL_WK = int(sellin[sellin["quantity"] > 0]["year_week"].max())

# Métricas del modelo (CV)
_agg = cv_rep.get("aggregate_metrics", {})
M_MASE = _agg.get("mase", {}).get("mean")
M_AWAPE = _agg.get("active_wape", {}).get("mean")
M_F1 = _agg.get("demand_f1", {}).get("mean")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:8px 4px 18px 4px;">
          <div style="font-size:1.35rem;font-weight:800;color:#fff;letter-spacing:-0.5px;">
            📊 Demand IQ
          </div>
          <div style="font-size:0.8rem;color:#9fb3d9;margin-top:2px;">
            Samsung Colombia · SIC2025
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navegación",
        ["🏠  Resumen Ejecutivo",
         "📈  Comportamiento de Ventas",
         "🔮  Proyección de Demanda",
         "🚨  Alertas de Inventario",
         "👥  Análisis de Clientes"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background:#ffffff0d;border:1px solid #ffffff1a;border-radius:12px;
                    padding:14px 16px;font-size:0.8rem;color:#c7d6f5;">
          <div style="font-weight:700;color:#fff;margin-bottom:6px;">🧠 Modelo activo</div>
          LightGBM Two-Stage<br>
          <span style="color:#9fb3d9">MASE</span>
            <b style="color:#5ee0a8">{M_MASE:.3f}</b> ·
          <span style="color:#9fb3d9">F1</span>
            <b style="color:#5ee0a8">{M_F1:.2f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("🔄  Actualizar datos", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        f"""
        <div style="font-size:0.72rem;color:#7e93bd;margin-top:14px;line-height:1.5;">
          Datos reales hasta <b style="color:#c7d6f5">{fmt_yw(LAST_REAL_WK)}</b><br>
          Pronóstico hasta <b style="color:#c7d6f5">W33·2026</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
if page.startswith("🏠"):
    st.markdown(
        """
        <div class="hero">
          <h1>Resumen Ejecutivo</h1>
          <p>Visión integral del negocio: demanda proyectada, riesgo de inventario y salud de clientes.</p>
          <span class="pill">📦 17,001 productos</span>
          <span class="pill">🏢 98 clientes</span>
          <span class="pill">🗓️ 52 semanas de pronóstico</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_skus = sellin[["Channel", "Material Description"]].drop_duplicates().shape[0]
    n_ch = sellin["Channel"].nunique()

    # Tendencia: últimas 4 semanas vs mismas 4 del año anterior
    l4_start = weeks_ago(LAST_REAL_WK, 3)
    py_end = weeks_ago(LAST_REAL_WK, 52)
    py_start = weeks_ago(LAST_REAL_WK, 55)
    last4 = sellin[sellin["year_week"] >= l4_start]["quantity"].sum()
    prev4 = sellin[(sellin["year_week"] >= py_start) & (sellin["year_week"] <= py_end)]["quantity"].sum()
    trend = ((last4 - prev4) / prev4 * 100) if prev4 > 0 else 0

    total_fc = blended["forecast_q50"].sum() if blended is not None else 0
    crit_n = int((risk["risk_level"] == "CRITICAL").sum()) if risk is not None else 0
    churn_alto = int((churn["churn_risk"] == "ALTO").sum()) if churn is not None else 0

    kpi_row([
        T.kpi_card(f"{n_skus:,}", "Productos monitoreados", "SKUs activos", bar_color=T.BLUE),
        T.kpi_card(f"{n_ch}", "Clientes", "canales de venta", bar_color=T.BLUE),
        T.kpi_card(human(total_fc, 2), "Demanda proyectada", "próximas 52 semanas",
                   delta="Q50 · modelo ML", delta_dir="neutral", bar_color=T.GREEN),
        T.kpi_card(f"{crit_n}", "SKUs en riesgo crítico", "quiebre inminente",
                   delta="acción urgente", delta_dir="down", bar_color=T.RED),
    ])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Insights ──
    st.markdown(T.section("💡 Lo que la dirección debe saber"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        dir_txt = "al alza 📈" if trend >= 0 else "a la baja 📉"
        kind = "good" if trend >= 0 else "warn"
        st.markdown(
            T.insight(
                f"Las ventas de las últimas 4 semanas muestran una tendencia <b>{dir_txt}</b> "
                f"de <b>{trend:+.1f}%</b> frente al mismo período del año anterior.",
                kind=kind, head="Tendencia comercial:"),
            unsafe_allow_html=True)

        if blended is not None:
            wk_avg = blended.groupby("year_week")["forecast_q50"].sum().mean()
            st.markdown(
                T.insight(
                    f"Se proyecta una demanda promedio de <b>{wk_avg:,.0f} unidades por semana</b> "
                    f"durante las próximas 52 semanas, según el modelo predictivo.",
                    head="Proyección de demanda:"),
                unsafe_allow_html=True)
    with c2:
        if crit_n > 0:
            st.markdown(
                T.insight(
                    f"<b>{crit_n} productos</b> están en riesgo crítico de quiebre de stock "
                    f"(menos de 2 semanas de cobertura bajo escenario conservador Q90). "
                    f"Requieren reposición inmediata.",
                    kind="alert", head="Alerta de inventario:"),
                unsafe_allow_html=True)
        if churn_alto > 0:
            st.markdown(
                T.insight(
                    f"<b>{churn_alto} clientes</b> presentan riesgo alto de abandono "
                    f"(más de 26 semanas sin comprar). El área comercial debería activar "
                    f"un plan de retención.",
                    kind="warn", head="Salud de clientes:"),
                unsafe_allow_html=True)

    # ── Gráfico tendencia + proyección ──
    st.markdown(T.section("📊 Tendencia histórica y proyección"), unsafe_allow_html=True)
    hist_start = weeks_ago(LAST_REAL_WK, 26)
    hist = (sellin[sellin["year_week"] >= hist_start]
            .groupby("year_week")["quantity"].sum().reset_index()
            .rename(columns={"quantity": "valor"}))
    hist["tipo"] = "Histórico"
    hist["idx"] = range(len(hist))

    if blended is not None:
        fc = (blended.groupby("year_week")["forecast_q50"].sum().reset_index()
              .rename(columns={"forecast_q50": "valor"}))
        fc["tipo"] = "Proyección"
        fc["idx"] = range(len(hist), len(hist) + len(fc))
        combo = pd.concat([hist, fc])
        ci = (blended.groupby("year_week")[["forecast_q10", "forecast_q90"]].sum().reset_index())
        ci["idx"] = range(len(hist), len(hist) + len(ci))
    else:
        combo = hist
        ci = None

    combo["semana"] = combo["year_week"].apply(fmt_yw)

    layers = []
    if ci is not None:
        band = (alt.Chart(ci).mark_area(opacity=0.16, color=T.CHART_GREEN)
                .encode(x=alt.X("idx:Q", axis=None),
                        y=alt.Y("forecast_q10:Q", title="Unidades"),
                        y2="forecast_q90:Q"))
        layers.append(band)

    line = (alt.Chart(combo).mark_line(strokeWidth=2.5, point=False)
            .encode(
                x=alt.X("idx:Q", axis=alt.Axis(title="Semana", labels=False, ticks=False)),
                y=alt.Y("valor:Q", title="Unidades"),
                color=alt.Color("tipo:N",
                                scale=alt.Scale(domain=["Histórico", "Proyección"],
                                                range=[T.CHART_BLUE, T.CHART_GREEN]),
                                legend=alt.Legend(orient="top", title=None)),
                strokeDash=alt.StrokeDash("tipo:N",
                                          scale=alt.Scale(domain=["Histórico", "Proyección"],
                                                          range=[[1, 0], [6, 4]]),
                                          legend=None),
                tooltip=[alt.Tooltip("semana:N", title="Semana"),
                         alt.Tooltip("tipo:N", title="Tipo"),
                         alt.Tooltip("valor:Q", title="Unidades", format=",.0f")]))
    layers.append(line)

    chart = base_chart(combo, 320)
    final = alt.layer(*layers).resolve_scale(y="shared")
    st.altair_chart(
        final.properties(height=320).configure_view(strokeWidth=0).configure_axis(
            grid=True, gridColor=T.CHART_GRID, domainColor=T.CHART_GRID,
            labelColor=T.SLATE, titleColor=T.SLATE).configure_legend(
            labelColor=T.SLATE, labelFontSize=12),
        width='stretch')
    st.caption("La banda verde representa el intervalo de confianza Q10–Q90 del modelo. "
               "La línea punteada es la proyección a 52 semanas.")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — COMPORTAMIENTO DE VENTAS
# ══════════════════════════════════════════════════════════════════════════════
elif page.startswith("📈"):
    st.markdown(
        """
        <div class="hero">
          <h1>Comportamiento de Ventas</h1>
          <p>Análisis histórico de la demanda 2023–2025 por temporada, producto y cliente.</p>
        </div>
        """, unsafe_allow_html=True)

    sellin2 = sellin.copy()
    sellin2["year"] = sellin2["year_week"] // 100
    sellin2["week"] = sellin2["year_week"] % 100

    annual = sellin2.groupby("year")["quantity"].sum()
    yoy = annual.pct_change().mul(100)

    cards = []
    bar_cols = [T.BLUE, T.BLUE, T.GREEN]
    for i, yr in enumerate(annual.index):
        delta = f"{yoy[yr]:+.1f}% YoY" if yr in yoy and pd.notna(yoy[yr]) else "año base"
        ddir = "up" if (yr in yoy and pd.notna(yoy[yr]) and yoy[yr] >= 0) else \
               ("down" if (yr in yoy and pd.notna(yoy[yr])) else "neutral")
        cards.append(T.kpi_card(human(annual[yr], 2), f"Ventas {yr}", "unidades Sell-in",
                                delta=delta, delta_dir=ddir, bar_color=bar_cols[i % 3]))
    kpi_row(cards)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── YoY por semana ──
    st.markdown(T.section("📅 Estacionalidad — comparación año contra año"), unsafe_allow_html=True)
    yoy_df = (sellin2.groupby(["year", "week"])["quantity"].sum().reset_index())
    yoy_df["year"] = yoy_df["year"].astype(str)
    yoy_chart = (alt.Chart(yoy_df).mark_line(strokeWidth=2, opacity=0.9)
                 .encode(
                     x=alt.X("week:Q", title="Semana del año"),
                     y=alt.Y("quantity:Q", title="Unidades vendidas"),
                     color=alt.Color("year:N", title="Año",
                                     scale=alt.Scale(scheme="blues")),
                     tooltip=["year:N", "week:Q",
                              alt.Tooltip("quantity:Q", format=",.0f")]))
    st.altair_chart(
        yoy_chart.properties(height=300).configure_view(strokeWidth=0).configure_axis(
            grid=True, gridColor=T.CHART_GRID, domainColor=T.CHART_GRID,
            labelColor=T.SLATE, titleColor=T.SLATE).configure_legend(labelColor=T.SLATE),
        width='stretch')

    # ── Top productos y canales ──
    ca, cb = st.columns(2)
    with ca:
        st.markdown(T.section("🏆 Top 10 productos"), unsafe_allow_html=True)
        topp = (sellin.groupby("Material Description", observed=True)["quantity"].sum()
                .nlargest(10).reset_index())
        topp["short"] = topp["Material Description"].str[:34]
        ch = (alt.Chart(topp).mark_bar(color=T.CHART_BLUE, cornerRadiusEnd=4)
              .encode(
                  x=alt.X("quantity:Q", title="Unidades"),
                  y=alt.Y("short:N", sort="-x", title=None),
                  tooltip=[alt.Tooltip("Material Description:N", title="Producto"),
                           alt.Tooltip("quantity:Q", format=",.0f")]))
        st.altair_chart(
            ch.properties(height=320).configure_view(strokeWidth=0).configure_axis(
                grid=True, gridColor=T.CHART_GRID, labelColor=T.SLATE, titleColor=T.SLATE),
            width='stretch')
    with cb:
        st.markdown(T.section("🏢 Top 10 clientes"), unsafe_allow_html=True)
        topc = (sellin.groupby("Channel", observed=True)["quantity"].sum().nlargest(10).reset_index())
        ch2 = (alt.Chart(topc).mark_bar(color=T.CHART_GREEN, cornerRadiusEnd=4)
               .encode(
                   x=alt.X("quantity:Q", title="Unidades"),
                   y=alt.Y("Channel:N", sort="-x", title=None),
                   tooltip=["Channel:N", alt.Tooltip("quantity:Q", format=",.0f")]))
        st.altair_chart(
            ch2.properties(height=320).configure_view(strokeWidth=0).configure_axis(
                grid=True, gridColor=T.CHART_GRID, labelColor=T.SLATE, titleColor=T.SLATE),
            width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — PROYECCIÓN DE DEMANDA
# ══════════════════════════════════════════════════════════════════════════════
elif page.startswith("🔮"):
    st.markdown(
        """
        <div class="hero">
          <h1>Proyección de Demanda</h1>
          <p>Pronóstico semanal W34·2025 → W33·2026 con intervalos de confianza Q10–Q90.</p>
        </div>
        """, unsafe_allow_html=True)

    if blended is None:
        st.warning("Los pronósticos aún no están disponibles.")
        st.stop()

    fc = blended.copy()

    c1, c2 = st.columns(2)
    clientes = ["Todos los clientes"] + sorted(fc["Channel"].unique())
    sel_cl = c1.selectbox("Cliente", clientes)
    fcf = fc if sel_cl == "Todos los clientes" else fc[fc["Channel"] == sel_cl]
    productos = ["Todos los productos"] + sorted(fcf["Material Description"].unique())
    sel_pr = c2.selectbox("Producto", productos)
    if sel_pr != "Todos los productos":
        fcf = fcf[fcf["Material Description"] == sel_pr]

    total = fcf["forecast_q50"].sum()
    avg_wk = fcf.groupby("year_week")["forecast_q50"].sum().mean()

    kpi_row([
        T.kpi_card(human(total, 2), "Demanda total proyectada", "52 semanas (Q50)",
                   bar_color=T.GREEN),
        T.kpi_card(f"{avg_wk:,.0f}", "Promedio semanal", "unidades/semana", bar_color=T.BLUE),
        T.kpi_card(f"{M_AWAPE:.1f}%", "Error del modelo", "active-WAPE (validación CV)",
                   delta="5-fold CV", delta_dir="neutral", bar_color=T.AMBER),
        T.kpi_card(f"{M_MASE:.3f}", "MASE", "< 1.0 supera al baseline",
                   delta="3.4× mejor", delta_dir="up", bar_color=T.GREEN),
    ])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(T.section("📈 Proyección con intervalo de confianza"), unsafe_allow_html=True)

    wk = (fcf.groupby("year_week")[["forecast_q10", "forecast_q50", "forecast_q90"]]
          .sum().reset_index())
    wk["idx"] = range(len(wk))
    wk["semana"] = wk["year_week"].apply(fmt_yw)

    band = (alt.Chart(wk).mark_area(opacity=0.18, color=T.CHART_GREEN)
            .encode(x=alt.X("idx:Q", axis=alt.Axis(title="Semana", labels=False, ticks=False)),
                    y=alt.Y("forecast_q10:Q", title="Unidades"), y2="forecast_q90:Q"))
    line = (alt.Chart(wk).mark_line(strokeWidth=2.5, color=T.CHART_GREEN)
            .encode(x="idx:Q", y="forecast_q50:Q",
                    tooltip=[alt.Tooltip("semana:N", title="Semana"),
                             alt.Tooltip("forecast_q10:Q", title="Q10 (pesimista)", format=",.0f"),
                             alt.Tooltip("forecast_q50:Q", title="Q50 (esperado)", format=",.0f"),
                             alt.Tooltip("forecast_q90:Q", title="Q90 (optimista)", format=",.0f")]))
    st.altair_chart(
        alt.layer(band, line).properties(height=330).configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor=T.CHART_GRID, domainColor=T.CHART_GRID,
                        labelColor=T.SLATE, titleColor=T.SLATE),
        width='stretch')
    st.caption("Banda verde = rango de confianza Q10 (escenario pesimista) a Q90 (optimista). "
               "Línea = pronóstico esperado Q50.")

    # ── Top productos proyectados ──
    st.markdown(T.section("🎯 Productos con mayor demanda estimada"), unsafe_allow_html=True)
    topf = (fc.groupby("Material Description")["forecast_q50"].sum()
            .nlargest(12).reset_index())
    topf["short"] = topf["Material Description"].str[:42]
    chf = (alt.Chart(topf).mark_bar(color=T.CHART_GREEN, cornerRadiusEnd=4)
           .encode(x=alt.X("forecast_q50:Q", title="Unidades estimadas (52 sem)"),
                   y=alt.Y("short:N", sort="-x", title=None),
                   tooltip=[alt.Tooltip("Material Description:N", title="Producto"),
                            alt.Tooltip("forecast_q50:Q", format=",.0f")]))
    st.altair_chart(
        chf.properties(height=360).configure_view(strokeWidth=0).configure_axis(
            grid=True, gridColor=T.CHART_GRID, labelColor=T.SLATE, titleColor=T.SLATE),
        width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — ALERTAS DE INVENTARIO
# ══════════════════════════════════════════════════════════════════════════════
elif page.startswith("🚨"):
    st.markdown(
        """
        <div class="hero">
          <h1>Alertas de Inventario</h1>
          <p>Riesgo de quiebre de stock bajo escenario conservador de demanda (Q90).</p>
        </div>
        """, unsafe_allow_html=True)

    if risk is None:
        st.warning("El análisis de inventario aún no está disponible.")
        st.stop()

    rv = risk.copy()
    clientes = ["Todos los clientes"] + sorted(rv["Channel"].unique())
    sel = st.selectbox("Filtrar por cliente", clientes)
    if sel != "Todos los clientes":
        rv = rv[rv["Channel"] == sel]

    dist = rv["risk_level"].value_counts()
    crit = int(dist.get("CRITICAL", 0))
    high = int(dist.get("HIGH", 0))
    deficit = (rv["forecast_q90_total"] - rv["current_inventory"]).clip(lower=0).sum()
    cov_med = rv["weeks_of_supply_conservative"].median()

    kpi_row([
        T.kpi_card(f"{crit}", "Riesgo CRÍTICO", "< 2 semanas cobertura",
                   delta="acción inmediata", delta_dir="down", bar_color=T.RED),
        T.kpi_card(f"{high}", "Riesgo ALTO", "2–4 semanas cobertura",
                   delta="monitorear", delta_dir="down", bar_color=T.AMBER),
        T.kpi_card(human(deficit, 1), "Déficit estimado", "unidades a reponer",
                   bar_color=T.BLUE),
        T.kpi_card(f"{cov_med:.1f}", "Cobertura mediana", "semanas (escenario Q90)",
                   bar_color=T.GREEN),
    ])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    ca, cb = st.columns([1, 1.4])
    with ca:
        st.markdown(T.section("🎚️ Distribución de riesgo"), unsafe_allow_html=True)
        dist_df = dist.reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0).reset_index()
        dist_df.columns = ["nivel", "n"]
        donut = (alt.Chart(dist_df).mark_arc(innerRadius=60, cornerRadius=3)
                 .encode(
                     theta="n:Q",
                     color=alt.Color("nivel:N",
                                     scale=alt.Scale(domain=list(T.RISK_COLORS.keys()),
                                                     range=list(T.RISK_COLORS.values())),
                                     legend=alt.Legend(title="Nivel", orient="bottom")),
                     tooltip=["nivel:N", alt.Tooltip("n:Q", title="SKUs")]))
        st.altair_chart(
            donut.properties(height=300).configure_view(strokeWidth=0)
            .configure_legend(labelColor=T.SLATE, titleColor=T.INK),
            width='stretch')
    with cb:
        st.markdown(T.section("🏢 SKUs en riesgo por cliente"), unsafe_allow_html=True)
        ch_risk = (risk[risk["risk_level"].isin(["CRITICAL", "HIGH"])]
                   .groupby("Channel").size().nlargest(10).reset_index(name="n"))
        if len(ch_risk):
            chb = (alt.Chart(ch_risk).mark_bar(color=T.CHART_RED, cornerRadiusEnd=4)
                   .encode(x=alt.X("n:Q", title="SKUs en riesgo"),
                           y=alt.Y("Channel:N", sort="-x", title=None),
                           tooltip=["Channel:N", alt.Tooltip("n:Q", title="SKUs")]))
            st.altair_chart(
                chb.properties(height=300).configure_view(strokeWidth=0).configure_axis(
                    grid=True, gridColor=T.CHART_GRID, labelColor=T.SLATE, titleColor=T.SLATE),
                width='stretch')

    # ── Tabla de críticos ──
    st.markdown(T.section("📋 Productos que requieren reposición urgente"), unsafe_allow_html=True)
    crit_tbl = rv[rv["risk_level"].isin(["CRITICAL", "HIGH"])].copy()
    crit_tbl = crit_tbl.sort_values("weeks_of_supply_conservative").head(20)
    crit_tbl["A comprar (uds)"] = (crit_tbl["forecast_q90_total"]
                                   - crit_tbl["current_inventory"]).clip(lower=0).round(0).astype(int)
    crit_tbl["Quiebre est."] = crit_tbl["stockout_week"].apply(
        lambda x: fmt_yw(x) if pd.notna(x) else "Inminente")
    show = crit_tbl[["risk_level", "Channel", "Material Description",
                     "current_inventory", "weeks_of_supply_conservative",
                     "A comprar (uds)", "Quiebre est."]].copy()
    show.columns = ["Nivel", "Cliente", "Producto", "Inv. actual",
                    "Cobertura (sem)", "A comprar (uds)", "Quiebre est."]
    show["Inv. actual"] = show["Inv. actual"].round(0).astype(int)
    show["Cobertura (sem)"] = show["Cobertura (sem)"].round(2)
    show["Producto"] = show["Producto"].str[:42]

    st.dataframe(
        show, width='stretch', hide_index=True,
        column_config={
            "Nivel": st.column_config.TextColumn(width="small"),
            "Cobertura (sem)": st.column_config.NumberColumn(format="%.2f"),
        })


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — ANÁLISIS DE CLIENTES (CHURN)
# ══════════════════════════════════════════════════════════════════════════════
elif page.startswith("👥"):
    st.markdown(
        """
        <div class="hero">
          <h1>Análisis de Clientes</h1>
          <p>Riesgo de abandono (churn) y comportamiento de compra por cliente.</p>
        </div>
        """, unsafe_allow_html=True)

    if churn is None:
        st.warning("El análisis de clientes aún no está disponible.")
        st.stop()

    dist = churn["churn_risk"].value_counts()
    alto = int(dist.get("ALTO", 0))
    medio = int(dist.get("MEDIO", 0))
    activo = int(dist.get("ACTIVO", 0))
    total_cl = len(churn)

    kpi_row([
        T.kpi_card(f"{total_cl}", "Clientes analizados", "con historial de compra",
                   bar_color=T.BLUE),
        T.kpi_card(f"{alto}", "Riesgo ALTO", f"{alto/total_cl*100:.0f}% del total",
                   delta="retención urgente", delta_dir="down", bar_color=T.RED),
        T.kpi_card(f"{medio}", "Riesgo MEDIO", "tendencia a la baja",
                   delta="monitorear", delta_dir="down", bar_color=T.AMBER),
        T.kpi_card(f"{activo}", "Clientes activos", f"{activo/total_cl*100:.0f}% estables",
                   delta="saludables", delta_dir="up", bar_color=T.GREEN),
    ])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    ca, cb = st.columns([1, 1.4])
    with ca:
        st.markdown(T.section("🎚️ Distribución de riesgo"), unsafe_allow_html=True)
        dd = dist.reindex(["ALTO", "MEDIO", "BAJO", "ACTIVO"]).dropna().reset_index()
        dd.columns = ["nivel", "n"]
        donut = (alt.Chart(dd).mark_arc(innerRadius=60, cornerRadius=3)
                 .encode(
                     theta="n:Q",
                     color=alt.Color("nivel:N",
                                     scale=alt.Scale(domain=list(T.CHURN_COLORS.keys()),
                                                     range=list(T.CHURN_COLORS.values())),
                                     legend=alt.Legend(title="Riesgo", orient="bottom")),
                     tooltip=["nivel:N", alt.Tooltip("n:Q", title="Clientes")]))
        st.altair_chart(
            donut.properties(height=300).configure_view(strokeWidth=0)
            .configure_legend(labelColor=T.SLATE, titleColor=T.INK),
            width='stretch')
    with cb:
        st.markdown(T.section("📉 Semanas sin comprar vs cambio de volumen"), unsafe_allow_html=True)
        sc = churn.copy()
        sc["abs_vol"] = sc["avg_weekly_volume"].clip(lower=1)
        scatter = (alt.Chart(sc).mark_circle(opacity=0.7)
                   .encode(
                       x=alt.X("weeks_since_last_purchase:Q", title="Semanas sin comprar"),
                       y=alt.Y("volume_change_pct:Q", title="Cambio de volumen %"),
                       size=alt.Size("abs_vol:Q", title="Volumen", legend=None,
                                     scale=alt.Scale(range=[30, 400])),
                       color=alt.Color("churn_risk:N",
                                       scale=alt.Scale(domain=list(T.CHURN_COLORS.keys()),
                                                       range=list(T.CHURN_COLORS.values())),
                                       legend=alt.Legend(title="Riesgo", orient="top")),
                       tooltip=[alt.Tooltip("Channel:N", title="Cliente"),
                                alt.Tooltip("weeks_since_last_purchase:Q", title="Sem. sin comprar"),
                                alt.Tooltip("volume_change_pct:Q", title="Cambio vol %", format="+.1f"),
                                alt.Tooltip("churn_risk:N", title="Riesgo")]))
        st.altair_chart(
            scatter.properties(height=300).configure_view(strokeWidth=0).configure_axis(
                grid=True, gridColor=T.CHART_GRID, labelColor=T.SLATE, titleColor=T.SLATE)
            .configure_legend(labelColor=T.SLATE, titleColor=T.INK),
            width='stretch')

    # ── Tabla clientes en riesgo ──
    st.markdown(T.section("📋 Clientes que requieren acción comercial"), unsafe_allow_html=True)
    risk_cl = churn[churn["churn_risk"].isin(["ALTO", "MEDIO"])].copy()
    risk_cl = risk_cl.sort_values(["churn_risk", "weeks_since_last_purchase"],
                                  ascending=[True, False])
    risk_cl["Últ. compra"] = risk_cl["last_purchase_week"].apply(fmt_yw)
    show = risk_cl[["churn_risk", "Channel", "Últ. compra",
                    "weeks_since_last_purchase", "active_weeks_pct",
                    "avg_weekly_volume", "volume_change_pct"]].copy()
    show.columns = ["Riesgo", "Cliente", "Últ. compra", "Sem. sin comprar",
                    "% sem activas", "Vol. promedio", "Cambio vol %"]
    show["% sem activas"] = show["% sem activas"].round(1)
    show["Vol. promedio"] = show["Vol. promedio"].round(1)
    show["Cambio vol %"] = show["Cambio vol %"].round(1)
    st.dataframe(show, width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div class="footer">
      <b>Demand IQ</b> · Pronóstico de Demanda Samsung Colombia · Hackathon SIC2025<br>
      Modelo LightGBM Two-Stage · MASE {M_MASE:.3f} · {len(sellin['Channel'].unique())} clientes ·
      datos hasta {fmt_yw(LAST_REAL_WK)}
    </div>
    """, unsafe_allow_html=True)
