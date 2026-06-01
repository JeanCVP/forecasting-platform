"""
Sistema de diseño enterprise — Pronóstico de Demanda Samsung Colombia
Paleta corporativa, CSS global y helpers de componentes visuales.
"""
from __future__ import annotations

# ─── Paleta corporativa ───────────────────────────────────────────────────────
NAVY        = "#0b1f3a"   # azul corporativo profundo
BLUE        = "#1428a0"   # Samsung blue
BLUE_LIGHT  = "#2f6bff"
ACCENT      = "#00b4d8"   # cian de acento
GREEN       = "#0e9f6e"
AMBER       = "#f59e0b"
RED         = "#e02424"
SLATE       = "#475569"
SLATE_LIGHT = "#94a3b8"
BG          = "#f4f6fb"
CARD_BG     = "#ffffff"
INK         = "#0f172a"

# Escala para gráficos
CHART_BLUE   = "#1428a0"
CHART_GREEN  = "#0e9f6e"
CHART_AMBER  = "#f59e0b"
CHART_RED    = "#e02424"
CHART_GRID   = "#e6eaf2"

RISK_COLORS = {
    "CRITICAL": "#e02424", "HIGH": "#f59e0b", "MEDIUM": "#eab308", "LOW": "#0e9f6e",
}
CHURN_COLORS = {
    "ALTO": "#e02424", "MEDIO": "#f59e0b", "BAJO": "#eab308", "ACTIVO": "#0e9f6e",
}


def inject_css() -> str:
    """CSS global del dashboard enterprise."""
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    .stApp {{ background: {BG}; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, #11295180 100%), {NAVY};
        border-right: 1px solid #ffffff10;
    }}
    section[data-testid="stSidebar"] * {{ color: #e8eef9; }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.95rem; padding: 2px 0;
    }}

    /* ── Header hero ── */
    .hero {{
        background: linear-gradient(110deg, {NAVY} 0%, {BLUE} 70%, {BLUE_LIGHT} 130%);
        border-radius: 18px; padding: 26px 32px; margin-bottom: 22px;
        box-shadow: 0 12px 30px -10px {BLUE}55;
        position: relative; overflow: hidden;
    }}
    .hero::after {{
        content: ""; position: absolute; right: -40px; top: -40px;
        width: 220px; height: 220px; border-radius: 50%;
        background: radial-gradient(circle, {ACCENT}40 0%, transparent 70%);
    }}
    .hero h1 {{ color: #fff; font-size: 1.7rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }}
    .hero p  {{ color: #c7d6f5; font-size: 0.95rem; margin: 6px 0 0 0; }}
    .hero .pill {{
        display: inline-block; background: #ffffff1a; color: #dbe6ff;
        padding: 4px 12px; border-radius: 20px; font-size: 0.78rem;
        font-weight: 600; margin-top: 12px; margin-right: 8px; border: 1px solid #ffffff20;
    }}

    /* ── KPI cards ── */
    .kpi-grid {{ display: grid; gap: 16px; }}
    .kpi {{
        background: {CARD_BG}; border-radius: 16px; padding: 20px 22px;
        box-shadow: 0 4px 18px -8px #0f172a26; border: 1px solid #eef1f7;
        position: relative; transition: transform .15s ease, box-shadow .15s ease;
        height: 100%;
    }}
    .kpi:hover {{ transform: translateY(-3px); box-shadow: 0 14px 28px -12px #0f172a33; }}
    .kpi .label {{ color: {SLATE}; font-size: 0.78rem; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi .value {{ color: {INK}; font-size: 2rem; font-weight: 800; line-height: 1.1;
                   margin: 8px 0 4px 0; letter-spacing: -1px; }}
    .kpi .sub   {{ color: {SLATE_LIGHT}; font-size: 0.8rem; font-weight: 500; }}
    .kpi .delta {{ font-size: 0.82rem; font-weight: 700; padding: 2px 8px;
                   border-radius: 8px; display: inline-block; margin-top: 8px; }}
    .kpi .delta.up   {{ color: {GREEN}; background: {GREEN}14; }}
    .kpi .delta.down {{ color: {RED};   background: {RED}14; }}
    .kpi .delta.neutral {{ color: {SLATE}; background: {SLATE}14; }}
    .kpi .bar {{ position: absolute; left: 0; top: 18px; bottom: 18px; width: 4px;
                 border-radius: 4px; }}

    /* ── Section header ── */
    .section {{
        font-size: 1.05rem; font-weight: 700; color: {INK};
        margin: 26px 0 14px 0; padding-bottom: 8px;
        border-bottom: 2px solid #e6eaf2; display: flex; align-items: center; gap: 10px;
    }}
    .section .dot {{ width: 8px; height: 8px; border-radius: 50%; background: {BLUE}; }}

    /* ── Insight box ── */
    .insight {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border-left: 4px solid {BLUE}; border-radius: 12px;
        padding: 16px 20px; margin-bottom: 14px;
        box-shadow: 0 2px 10px -6px #0f172a22;
    }}
    .insight.alert {{ border-left-color: {RED}; }}
    .insight.warn  {{ border-left-color: {AMBER}; }}
    .insight.good  {{ border-left-color: {GREEN}; }}
    .insight p {{ margin: 0; color: {INK}; font-size: 0.92rem; line-height: 1.5; }}
    .insight .head {{ font-weight: 700; }}

    /* ── Badges ── */
    .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.72rem;
              font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; }}

    /* ── Dataframe polish ── */
    [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden;
        box-shadow: 0 2px 12px -8px #0f172a22; }}

    /* ── Buttons ── */
    .stButton button {{
        background: {BLUE}; color: #fff; border: none; border-radius: 10px;
        font-weight: 600; padding: 8px 16px; transition: background .15s ease;
    }}
    .stButton button:hover {{ background: {BLUE_LIGHT}; color: #fff; }}

    /* ── Footer ── */
    .footer {{ text-align: center; color: {SLATE_LIGHT}; font-size: 0.78rem;
               margin-top: 36px; padding-top: 16px; border-top: 1px solid #e6eaf2; }}
</style>
"""


def kpi_card(value: str, label: str, sub: str = "", delta: str = "",
             delta_dir: str = "neutral", bar_color: str = BLUE) -> str:
    """Genera el HTML de una tarjeta KPI enterprise."""
    delta_html = f'<div class="delta {delta_dir}">{delta}</div>' if delta else ""
    sub_html   = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi">
      <div class="bar" style="background:{bar_color}"></div>
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {sub_html}
      {delta_html}
    </div>
    """


def section(title: str) -> str:
    return f'<div class="section"><span class="dot"></span>{title}</div>'


def insight(text: str, kind: str = "", head: str = "") -> str:
    head_html = f'<span class="head">{head}</span> ' if head else ""
    return f'<div class="insight {kind}"><p>{head_html}{text}</p></div>'
