"""
Customer Churn Analyzer
Analiza el comportamiento de compra histórico por cliente (Channel) usando datos
de Sell-in para detectar patrones de abandono (churn) y clasificar el riesgo.

Métricas por cliente:
  - last_purchase_week      : última semana con quantity > 0
  - weeks_since_last_purchase: semanas desde la última compra hasta W33-2025 (202533)
  - active_weeks_pct        : % de semanas activas sobre el total de semanas disponibles
  - trend_slope             : pendiente de regresión lineal sobre los últimos 26 periodos
                              (negativa = cliente bajando en volumen)
  - avg_weekly_volume       : promedio semanal histórico
  - last_13w_avg            : promedio de las últimas 13 semanas con datos
  - volume_change_pct       : cambio % entre last_13w_avg y el promedio de las 13
                              semanas anteriores
  - churn_risk              : ALTO / MEDIO / BAJO / ACTIVO

Umbrales de clasificación:
  - ALTO   : sin compras en >26 semanas
  - MEDIO  : sin compras en 13–26 semanas O tendencia muy negativa (slope < -0.5 y
             volume_change_pct < -40%)
  - ACTIVO : compró en las últimas 13 semanas y volumen estable o creciente
  - BAJO   : todo lo demás (activo pero con cierta debilidad)

Output:
  - data/forecasts/churn_analysis.parquet
  - reports/churn_report.json
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
SILVER_PATH   = Path("data/silver/silver_dataset.parquet")
FORECASTS_DIR = Path("data/forecasts")
REPORTS_DIR   = Path("reports")

# ── Parámetros ─────────────────────────────────────────────────────────────────
REFERENCE_WEEK  = 202533        # W33-2025 (semana de corte)
CATEGORY        = "Sell-in"

# Umbrales de churn (en semanas)
CHURN_HIGH      = 26            # >26 semanas sin compras → ALTO
CHURN_MEDIUM    = 13            # 13-26 semanas sin compras → MEDIO

# Umbrales de tendencia para clasificar MEDIO adicional
SLOPE_VERY_NEG  = -0.5          # pendiente muy negativa
VCHANGE_VERY_NEG = -40.0        # caída de volumen > 40%


# ── Helpers ──────────────────────────────────────────────────────────────────

def _year_week_to_index(year_week: int) -> int:
    """
    Convierte un year_week entero (YYYYWW) a un índice secuencial de semanas.
    Asume máximo 52 semanas por año (sin ISO 53).
    """
    year = year_week // 100
    week = year_week % 100
    return year * 52 + week


def _weeks_between(yw_from: int, yw_to: int) -> int:
    """
    Calcula la diferencia en semanas entre dos year_week enteros (YYYYWW).
    Retorna un entero; puede ser negativo si yw_from > yw_to.
    """
    return _year_week_to_index(yw_to) - _year_week_to_index(yw_from)


def _linear_slope(values: np.ndarray) -> float:
    """
    Pendiente de una regresión lineal simple sobre la secuencia dada.
    Retorna np.nan si hay menos de 2 observaciones.
    """
    n = len(values)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    # Fórmula cerrada: slope = cov(x, y) / var(x)
    x_mean = x.mean()
    y_mean = values.mean()
    num = np.sum((x - x_mean) * (values - y_mean))
    den = np.sum((x - x_mean) ** 2)
    if den == 0:
        return 0.0
    return float(num / den)


def _compute_client_metrics(
    series: pd.Series,          # pd.Series indexada por year_week (int), ordenada
    all_weeks: list[int],       # lista completa de semanas del universo
    reference_week: int,
) -> dict:
    """
    Calcula todas las métricas de churn para un cliente a partir de su serie
    temporal de Sell-in (agregada sobre todos los SKUs del cliente).

    Parameters
    ----------
    series     : Serie con índice year_week (int), values = quantity sumada
    all_weeks  : todas las semanas disponibles en el dataset (universo completo)
    reference_week : semana de referencia (202533)

    Returns
    -------
    dict con todas las métricas calculadas
    """
    # Rellenar semanas faltantes con 0
    full = series.reindex(all_weeks, fill_value=0.0)

    # ── last_purchase_week ───────────────────────────────────────────────────
    nonzero_weeks = full[full > 0]
    if len(nonzero_weeks) == 0:
        last_purchase_week = None
        weeks_since_last   = _weeks_between(all_weeks[0], reference_week)
    else:
        last_purchase_week = int(nonzero_weeks.index[-1])
        weeks_since_last   = _weeks_between(last_purchase_week, reference_week)

    # ── active_weeks_pct ─────────────────────────────────────────────────────
    total_weeks  = len(all_weeks)
    active_count = int((full > 0).sum())
    active_weeks_pct = round(active_count / total_weeks * 100, 2) if total_weeks > 0 else 0.0

    # ── avg_weekly_volume ────────────────────────────────────────────────────
    avg_weekly_volume = round(float(full.mean()), 4)

    # ── trend_slope (últimos 26 periodos) ────────────────────────────────────
    last_26 = full.iloc[-26:].values.astype(float) if len(full) >= 26 else full.values.astype(float)
    trend_slope = round(_linear_slope(last_26), 6)

    # ── last_13w_avg y volume_change_pct ────────────────────────────────────
    if len(full) >= 13:
        last_13  = full.iloc[-13:].values.astype(float)
        prev_13  = full.iloc[-26:-13].values.astype(float) if len(full) >= 26 else full.iloc[:-13].values.astype(float)
    else:
        last_13  = full.values.astype(float)
        prev_13  = np.array([], dtype=float)

    last_13w_avg = round(float(last_13.mean()), 4) if len(last_13) > 0 else 0.0
    prev_13_avg  = round(float(prev_13.mean()), 4) if len(prev_13) > 0 else 0.0

    if prev_13_avg > 0:
        volume_change_pct = round((last_13w_avg - prev_13_avg) / prev_13_avg * 100, 2)
    elif last_13w_avg > 0:
        # cliente nuevo o sin historial previo pero activo ahora
        volume_change_pct = 100.0
    else:
        volume_change_pct = 0.0

    return {
        "last_purchase_week":       last_purchase_week,
        "weeks_since_last_purchase": int(weeks_since_last),
        "active_weeks_pct":         active_weeks_pct,
        "trend_slope":              trend_slope,
        "avg_weekly_volume":        avg_weekly_volume,
        "last_13w_avg":             last_13w_avg,
        "volume_change_pct":        volume_change_pct,
    }


def _classify_churn_risk(
    weeks_since_last: int,
    trend_slope: float,
    volume_change_pct: float,
) -> str:
    """
    Clasifica el riesgo de churn de un cliente:

    - ALTO   : sin compras en >26 semanas
    - MEDIO  : sin compras 13–26 semanas, O tendencia muy negativa
    - ACTIVO : compró en las últimas 13 semanas (y no cayendo bruscamente)
    - BAJO   : activo y con cierta debilidad moderada
    """
    if weeks_since_last > CHURN_HIGH:
        return "ALTO"

    if CHURN_MEDIUM < weeks_since_last <= CHURN_HIGH:
        return "MEDIO"

    # Activo (< 13 semanas sin comprar): verificar si hay tendencia muy negativa
    if not np.isnan(trend_slope) and trend_slope < SLOPE_VERY_NEG and volume_change_pct < VCHANGE_VERY_NEG:
        return "MEDIO"

    # Compró en las últimas 13 semanas
    if weeks_since_last <= CHURN_MEDIUM:
        return "ACTIVO"

    return "BAJO"


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_churn_analysis() -> dict:
    """
    Ejecuta el análisis completo de churn y escribe los artefactos de salida.

    Returns
    -------
    dict con el resumen del reporte (mismo contenido que churn_report.json)
    """
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cargar silver ─────────────────────────────────────────────────────────
    log.info("Cargando silver dataset...")
    silver = pd.read_parquet(SILVER_PATH)

    # Normalizar year_week a entero (puede venir como "202314" string o 202314.0 float)
    silver["year_week"] = (
        silver["year_week"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .astype(int)
    )

    # ── Filtrar Sell-in ───────────────────────────────────────────────────────
    log.info(f"Filtrando categoría '{CATEGORY}'...")
    sellin = silver[silver["Category"] == CATEGORY].copy()
    log.info(f"  Filas Sell-in: {len(sellin):,}")

    # ── Agregar por cliente × semana (suma de todos los SKUs) ─────────────────
    log.info("Agregando cantidades por cliente y semana...")
    client_weekly = (
        sellin
        .groupby(["Channel", "year_week"], as_index=False)["quantity"]
        .sum()
    )

    # Solo semanas hasta REFERENCE_WEEK (para no incluir datos futuros)
    client_weekly = client_weekly[client_weekly["year_week"] <= REFERENCE_WEEK]

    # Universo completo de semanas disponibles en los datos (ordenadas)
    all_weeks = sorted(client_weekly["year_week"].unique().tolist())
    log.info(f"  Semanas en el universo: {len(all_weeks)} ({all_weeks[0]} → {all_weeks[-1]})")

    # Solo clientes que alguna vez compraron Sell-in (excluir nunca-compraron)
    ever_bought = client_weekly[client_weekly["quantity"] > 0]["Channel"].unique()
    clients = sorted(ever_bought.tolist())
    never_bought_n = sellin["Channel"].nunique() - len(clients)
    log.info(f"  Clientes con historial Sell-in: {len(clients)} ({never_bought_n} sin compras Sell-in)")

    # ── Calcular métricas por cliente ─────────────────────────────────────────
    log.info("Calculando métricas de churn por cliente...")
    records = []
    for client in clients:
        mask   = client_weekly["Channel"] == client
        series = (
            client_weekly[mask]
            .set_index("year_week")["quantity"]
            .sort_index()
        )
        metrics = _compute_client_metrics(series, all_weeks, REFERENCE_WEEK)
        metrics["churn_risk"] = _classify_churn_risk(
            metrics["weeks_since_last_purchase"],
            metrics["trend_slope"],
            metrics["volume_change_pct"],
        )
        metrics["Channel"] = client
        records.append(metrics)

    # ── Construir DataFrame ───────────────────────────────────────────────────
    result = pd.DataFrame(records)

    # Reordenar columnas con Channel primero
    col_order = [
        "Channel",
        "last_purchase_week",
        "weeks_since_last_purchase",
        "active_weeks_pct",
        "trend_slope",
        "avg_weekly_volume",
        "last_13w_avg",
        "volume_change_pct",
        "churn_risk",
    ]
    result = result[col_order]

    # Ordenar por riesgo y luego por semanas sin comprar descendente
    risk_order = {"ALTO": 0, "MEDIO": 1, "BAJO": 2, "ACTIVO": 3}
    result["_risk_ord"] = result["churn_risk"].map(risk_order)
    result = (
        result
        .sort_values(["_risk_ord", "weeks_since_last_purchase"], ascending=[True, False])
        .drop(columns="_risk_ord")
        .reset_index(drop=True)
    )

    # ── Guardar parquet ───────────────────────────────────────────────────────
    out_parquet = FORECASTS_DIR / "churn_analysis.parquet"
    result.to_parquet(out_parquet, index=False)
    log.info(f"Churn analysis → {out_parquet} ({len(result):,} clientes)")

    # ── Generar reporte JSON ──────────────────────────────────────────────────
    risk_counts = result["churn_risk"].value_counts().to_dict()

    top_alto = (
        result[result["churn_risk"] == "ALTO"]
        [[
            "Channel",
            "last_purchase_week",
            "weeks_since_last_purchase",
            "active_weeks_pct",
            "avg_weekly_volume",
            "last_13w_avg",
            "volume_change_pct",
            "trend_slope",
        ]]
        .head(10)
        .to_dict("records")
    )

    report = {
        "pipeline":        "churn_analysis",
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "reference_week":  REFERENCE_WEEK,
        "category":        CATEGORY,
        "weeks_in_universe": len(all_weeks),
        "universe_range":  f"{all_weeks[0]}–{all_weeks[-1]}",
        "total_clients":   len(result),
        "never_bought_sellin": never_bought_n,
        "risk_distribution": {
            "ALTO":   risk_counts.get("ALTO",   0),
            "MEDIO":  risk_counts.get("MEDIO",  0),
            "BAJO":   risk_counts.get("BAJO",   0),
            "ACTIVO": risk_counts.get("ACTIVO", 0),
        },
        "pct_high_risk": round(
            risk_counts.get("ALTO", 0) / len(result) * 100, 1
        ) if len(result) > 0 else 0.0,
        "top_10_alto_risk": top_alto,
    }

    out_report = REPORTS_DIR / "churn_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Churn report   → {out_report}")

    # ── Log de resumen ────────────────────────────────────────────────────────
    dist = report["risk_distribution"]
    log.info(
        f"Distribución de riesgo — "
        f"ALTO: {dist['ALTO']} | MEDIO: {dist['MEDIO']} | "
        f"BAJO: {dist['BAJO']} | ACTIVO: {dist['ACTIVO']}"
    )

    return report


# ── Entry point standalone ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    result = run_churn_analysis()

    dist  = result["risk_distribution"]
    top10 = result["top_10_alto_risk"]

    print(f"\n{'='*60}")
    print(f"  ANÁLISIS DE CHURN — Semana de referencia: {result['reference_week']}")
    print(f"{'='*60}")
    print(f"  Clientes analizados : {result['total_clients']}")
    print(f"  Semanas en universo : {result['weeks_in_universe']} ({result['universe_range']})")
    print()
    print(f"  Distribución de riesgo:")
    print(f"    ALTO   : {dist['ALTO']:>4}  ({result['pct_high_risk']}%)")
    print(f"    MEDIO  : {dist['MEDIO']:>4}")
    print(f"    BAJO   : {dist['BAJO']:>4}")
    print(f"    ACTIVO : {dist['ACTIVO']:>4}")
    print()

    if top10:
        print(f"  Top 10 clientes en riesgo ALTO:")
        print(f"  {'Cliente':<35} {'Últ.Compra':>10} {'SemSin':>7} {'Act%':>6} {'Vol.Med':>9} {'Últ13w':>9} {'Cambio%':>8}")
        print(f"  {'-'*35} {'-'*10} {'-'*7} {'-'*6} {'-'*9} {'-'*9} {'-'*8}")
        for r in top10:
            lp  = str(r["last_purchase_week"]) if r["last_purchase_week"] else "N/A"
            print(
                f"  {str(r['Channel']):<35} "
                f"{lp:>10} "
                f"{r['weeks_since_last_purchase']:>7} "
                f"{r['active_weeks_pct']:>6.1f} "
                f"{r['avg_weekly_volume']:>9.1f} "
                f"{r['last_13w_avg']:>9.1f} "
                f"{r['volume_change_pct']:>7.1f}%"
            )
    print(f"\n  Parquet : data/forecasts/churn_analysis.parquet")
    print(f"  Reporte : reports/churn_report.json")
    print(f"{'='*60}\n")
