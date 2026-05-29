"""
Inventory Risk Scorer
Cruza el inventario actual (Channel Inv.) con el forecast de demanda (Sell-in)
para identificar SKUs en riesgo de quiebre de stock.

Métricas por SKU:
  - current_inventory    : última lectura no-nula de Channel Inv.
  - forecast_13wk        : suma forecast_lgbm en las próximas 13 semanas
  - avg_weekly_demand    : promedio semanal del forecast
  - weeks_of_supply      : current_inventory / avg_weekly_demand
  - coverage_ratio       : current_inventory / forecast_13wk  (1.0 = exacto)
  - stockout_week        : primera semana donde el inventario se agota
  - risk_level           : CRITICAL / HIGH / MEDIUM / LOW

Output: data/forecasts/inventory_risk.parquet + reports/inventory_risk_report.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SILVER_PATH    = Path("data/silver/silver_dataset.parquet")
FORECASTS_PATH = Path("data/forecasts/forecasts.parquet")
FORECASTS_DIR  = Path("data/forecasts")
REPORTS_DIR    = Path("reports")

# Umbrales de riesgo (semanas de cobertura)
THRESHOLDS = {
    "CRITICAL": 2,   # < 2 semanas
    "HIGH":     4,   # 2–4 semanas
    "MEDIUM":   8,   # 4–8 semanas
    # >= 8 semanas → LOW
}


def _risk_level(weeks_of_supply: float) -> str:
    if weeks_of_supply < THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if weeks_of_supply < THRESHOLDS["HIGH"]:
        return "HIGH"
    if weeks_of_supply < THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def _stockout_week(current_inv: float, weekly_forecasts: list[float],
                   future_weeks: list[int]) -> int | None:
    """
    Simula el drawdown de inventario semana a semana.
    Retorna el year_week donde se agota, o None si alcanza para todo el horizonte.
    """
    remaining = current_inv
    for fw, demand in zip(future_weeks, weekly_forecasts):
        remaining -= demand
        if remaining <= 0:
            return fw
    return None


def run_inventory_risk_scoring() -> dict:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ──────────────────────────────────────────────────────────
    log.info("Cargando silver dataset...")
    silver = pd.read_parquet(SILVER_PATH)
    silver["year_week"] = (
        silver["year_week"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .astype(int)
    )

    log.info("Cargando forecasts...")
    forecasts = pd.read_parquet(FORECASTS_PATH)

    # ── Inventario actual: última lectura no-cero por SKU ─────────────────────
    log.info("Calculando inventario actual por SKU...")
    inv_df = silver[silver["Category"] == "Channel Inv."].copy()

    # Última semana con inventario > 0 por SKU
    inv_nonzero = inv_df[inv_df["quantity"] > 0]
    last_inv = (
        inv_nonzero.sort_values("year_week")
        .groupby(["Channel", "Material Description"])
        .agg(
            current_inventory=("quantity", "last"),
            last_inv_week=("year_week", "last"),
        )
        .reset_index()
    )
    log.info(f"  SKUs con inventario activo: {len(last_inv):,}")

    # ── Forecast total 13 semanas por SKU ─────────────────────────────────────
    log.info("Agregando forecasts por SKU...")
    fc_agg = (
        forecasts.groupby(["Channel", "Material Description"])
        .agg(
            forecast_13wk=("forecast_naive", "sum"),
            avg_weekly_demand=("forecast_naive", "mean"),
            forecast_start=("year_week", "min"),
            forecast_end=("year_week", "max"),
            n_weeks=("horizon_step", "count"),
        )
        .reset_index()
    )

    # ── Join inventario × forecast ────────────────────────────────────────────
    log.info("Cruzando inventario con forecast...")
    risk = last_inv.merge(fc_agg, on=["Channel", "Material Description"], how="inner")

    # ── Métricas de riesgo ────────────────────────────────────────────────────
    eps = 1e-6
    risk["coverage_ratio"] = (
        risk["current_inventory"] / (risk["forecast_13wk"] + eps)
    ).round(4)

    risk["weeks_of_supply"] = (
        risk["current_inventory"] / (risk["avg_weekly_demand"] + eps)
    ).round(2)

    risk["risk_level"] = risk["weeks_of_supply"].apply(_risk_level)

    # ── Semana de quiebre (stockout simulation) ───────────────────────────────
    log.info("Calculando semana de quiebre por SKU...")
    fc_by_sku = (
        forecasts.sort_values("horizon_step")
        .groupby(["Channel", "Material Description"])[["year_week", "forecast_naive"]]
        .apply(lambda g: list(zip(g["year_week"], g["forecast_naive"])), include_groups=False)
        .to_dict()
    )

    stockout_weeks = []
    for _, row in risk.iterrows():
        key = (row["Channel"], row["Material Description"])
        if key not in fc_by_sku:
            stockout_weeks.append(None)
            continue
        wk_fc = fc_by_sku[key]
        fws    = [x[0] for x in wk_fc]
        fdemands = [x[1] for x in wk_fc]
        sw = _stockout_week(row["current_inventory"], fdemands, fws)
        stockout_weeks.append(sw)

    risk["stockout_week"] = stockout_weeks

    # ── Ordenar por riesgo ────────────────────────────────────────────────────
    risk_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    risk["_risk_ord"] = risk["risk_level"].map(risk_order)
    risk = risk.sort_values(["_risk_ord", "weeks_of_supply"]).drop(columns="_risk_ord")
    risk = risk.reset_index(drop=True)

    # ── Guardar ───────────────────────────────────────────────────────────────
    out = FORECASTS_DIR / "inventory_risk.parquet"
    risk.to_parquet(out, index=False)
    log.info(f"Inventory risk → {out} ({len(risk):,} SKUs)")

    # ── Resumen ───────────────────────────────────────────────────────────────
    counts = risk["risk_level"].value_counts().to_dict()
    critical_skus = (
        risk[risk["risk_level"] == "CRITICAL"]
        [["Channel", "Material Description", "current_inventory",
          "forecast_13wk", "weeks_of_supply", "stockout_week"]]
        .head(20)
        .to_dict("records")
    )

    report = {
        "pipeline":        "inventory_risk_scoring",
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "forecast_window": f"{risk['forecast_start'].iloc[0]}–{risk['forecast_end'].iloc[0]}",
        "n_skus_scored":   len(risk),
        "risk_distribution": {
            "CRITICAL": counts.get("CRITICAL", 0),
            "HIGH":     counts.get("HIGH",     0),
            "MEDIUM":   counts.get("MEDIUM",   0),
            "LOW":      counts.get("LOW",      0),
        },
        "pct_at_risk": round(
            (counts.get("CRITICAL", 0) + counts.get("HIGH", 0)) / len(risk) * 100, 1
        ),
        "top_critical_skus": critical_skus,
    }

    with open(REPORTS_DIR / "inventory_risk_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Risk report → {REPORTS_DIR}/inventory_risk_report.json")

    dist = report["risk_distribution"]
    log.info(
        f"Distribución de riesgo — "
        f"CRITICAL: {dist['CRITICAL']} | HIGH: {dist['HIGH']} | "
        f"MEDIUM: {dist['MEDIUM']} | LOW: {dist['LOW']}"
    )

    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = run_inventory_risk_scoring()
    dist = result["risk_distribution"]
    print(f"\nSKUs analizados: {result['n_skus_scored']:,}")
    print(f"En riesgo (CRITICAL+HIGH): {result['pct_at_risk']}%")
    print(f"  CRITICAL: {dist['CRITICAL']:,}")
    print(f"  HIGH:     {dist['HIGH']:,}")
    print(f"  MEDIUM:   {dist['MEDIUM']:,}")
    print(f"  LOW:      {dist['LOW']:,}")
