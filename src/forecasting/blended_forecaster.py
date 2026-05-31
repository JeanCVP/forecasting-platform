"""
Blended Forecaster — Loop 8
=============================
Combina Loop 7 (recursive Q10/Q50/Q90) con Seasonal Naïve en un único archivo:

  Cobertura final (loop8_blended_forecasts.parquet):
  ┌─────────────────────────────────────────────────┐
  │ W34·2025–W52·2025 │ Loop 7 (330 SKUs activos)  │
  │ W01·2026–W17·2026 │ L7 (330) + SN (16,671)     │
  │ W18·2026–W52·2026 │ Seasonal Naïve (17,001)    │
  └─────────────────────────────────────────────────┘

Intervalos de confianza:
  - Loop 7 SKUs : Q10/Q90 del regresor cuantil
  - SN SKUs     : Q10 = SN × 0.45, Q90 = SN × 1.60 (derivado de bias Loop 5)

Métricas de cobertura:
  - total_q50 / total_q90 por semana y por canal
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FORECASTS_PATH  = Path("data/forecasts/forecasts.parquet")
LOOP7_FC_PATH   = Path("data/forecasts/loop7_recursive_forecasts.parquet")
BLENDED_PATH    = Path("data/forecasts/loop8_blended_forecasts.parquet")
REPORTS_DIR     = Path("reports")

# Confidence interval multipliers for Seasonal Naïve (from Loop 5 Q10/Q90 analysis)
SN_Q10_FACTOR = 0.45   # lower bound ≈ 45% of point estimate
SN_Q90_FACTOR = 1.60   # upper bound ≈ 160% of point estimate


def run_blended_forecast() -> dict:
    BLENDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Seasonal Naïve ───────────────────────────────────────────────────
    log.info("Loading Seasonal Naïve forecasts...")
    sn = pd.read_parquet(FORECASTS_PATH)
    sn["year_week"] = sn["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    sn_base = sn[["Channel", "Material Description", "year_week", "forecast_naive"]].copy()
    sn_base = sn_base.rename(columns={"forecast_naive": "forecast_q50"})
    sn_base["forecast_q10"] = (sn_base["forecast_q50"] * SN_Q10_FACTOR).astype("float32")
    sn_base["forecast_q90"] = (sn_base["forecast_q50"] * SN_Q90_FACTOR).astype("float32")
    sn_base["source"] = "seasonal_naive"
    log.info(f"  SN: {len(sn_base):,} rows | weeks {sn_base['year_week'].min()}–{sn_base['year_week'].max()}")

    # ── Load Loop 7 recursive ─────────────────────────────────────────────────
    if not LOOP7_FC_PATH.exists():
        log.warning("Loop 7 forecasts not found — using Seasonal Naïve only")
        sn_base.to_parquet(BLENDED_PATH, index=False)
        return _summary(sn_base)

    log.info("Loading Loop 7 recursive forecasts...")
    l7 = pd.read_parquet(LOOP7_FC_PATH)
    l7["year_week"] = l7["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    l7["source"] = "loop7_recursive"
    log.info(f"  L7: {len(l7):,} rows | weeks {l7['year_week'].min()}–{l7['year_week'].max()} | {l7.groupby(['Channel','Material Description']).ngroups:,} SKUs")

    # ── Merge: SN augmented with L7 where available ───────────────────────────
    log.info("Blending forecasts...")
    merged = sn_base.merge(
        l7[["Channel", "Material Description", "year_week",
            "forecast_q10", "forecast_q50", "forecast_q90"]].rename(
            columns={"forecast_q50": "l7_q50",
                     "forecast_q10": "l7_q10",
                     "forecast_q90": "l7_q90"}
        ),
        on=["Channel", "Material Description", "year_week"],
        how="left",
    )
    has_l7 = merged["l7_q50"].notna()
    merged.loc[has_l7, "forecast_q50"] = merged.loc[has_l7, "l7_q50"]
    merged.loc[has_l7, "forecast_q10"] = merged.loc[has_l7, "l7_q10"]
    merged.loc[has_l7, "forecast_q90"] = merged.loc[has_l7, "l7_q90"]
    merged.loc[has_l7, "source"]       = "loop7_recursive"
    merged = merged.drop(columns=["l7_q50", "l7_q10", "l7_q90"])
    log.info(f"  SN rows with L7 override: {has_l7.sum():,} / {len(merged):,}")

    # ── Add L7-only weeks (W34·2025–W52·2025, not in SN) ─────────────────────
    sn_min_yw = int(sn_base["year_week"].min())
    l7_extra  = l7[l7["year_week"] < sn_min_yw][
        ["Channel", "Material Description", "year_week",
         "forecast_q10", "forecast_q50", "forecast_q90", "source"]
    ].copy()
    log.info(f"  L7-only rows (pre-2026): {len(l7_extra):,}")

    # ── Final blend ───────────────────────────────────────────────────────────
    blended = pd.concat([l7_extra, merged], ignore_index=True)
    blended = blended.sort_values(
        ["Channel", "Material Description", "year_week"]
    ).reset_index(drop=True)

    # Cast types
    for col in ["forecast_q10", "forecast_q50", "forecast_q90"]:
        blended[col] = blended[col].fillna(0).astype("float32")

    blended.to_parquet(BLENDED_PATH, index=False)
    log.info(f"Blended forecast saved → {BLENDED_PATH} ({len(blended):,} rows)")

    return _summary(blended)


def _summary(blended: pd.DataFrame) -> dict:
    src_counts = blended["source"].value_counts().to_dict()
    wk_agg = (
        blended.groupby("year_week")[["forecast_q10", "forecast_q50", "forecast_q90"]]
        .sum()
        .reset_index()
    )

    result = {
        "pipeline":         "blended_forecast_loop8",
        "run_at":           datetime.now(timezone.utc).isoformat(),
        "total_rows":       len(blended),
        "n_skus":           blended.groupby(["Channel", "Material Description"]).ngroups,
        "n_weeks":          blended["year_week"].nunique(),
        "week_range":       f"{blended['year_week'].min()}–{blended['year_week'].max()}",
        "source_breakdown": src_counts,
        "total_q50":        round(float(blended["forecast_q50"].sum()), 0),
        "total_q90":        round(float(blended["forecast_q90"].sum()), 0),
        "avg_weekly_q50":   round(float(wk_agg["forecast_q50"].mean()), 0),
        "avg_weekly_q90":   round(float(wk_agg["forecast_q90"].mean()), 0),
        "output_path":      str(BLENDED_PATH),
    }

    out = REPORTS_DIR / "loop8_blended_report.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Report → {out}")
    return result


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    r = run_blended_forecast()
    print(f"\n{'='*60}")
    print(f"  LOOP 8 — BLENDED FORECAST")
    print(f"  Rows: {r['total_rows']:,}  SKUs: {r['n_skus']:,}  Weeks: {r['n_weeks']}")
    print(f"  Range: {r['week_range']}")
    print(f"  Sources: {r['source_breakdown']}")
    print(f"  Total Q50: {r['total_q50']:,.0f}  |  Total Q90: {r['total_q90']:,.0f}")
    print(f"{'='*60}\n")
