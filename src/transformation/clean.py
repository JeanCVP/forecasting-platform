"""
Silver Layer Transformation Pipeline
Reads Bronze CSVs, fixes dtypes (especially comma-formatted 2025 numerics),
melts wide→long, deduplicates, validates temporal consistency,
detects anomalies, produces silver_dataset.parquet + DQ reports.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
REPORTS_DIR = Path("reports")
ID_COLS = ["Channel", "Material Description", "Category"]
VALID_CATEGORIES = {"Sell-in", "Cust. Sales", "Channel Inv."}
ANOMALY_STD = 4.0

_COLORS = {
    "BLACK","WHITE","BLUE","BEIGE","GREEN","RED","YELLOW","GOLD","SILVER",
    "GRAY","GREY","PINK","PURPLE","VIOLET","CREAM","ORANGE","BROWN","NAVY",
    "LIGHT BLUE","DARK BLUE","LIGHT GREEN","PHANTOM","LAVENDER","GRAPHITE",
    "TITANIUM","MYSTIC","AWESOME",
}
_COUNTRY_CODES = {
    "LTC","CO","COO","BR","PE","AR","CL","MX","EC","VE","BOL",
    "PAN","GTM","CRI","HND","SLV","NIC","DOM","PRI","USA","R410A",
}


def _parse_material(desc: str) -> tuple[str, str, str, str]:
    """Extract (category, model, color, country) from concatenated description."""
    if not isinstance(desc, str) or not desc.strip():
        return ("", "", "", "")
    parts = [p.strip() for p in desc.split(",")]
    category = parts[0] if parts else ""
    model    = parts[1] if len(parts) > 1 else ""
    # Color: first token that matches known colors (case-insensitive)
    color = ""
    for p in parts[2:]:
        if p.upper() in _COLORS:
            color = p
            break
    # Country: last token if it looks like a country code
    country = ""
    if parts:
        last = parts[-1].strip()
        if last.upper() in _COUNTRY_CODES or (last.isalpha() and 2 <= len(last) <= 3):
            country = last
    return category, model, color, country

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)])
log = logging.getLogger(__name__)


def _fix_numeric_col(series: pd.Series) -> pd.Series:
    """Convert comma-formatted or float-strings to float64."""
    if series.dtype == object:
        return (series.astype(str).str.strip()
                .str.replace(",", "", regex=False)
                .replace({"nan": "0", "None": "0", "": "0"})
                .astype(float).fillna(0.0))
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _load_bronze(year: str) -> pd.DataFrame:
    for ext in [".parquet", ".csv"]:
        path = BRONZE_DIR / f"bronze_{year}{ext}"
        if path.exists():
            df = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path, dtype=str)
            log.info(f"  [{year}] Loaded bronze: {len(df):,} rows")
            return df
    raise FileNotFoundError(f"No bronze file for year {year}")


def _melt_and_fix(df: pd.DataFrame, year: str) -> pd.DataFrame:
    week_cols = [c for c in df.columns if c not in ID_COLS]
    # Fix dtypes before melt
    for col in week_cols:
        df[col] = _fix_numeric_col(df[col])
    
    melted = df.melt(id_vars=ID_COLS, value_vars=week_cols,
                     var_name="year_week", value_name="quantity")
    melted["year"] = melted["year_week"].str[:4].astype(int)
    melted["week_num"] = melted["year_week"].str[4:].astype(int)
    melted["quantity"] = melted["quantity"].fillna(0.0).clip(lower=0.0)
    log.info(f"  [{year}] Melted: {len(df):,} wide → {len(melted):,} long rows")
    return melted


def _detect_truncation(df: pd.DataFrame) -> dict:
    df25 = df[df["year"] == 2025]
    if len(df25) == 0:
        return {"truncation_detected": False}
    weekly = df25.groupby("week_num")["quantity"].sum()
    nonzero_weeks = weekly[weekly > 0]
    zero_tail = []
    for w in sorted(weekly.index, reverse=True):
        if weekly[w] == 0:
            zero_tail.append(int(w))
        else:
            break
    return {
        "truncation_detected": len(zero_tail) > 0,
        "zero_tail_weeks_2025": sorted(zero_tail),
        "last_nonzero_week_2025": int(nonzero_weeks.index.max()) if len(nonzero_weeks) > 0 else None,
        "n_nonzero_weeks_2025": int(len(nonzero_weeks)),
    }


def _compute_dq(df: pd.DataFrame) -> dict:
    n = len(df)
    null_qty = int(df["quantity"].isna().sum())
    neg_qty = int((df["quantity"] < 0).sum())
    zero_qty = int((df["quantity"] == 0).sum())
    pct_null = null_qty / n if n > 0 else 0
    pct_zero = zero_qty / n if n > 0 else 0
    score = max(0.0, 1.0 - pct_null * 5 - (neg_qty/n)*5 - max(0, (pct_zero-0.6)*0.5))
    return {
        "total_rows": n, "null_quantity": null_qty,
        "negative_quantity": neg_qty, "zero_quantity": zero_qty,
        "pct_null_quantity": round(pct_null, 4),
        "pct_zero_quantity": round(pct_zero, 4),
        "quality_score": round(score, 4),
    }


def _detect_anomalies(df: pd.DataFrame) -> dict:
    nonzero = df[df["quantity"] > 0].copy()
    if len(nonzero) == 0:
        return {"anomaly_count": 0, "anomaly_pct": 0.0}
    stats = nonzero.groupby(["Channel", "Category"])["quantity"].agg(["mean", "std"]).reset_index()
    stats.columns = ["Channel", "Category", "mu", "sigma"]
    merged = nonzero.merge(stats, on=["Channel", "Category"], how="left")
    merged["z"] = (merged["quantity"] - merged["mu"]) / (merged["sigma"] + 1e-9)
    anomalies = merged[merged["z"].abs() > ANOMALY_STD]
    top = (anomalies.nlargest(20, "z")
           [["Channel", "Material Description", "Category", "year_week", "quantity", "z"]]
           .to_dict("records"))
    return {
        "anomaly_count": len(anomalies),
        "anomaly_pct": round(len(anomalies)/len(nonzero), 4),
        "z_score_threshold": ANOMALY_STD,
        "top_anomalies": top,
    }


def run_cleaning() -> pd.DataFrame:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for year in ["2023", "2024", "2025"]:
        try:
            df_wide = _load_bronze(year)
            df_long = _melt_and_fix(df_wide, year)
            frames.append(df_long)
        except FileNotFoundError as e:
            log.error(str(e))

    if not frames:
        raise RuntimeError("No bronze files found. Run ingestion first.")

    df = pd.concat(frames, ignore_index=True)
    log.info(f"Combined: {len(df):,} rows")

    # Dedup: aggregate by key sum
    key = ID_COLS + ["year_week"]
    n_before = len(df)
    df = df.groupby(key, as_index=False).agg(
        quantity=("quantity", "sum"),
        year=("year", "first"),
        week_num=("week_num", "first")
    )
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        log.warning(f"Dedup: removed {n_dupes:,} duplicate rows")

    # Filter valid categories
    df = df[df["Category"].isin(VALID_CATEGORIES)]

    # Parse Material Description attributes
    parsed = df["Material Description"].apply(
        lambda d: pd.Series(_parse_material(d),
                            index=["product_category","product_model","product_color","product_country"])
    )
    df = pd.concat([df, parsed], axis=1)

    # Sort
    df = df.sort_values(["Channel", "Material Description", "Category", "year", "week_num"])
    df = df.reset_index(drop=True)

    truncation = _detect_truncation(df)
    dq = _compute_dq(df)
    anomalies = _detect_anomalies(df)

    log.info(f"Quality score: {dq['quality_score']} | Anomalies: {anomalies['anomaly_count']:,}")
    log.info(f"Truncation: {truncation}")

    # Save silver
    silver_path = SILVER_DIR / "silver_dataset.parquet"
    try:
        df.to_parquet(silver_path, index=False, engine="pyarrow")
    except Exception:
        silver_path = SILVER_DIR / "silver_dataset.csv"
        df.to_csv(silver_path, index=False)
        log.warning("Saved as CSV (pyarrow unavailable)")
    log.info(f"Silver saved: {silver_path}")

    dq_report = {
        "pipeline": "silver_cleaning",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "rows_total": len(df), "duplicates_removed": n_dupes,
        "temporal_validation": {
            "years_present": sorted(df["year"].unique().tolist()),
            "week_range": [int(df["week_num"].min()), int(df["week_num"].max())],
        },
        "truncation_analysis": truncation,
        "dq_metrics": dq,
        "silver_path": str(silver_path),
    }
    anomaly_report = {"pipeline": "silver_cleaning",
                      "run_at": datetime.now(timezone.utc).isoformat(), **anomalies}

    with open(REPORTS_DIR / "dq_report.json", "w") as f:
        json.dump(dq_report, f, indent=2, default=str)
    with open(REPORTS_DIR / "anomaly_report.json", "w") as f:
        json.dump(anomaly_report, f, indent=2, default=str)

    log.info(f"DQ report      → {REPORTS_DIR}/dq_report.json")
    log.info(f"Anomaly report → {REPORTS_DIR}/anomaly_report.json")
    return df


if __name__ == "__main__":
    run_cleaning()
