"""
Feature Engineering Pipeline — Optimized with DuckDB + pandas hybrid.
All features use SHIFTED windows (no leakage).
Uses integer group encoding + vectorized numpy for performance on 4.6M rows.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SILVER_PATH = Path("data/silver/silver_dataset.parquet")
GOLD_DIR = Path("data/gold")
REPORTS_DIR = Path("reports")
ID_COLS = ["Channel", "Material Description", "Category"]
WEEKS_IN_YEAR = 52.18

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stderr)])
log = logging.getLogger(__name__)


def _encode_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer group code for (Channel, SKU, Category) triplet."""
    return df.groupby(ID_COLS, sort=False).ngroup().values


def _compute_features_per_group(qty: np.ndarray) -> dict[str, np.ndarray]:
    """
    Given a 1D sorted time series for one SKU group,
    return all lag and rolling features (all SHIFTED — no leakage).
    """
    n = len(qty)
    lag1 = np.empty(n); lag1[:] = np.nan
    lag4 = np.empty(n); lag4[:] = np.nan
    lag52 = np.empty(n); lag52[:] = np.nan
    rm4 = np.empty(n); rm4[:] = np.nan
    rm12 = np.empty(n); rm12[:] = np.nan
    rs12 = np.empty(n); rs12[:] = np.nan
    wsls = np.zeros(n)
    interm = np.zeros(n)

    # Shifted lags
    if n > 1: lag1[1:] = qty[:-1]
    if n > 4: lag4[4:] = qty[:-4]
    if n > 52: lag52[52:] = qty[:-52]

    # Shifted rolling (window over qty[0..i-1])
    for i in range(1, n):
        past = qty[max(0, i-4):i]       # past 4 weeks
        rm4[i] = np.mean(past)
        past12 = qty[max(0, i-12):i]    # past 12 weeks
        rm12[i] = np.mean(past12)
        if len(past12) >= 2:
            rs12[i] = np.std(past12, ddof=1)

    # weeks_since_last_sale (shifted: count weeks since last positive BEFORE current)
    last_sale = -1
    for i in range(n):
        if i > 0 and qty[i-1] > 0:
            last_sale = i - 1
        wsls[i] = (i - last_sale - 1) if last_sale >= 0 else i

    # intermittent_flag: >50% zeros in past 12 weeks
    for i in range(1, n):
        past12 = qty[max(0, i-12):i]
        if len(past12) > 0:
            interm[i] = 1.0 if np.mean(past12 == 0) > 0.5 else 0.0

    return {
        "lag_1": lag1, "lag_4": lag4, "lag_52": lag52,
        "rolling_mean_4": rm4, "rolling_mean_12": rm12, "rolling_std_12": rs12,
        "weeks_since_last_sale": wsls, "intermittent_flag": interm,
    }


def run_feature_engineering(sample_n: int = 0) -> pd.DataFrame:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading silver dataset...")
    df = pd.read_parquet(SILVER_PATH)
    df["quantity"] = df["quantity"].astype(float)
    if sample_n > 0:
        log.info(f"Sampling {sample_n:,} rows for fast mode")
        df = df.head(sample_n)

    df = df.sort_values(ID_COLS + ["year", "week_num"]).reset_index(drop=True)
    log.info(f"  → {len(df):,} rows sorted")

    # Encode groups
    group_ids = _encode_groups(df)
    qty_arr = df["quantity"].values.astype(float)
    n = len(df)

    # Allocate output arrays
    feat_arrays: dict[str, np.ndarray] = {
        "lag_1": np.full(n, np.nan), "lag_4": np.full(n, np.nan),
        "lag_52": np.full(n, np.nan), "rolling_mean_4": np.full(n, np.nan),
        "rolling_mean_12": np.full(n, np.nan), "rolling_std_12": np.full(n, np.nan),
        "weeks_since_last_sale": np.zeros(n), "intermittent_flag": np.zeros(n),
    }

    # Process each group
    unique_groups = np.unique(group_ids)
    log.info(f"Computing features for {len(unique_groups):,} SKU groups...")

    for g_idx, g in enumerate(unique_groups):
        mask = group_ids == g
        idx = np.where(mask)[0]
        group_qty = qty_arr[idx]
        feats = _compute_features_per_group(group_qty)
        for feat_name, feat_vals in feats.items():
            feat_arrays[feat_name][idx] = feat_vals

        if (g_idx + 1) % 5000 == 0:
            log.info(f"  Progress: {g_idx+1:,}/{len(unique_groups):,} groups")

    log.info("Assigning features to dataframe...")
    for feat_name, feat_vals in feat_arrays.items():
        df[feat_name] = feat_vals

    # Seasonal encoding (vectorized)
    log.info("Computing seasonal encoding...")
    df["week_sin"] = np.sin(2 * np.pi * df["week_num"] / WEEKS_IN_YEAR)
    df["week_cos"] = np.cos(2 * np.pi * df["week_num"] / WEEKS_IN_YEAR)

    # Inventory days of supply
    log.info("Computing inventory_days_of_supply...")
    sellin = df[df["Category"] == "Sell-in"][
        ["Channel", "Material Description", "year_week", "quantity"]
    ].rename(columns={"quantity": "qty_sellin"})
    inv = df[df["Category"] == "Channel Inv."][
        ["Channel", "Material Description", "year_week", "quantity"]
    ].rename(columns={"quantity": "qty_inv"})
    dos = sellin.merge(inv, on=["Channel", "Material Description", "year_week"], how="left")
    dos["inventory_days_of_supply"] = np.where(
        dos["qty_sellin"] > 0, dos["qty_inv"] / (dos["qty_sellin"] / 7.0), np.nan
    )
    df = df.merge(
        dos[["Channel", "Material Description", "year_week", "inventory_days_of_supply"]],
        on=["Channel", "Material Description", "year_week"], how="left"
    )

    feature_cols = [c for c in df.columns
                    if c not in ID_COLS + ["year_week", "year", "week_num", "quantity"]]
    null_counts = {col: int(df[col].isna().sum()) for col in feature_cols}

    log.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # Save
    gold_path = GOLD_DIR / "gold_features.parquet"
    df.to_parquet(gold_path, index=False)
    log.info(f"Gold saved: {gold_path} ({gold_path.stat().st_size/1e6:.1f}MB)")

    feature_report = {
        "pipeline": "feature_engineering",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(df), "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "null_counts_per_feature": null_counts,
        "gold_path": str(gold_path),
    }
    with open(REPORTS_DIR / "feature_report.json", "w") as f:
        json.dump(feature_report, f, indent=2, default=str)

    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="Row sample for testing (0=full)")
    args = p.parse_args()
    run_feature_engineering(sample_n=args.sample)
