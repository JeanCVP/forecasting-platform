"""
Forecast Generator
Generates 13-week forward forecasts for all Sell-in SKUs.
Primary model: LightGBM global (recursive h-step).
Benchmark:     SeasonalNaive (lag-52).
Output: data/forecasts/forecasts.parquet
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SILVER_PATH   = Path("data/silver/silver_dataset.parquet")
MODEL_PATH    = Path("data/models/lgbm_global.txt")
FORECASTS_DIR = Path("data/forecasts")
REPORTS_DIR   = Path("reports")

CATEGORY        = "Sell-in"
HORIZON         = 71   # 19 semanas resto 2025 (W34-W52) + 52 semanas 2026
MIN_HISTORY     = 26
WEEKS_IN_YEAR   = 52.18

FEATURE_COLS = [
    "lag_1", "lag_4", "lag_52",
    "rolling_mean_4", "rolling_mean_12", "rolling_std_12",
    "weeks_since_last_sale", "intermittent_flag",
    "week_sin", "week_cos",
    "inventory_days_of_supply",
]


# ─── Year-week helpers ────────────────────────────────────────────────────────

def _yw_to_date(yw: int) -> date:
    """YYYYWW → Monday of that ISO week."""
    year, week = divmod(yw, 100)
    return date.fromisocalendar(year, week, 1)


def _date_to_yw(d: date) -> int:
    iso = d.isocalendar()
    return iso[0] * 100 + iso[1]


def _next_yw(yw: int) -> int:
    return _date_to_yw(_yw_to_date(yw) + timedelta(weeks=1))


def _future_weeks(from_yw: int, horizon: int) -> list[int]:
    weeks = []
    current = from_yw
    for _ in range(horizon):
        current = _next_yw(current)
        weeks.append(current)
    return weeks


def _week_num(yw: int) -> int:
    return yw % 100


# ─── Feature computation for one future step ─────────────────────────────────

def _build_features(ext: list[float], target_yw: int) -> list[float]:
    """
    Given extended history (actuals + predictions so far),
    compute the feature vector for the next step.
    ext[-1] is the most recent known value.
    """
    n = len(ext)
    arr = np.array(ext, dtype=float)

    lag_1  = arr[-1]   if n >= 1  else 0.0
    lag_4  = arr[-4]   if n >= 4  else 0.0
    lag_52 = arr[-52]  if n >= 52 else 0.0

    rm4  = float(np.mean(arr[-4:]))  if n >= 1 else 0.0
    rm12 = float(np.mean(arr[-12:])) if n >= 1 else 0.0
    rs12 = float(np.std(arr[-12:], ddof=1)) if n >= 2 else 0.0

    # weeks_since_last_sale: steps since last positive in ext
    nonzero_idx = np.where(arr > 0)[0]
    wsls = float(n - nonzero_idx[-1] - 1) if len(nonzero_idx) > 0 else float(n)

    # intermittent_flag: >50% zeros in last 12
    past12 = arr[-12:] if n >= 1 else np.array([0.0])
    interm = 1.0 if float(np.mean(past12 == 0)) > 0.5 else 0.0

    wn = _week_num(target_yw)
    w_sin = float(np.sin(2 * np.pi * wn / WEEKS_IN_YEAR))
    w_cos = float(np.cos(2 * np.pi * wn / WEEKS_IN_YEAR))

    dos = 0.0  # inventory days-of-supply unknown for future weeks

    return [lag_1, lag_4, lag_52, rm4, rm12, rs12, wsls, interm, w_sin, w_cos, dos]


# ─── Per-SKU forecasters ──────────────────────────────────────────────────────

def _forecast_naive(history: np.ndarray, future_weeks: list[int]) -> list[float]:
    """SeasonalNaive: y[t+h] = y[t+h-52] or last value if history short."""
    n = len(history)
    preds = []
    for i, _ in enumerate(future_weeks, start=1):
        if n >= 52:
            pred = float(max(0.0, history[-(52 - (i - 1) % 52)]))
        elif n > 0:
            pred = float(history[-1])
        else:
            pred = 0.0
        preds.append(pred)
    return preds


def _forecast_lgbm(history: np.ndarray, future_weeks: list[int], model) -> list[float]:
    """Recursive LightGBM: predict step by step, feeding each prediction as lag."""
    ext = list(history.astype(float))
    preds = []
    for fw in future_weeks:
        feats = _build_features(ext, fw)
        pred = float(max(0.0, model.predict([feats])[0]))
        preds.append(pred)
        ext.append(pred)
    return preds


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_forecast_generation(horizon: int = HORIZON) -> dict:
    import lightgbm as lgb

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    log.info("Loading silver Sell-in data...")
    df = pd.read_parquet(SILVER_PATH)
    df = df[df["Category"] == CATEGORY].copy()
    df["year_week"] = (
        df["year_week"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .astype(int)
    )
    df = df.sort_values(["Channel", "Material Description", "year", "week_num"])

    # Last week with actual non-zero sales = real business cutoff
    last_yw = int(df[df["quantity"] > 0]["year_week"].max())
    future_wks = _future_weeks(last_yw, horizon)
    log.info(f"Last data week: {last_yw} | Forecast: {future_wks[0]}–{future_wks[-1]} ({horizon} weeks)")

    # Load LightGBM model
    log.info(f"Loading model: {MODEL_PATH}")
    model = lgb.Booster(model_file=str(MODEL_PATH))

    # Get all SKUs
    skus = df[["Channel", "Material Description"]].drop_duplicates()
    n_skus = len(skus)
    log.info(f"Generating forecasts for {n_skus:,} SKUs...")

    records = []
    skipped = 0

    for idx, (_, row) in enumerate(skus.iterrows()):
        ch, mat = row["Channel"], row["Material Description"]

        sku_df = df[
            (df["Channel"] == ch) &
            (df["Material Description"] == mat)
        ].sort_values("year_week")

        history = sku_df["quantity"].values.astype(float)

        if len(history) < MIN_HISTORY:
            skipped += 1
            continue

        naive_preds = _forecast_naive(history, future_wks)
        lgbm_preds  = _forecast_lgbm(history, future_wks, model)

        for h, (fw, fn, fl) in enumerate(zip(future_wks, naive_preds, lgbm_preds), start=1):
            records.append({
                "Channel":              ch,
                "Material Description": mat,
                "year_week":            fw,
                "horizon_step":         h,
                "forecast_naive":       round(fn, 2),
                "forecast_lgbm":        round(fl, 2),
            })

        if (idx + 1) % 2000 == 0:
            log.info(f"  {idx+1:,}/{n_skus:,} SKUs processed")

    log.info(f"Done. {n_skus - skipped:,} SKUs forecast, {skipped:,} skipped (insufficient history)")

    forecasts_df = pd.DataFrame(records)

    # Add week_num for dashboard convenience
    forecasts_df["week_num"] = forecasts_df["year_week"] % 100
    forecasts_df["year"]     = forecasts_df["year_week"] // 100

    out = FORECASTS_DIR / "forecasts.parquet"
    forecasts_df.to_parquet(out, index=False)
    log.info(f"Forecasts saved → {out} ({len(forecasts_df):,} rows)")

    # Summary stats
    total_naive = float(forecasts_df["forecast_naive"].sum())
    total_lgbm  = float(forecasts_df["forecast_lgbm"].sum())
    top_skus = (
        forecasts_df.groupby(["Channel", "Material Description"])["forecast_lgbm"]
        .sum()
        .nlargest(10)
        .reset_index()
        .rename(columns={"forecast_lgbm": "total_forecast"})
        .to_dict("records")
    )

    report = {
        "pipeline":        "forecast_generation",
        "last_data_week":  last_yw,
        "forecast_start":  future_wks[0],
        "forecast_end":    future_wks[-1],
        "horizon_weeks":   horizon,
        "n_skus_forecast": n_skus - skipped,
        "n_skus_skipped":  skipped,
        "total_rows":      len(forecasts_df),
        "total_forecast_naive": round(total_naive, 0),
        "total_forecast_lgbm":  round(total_lgbm, 0),
        "top10_skus_by_volume": top_skus,
    }

    with open(REPORTS_DIR / "forecast_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Forecast report → {REPORTS_DIR}/forecast_report.json")

    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = run_forecast_generation()
    print(f"\nForecast window: W{result['forecast_start']} → W{result['forecast_end']}")
    print(f"SKUs forecast:   {result['n_skus_forecast']:,}")
    print(f"Total volume (LightGBM): {result['total_forecast_lgbm']:,.0f} units")
