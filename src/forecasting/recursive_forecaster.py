"""
Recursive Multi-Step Forecaster — Loop 7
==========================================
Genera pronósticos Q10/Q50/Q90 para W34·2025 → W52·2026 (71 semanas)
usando los modelos Loop 5 entrenados.

Algoritmo vectorizado:
  Para cada semana futura t:
    1. Construir matriz de features para los 17K SKUs desde su historial actualizado
    2. Clasificador → probabilidad de demanda (calibrado)
    3. Regresor Q50 → cantidad esperada | demanda ocurre
    4. Regresor Q10/Q90 → intervalo de confianza
    5. Aplicar bias correction al Q50
    6. Actualizar historial con Q50 (lag para siguiente semana)

La recursión propaga la incertidumbre: errores en t afectan t+1, t+2, ...
Por esto los intervalos Q10/Q90 se amplían naturalmente en horizontes largos.
"""
from __future__ import annotations

import logging
import pickle
import time
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.forecasting.bias_corrector import load_bias_factors

log = logging.getLogger(__name__)

GOLD_PATH     = Path("data/gold/gold_features.parquet")
FORECASTS_DIR = Path("data/forecasts")
MODELS_DIR    = Path("data/models")
REPORTS_DIR   = Path("reports")

CATEGORY       = "Sell-in"
OPT_THRESHOLD  = 0.0506   # from loop5_enhanced_report.json
PROB_THRESH_REGULAR = 0.3  # segment boundary for bias correction

FEAT_NAMES = [
    "lag_1", "lag_4", "lag_52",
    "rolling_mean_4", "rolling_mean_12", "rolling_std_12",
    "weeks_since_last_sale", "intermittent_flag",
    "week_sin", "week_cos", "inventory_days_of_supply",
    "lag_nonzero_1", "lag_nonzero_4", "lag_nonzero_52",
    "demand_rate_12", "zscore_vs_channel",
    "log_lag_1", "log_lag_4", "log_rolling_mean_12",
    "cv_12", "channel_nonzero_avg",
]
N_FEATS   = len(FEAT_NAMES)
WEEKS_IN_YEAR = 52.18
EPS = 1e-9


def _next_weeks(last_yw: int, n: int) -> list[int]:
    yr, wk = last_yw // 100, last_yw % 100
    d = date.fromisocalendar(yr, wk, 1)
    out = []
    for i in range(1, n + 1):
        d2 = d + timedelta(weeks=i)
        iso = d2.isocalendar()
        out.append(iso[0] * 100 + iso[1])
    return out


def _build_feature_matrix(
    histories:   np.ndarray,   # (n_skus, 104) — last 104 weeks per SKU (ordered oldest→newest)
    future_yw:   int,
    ch_means:    np.ndarray,   # (n_skus,)
    ch_stds:     np.ndarray,   # (n_skus,)
    ch_nz_avgs:  np.ndarray,   # (n_skus,)
) -> np.ndarray:
    """
    Build (n_skus, N_FEATS) feature matrix for a single future week.
    histories[:, -1] = most recent quantity, histories[:, 0] = oldest.
    """
    n = histories.shape[0]
    X = np.zeros((n, N_FEATS), dtype="float32")

    # Lag features (no leakage: last known value before this week)
    lag1  = histories[:, -1]
    lag4  = histories[:, -4]  if histories.shape[1] >= 4  else np.zeros(n)
    lag52 = histories[:, -52] if histories.shape[1] >= 52 else np.zeros(n)

    # Rolling stats (last 4 / 12 weeks)
    rm4  = histories[:, -4:].mean(axis=1)
    rm12 = histories[:, -12:].mean(axis=1) if histories.shape[1] >= 12 else histories.mean(axis=1)
    rs12 = histories[:, -12:].std(axis=1)  if histories.shape[1] >= 12 else histories.std(axis=1)

    # weeks_since_last_sale
    # Count from the right until we find a non-zero
    def wsls_vec(h: np.ndarray) -> np.ndarray:
        wsls = np.zeros(h.shape[0], dtype="float32")
        for col in range(h.shape[1] - 1, -1, -1):
            active = (wsls == 0) & (col < h.shape[1] - 1)
            wsls = np.where(h[:, col] > 0, h.shape[1] - 1 - col, wsls)
            if np.all(wsls > 0):
                break
        return wsls
    wsls = wsls_vec(histories)

    # intermittent_flag (>50% zeros in last 12 weeks)
    zeros12 = (histories[:, -12:] == 0).mean(axis=1)
    interm  = (zeros12 > 0.5).astype("float32")

    # Seasonal encoding
    wk_num = future_yw % 100
    wk_sin = np.full(n, np.sin(2 * np.pi * wk_num / WEEKS_IN_YEAR), dtype="float32")
    wk_cos = np.full(n, np.cos(2 * np.pi * wk_num / WEEKS_IN_YEAR), dtype="float32")

    # inventory_days_of_supply → not available for future weeks
    inv_dos = np.zeros(n, dtype="float32")

    # Loop 3 features
    nz1  = (lag1  > 0).astype("float32")
    nz4  = (lag4  > 0).astype("float32")
    nz52 = (lag52 > 0).astype("float32")
    dr12 = 1.0 - interm

    # zscore_vs_channel
    zscore = (rm12 - ch_means) / (ch_stds + EPS)

    # Loop 5 features
    log_l1  = np.log1p(lag1)
    log_l4  = np.log1p(lag4)
    log_rm12 = np.log1p(rm12)
    cv12    = np.clip(rs12 / (rm12 + EPS), 0, 10)

    X[:, 0]  = lag1
    X[:, 1]  = lag4
    X[:, 2]  = lag52
    X[:, 3]  = rm4
    X[:, 4]  = rm12
    X[:, 5]  = rs12
    X[:, 6]  = wsls
    X[:, 7]  = interm
    X[:, 8]  = wk_sin
    X[:, 9]  = wk_cos
    X[:, 10] = inv_dos
    X[:, 11] = nz1
    X[:, 12] = nz4
    X[:, 13] = nz52
    X[:, 14] = dr12
    X[:, 15] = zscore
    X[:, 16] = log_l1
    X[:, 17] = log_l4
    X[:, 18] = log_rm12
    X[:, 19] = cv12
    X[:, 20] = ch_nz_avgs

    return np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)


def run_recursive_forecast(horizon: int = 71) -> dict:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────────────
    log.info("Loading Loop 5 models...")
    clf = lgb.Booster(model_file=str(MODELS_DIR / "loop5_stage1_classifier.txt"))
    with open(MODELS_DIR / "loop5_stage1_calibrator.pkl", "rb") as fp:
        cal = pickle.load(fp)
    reg    = lgb.Booster(model_file=str(MODELS_DIR / "loop5_stage2_regressor.txt"))
    reg_q10 = lgb.Booster(model_file=str(MODELS_DIR / "loop5_stage2_q10.txt"))
    reg_q90 = lgb.Booster(model_file=str(MODELS_DIR / "loop5_stage2_q90.txt"))

    # ── Load bias correction factors ───────────────────────────────────────────
    bc = load_bias_factors()
    log.info(f"Bias factors — global={bc['global']}  regular={bc['regular']}  sparse={bc['sparse']}")

    # ── Load gold features to init histories ───────────────────────────────────
    log.info("Loading gold features to initialize SKU histories...")
    gold = pd.read_parquet(GOLD_PATH)
    gold = gold[gold["Category"] == CATEGORY].copy()
    gold["year_week"] = (
        gold["year_week"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    )
    gold["quantity"] = gold["quantity"].fillna(0).astype("float32")
    log.info(f"  Sell-in rows: {len(gold):,}")

    # Channel-level stats (from training data, fixed for all future weeks)
    ch_rm12_stats = (
        gold.groupby("Channel")["rolling_mean_12"]
        .agg(ch_mean="mean", ch_std="std")
    )
    ch_nz_stats = (
        gold[gold["quantity"] > 0]
        .groupby("Channel")["quantity"].mean()
        .rename("ch_nz_avg")
    )
    ch_stats_df = ch_rm12_stats.join(ch_nz_stats, how="left").fillna(0)

    # Build SKU list and history arrays
    log.info("Building SKU history arrays...")
    sorted_yws = sorted(gold["year_week"].unique())
    yw_to_idx  = {yw: i for i, yw in enumerate(sorted_yws)}
    n_weeks    = len(sorted_yws)

    sku_groups = gold.groupby(["Channel", "Material Description"])
    sku_keys   = list(sku_groups.groups.keys())
    n_skus     = len(sku_keys)
    log.info(f"  SKUs: {n_skus:,} | History weeks: {n_weeks}")

    # Store per-SKU quantities indexed by year_week
    sku_qty_by_yw: list[dict[int, float]] = []
    for (ch, mat), grp in sku_groups:
        qty_map = dict(zip(grp["year_week"].tolist(), grp["quantity"].tolist()))
        sku_qty_by_yw.append(qty_map)

    # Channel arrays for vectorized feature building
    ch_means   = np.array([ch_stats_df.loc[ch, "ch_mean"] if ch in ch_stats_df.index else 0.0
                           for (ch, _) in sku_keys], dtype="float32")
    ch_stds    = np.array([ch_stats_df.loc[ch, "ch_std"]  if ch in ch_stats_df.index else 1.0
                           for (ch, _) in sku_keys], dtype="float32")
    ch_nz_avgs = np.array([ch_stats_df.loc[ch, "ch_nz_avg"] if ch in ch_stats_df.index else 0.0
                           for (ch, _) in sku_keys], dtype="float32")

    # ── Generate future weeks ──────────────────────────────────────────────────
    last_yw = int(gold[gold["quantity"] > 0]["year_week"].max())
    future_weeks = _next_weeks(last_yw, horizon)
    log.info(f"Last actual week: {last_yw} | Forecasting {len(future_weeks)} weeks: {future_weeks[0]} → {future_weeks[-1]}")

    # History buffer: last 104 weeks per SKU, only up to last_yw (no future zeros)
    BUF = 104
    actual_yws = [yw for yw in sorted_yws if yw <= last_yw]
    buf_yws    = actual_yws[-BUF:]
    histories  = np.zeros((n_skus, BUF), dtype="float32")
    for i, qty_map in enumerate(sku_qty_by_yw):
        for j, yw in enumerate(buf_yws):
            histories[i, j] = qty_map.get(yw, 0.0)
    log.info(f"  History buffer: {buf_yws[0]} → {buf_yws[-1]} ({len(buf_yws)} weeks)")

    # ── Recursive inference ────────────────────────────────────────────────────
    t0 = time.time()
    all_records = []

    for week_idx, future_yw in enumerate(future_weeks):
        X = _build_feature_matrix(histories, future_yw, ch_means, ch_stds, ch_nz_avgs)

        # Stage 1: Demand probability
        raw_probs  = clf.predict(X).astype("float32")
        cal_probs  = cal.predict(raw_probs).astype("float32")

        # Diagnostic on first week
        if week_idx == 0:
            log.info(f"  [diag week 1] raw_prob: min={raw_probs.min():.4f} max={raw_probs.max():.4f} "
                     f"mean={raw_probs.mean():.4f} | cal_prob: min={cal_probs.min():.4f} "
                     f"max={cal_probs.max():.4f} mean={cal_probs.mean():.4f} | thresh={OPT_THRESHOLD}")

        demand_mask = cal_probs >= OPT_THRESHOLD

        # Stage 2: Quantity
        q50_raw = reg.predict(X).astype("float32")
        q10_raw = reg_q10.predict(X).astype("float32")
        q90_raw = reg_q90.predict(X).astype("float32")

        q50 = np.expm1(q50_raw).clip(0)
        q10 = np.expm1(q10_raw).clip(0)
        q90 = np.expm1(q90_raw).clip(0)

        # Bias correction on Q50 (per segment)
        bc_array = np.where(cal_probs > PROB_THRESH_REGULAR, bc["regular"], bc["sparse"]).astype("float32")
        q50_corrected = q50 * bc_array

        # Apply demand threshold
        final_q50 = np.where(demand_mask, q50_corrected, 0.0)
        final_q10 = np.where(demand_mask, q10,           0.0)
        final_q90 = np.where(demand_mask, q90,           0.0)

        # Store results
        for i, (ch, mat) in enumerate(sku_keys):
            if final_q50[i] > 0 or final_q90[i] > 0:   # skip pure-zero rows
                all_records.append((ch, mat, future_yw,
                                    float(final_q10[i]),
                                    float(final_q50[i]),
                                    float(final_q90[i])))

        # Update histories: shift left, append Q50 as the new "actual"
        histories = np.roll(histories, -1, axis=1)
        histories[:, -1] = final_q50

        if (week_idx + 1) % 10 == 0:
            log.info(f"  Week {week_idx+1}/{len(future_weeks)}: {future_yw} — "
                     f"{demand_mask.sum():,} SKUs with demand")

    elapsed = time.time() - t0
    log.info(f"Recursive forecast complete in {elapsed:.1f}s ({len(all_records):,} non-zero records)")

    # ── Save forecasts ─────────────────────────────────────────────────────────
    df_out = pd.DataFrame(
        all_records,
        columns=["Channel", "Material Description", "year_week",
                 "forecast_q10", "forecast_q50", "forecast_q90"],
    )
    df_out["year_week"] = df_out["year_week"].astype("int32")
    df_out["forecast_q10"] = df_out["forecast_q10"].astype("float32")
    df_out["forecast_q50"] = df_out["forecast_q50"].astype("float32")
    df_out["forecast_q90"] = df_out["forecast_q90"].astype("float32")

    out_path = FORECASTS_DIR / "loop7_recursive_forecasts.parquet"
    df_out.to_parquet(out_path, index=False)
    log.info(f"Saved → {out_path} ({df_out['year_week'].nunique()} weeks, {len(df_out):,} rows)")

    # Summary stats
    wk_totals = df_out.groupby("year_week")[["forecast_q10", "forecast_q50", "forecast_q90"]].sum()
    avg_wk_q50 = float(wk_totals["forecast_q50"].mean())
    total_q50  = float(df_out["forecast_q50"].sum())
    n_active_skus = int((df_out.groupby("Material Description")["forecast_q50"].sum() > 0).sum())

    result = {
        "pipeline":         "recursive_forecast_loop7",
        "last_actual_week": last_yw,
        "forecast_start":   future_weeks[0],
        "forecast_end":     future_weeks[-1],
        "horizon_weeks":    horizon,
        "n_skus_active":    n_active_skus,
        "total_q50":        round(total_q50, 0),
        "avg_weekly_q50":   round(avg_wk_q50, 0),
        "bias_factors":     bc,
        "elapsed_s":        round(elapsed, 1),
        "output_path":      str(out_path),
        "output_rows":      len(df_out),
    }

    import json
    out_rep = REPORTS_DIR / "loop7_recursive_forecast_report.json"
    with open(out_rep, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Report → {out_rep}")
    return result


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    r = run_recursive_forecast()
    print(f"\n{'='*60}")
    print(f"  LOOP 7 — RECURSIVE FORECAST")
    print(f"{'='*60}")
    print(f"  Horizon:       {r['forecast_start']} → {r['forecast_end']} ({r['horizon_weeks']} weeks)")
    print(f"  Active SKUs:   {r['n_active_skus']:,}")
    print(f"  Total Q50:     {r['total_q50']:,.0f} units")
    print(f"  Avg/week Q50:  {r['avg_weekly_q50']:,.0f} units")
    print(f"  Bias factors:  {r['bias_factors']}")
    print(f"  Runtime:       {r['elapsed_s']}s")
    print(f"{'='*60}\n")
