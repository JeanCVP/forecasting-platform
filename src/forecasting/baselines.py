"""
Baseline Forecasting Suite — 5 Models
All models implement BaseForecaster interface.
Designed for intermittent demand (median non-zero rate = 1%).

Models:
  1. SeasonalNaive    — y_hat[t+h] = y[t+h-52]
  2. RollingSeasonalMean — blend of seasonal + rolling
  3. Croston          — separate smoothing of non-zero demand and intervals
  4. SBA              — Syntetos-Boylan Approximation (bias-corrected Croston)
  5. TSB              — Teunter-Syntetos-Babai (probability-weighted)

Each model:
  - Accepts 1D numpy array (sorted time series)
  - Returns numpy array of length `horizon`
  - Is stateless (no side effects)
  - Handles edge cases: all-zero, single obs, very short history
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ─── Base Interface ───────────────────────────────────────────────────────────
class BaseForecaster(ABC):
    name: str = "base"
    supports_intermittent: bool = False

    @abstractmethod
    def fit_predict(
        self,
        series: np.ndarray,
        horizon: int = 13,
        **kwargs,
    ) -> np.ndarray:
        """
        Fit on `series` (sorted, no future data) and return `horizon` forecasts.
        All values should be >= 0.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def _safe_return(self, values: np.ndarray, horizon: int) -> np.ndarray:
        """Clip negatives, ensure length, fill NaN with 0."""
        out = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, 0.0, None)
        if len(out) < horizon:
            out = np.concatenate([out, np.zeros(horizon - len(out))])
        return out[:horizon]


# ─── 1. Seasonal Naïve ────────────────────────────────────────────────────────
class SeasonalNaive(BaseForecaster):
    """
    Forecast y_hat[t+h] = y[t+h-S] (last observed same-season value).
    Season S=52 for weekly data.
    Falls back to last value if history < S.
    """
    name = "seasonal_naive"
    supports_intermittent = True

    def __init__(self, season: int = 52):
        self.season = season

    def fit_predict(self, series: np.ndarray, horizon: int = 13, **kwargs) -> np.ndarray:
        n = len(series)
        if n == 0:
            return np.zeros(horizon)
        if n < self.season:
            # Not enough history: repeat last value
            return self._safe_return(np.full(horizon, series[-1]), horizon)
        # y_hat[h] = series[-(season - h % season)]
        preds = np.array([series[-self.season + (h % self.season)] for h in range(horizon)])
        return self._safe_return(preds, horizon)


# ─── 2. Rolling Seasonal Mean ─────────────────────────────────────────────────
class RollingSeasonalMean(BaseForecaster):
    """
    Weighted blend:
      preds[h] = alpha * seasonal_ref[h] + (1-alpha) * rolling_mean
    Where:
      - seasonal_ref = average of same week across available years
      - rolling_mean = mean of last `window` non-zero observations
    """
    name = "rolling_seasonal_mean"
    supports_intermittent = True

    def __init__(self, season: int = 52, window: int = 12, alpha: float = 0.5):
        self.season = season
        self.window = window
        self.alpha = alpha

    def fit_predict(self, series: np.ndarray, horizon: int = 13, **kwargs) -> np.ndarray:
        n = len(series)
        if n == 0:
            return np.zeros(horizon)

        # Rolling mean of last `window` observations (including zeros)
        w = min(self.window, n)
        rolling_mean = float(np.mean(series[-w:]))

        # Seasonal reference per lag position
        preds = []
        for h in range(horizon):
            lag = self.season - (h % self.season)
            # Collect all observations at this seasonal position
            refs = []
            pos = n - lag
            while pos >= 0:
                if pos < n:
                    refs.append(series[pos])
                pos -= self.season
            seasonal_ref = float(np.mean(refs)) if refs else rolling_mean
            blend = self.alpha * seasonal_ref + (1 - self.alpha) * rolling_mean
            preds.append(blend)

        return self._safe_return(np.array(preds), horizon)


# ─── 3. Croston's Method ──────────────────────────────────────────────────────
class Croston(BaseForecaster):
    """
    Croston (1972): Separate exponential smoothing of:
      - z̃: demand size (non-zero observations only)
      - p̃: inter-demand interval (periods between non-zeros)
    Forecast = z̃ / p̃

    Note: Croston is biased upward — SBA corrects this.
    """
    name = "croston"
    supports_intermittent = True

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit_predict(self, series: np.ndarray, horizon: int = 13, **kwargs) -> np.ndarray:
        n = len(series)
        if n == 0:
            return np.zeros(horizon)

        nonzero_idx = np.where(series > 0)[0]
        if len(nonzero_idx) == 0:
            return np.zeros(horizon)
        if len(nonzero_idx) == 1:
            return self._safe_return(
                np.full(horizon, series[nonzero_idx[0]] / nonzero_idx[0] if nonzero_idx[0] > 0 else series[nonzero_idx[0]]),
                horizon,
            )

        # Initialize from first non-zero
        z = float(series[nonzero_idx[0]])  # demand level
        p = float(nonzero_idx[0] + 1)     # interval level (periods to first demand)

        prev_idx = nonzero_idx[0]
        for idx in nonzero_idx[1:]:
            interval = float(idx - prev_idx)
            z = self.alpha * series[idx] + (1 - self.alpha) * z
            p = self.alpha * interval + (1 - self.alpha) * p
            prev_idx = idx

        forecast = z / max(p, 1e-9)
        return self._safe_return(np.full(horizon, forecast), horizon)


# ─── 4. SBA (Syntetos-Boylan Approximation) ───────────────────────────────────
class SBA(BaseForecaster):
    """
    SBA (Syntetos & Boylan 2005): Bias-corrected Croston.
    Forecast = (1 - alpha/2) * z̃ / p̃

    Proven to outperform Croston for intermittent demand
    by correcting the systematic upward bias.
    """
    name = "sba"
    supports_intermittent = True

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._croston = Croston(alpha=alpha)

    def fit_predict(self, series: np.ndarray, horizon: int = 13, **kwargs) -> np.ndarray:
        # Get Croston forecast then apply SBA correction
        croston_forecast = self._croston.fit_predict(series, horizon)
        correction = 1.0 - (self.alpha / 2.0)
        return self._safe_return(croston_forecast * correction, horizon)


# ─── 5. TSB (Teunter-Syntetos-Babai) ─────────────────────────────────────────
class TSB(BaseForecaster):
    """
    TSB (Teunter, Syntetos & Babai 2011):
    Updates demand probability and demand level separately.
    Better handles items that become obsolete (demand probability → 0).

    State:
      p_t: probability of demand in period t
      d_t: expected demand size given demand occurs
    Forecast = p_t * d_t
    """
    name = "tsb"
    supports_intermittent = True

    def __init__(self, alpha: float = 0.1, beta: float = 0.1):
        self.alpha = alpha  # smoothing for demand level
        self.beta = beta   # smoothing for demand probability

    def fit_predict(self, series: np.ndarray, horizon: int = 13, **kwargs) -> np.ndarray:
        n = len(series)
        if n == 0:
            return np.zeros(horizon)

        nonzero_idx = np.where(series > 0)[0]
        if len(nonzero_idx) == 0:
            return np.zeros(horizon)

        # Initialize
        first_nz = nonzero_idx[0]
        p = 1.0 / (first_nz + 1) if first_nz > 0 else 1.0  # initial demand probability
        d = float(series[first_nz])                          # initial demand level

        for t in range(first_nz + 1, n):
            demand_occurred = series[t] > 0
            if demand_occurred:
                p = (1 - self.beta) * p + self.beta * 1.0
                d = (1 - self.alpha) * d + self.alpha * float(series[t])
            else:
                p = (1 - self.beta) * p + self.beta * 0.0
                # d unchanged when no demand

        forecast = p * d
        return self._safe_return(np.full(horizon, forecast), horizon)


# ─── Model registry ───────────────────────────────────────────────────────────
ALL_MODELS: dict[str, BaseForecaster] = {
    "seasonal_naive":       SeasonalNaive(season=52),
    "rolling_seasonal_mean": RollingSeasonalMean(season=52, window=12, alpha=0.5),
    "croston":              Croston(alpha=0.1),
    "sba":                  SBA(alpha=0.1),
    "tsb":                  TSB(alpha=0.1, beta=0.1),
}


def get_model(name: str) -> BaseForecaster:
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(ALL_MODELS)}")
    return ALL_MODELS[name]
