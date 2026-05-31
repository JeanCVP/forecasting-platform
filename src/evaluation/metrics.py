"""
Forecast Evaluation Metrics
Intermittent-demand-aware metrics for weekly sell-in data.

Metrics:
  sMAPE  — Symmetric MAPE (handles zeros better than MAPE)
  WAPE   — Weighted Absolute Percentage Error (volume-weighted, robust to zeros)
  MASE   — Mean Absolute Scaled Error (scale-free, compares to naïve)
  bias   — Mean signed error / mean actual (directional bias %)
  SLP    — Service Level Proxy (% of weeks where forecast >= actual)
  MAE    — Mean Absolute Error (absolute scale)
  RMSE   — Root Mean Squared Error

Design:
  - All functions handle zeros and NaNs gracefully
  - MASE uses seasonal naïve (lag-52) as the scaling denominator
  - sMAPE uses (|actual| + |forecast|) / 2 denominator to avoid division by zero
  - All return finite floats (no NaN/Inf in output)
"""
from __future__ import annotations

import numpy as np


EPS = 1e-9  # avoid division by zero


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Symmetric MAPE = mean( 2*|a-f| / (|a| + |f| + eps) ) * 100
    Range: [0, 200]. Returns 0 if all actuals are 0.
    """
    denom = np.abs(actual) + np.abs(forecast) + EPS
    return float(np.mean(2.0 * np.abs(actual - forecast) / denom) * 100)


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Weighted APE = sum(|a-f|) / (sum(a) + eps) * 100
    Volume-weighted — naturally handles zeros (zero rows contribute 0 to numerator).
    Equivalent to MAE / mean(actual).
    """
    total_actual = np.sum(np.abs(actual))
    if total_actual < EPS:
        return 0.0
    return float(np.sum(np.abs(actual - forecast)) / total_actual * 100)


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    train_series: np.ndarray | None = None,
    season: int = 52,
) -> float:
    """
    Mean Absolute Scaled Error = MAE(model) / MAE(seasonal_naive)
    Scale denominator: MAE of seasonal naïve on training data.
    If train_series not provided, uses actual itself for scaling (less ideal).
    Values < 1 mean the model beats seasonal naïve.
    """
    mae_model = float(np.mean(np.abs(actual - forecast)))

    if train_series is not None and len(train_series) > season:
        naive_errors = np.abs(train_series[season:] - train_series[:-season])
        mae_naive = float(np.mean(naive_errors))
    else:
        # Fallback: use in-sample naive on actuals
        if len(actual) > 1:
            mae_naive = float(np.mean(np.abs(np.diff(actual))))
        else:
            mae_naive = float(np.mean(np.abs(actual))) + EPS

    return mae_model / (mae_naive + EPS)


def bias_pct(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Mean signed error as % of mean actual.
    Positive = over-forecast, negative = under-forecast.
    """
    mean_actual = float(np.mean(np.abs(actual)))
    if mean_actual < EPS:
        return 0.0
    return float(np.mean(forecast - actual)) / mean_actual * 100


def service_level_proxy(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    % of weeks where forecast >= actual (stock-cover proxy).
    Higher = more conservative (less stockout risk).
    Range: [0, 1].
    """
    if len(actual) == 0:
        return 0.0
    return float(np.mean(forecast >= actual))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def active_wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """WAPE restricted to weeks where actual > 0 (ignores structural zeros)."""
    mask = actual > 0
    if mask.sum() == 0:
        return 0.0
    return wape(actual[mask], forecast[mask])


def demand_f1(actual: np.ndarray, forecast: np.ndarray, threshold: float = 0.5) -> float:
    """F1 for binary demand occurrence: actual>0 vs forecast>threshold."""
    from sklearn.metrics import f1_score
    y_true = (actual > 0).astype(int)
    y_pred = (forecast > threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0.0))


def pinball_loss(actual: np.ndarray, forecast: np.ndarray, quantile: float = 0.5) -> float:
    """Pinball (quantile) loss at given quantile level."""
    diff = actual - forecast
    return float(np.mean(np.where(diff >= 0, quantile * diff, (quantile - 1.0) * diff)))


def compute_all_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    train_series: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute full metric suite. All inputs are 1D numpy arrays.
    Returns dict with all metrics as finite floats.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    # Clip negatives in forecast (models should not produce negatives)
    forecast = np.clip(forecast, 0.0, None)

    # Handle empty arrays
    if len(actual) == 0 or len(forecast) == 0:
        return {k: 0.0 for k in ("smape", "wape", "mase", "bias", "slp", "mae", "rmse")}

    metrics = {
        "smape":       smape(actual, forecast),
        "wape":        wape(actual, forecast),
        "active_wape": active_wape(actual, forecast),
        "mase":        mase(actual, forecast, train_series),
        "bias":        bias_pct(actual, forecast),
        "slp":         service_level_proxy(actual, forecast),
        "demand_f1":   demand_f1(actual, forecast),
        "pinball_50":  pinball_loss(actual, forecast, quantile=0.5),
        "mae":         mae(actual, forecast),
        "rmse":        rmse(actual, forecast),
    }

    # Sanitize: replace any NaN/Inf with 0
    return {
        k: float(v) if np.isfinite(v) else 0.0
        for k, v in metrics.items()
    }


def compute_per_sku_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    channel: str,
    material: str,
) -> dict[str, float]:
    """Per-SKU metrics with identification."""
    metrics = compute_all_metrics(actual, forecast)
    metrics["channel"] = channel
    metrics["material"] = material
    metrics["n_obs"] = len(actual)
    metrics["actual_sum"] = float(np.sum(actual))
    metrics["forecast_sum"] = float(np.sum(forecast))
    return metrics
