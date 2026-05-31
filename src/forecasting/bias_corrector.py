"""
Bias Corrector — Loop 7
========================
Calcula factores de corrección multiplicativa a partir de las predicciones
de validación del fold 5 (loop6_ci_predictions.parquet).

Segmentos:
  regular : demand_prob > 0.3  →  bc_factor ≈ 1.88x
  sparse  : demand_prob ≤ 0.3  →  bc_factor ≈ 5.0x (capped)

El bias de -57% de Loop 5 se corrige multiplicando la predicción Q50 por el factor.
Los Q10/Q90 no se corrigen (representan el rango, no el punto central).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CI_PATH     = Path("data/forecasts/loop6_ci_predictions.parquet")
REPORTS_DIR = Path("reports")

PROB_THRESHOLD_REGULAR = 0.3   # demand_prob > this → "regular" segment
MAX_FACTOR             = 5.0   # cap for sparse segment (8x is too aggressive)
MIN_FACTOR             = 0.8   # floor to avoid over-correction


def compute_bias_factors() -> dict[str, float]:
    """
    Compute per-segment bias correction factors from fold-5 CI predictions.
    Returns {"regular": float, "sparse": float, "global": float}.
    """
    if not CI_PATH.exists():
        log.warning(f"CI predictions not found at {CI_PATH} — using factor=1.0")
        return {"regular": 1.0, "sparse": 1.0, "global": 1.0}

    ci = pd.read_parquet(CI_PATH)
    # Only rows where actual demand occurred
    nz = ci[ci["quantity_actual"] > 0].copy()

    if len(nz) == 0:
        return {"regular": 1.0, "sparse": 1.0, "global": 1.0}

    eps = 1e-9
    factors: dict[str, float] = {}

    # Global factor
    mean_act = float(nz["quantity_actual"].mean())
    mean_pred = float(nz["forecast_q50"].mean())
    factors["global"] = float(np.clip(mean_act / (mean_pred + eps), MIN_FACTOR, MAX_FACTOR))

    # By segment
    regular = nz[nz["demand_prob"] >  PROB_THRESHOLD_REGULAR]
    sparse  = nz[nz["demand_prob"] <= PROB_THRESHOLD_REGULAR]

    for name, df in [("regular", regular), ("sparse", sparse)]:
        if len(df) >= 20:
            ma = float(df["quantity_actual"].mean())
            mp = float(df["forecast_q50"].mean())
            f  = float(np.clip(ma / (mp + eps), MIN_FACTOR, MAX_FACTOR))
        else:
            f = factors["global"]
        factors[name] = round(f, 4)

    log.info(
        f"Bias correction factors — global={factors['global']:.3f}  "
        f"regular={factors['regular']:.3f}  sparse={factors['sparse']:.3f}"
    )
    return {k: round(v, 4) for k, v in factors.items()}


def save_bias_factors(factors: dict[str, float]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "loop7_bias_factors.json"
    with open(out, "w") as f:
        json.dump(factors, f, indent=2)
    log.info(f"Bias factors saved → {out}")
    return out


def load_bias_factors() -> dict[str, float]:
    p = REPORTS_DIR / "loop7_bias_factors.json"
    if p.exists():
        return json.load(open(p))
    return compute_bias_factors()
