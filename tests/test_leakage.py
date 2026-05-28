"""
Leakage Prevention Tests — CRITICAL: All must pass before training.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SILVER_PARQUET = Path("data/silver/silver_dataset.parquet")
SILVER_CSV = Path("data/silver/silver_dataset.csv")
GOLD_PARQUET = Path("data/gold/gold_features.parquet")
GOLD_CSV = Path("data/gold/gold_features.csv")


def _load(parquet_path, csv_path):
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@pytest.fixture(scope="module")
def silver():
    df = _load(SILVER_PARQUET, SILVER_CSV)
    if df is None:
        pytest.skip("Silver dataset not found")
    return df


@pytest.fixture(scope="module")
def gold():
    df = _load(GOLD_PARQUET, GOLD_CSV)
    if df is None:
        pytest.skip("Gold features not found")
    return df


@pytest.fixture(scope="module")
def sample_sku(silver):
    key = silver.iloc[0][["Channel", "Material Description", "Category"]]
    return (silver[
        (silver["Channel"] == key["Channel"]) &
        (silver["Material Description"] == key["Material Description"]) &
        (silver["Category"] == key["Category"])
    ].sort_values(["year", "week_num"]).reset_index(drop=True))


# ─── 1. Temporal Ordering ─────────────────────────────────────────────────────
def test_silver_temporal_ordering(silver):
    """Within each SKU, year×week must be non-decreasing."""
    grps = ["Channel", "Material Description", "Category"]
    df = silver.sort_values(grps + ["year", "week_num"])
    df["yw_int"] = df["year"] * 100 + df["week_num"]
    violations = df.groupby(grps)["yw_int"].apply(lambda x: (x.diff().dropna() < 0).sum()).sum()
    assert violations == 0, f"Temporal ordering violated in {int(violations)} transitions"


def test_year_week_string_consistency(silver):
    """year and week_num must match year_week string."""
    year_from_str = silver["year_week"].astype(str).str[:4].astype(int)
    week_from_str = silver["year_week"].astype(str).str[4:].astype(int)
    year_mismatch = (silver["year"] != year_from_str).sum()
    week_mismatch = (silver["week_num"] != week_from_str).sum()
    assert year_mismatch == 0, f"year mismatch in {year_mismatch} rows"
    assert week_mismatch == 0, f"week_num mismatch in {week_mismatch} rows"


# ─── 2. Lag Feature Correctness ───────────────────────────────────────────────
def test_lag_1_references_past_only(gold, sample_sku):
    """lag_1 at row i must equal quantity at row i-1 for same SKU."""
    if "lag_1" not in gold.columns:
        pytest.skip("lag_1 not in gold features")
    grps = ["Channel", "Material Description", "Category"]
    key = sample_sku.iloc[0][grps]
    sku_gold = (gold[
        (gold["Channel"] == key["Channel"]) &
        (gold["Material Description"] == key["Material Description"]) &
        (gold["Category"] == key["Category"])
    ].sort_values(["year", "week_num"]).reset_index(drop=True))

    if len(sku_gold) < 3:
        pytest.skip("Insufficient rows")

    # First row must have null lag_1
    first_lag = sku_gold.iloc[0]["lag_1"]
    assert pd.isna(first_lag), f"lag_1 at first row should be NaN, got {first_lag}"

    # Rows 1..N: lag_1[i] == quantity[i-1]
    mismatches = []
    for i in range(1, min(len(sku_gold), 20)):
        expected = sku_gold.iloc[i-1]["quantity"]
        actual = sku_gold.iloc[i]["lag_1"]
        if not pd.isna(actual) and not pd.isna(expected):
            if abs(float(actual) - float(expected)) > 1e-6:
                mismatches.append((i, actual, expected))
    assert len(mismatches) == 0, f"lag_1 mismatch in {len(mismatches)} rows: {mismatches[:3]}"


def test_lag_52_null_at_sku_start(gold):
    """lag_52 must be NaN at first 52 rows of each SKU (no prior history)."""
    if "lag_52" not in gold.columns:
        pytest.skip("lag_52 not in gold features")
    grps = ["Channel", "Material Description", "Category"]
    first_rows = (
        gold.sort_values(grps + ["year", "week_num"])
        .groupby(grps)
        .nth(0)
        .reset_index()
    )
    non_null_lag52 = first_rows["lag_52"].notna() & (first_rows["lag_52"] != 0)
    assert non_null_lag52.sum() == 0, \
        f"{non_null_lag52.sum()} SKUs have non-NaN lag_52 at first row"


# ─── 3. Rolling Window Shift ──────────────────────────────────────────────────
def test_rolling_mean_4_shifted(gold, sample_sku):
    """rolling_mean_4 must equal mean of PAST 4 weeks (not including current)."""
    if "rolling_mean_4" not in gold.columns:
        pytest.skip("rolling_mean_4 not in gold features")
    grps = ["Channel", "Material Description", "Category"]
    key = sample_sku.iloc[0][grps]
    sku = (gold[
        (gold["Channel"] == key["Channel"]) &
        (gold["Material Description"] == key["Material Description"]) &
        (gold["Category"] == key["Category"])
    ].sort_values(["year", "week_num"]).reset_index(drop=True))

    if len(sku) < 6:
        pytest.skip("Insufficient rows")

    qty = sku["quantity"].values
    rm4 = sku["rolling_mean_4"].values
    mismatches = []
    for i in range(4, min(len(qty), 20)):
        if not np.isnan(rm4[i]):
            expected = float(np.mean(qty[i-4:i]))
            if abs(float(rm4[i]) - expected) > 0.5:
                mismatches.append((i, rm4[i], expected))
    assert len(mismatches) == 0, f"rolling_mean_4 not shifted in {len(mismatches)} rows: {mismatches[:3]}"


# ─── 4. Cutoff Correctness ────────────────────────────────────────────────────
def test_no_future_columns_in_gold(gold):
    """Gold features must not contain any 'future' columns."""
    future_cols = [c for c in gold.columns if "future" in c.lower()]
    assert len(future_cols) == 0, f"Future-referencing columns: {future_cols}"


def test_train_test_split_no_overlap(gold):
    """If split column exists: train max yw < test min yw."""
    if "split" not in gold.columns:
        pytest.skip("split column not yet applied")
    train = gold[gold["split"] == "train"]
    test = gold[gold["split"] == "test"]
    train_max = (train["year"] * 100 + train["week_num"]).max()
    test_min = (test["year"] * 100 + test["week_num"]).min()
    assert train_max < test_min, f"Train/test overlap: train_max={train_max} >= test_min={test_min}"


# ─── 5. Feature Causality ─────────────────────────────────────────────────────
def test_intermittent_flag_is_binary(gold):
    """intermittent_flag must be 0 or 1."""
    if "intermittent_flag" not in gold.columns:
        pytest.skip("intermittent_flag not in gold")
    valid = gold["intermittent_flag"].isin([0, 1, 0.0, 1.0, True, False]) | gold["intermittent_flag"].isna()
    invalid_count = (~valid).sum()
    assert invalid_count == 0, f"intermittent_flag has {invalid_count} non-binary values"


def test_weeks_since_last_sale_non_negative(gold):
    """weeks_since_last_sale must be >= 0."""
    if "weeks_since_last_sale" not in gold.columns:
        pytest.skip("weeks_since_last_sale not in gold")
    neg = (gold["weeks_since_last_sale"] < 0).sum()
    assert neg == 0, f"weeks_since_last_sale has {neg} negative values"


# ─── 6. Silver Integrity ──────────────────────────────────────────────────────
def test_no_duplicate_keys_silver(silver):
    """Each (Channel, SKU, Category, year_week) must be unique in silver."""
    key = ["Channel", "Material Description", "Category", "year_week"]
    dupes = silver.duplicated(subset=key).sum()
    assert dupes == 0, f"Silver has {dupes} duplicate keys"


def test_silver_quantity_non_negative(silver):
    """Quantity must be >= 0 after cleaning."""
    neg = (silver["quantity"] < 0).sum()
    assert neg == 0, f"{neg} rows have negative quantity in silver"
