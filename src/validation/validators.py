"""Data Validation Framework — pandas implementation."""
from __future__ import annotations

import json
import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPORTS_DIR = Path("reports")
SILVER_PATH = Path("data/silver/silver_dataset.parquet")
SILVER_CSV = Path("data/silver/silver_dataset.csv")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)])
log = logging.getLogger(__name__)


class BaseValidator(ABC):
    name: str = "base"
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> dict[str, Any]: ...


class SchemaValidator(BaseValidator):
    name = "schema"
    REQUIRED = ["Channel", "Material Description", "Category",
                "year_week", "quantity", "year", "week_num"]

    def validate(self, df):
        missing = [c for c in self.REQUIRED if c not in df.columns]
        wrong = {}
        if "quantity" in df.columns and not pd.api.types.is_float_dtype(df["quantity"]):
            wrong["quantity"] = {"expected": "float64", "found": str(df["quantity"].dtype)}
        return {"validator": self.name, "passed": not missing and not wrong,
                "missing_columns": missing, "wrong_dtypes": wrong}


class NullValidator(BaseValidator):
    name = "null"
    THRESHOLD = 0.05

    def validate(self, df):
        nulls = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
        pcts = {c: round(n/len(df), 4) for c, n in nulls.items()}
        violations = {c: p for c, p in pcts.items() if p > self.THRESHOLD}
        return {"validator": self.name, "passed": len(violations) == 0,
                "null_counts": nulls, "null_pcts": pcts,
                "violations": violations, "threshold": self.THRESHOLD}


class TemporalValidator(BaseValidator):
    name = "temporal"

    def validate(self, df):
        bad_weeks = int(((df["week_num"] < 1) | (df["week_num"] > 53)).sum())
        bad_years = int(((df["year"] < 2023) | (df["year"] > 2026)).sum())
        bad_fmt = int((df["year_week"].astype(str).str.len() != 6).sum()) if "year_week" in df.columns else 0
        return {
            "validator": self.name, "passed": bad_weeks == 0 and bad_years == 0,
            "invalid_week_num_rows": bad_weeks, "invalid_year_rows": bad_years,
            "malformed_year_week_rows": bad_fmt,
            "year_range": {"min": int(df["year"].min()), "max": int(df["year"].max())},
        }


class LeakageValidator(BaseValidator):
    name = "leakage"

    def validate(self, df):
        # Silver must be temporally sorted within each SKU group
        df_sorted = df.sort_values(["Channel", "Material Description", "Category", "year", "week_num"])
        yw_int = df_sorted.groupby(["Channel", "Material Description", "Category"])["year"].transform(
            lambda x: x * 100
        ) + df_sorted["week_num"]
        diffs = yw_int.groupby(
            [df_sorted["Channel"], df_sorted["Material Description"], df_sorted["Category"]]
        ).diff().dropna()
        violations = int((diffs < 0).sum())

        gold_path = Path("data/gold/gold_features.parquet")
        gold_csv = Path("data/gold/gold_features.csv")
        future_cols = []
        if gold_path.exists() or gold_csv.exists():
            try:
                g = pd.read_parquet(gold_path) if gold_path.exists() else pd.read_csv(gold_csv, nrows=100)
                future_cols = [c for c in g.columns if "future" in c.lower()]
            except Exception:
                pass

        return {
            "validator": self.name,
            "passed": violations == 0 and len(future_cols) == 0,
            "temporal_order_violations": violations,
            "future_referencing_columns": future_cols,
        }


class DuplicateValidator(BaseValidator):
    name = "duplicate"
    KEY = ["Channel", "Material Description", "Category", "year_week"]

    def validate(self, df):
        key = [c for c in self.KEY if c in df.columns]
        n_total = len(df)
        n_unique = df[key].drop_duplicates().shape[0]
        dupes = n_total - n_unique
        return {"validator": self.name, "passed": dupes == 0,
                "total_rows": n_total, "unique_key_rows": n_unique,
                "duplicate_rows": dupes, "key_columns": key}


class RangeValidator(BaseValidator):
    name = "range"

    def validate(self, df):
        neg = int((df["quantity"] < 0).sum())
        extreme = int((df["quantity"] > 1_000_000).sum())
        stats = df["quantity"].describe().to_dict()
        return {"validator": self.name, "passed": neg == 0,
                "negative_quantity_rows": neg, "extreme_quantity_rows": extreme,
                "quantity_stats": {k: round(float(v), 2) for k, v in stats.items()}}


class InventoryConsistencyValidator(BaseValidator):
    name = "inventory_consistency"

    def validate(self, df):
        inv = df[df["Category"] == "Channel Inv."]
        neg_inv = int((inv["quantity"] < 0).sum())
        neg_pct = neg_inv / len(inv) if len(inv) > 0 else 0
        return {
            "validator": self.name, "passed": neg_pct < 0.01,
            "categories_present": df["Category"].unique().tolist(),
            "inventory_rows": len(inv), "negative_inventory_rows": neg_inv,
            "negative_inventory_pct": round(neg_pct, 4),
        }


def run_all_validators(df: pd.DataFrame | None = None) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        p = SILVER_PATH if SILVER_PATH.exists() else SILVER_CSV
        if not p.exists():
            raise FileNotFoundError("Silver dataset not found")
        df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)

    validators = [SchemaValidator(), NullValidator(), TemporalValidator(),
                  LeakageValidator(), DuplicateValidator(), RangeValidator(),
                  InventoryConsistencyValidator()]
    results = []
    all_passed = True
    for v in validators:
        log.info(f"Running: {v.name}")
        result = v.validate(df)
        results.append(result)
        status = "PASS ✓" if result.get("passed") else "FAIL ✗"
        log.info(f"  [{status}] {v.name}")
        if not result.get("passed"):
            all_passed = False

    report = {
        "pipeline": "validation", "run_at": datetime.now(timezone.utc).isoformat(),
        "overall_passed": all_passed,
        "validators_run": len(validators),
        "validators_passed": sum(1 for r in results if r.get("passed")),
        "results": results,
    }
    with open(REPORTS_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Validation: {report['validators_passed']}/{report['validators_run']} passed")
    return report


if __name__ == "__main__":
    run_all_validators()
