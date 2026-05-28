"""
Bronze Layer Ingestion Pipeline
Reads raw CSVs, validates schema, detects type issues,
saves to Bronze parquet (via pandas), generates ingestion metadata and lineage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ─── Config ───────────────────────────────────────────────────────────────────
RAW_FILES = {"2023": "2023.csv", "2024": "2024.csv", "2025": "2025.csv"}
BRONZE_DIR = Path("data/bronze")
REPORTS_DIR = Path("reports")
ID_COLS = ["Channel", "Material Description", "Category"]
VALID_CATEGORIES = {"Sell-in", "Cust. Sales", "Channel Inv."}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_dtype_issues(df: pd.DataFrame, year: str) -> list[dict]:
    issues = []
    week_cols = [c for c in df.columns if c not in ID_COLS]
    for col in week_cols:
        if df[col].dtype == object:
            # Count comma-formatted values
            sample = df[col].dropna().head(100)
            comma_count = sample.astype(str).str.contains(",").sum()
            issues.append({
                "year": year, "column": col,
                "dtype_found": "object", "dtype_expected": "float64",
                "comma_formatted_sample_count": int(comma_count),
            })
    return issues


def _validate_schema(df: pd.DataFrame, year: str) -> dict:
    missing_cols = [c for c in ID_COLS if c not in df.columns]
    invalid_cats = list(set(df["Category"].dropna().unique()) - VALID_CATEGORIES) if "Category" in df.columns else []
    null_id = {col: int(df[col].isna().sum()) for col in ID_COLS if col in df.columns}
    week_cols = [c for c in df.columns if c not in ID_COLS]
    return {
        "year": year, "rows": len(df), "week_columns": len(week_cols),
        "missing_required_columns": missing_cols,
        "invalid_category_values": invalid_cats,
        "null_counts_id_columns": null_id,
        "is_valid": len(missing_cols) == 0 and len(invalid_cats) == 0,
    }


def _compute_dataset_hash(df: pd.DataFrame) -> str:
    fingerprint = f"{list(df.columns)}{len(df)}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()


def run_ingestion(data_dir: str = "data/raw") -> dict:
    data_path = Path(data_dir)
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    ingestion_report = {
        "pipeline": "bronze_ingestion",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_path),
        "files": {},
        "lineage": [],
    }
    schema_validation = {
        "pipeline": "schema_validation",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
        "dtype_issues": [],
        "overall_valid": True,
    }

    for year, fname in RAW_FILES.items():
        fpath = data_path / fname
        if not fpath.exists():
            log.warning(f"File not found: {fpath} — skipping")
            continue

        log.info(f"=== Processing {fname} (year={year}) ===")
        file_hash = _sha256_file(fpath)
        log.info(f"  SHA-256: {file_hash[:16]}... | size: {fpath.stat().st_size/1e6:.1f}MB")

        # Read with all strings to preserve raw data
        df = pd.read_csv(fpath, dtype=str, na_values=["", "NULL", "null", "NA"])
        log.info(f"  Read: {len(df):,} rows × {len(df.columns)} columns")

        sv = _validate_schema(df, year)
        schema_validation["results"].append(sv)
        if not sv["is_valid"]:
            schema_validation["overall_valid"] = False
            log.warning(f"  Schema issues: {sv}")
        else:
            log.info(f"  Schema OK: {sv['rows']:,} rows, {sv['week_columns']} weeks")

        dtype_issues = _detect_dtype_issues(df, year)
        if dtype_issues:
            log.warning(f"  {len(dtype_issues)} week columns have dtype issues in {year}")
            schema_validation["dtype_issues"].extend(dtype_issues[:5])  # sample

        ds_hash = _compute_dataset_hash(df)

        # Save bronze as parquet (using pickle fallback)
        bronze_path = BRONZE_DIR / f"bronze_{year}.parquet"
        try:
            df.to_parquet(bronze_path, index=False, engine="pyarrow")
        except Exception:
            # Fallback: save as CSV if pyarrow unavailable
            bronze_path = BRONZE_DIR / f"bronze_{year}.csv"
            df.to_csv(bronze_path, index=False)
        log.info(f"  Saved bronze: {bronze_path}")

        ingestion_report["files"][year] = {
            "source_file": str(fpath), "source_hash_sha256": file_hash,
            "dataset_hash": ds_hash, "rows": len(df), "columns": len(df.columns),
            "bronze_path": str(bronze_path),
        }
        ingestion_report["lineage"].append({
            "event": "raw_to_bronze", "year": year,
            "source": str(fpath), "destination": str(bronze_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rows_written": len(df), "source_hash": file_hash,
        })

    with open(REPORTS_DIR / "ingestion_report.json", "w") as f:
        json.dump(ingestion_report, f, indent=2, default=str)
    with open(REPORTS_DIR / "schema_validation.json", "w") as f:
        json.dump(schema_validation, f, indent=2, default=str)

    log.info(f"Ingestion complete. Files: {list(ingestion_report['files'].keys())}")
    return ingestion_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    run_ingestion(data_dir=args.data_dir)
