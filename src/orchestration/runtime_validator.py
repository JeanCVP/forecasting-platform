from pathlib import Path
import sqlite3
import logging

logger = logging.getLogger(__name__)


def validate_runtime():
    """
    Minimal Loop 2 runtime validation.
    Prevents pipeline crash while validating critical runtime assets.
    """

    checks = {}

    # ─────────────────────────────────────────────
    # Silver dataset exists
    # ─────────────────────────────────────────────
    silver_path = Path("data/silver/silver_dataset.parquet")
    silver_csv = Path("data/silver/silver_dataset.csv")

    checks["silver_exists"] = silver_path.exists() or silver_csv.exists()

    # ─────────────────────────────────────────────
    # MLflow sqlite exists or can initialize
    # ─────────────────────────────────────────────
    mlruns = Path("mlruns")
    mlruns.mkdir(exist_ok=True)

    db_path = mlruns / "mlflow_runs.db"

    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS healthcheck (
                id INTEGER PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

        checks["mlflow_sqlite"] = True

    except Exception as e:
        logger.exception(e)
        checks["mlflow_sqlite"] = False

    # ─────────────────────────────────────────────
    # Reports dir
    # ─────────────────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    checks["reports_dir"] = reports_dir.exists()

    # ─────────────────────────────────────────────
    # Final result
    # ─────────────────────────────────────────────
    all_ok = all(checks.values())

    logger.info(f"Runtime validation checks: {checks}")

    return {
        "status": "success" if all_ok else "failed",
        "checks": checks,
    }
