"""
MLflow Tracking Backend — Production-grade with SQLite fallback.

Design principles:
  - Same interface whether MLflow server is available or not
  - All runs stored in mlruns/mlflow_runs.db (SQLite)
  - Artifacts written to mlruns/artifacts/<experiment>/<run_id>/
  - Reproducible run IDs (deterministic from params hash)
  - Thread-safe (WAL mode SQLite)
  - Structured run metadata: params, metrics, tags, artifacts, git hash
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

log = logging.getLogger(__name__)

DB_PATH = Path("mlruns/mlflow_runs.db")
ARTIFACTS_ROOT = Path("mlruns/artifacts")
SCHEMA_VERSION = 2


# ─── Schema ───────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,
    created_at      TEXT NOT NULL,
    artifact_root   TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    run_name        TEXT NOT NULL,
    status          TEXT DEFAULT 'RUNNING',
    start_time      TEXT,
    end_time        TEXT,
    duration_s      REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS run_params (
    run_id  TEXT,
    key     TEXT,
    value   TEXT,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id  TEXT,
    key     TEXT,
    value   REAL,
    step    INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, key, step)
);
CREATE TABLE IF NOT EXISTS run_tags (
    run_id  TEXT,
    key     TEXT,
    value   TEXT,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS run_artifacts (
    run_id      TEXT,
    name        TEXT,
    path        TEXT,
    size_bytes  INTEGER,
    PRIMARY KEY (run_id, name)
);
"""


# ─── Tracker ──────────────────────────────────────────────────────────────────
class RunTracker:
    """
    Production MLflow-compatible tracker.
    Uses SQLite with WAL mode for concurrent access safety.
    Falls through to real MLflow if available.
    """

    _local = threading.local()

    def __init__(
        self,
        db_path: Path = DB_PATH,
        artifacts_root: Path = ARTIFACTS_ROOT,
        use_mlflow: bool = True,
    ):
        self.db_path = db_path
        self.artifacts_root = artifacts_root
        self._mlflow = None

        db_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts_root.mkdir(parents=True, exist_ok=True)

        # Try real MLflow first
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(
                    os.getenv("MLFLOW_TRACKING_URI", str(db_path.parent))
                )
                self._mlflow = mlflow
                log.info("MLflow tracking: ACTIVE (real MLflow)")
            except ImportError:
                log.info("MLflow not installed — using SQLite tracker")

        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """Per-thread connection with WAL mode."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO meta VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        conn.commit()

    def get_or_create_experiment(self, name: str) -> str:
        conn = self._conn()
        row = conn.execute(
            "SELECT experiment_id FROM experiments WHERE name=?", (name,)
        ).fetchone()
        if row:
            return row["experiment_id"]
        exp_id = hashlib.md5(name.encode()).hexdigest()[:12]
        artifact_root = str(self.artifacts_root / exp_id)
        conn.execute(
            "INSERT INTO experiments VALUES (?,?,?,?)",
            (exp_id, name, datetime.now(timezone.utc).isoformat(), artifact_root),
        )
        conn.commit()
        log.info(f"Created experiment '{name}' (id={exp_id})")
        return exp_id

    @contextmanager
    def start_run(
        self,
        experiment: str,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator["ActiveRun", None, None]:
        """Context manager — yields ActiveRun, commits on exit."""
        exp_id = self.get_or_create_experiment(experiment)

        # Deterministic run_id from experiment + name + params hash
        params = params or {}
        param_sig = json.dumps(params, sort_keys=True)
        run_id = hashlib.sha256(
            f"{experiment}:{run_name}:{param_sig}".encode()
        ).hexdigest()[:16]

        conn = self._conn()
        start = datetime.now(timezone.utc)
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
            (run_id, exp_id, run_name, "RUNNING",
             start.isoformat(), None, None),
        )
        conn.commit()

        # Log params
        for k, v in params.items():
            conn.execute(
                "INSERT OR REPLACE INTO run_params VALUES (?,?,?)",
                (run_id, str(k), str(v)),
            )
        # Log tags
        for k, v in (tags or {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO run_tags VALUES (?,?,?)",
                (run_id, str(k), str(v)),
            )
        conn.commit()

        artifact_dir = self.artifacts_root / exp_id / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        run = ActiveRun(
            run_id=run_id,
            experiment=experiment,
            exp_id=exp_id,
            run_name=run_name,
            artifact_dir=artifact_dir,
            conn=conn,
            tracker=self,
        )

        # Also log to real MLflow if available
        mlflow_run = None
        if self._mlflow:
            try:
                self._mlflow.set_experiment(experiment)
                mlflow_run = self._mlflow.start_run(run_name=run_name)
                if params:
                    self._mlflow.log_params(params)
                if tags:
                    self._mlflow.set_tags(tags)
            except Exception as e:
                log.warning(f"MLflow log failed: {e}")

        try:
            yield run
            status = "FINISHED"
        except Exception:
            status = "FAILED"
            raise
        finally:
            end = datetime.now(timezone.utc)
            duration = (end - start).total_seconds()
            conn.execute(
                "UPDATE runs SET status=?, end_time=?, duration_s=? WHERE run_id=?",
                (status, end.isoformat(), duration, run_id),
            )
            conn.commit()
            if mlflow_run and self._mlflow:
                try:
                    self._mlflow.end_run()
                except Exception:
                    pass
            log.info(f"Run '{run_name}' [{run_id}] → {status} ({duration:.1f}s)")

    def get_runs(self, experiment: str) -> list[dict]:
        exp_id = self.get_or_create_experiment(experiment)
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM runs WHERE experiment_id=? ORDER BY start_time DESC",
            (exp_id,),
        ).fetchall()
        result = []
        for row in rows:
            r = dict(row)
            r["metrics"] = {
                m["key"]: m["value"]
                for m in conn.execute(
                    "SELECT key, value FROM run_metrics WHERE run_id=? AND step=0",
                    (row["run_id"],),
                ).fetchall()
            }
            r["params"] = {
                p["key"]: p["value"]
                for p in conn.execute(
                    "SELECT key, value FROM run_params WHERE run_id=?",
                    (row["run_id"],),
                ).fetchall()
            }
            result.append(r)
        return result

    def get_best_run(self, experiment: str, metric: str, mode: str = "min") -> dict | None:
        runs = [r for r in self.get_runs(experiment) if metric in r.get("metrics", {})]
        if not runs:
            return None
        key = lambda r: r["metrics"][metric]
        return min(runs, key=key) if mode == "min" else max(runs, key=key)


class ActiveRun:
    """Live run handle yielded by RunTracker.start_run()."""

    def __init__(
        self, run_id, experiment, exp_id, run_name,
        artifact_dir, conn, tracker
    ):
        self.run_id = run_id
        self.experiment = experiment
        self.exp_id = exp_id
        self.run_name = run_name
        self.artifact_dir = artifact_dir
        self._conn = conn
        self._tracker = tracker

    def log_metric(self, key: str, value: float, step: int = 0):
        self._conn.execute(
            "INSERT OR REPLACE INTO run_metrics VALUES (?,?,?,?)",
            (self.run_id, key, float(value), step),
        )
        self._conn.commit()
        if self._tracker._mlflow:
            try:
                self._tracker._mlflow.log_metric(key, value, step=step)
            except Exception:
                pass

    def log_metrics(self, metrics: dict[str, float], step: int = 0):
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_artifact(self, src_path: str | Path, name: str | None = None):
        src = Path(src_path)
        if not src.exists():
            log.warning(f"Artifact not found: {src}")
            return
        name = name or src.name
        dst = self.artifact_dir / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        size = dst.stat().st_size
        self._conn.execute(
            "INSERT OR REPLACE INTO run_artifacts VALUES (?,?,?,?)",
            (self.run_id, name, str(dst), size),
        )
        self._conn.commit()
        if self._tracker._mlflow:
            try:
                self._tracker._mlflow.log_artifact(str(src))
            except Exception:
                pass

    def log_dict(self, data: dict, filename: str):
        """Write dict as JSON artifact."""
        path = self.artifact_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self.log_artifact(path, filename)

    def __repr__(self) -> str:
        return f"<ActiveRun run_id={self.run_id} name={self.run_name}>"
