# Audit Framework
**Project:** AI-DLC-FORECAST-COL-001
**Iteration:** 3
**Last Updated:** 2026-05-25
**AI-DLC Traceability ID:** AUDIT-ITER3-001

---

## 1. Principios de Auditoría

El sistema de auditoría garantiza que **cada forecast pueda ser trazado de vuelta a sus datos fuente, el modelo que lo generó, y la versión de features usada**. Esto es crítico para:

- Regulatorio: justificar decisiones de replenishment ante auditores
- Operacional: debuggear forecasts incorrectos
- ML: reproducir cualquier run de entrenamiento exactamente
- Governance: demostrar data lineage completo

---

## 2. Eventos Auditables

| Categoría | Evento | Criticidad |
|---|---|---|
| **Ingesta** | Nuevo archivo CSV procesado | Alta |
| **Ingesta** | Schema violation detectada | Alta |
| **Transformación** | Bronze → Silver ejecutado | Alta |
| **Transformación** | Silver → Gold ejecutado | Alta |
| **Transformación** | Duplicados agregados (count) | Alta |
| **Transformación** | Negativos de inventario corregidos | Alta |
| **Transformación** | Valores censored flaggeados | Alta |
| **Features** | Feature store regenerado | Alta |
| **Features** | Hash del feature store registrado | Alta |
| **Training** | Nuevo experimento MLflow iniciado | Alta |
| **Training** | Modelo promovido a Production | Crítica |
| **Training** | Modelo archivado | Media |
| **Forecast** | Forecast batch generado | Alta |
| **Forecast** | Override manual aplicado | Alta |
| **Forecast** | Forecast exportado a BI | Media |
| **Monitoring** | MAPE threshold breach | Alta |
| **Monitoring** | Drift alert disparada | Alta |
| **Monitoring** | Retraining trigger activado | Alta |
| **Governance** | Data contract violation | Crítica |
| **Governance** | DQ score bajo umbral | Alta |

---

## 3. Audit Log Schema

```python
# src/audit/logger.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json, hashlib, uuid
import polars as pl

@dataclass
class AuditEvent:
    event_id:         str           # UUID v4
    event_type:       str           # categoría del evento
    event_subtype:    str           # evento específico
    timestamp:        datetime      # UTC
    actor:            str           # 'prefect_flow', 'user:email', 'mlflow_run'
    pipeline_run_id:  Optional[str] # Prefect flow run ID
    mlflow_run_id:    Optional[str] # MLflow run ID si aplica
    input_artifact:   Optional[str] # path o DVC hash del input
    output_artifact:  Optional[str] # path o DVC hash del output
    input_hash:       Optional[str] # SHA-256 del input
    output_hash:      Optional[str] # SHA-256 del output
    params_json:      Optional[str] # parámetros relevantes como JSON
    status:           str           # 'SUCCESS', 'FAILURE', 'WARNING'
    error_message:    Optional[str] # mensaje de error si aplica
    duration_seconds: Optional[float]
    row_count_in:     Optional[int]
    row_count_out:    Optional[int]
    notes:            Optional[str]

    def to_dict(self) -> dict:
        return {
            'event_id':         self.event_id,
            'event_type':       self.event_type,
            'event_subtype':    self.event_subtype,
            'timestamp':        self.timestamp.isoformat(),
            'actor':            self.actor,
            'pipeline_run_id':  self.pipeline_run_id,
            'mlflow_run_id':    self.mlflow_run_id,
            'input_artifact':   self.input_artifact,
            'output_artifact':  self.output_artifact,
            'input_hash':       self.input_hash,
            'output_hash':      self.output_hash,
            'params_json':      self.params_json,
            'status':           self.status,
            'error_message':    self.error_message,
            'duration_seconds': self.duration_seconds,
            'row_count_in':     self.row_count_in,
            'row_count_out':    self.row_count_out,
            'notes':            self.notes,
        }


class AuditLogger:
    AUDIT_LOG_PATH = "data/audit/audit_log.parquet"
    
    def log(self, event: AuditEvent):
        """Append-only write to audit log."""
        new_row = pl.DataFrame([event.to_dict()])
        
        try:
            existing = pl.read_parquet(self.AUDIT_LOG_PATH)
            updated = pl.concat([existing, new_row])
        except FileNotFoundError:
            updated = new_row
        
        updated.write_parquet(self.AUDIT_LOG_PATH)
    
    def log_pipeline_step(
        self,
        event_type: str,
        event_subtype: str,
        input_path: str,
        output_path: str,
        row_count_in: int,
        row_count_out: int,
        status: str = "SUCCESS",
        actor: str = "prefect_flow",
        **kwargs
    ):
        """Convenience method for pipeline step logging."""
        self.log(AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            event_subtype=event_subtype,
            timestamp=datetime.utcnow(),
            actor=actor,
            input_artifact=input_path,
            output_artifact=output_path,
            input_hash=self._file_hash(input_path),
            output_hash=self._file_hash(output_path),
            row_count_in=row_count_in,
            row_count_out=row_count_out,
            status=status,
            **kwargs
        ))
    
    @staticmethod
    def _file_hash(path: str) -> Optional[str]:
        try:
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except FileNotFoundError:
            return None


# Global singleton
audit = AuditLogger()
```

---

## 4. Uso en Pipeline

```python
# Ejemplo: Bronze ingestion con audit
from src.audit.logger import audit
import time

def ingest_year_with_audit(filepath: str, year: int):
    start = time.time()
    
    # Pre-audit
    audit.log_pipeline_step(
        event_type="INGESTION",
        event_subtype="CSV_INGEST_START",
        input_path=filepath,
        output_path=f"data/bronze/sell_data_{year}.parquet",
        row_count_in=0,  # unknown at start
        row_count_out=0,
        status="IN_PROGRESS"
    )
    
    try:
        df = run_ingestion(filepath, year)
        
        audit.log_pipeline_step(
            event_type="INGESTION",
            event_subtype="CSV_INGEST_COMPLETE",
            input_path=filepath,
            output_path=f"data/bronze/sell_data_{year}.parquet",
            row_count_in=count_csv_rows(filepath),
            row_count_out=len(df),
            status="SUCCESS",
            duration_seconds=time.time() - start,
            params_json=json.dumps({"year": year, "dtype_fix_applied": year==2025})
        )
    except Exception as e:
        audit.log_pipeline_step(
            event_type="INGESTION",
            event_subtype="CSV_INGEST_FAILED",
            input_path=filepath,
            output_path=f"data/bronze/sell_data_{year}.parquet",
            row_count_in=0, row_count_out=0,
            status="FAILURE",
            error_message=str(e),
            duration_seconds=time.time() - start
        )
        raise
```

---

## 5. Trazabilidad de Forecast

Para cualquier forecast entregado, debe ser posible responder:

```
PREGUNTA: "¿Por qué el forecast de CUSTOMER5 × SM-A057M × W38 2025 es 150 unidades?"

RESPUESTA TRAZABLE:
1. Forecast generado por: run_id=abc123 (MLflow)
2. Modelo: sell_in_lgbm_global v2.1, promoted 2025-05-15
3. Feature store hash: f3a8c12d (DVC)
4. Silver hash: b7e91f44 (DVC)
5. Bronze 2025 hash: cc94a112 (DVC)
6. Source file: 2025.csv, ingested 2025-05-12T08:15:00Z
7. Training cutoff: 202533 (week 33, 2025)
8. Series segment: 'regular' (active_weeks_2024=42)
9. Key features: sell_in_lag_52=180, days_of_supply_lag_1=28, seasonal_index=0.85
10. Inventory adjustment applied: No (DOS=28, dentro de rango sano)
```

```python
# src/audit/tracer.py

def trace_forecast(
    channel: str,
    material: str,
    yearweek: str
) -> dict:
    """
    Return complete audit trail for a specific forecast.
    """
    # Load forecast output
    fc = pl.read_parquet("data/gold/forecast_output.parquet")
    row = fc.filter(
        (pl.col("channel") == channel) &
        (pl.col("material") == material) &
        (pl.col("yearweek") == yearweek)
    )
    
    if len(row) == 0:
        return {"error": "Forecast not found"}
    
    model_version = row["model_version"][0]
    
    # Load MLflow run
    client = MlflowClient()
    run = client.get_run(model_version)
    
    # Load audit log
    audit_log = pl.read_parquet("data/audit/audit_log.parquet")
    relevant_events = audit_log.filter(
        pl.col("mlflow_run_id") == model_version
    )
    
    return {
        "forecast": row.to_dicts()[0],
        "model_run_id": model_version,
        "feature_store_hash": run.data.tags.get("feature_store_hash"),
        "silver_hash": run.data.tags.get("silver_hash"),
        "training_cutoff": run.data.tags.get("training_cutoff"),
        "series_segment": row["segment"][0],
        "pipeline_events": relevant_events.to_dicts(),
        "model_params": run.data.params,
        "model_metrics": run.data.metrics,
    }
```

---

## 6. Audit Log Queries

```sql
-- DuckDB: ¿Cuándo fue ingresado el último CSV 2025?
SELECT * FROM read_parquet('data/audit/audit_log.parquet')
WHERE event_subtype = 'CSV_INGEST_COMPLETE' AND params_json LIKE '%2025%'
ORDER BY timestamp DESC LIMIT 5;

-- ¿Qué modelos fueron promovidos a production este año?
SELECT event_id, timestamp, params_json, actor
FROM read_parquet('data/audit/audit_log.parquet')
WHERE event_subtype = 'MODEL_PROMOTED'
ORDER BY timestamp DESC;

-- ¿Cuántos overrides manuales se hicieron esta semana?
SELECT COUNT(*) as n_overrides, actor
FROM read_parquet('data/audit/audit_log.parquet')
WHERE event_subtype = 'FORECAST_OVERRIDE'
  AND timestamp >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY actor;

-- ¿Qué errores de DQ hubo en el último mes?
SELECT event_subtype, COUNT(*) as count, MIN(timestamp) as first, MAX(timestamp) as last
FROM read_parquet('data/audit/audit_log.parquet')
WHERE event_type = 'DATA_QUALITY' AND status = 'FAILURE'
  AND timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY event_subtype ORDER BY count DESC;
```

---

## 7. Política de Retención

| Tipo de Log | Retención | Almacenamiento |
|---|---|---|
| Audit log completo | Permanente | `data/audit/audit_log.parquet` (DVC) |
| MLflow run artifacts | 24 meses | `mlruns/artifacts/` |
| MLflow metrics/params | Permanente | MLflow DB |
| DVC cache (datos) | 12 meses | `.dvc/cache/` |
| Forecast output histórico | 24 meses | `data/gold/forecast_output.parquet` |
| Prefect run logs | 90 días | Prefect DB |

---

*AI-DLC Traceability ID: AUDIT-ITER3-001 | Version: 3.0*
