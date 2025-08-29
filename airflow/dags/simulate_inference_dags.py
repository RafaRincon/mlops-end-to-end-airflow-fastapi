# /opt/airflow/dags/simulate_inference_dag.py
import os, io, json, uuid, time, random, logging
from datetime import datetime, timezone

import pendulum
import pandas as pd
import requests
import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator

API_URL = os.getenv("PREDICT_API_URL", "http://fastapi:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
INFERENCE_EXPERIMENT = os.getenv("INFERENCE_EXPERIMENT", "inference_logs")
INFERENCE_STORE_DIR = os.getenv("INFERENCE_STORE_DIR", "/opt/airflow/inference_store")

def _ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def _generate_row():
    return {
        "age": int(random.randint(18, 80)),
        "monthly_income": int(random.randint(3000, 25000)),
        "debt_to_income_ratio": float(round(random.uniform(0.05, 0.9), 3)),
        "transactions_per_month": int(random.randint(5, 120)),
        "complaints_last_6m": int(random.randint(0, 8)),
    }

def simulate_and_log(**context):
    logger = logging.getLogger("simulate_inference")
    ts = context["data_interval_start"]
    exec_ts_str = ts.to_datetime_string()   # e.g., 2025-08-28T21:35:00-06:00
    exec_date = ts.to_date_string()

    # Health (optional)
    try:
        health = requests.get(HEALTH_ENDPOINT, timeout=5)
        health_json = health.json() if health.ok else {}
    except Exception:
        health_json = {}
    loaded_version = health_json.get("loaded_version", "unknown")
    model_uri = health_json.get("model_uri", "unknown")

    # MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(INFERENCE_EXPERIMENT)

    rows = []
    ok_count = 0

    # Exactly 10 requests per run
    for _ in range(10):
        features = _generate_row()
        t0 = time.perf_counter()
        status = None
        prediction_val = None
        err_detail = None

        try:
            r = requests.post(PREDICT_ENDPOINT, json=features, timeout=5)
            status = r.status_code
            if r.ok:
                ok_count += 1
                prediction_val = r.json().get("prediction", None)
            else:
                err_detail = r.text[:500]
        except Exception as e:
            status = 599
            err_detail = str(e)[:500]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        rows.append({
            "prediction_id": str(uuid.uuid4()),
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "exec_ts": exec_ts_str,
            "status_code": status,
            "latency_ms": round(latency_ms, 2),
            "prediction": prediction_val,
            "error": err_detail,
            "model_version_loaded": loaded_version,
            "model_uri_api": model_uri,
            **features,
        })

        # small spacing (not required, keeps API neat)
        time.sleep(0.25)

    df = pd.DataFrame(rows)
    agg = {
        "count": len(df),
        "ok_count": int((df["status_code"] == 200).sum()),
        "error_count": int((df["status_code"] != 200).sum()),
        "mean_latency_ms": float(df["latency_ms"].mean() if len(df) else 0.0),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95) if len(df) else 0.0),
    }

    run_name = f"inference_batch__{exec_ts_str.replace(':','-')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("inference_count", agg["count"])
        mlflow.log_metric("inference_ok", agg["ok_count"])
        mlflow.log_metric("inference_error", agg["error_count"])
        mlflow.log_metric("latency_mean_ms", agg["mean_latency_ms"])
        mlflow.log_metric("latency_p95_ms", agg["p95_latency_ms"])
        mlflow.set_tag("source", "airflow_simulator")
        mlflow.set_tag("api_url", API_URL)
        mlflow.set_tag("api_model_uri", model_uri)
        mlflow.set_tag("api_model_version_loaded", loaded_version)
        # artifact: full row-level CSV
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        mlflow.log_text(buf.getvalue(), artifact_file=f"inference_rows/inference_{exec_ts_str}.csv")

    # persist locally (for upcoming “ground truth” monitor DAG)
    day_dir = os.path.join(INFERENCE_STORE_DIR, f"dt={exec_date}")
    _ensure_dirs(day_dir)
    out_path = os.path.join(day_dir, f"inference_{exec_ts_str.replace(':','-')}.csv")
    df.to_csv(out_path, index=False)

with DAG(
    dag_id="simulate_inference_to_mlflow",
    description="Every 5 minutes: send 10 predictions, log to MLflow, persist CSV.",
    start_date=pendulum.datetime(2025, 8, 28, tz="America/Mexico_City"),
    schedule_interval="*/5 * * * *",   # rest 5 minutes between batches
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops", "retries": 1},
    tags=["inference", "mlflow", "simulation"],
) as dag:
    simulate_and_log_task = PythonOperator(
        task_id="simulate_and_log",
        python_callable=simulate_and_log,
        provide_context=True,
    )