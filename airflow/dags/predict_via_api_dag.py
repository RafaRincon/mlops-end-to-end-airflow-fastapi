from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import random
import mlflow

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1)
}

def call_fastapi_predict():
    payload = {
    "feature1": round(random.uniform(0, 7), 2),
    "feature2": round(random.uniform(0, 7), 2),
    "feature3": round(random.uniform(0, 7), 2),
    "feature4": round(random.uniform(0, 7), 2)
}

    response = requests.post("http://api-fastapi:8000/predict", json=payload)
    print("Status code:", response.status_code)
    print("Raw response:", response.text)
    result = response.json()

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("predicciones-fastapi")

    with mlflow.start_run():
        mlflow.log_params(payload)
        mlflow.log_metric("prediccion", result.get("predicción", -1))

with DAG(
    dag_id="predict_api_every_5_minutes",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="*/5 * * * *",  # Cada 5 minutos
    catchup=False,
    tags=["fastapi", "mlflow", "prueba"],
) as dag:

    test_model = PythonOperator(
        task_id="probar_modelo_desplegado",
        python_callable=call_fastapi_predict
    )
