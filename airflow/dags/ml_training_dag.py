from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import io
import tempfile
import os

def train_and_log_model(**kwargs):
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, "/shared_models/model.joblib")
    print(f"Modelo guardado en: /shared_models/model.joblib")
    
    experiment_name = "iris_classification"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error al crear experimento: {e}")
        experiment_id = "0"
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_params(model.get_params())
        
        accuracy = model.score(X, y)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Precisión del modelo: {accuracy:.4f}")
        
        print(f"🏃 Modelo registrado en MLflow (solo métricas y parámetros)")
        print(f"🔗 Ver run: http://mlflow:5000/#/experiments/{experiment_id}/runs/{run.info.run_id}")

with DAG(
    dag_id="entrenamiento_modelo_iris",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "iris", "mlflow"],
) as dag:

    entrenar = PythonOperator(
        task_id="entrenar_y_loguear",
        python_callable=train_and_log_model
    )

    entrenar