from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_log_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    # MLflow tracking
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("iris-random-forest")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "modelo")

    # Guardar para FastAPI
    joblib.dump(model, "/mlflow/artifacts/model.joblib")

# Definir DAG
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
