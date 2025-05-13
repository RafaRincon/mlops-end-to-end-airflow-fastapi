# 🧠 MLOps End-to-End Project with Airflow, FastAPI, MLflow & Prometheus

This project implements a complete MLOps workflow running locally with Docker Compose. It includes:

- **Model training and orchestration using Apache Airflow**
- **Model serving with FastAPI**
- **Experiment tracking with MLflow**
- **API monitoring with Prometheus**

---

## 🚀 Technology Stack

| Component     | Purpose                                      |
|---------------|----------------------------------------------|
| **FastAPI**   | Serves predictions from a trained ML model   |
| **Airflow**   | Schedules and orchestrates ML tasks          |
| **MLflow**    | Logs model parameters, metrics, and runs     |
| **Prometheus**| Monitors API performance and availability    |
| **Docker**    | Containerizes and orchestrates services      |
| **Scikit-learn** | Trains a classification model (Iris)     |

---

## 📁 Project Structure

```

MLAirflow/
├── airflow/
│   └── dags/
├── fastapi\_app/
│   ├── app.py
│   └── requirements.txt
├── mlflow/                 # MLflow backend store
├── shared\_models/          # Trained model (model.joblib)
├── prometheus/
│   └── prometheus.yml
├── docker-compose.yml
└── README.md

````

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/RafaRincon/MLAirflow.git
cd MLAirflow
````

### 2. Initialize the Airflow database (one time only)

```bash
docker-compose run airflow airflow db init
```

### 3. Create an Airflow admin user

```bash
docker-compose run airflow airflow users create \
  --username admin \
  --firstname User \
  --lastname Name \
  --role Admin \
  --email youremail@example.com \
  --password admin
```

### 4. Start all services

```bash
docker-compose up --build
```

---

## 🌐 Access Services

| Service    | URL                                            |
| ---------- | ---------------------------------------------- |
| FastAPI    | [http://localhost:8000](http://localhost:8000) |
| Airflow UI | [http://localhost:8080](http://localhost:8080) |
| MLflow UI  | [http://localhost:5000](http://localhost:5000) |
| Prometheus | [http://localhost:9090](http://localhost:9090) |

---

## 🤖 Available Workflows

### `entrenamiento_modelo_iris`

* Trains a `RandomForestClassifier` on the Iris dataset
* Saves the trained model to `shared_models/model.joblib`
* Logs parameters and metrics to MLflow

### `predict_api_every_5_minutes`

* Runs every 5 minutes
* Generates random inputs and sends them to the FastAPI `/predict` endpoint
* Logs predictions and latency in MLflow

---

## 📈 API Monitoring with Prometheus

FastAPI exposes Prometheus-compatible metrics on port `8001` using the `prometheus_client` library.

Monitored metrics include:

* `api_requests_total`: number of `/predict` calls
* `api_request_duration_seconds`: response time histogram

---

## 🔐 Notes

* Run `docker-compose build fastapi` after editing `app.py`
* Ensure at least one request has been made to `/predict` to see Prometheus metrics

---

## 📌 Author

Rafael Rincón · [LinkedIn]([https://linkedin.com](https://www.linkedin.com/in/rafael-rinc%C3%B3n-ram%C3%ADrez-a8b052122/)) · [GitHub]([https://github.com](https://github.com/RafaRincon))

---

## 🏁 Next Steps

* ✅ Add Grafana dashboards for advanced visualization
* ✅ Enable MLflow Model Registry
* ✅ Implement input data validation in the Airflow training pipeline
