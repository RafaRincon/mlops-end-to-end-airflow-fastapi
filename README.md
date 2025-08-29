# 🧠 End-to-End MLOps (Airflow + FastAPI + MLflow + Prometheus)

This repository delivers a local MLOps workflow using Docker Compose. It covers:
- **Training & model selection** (scikit-learn) with **MLflow** tracking/registry
- **Serving** with **FastAPI** loading models from MLflow (via alias)
- **Prediction simulation** with an **Airflow** DAG (10 predictions every 5 minutes)
- **Monitoring** via **Prometheus** (requests, latency, errors, status codes)

---

## 🧩 Components

| Component      | Purpose                                                   |
|----------------|-----------------------------------------------------------|
| **MLflow**     | Experiment tracking, artifacts, and Model Registry        |
| **FastAPI**    | Online inference; loads latest **Production** alias       |
| **Airflow**    | Orchestrates the prediction simulator DAG                 |
| **Prometheus** | Metrics for the API: requests, latency, errors, statuses  |
| **Docker**     | Local orchestration for all services                      |

---

## 📁 Project Structure (key paths)

```
.
├── airflow/
│   └── dags/
│       └── simulate\_inference\_dag.py   # sends 10 predictions every 5 minutes
├── data/
│   └── Synthetic\_Customers\_Data.csv    # training data
├── fastapi\_app/
│   ├── app.py                          # FastAPI app (loads MLflow model)
│   └── requirements.txt
├── inference\_store/                    # CSVs from the simulator
├── mlflow/                             # MLflow backend store
├── prometheus/
│   └── prometheus.yml
├── src/
│   └── training/
│       └── train\_best\_model.py         # trains, logs, registers best model
├── docker-compose.yml
└── README.md
````

---

## ⚙️ Prerequisites

- Docker & Docker Compose

---

## 🚀 Workflow

### 1. Build all images
```bash
docker-compose build
````

### 2. Start only MLflow

```bash
docker-compose up -d mlflow
```

### 3. Train and (optionally) register the best model

Run after MLflow is reachable at `http://localhost:5000`:

```bash
python ./src/training/train_best_model.py \
  --data ./data/Synthetic_Customers_Data.csv \
  --experiment customer_models \
  --register customers_models \
  --test_size 0.2 \
  --cv_folds 2 \
  --alias Production
```

This:

* Trains multiple candidate models with CV.
* Logs metrics and artifacts to **MLflow**.
* Registers the best model in the Model Registry as `customers_models`.
* Updates alias **Production** so FastAPI can serve it.

> 🔎 If you omit `--register`, the model will not be registered, only logged.

### 4. Start FastAPI (now a model exists to load)

```bash
docker-compose up -d fastapi
```

FastAPI loads the MLflow alias (`models:/customers_models@Production`) automatically.

### 5. Initialize Airflow

```bash
docker-compose run --rm airflow airflow db init
```

Create the Airflow admin user:

```bash
docker-compose run --rm airflow airflow users create \
  --username admin \
  --firstname User \
  --lastname Name \
  --role Admin \
  --email youremail@example.com \
  --password admin
```

### 6. Start remaining services (Airflow, Prometheus)

```bash
docker-compose up -d
```

### 7. Enable the simulation DAG

In Airflow UI (`http://localhost:8080`), turn on `simulate_inference_to_mlflow`.

---

## 📡 Endpoints

* FastAPI → [http://localhost:8000](http://localhost:8000)
* Airflow UI → [http://localhost:8080](http://localhost:8080)
* MLflow UI → [http://localhost:5000](http://localhost:5000)
* Prometheus → [http://localhost:9090](http://localhost:9090)

---

## 📈 Prometheus Metrics

FastAPI exposes metrics at port `8001`. Key metrics:

* **`api_requests_total`** → total calls to `/predict`
* **`api_request_duration_seconds`** → histogram of request latencies
* **`api_errors_total`** → total unhandled errors
* **`api_status_total{path,method,status}`** → per-endpoint status codes

---

## 🧭 Typical Flow

1. `docker-compose build`
2. `docker-compose up -d mlflow`
3. Run `train_best_model.py` to train/register the best model
4. `docker-compose up -d fastapi`
5. Init Airflow DB + admin user
6. `docker-compose up -d` for Airflow + Prometheus
7. Enable the simulator DAG in Airflow
8. Watch predictions flow into MLflow, CSVs in `inference_store`, and metrics in Prometheus

---

## 🔮 Future Improvements

* Ground-truth monitor DAG (watch `data/truth_labels`)
* Auto retrain trigger when performance drops
* Unit tests for API/model predictions
* Alerting on error/latency thresholds
* Modular training pipeline (data prep → train → eval → register)
* Grafana dashboards

---

## 👤 Author

Rafael Rincón · [LinkedIn](https://www.linkedin.com/in/rafarinra/) · [GitHub](https://github.com/RafaRincon)