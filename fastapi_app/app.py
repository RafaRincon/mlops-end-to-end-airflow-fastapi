from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram
import time

# Cargar el modelo previamente entrenado contenido en shared_models
#model = joblib.load("model.joblib")
model = joblib.load("/app/shared_models/model.joblib")

REQUEST_COUNT = Counter("api_requests_total", "Total de peticiones al endpoint /predict")
REQUEST_LATENCY = Histogram("api_request_duration_seconds", "Duración de las peticiones al endpoint /predict")


app = FastAPI()
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.on_event("startup")
def start_metrics_server():
    start_http_server(8001)

@app.get("/")
def read_root():
    return {"message": "API de predicción activa"}

@app.post("/predict")
@REQUEST_LATENCY.time()  # mide duración
def predict(data: InputData):
    REQUEST_COUNT.inc()   # suma 1 a la métrica de conteo

    if model is None:
        return {"error": "Modelo no disponible"}
    
    input_array = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = model.predict(input_array)
    return {"predicción": int(prediction[0])}