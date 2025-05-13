from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo previamente entrenado
#model = joblib.load("model.joblib")
model = joblib.load("/app/shared_models/model.joblib")

app = FastAPI()
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.get("/")
def read_root():
    return {"message": "API de predicción activa"}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Modelo no disponible"}
    
    input_array = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = model.predict(input_array)
    return {"predicción": int(prediction[0])}