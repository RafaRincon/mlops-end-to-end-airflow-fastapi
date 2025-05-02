from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo previamente entrenado
#model = joblib.load("model.joblib")
model = joblib.load("/app/shared_models/model.joblib")

# Inicializar FastAPI
app = FastAPI()

# Definir el esquema de entrada con Pydantic
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Ruta de prueba
@app.get("/")
def read_root():
    return {"message": "API de predicción activa"}

# Endpoint para predicciones
@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.feature1, data.feature2, data.feature3]])
    prediction = model.predict(input_array)
    return {"predicción": int(prediction[0])}