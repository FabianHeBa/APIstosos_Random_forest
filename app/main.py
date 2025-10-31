from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import List
from decisionTree import SimpleRandomForest
import os

class PredictionInput(BaseModel):
      features: List[float]


IRIS_CLASSES = {
    0: "setosa",
    1: "versicolor", 
    2: "virginica"
}

model = joblib.load('modelo_entrenado.pkl')

app = FastAPI(
    title="APIstosos. Modelo de prediccion, conjunto Iris",
    description="Un API simple para demostración con /health y /predict",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de 'APIstosos'. Para hacer las predicciones use /predict"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/info")
async def info_check():
    return {
        "team": "APIstosos",
        "Model": "RandomForestClassifier", 
        "n_estimators": 100,
        "max_depth": 8
    }

@app.post("/predict")
async def make_predict(input_data: PredictionInput):
      
      features_array = np.array(input_data.features).reshape(1, -1)

      pred = model.predict(features_array)
      
      prediction_value = pred[0] 

      predicted_class = IRIS_CLASSES.get(prediction_value, "unknown")
     
      return {
            "prediction": predicted_class,
             }

if __name__ == "__main__":
     import uvicorn
     port = int(os.getenv("PORT",8000))
     uvicorn.run(app, host='0.0.0.0', port=port)



      




