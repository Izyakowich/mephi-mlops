from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys
sys.path.append(".")
from src.features.build_features import build_features

app = FastAPI(title="Credit Default Prediction API")

# Загрузка модели при старте
model_path = "models/credit_default_model.pkl"
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.get("/")
def read_root():
    return {"message": "Credit Default Prediction API is alive!"}

@app.post("/predict")
def predict(data: ClientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    input_df = pd.DataFrame([data.dict()])
    input_df = build_features(input_df) # Применяем Feature Engineering
    
    # Предсказание
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "default_prediction": int(prediction),
        "default_probability": float(probability)
    }