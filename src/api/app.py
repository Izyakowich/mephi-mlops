from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Путь к корню проекта
BASE_DIR = Path(__file__).resolve().parents[2]

# Путь к обученной модели
MODEL_PATH = BASE_DIR / "models" / "credit_default_model.pkl"


app = FastAPI(
    title="Credit Default Prediction API",
    description="API для предсказания вероятности дефолта клиента",
    version="1.0.0",
)


class ClientData(BaseModel):
    """Входные признаки клиента."""

    LIMIT_BAL: float = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)

    PAY_0: int = Field(ge=-2, le=8)
    PAY_2: int = Field(ge=-2, le=8)
    PAY_3: int = Field(ge=-2, le=8)
    PAY_4: int = Field(ge=-2, le=8)
    PAY_5: int = Field(ge=-2, le=8)
    PAY_6: int = Field(ge=-2, le=8)

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Ответ модели."""

    default_prediction: int
    default_probability: float


@lru_cache(maxsize=1)
def load_model():
    """Загрузить модель один раз и сохранить в памяти."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


@app.get("/")
def read_root():
    """Информация об API."""

    return {
        "message": "Credit Default Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Проверить доступность модели."""

    try:
        load_model()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        ) from exc

    return {"status": "ok", "model_loaded": True}


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(data: ClientData):
    """Предсказать вероятность дефолта клиента."""

    try:
        model = load_model()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        ) from exc

    # Преобразование JSON в DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Подготовка признаков в порядке обучения
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[list(model.feature_names_in_)]

    # Предсказание класса
    prediction = model.predict(input_df)

    # Вероятность дефолта (класс 1)
    probability = model.predict_proba(input_df)[0, 1]

    return PredictionResponse(
        default_prediction=int(prediction[0]),
        default_probability=float(probability),
    )
