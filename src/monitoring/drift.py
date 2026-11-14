import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.validation import TARGET


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
REPORT_PATH = Path("reports/drift_report.json")

MONITORED_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "PAY_AMT1",
]


def calculate_psi(reference, current, bins=10):
    """Рассчитать PSI по границам эталонного распределения."""

    reference = (
        pd.to_numeric(pd.Series(reference), errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )

    current = (
        pd.to_numeric(pd.Series(current), errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Недостаточно данных для расчёта PSI")

    # Границы интервалов определяются только по train
    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1),
        )
    )

    # Обработка признаков с постоянным значением
    if len(edges) < 2:
        value = reference[0]
        edges = np.array([value - 0.5, value + 0.5])

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    # Сглаживание, чтобы избежать log(0)
    epsilon = 1e-6

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    reference_pct = np.maximum(reference_pct, epsilon)
    current_pct = np.maximum(current_pct, epsilon)

    psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))

    return float(psi)


def get_drift_status(psi):
    """Интерпретировать значение PSI."""

    if psi < 0.1:
        return "stable"

    if psi < 0.25:
        return "warning"

    return "significant_drift"


def send_predictions(data, api_url):
    """Отправить клиентов в FastAPI и получить вероятности."""

    probabilities = []

    with requests.Session() as session:
        for index, (_, row) in enumerate(data.iterrows(), start=1):
            # Преобразование pandas-типов в JSON
            payload = json.loads(row.to_json())

            response = session.post(
                f"{api_url.rstrip('/')}/predict",
                json=payload,
                timeout=15,
            )

            response.raise_for_status()

            prediction = response.json()

            probabilities.append(float(prediction["default_probability"]))

            if index % 25 == 0:
                print(f"Обработано клиентов: {index}/{len(data)}")

    return probabilities


def monitor_drift(
    api_url="http://127.0.0.1:8000",
    sample_size=100,
):
    """Выполнить мониторинг признаков и запросов API."""

    print("Загрузка данных для мониторинга...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Берём случайную часть тестовой выборки
    sample_size = min(sample_size, len(test_df))

    if sample_size <= 0:
        raise ValueError("sample_size должен быть положительным")

    new_data = test_df.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    print(f"Объём эталонной выборки: {len(train_df)}")
    print(f"Объём новых данных: {len(new_data)}")

    # 1. Расчёт PSI по ключевым признакам
    feature_results = {}

    print("\nРасчёт PSI:")

    for feature in MONITORED_FEATURES:
        psi = calculate_psi(
            train_df[feature],
            new_data[feature],
        )

        status = get_drift_status(psi)

        feature_results[feature] = {
            "psi": round(psi, 6),
            "status": status,
        }

        print(f"{feature}: PSI={psi:.4f}, status={status}")

    # 2. Отправка новых данных в FastAPI
    print("\nОтправка запросов в FastAPI...")

    X_new = new_data.drop(columns=[TARGET])

    probabilities = send_predictions(
        X_new,
        api_url,
    )

    # 3. Формирование отчёта
    report = {
        "reference_samples": int(len(train_df)),
        "current_samples": int(len(new_data)),
        "api_url": api_url,
        "features": feature_results,
        "predictions": {
            "count": len(probabilities),
            "mean_default_probability": float(np.mean(probabilities)),
            "min_default_probability": float(np.min(probabilities)),
            "max_default_probability": float(np.max(probabilities)),
        },
        "has_significant_drift": any(
            item["status"] == "significant_drift" for item in feature_results.values()
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\nМониторинг завершён!")
    print(f"Отчёт сохранён: {REPORT_PATH}")
    print(
        "Средняя вероятность дефолта:",
        round(
            report["predictions"]["mean_default_probability"],
            4,
        ),
    )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Мониторинг дрифта кредитной модели")

    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    monitor_drift(
        api_url=args.api_url,
        sample_size=args.sample_size,
    )
