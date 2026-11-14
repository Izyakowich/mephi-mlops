import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)

from src.data.validation import TARGET
from src.models.pipeline import create_pipeline


MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


def calculate_metrics(y_true, y_pred, y_proba):
    """Рассчитать метрики бинарной классификации."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_roc_curve(y_true, y_proba, output_path):
    """Построить и сохранить ROC-кривую."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Logistic Regression (AUC = {auc:.4f})",
        linewidth=2,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Credit Default Prediction")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()


def train_model():
    """Подбор гиперпараметров, оценка и MLflow."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка выборок
    print("Загрузка данных")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    # 2. Настройка MLflow
    tracking_dir = Path("mlruns").resolve()

    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("Credit_Default_Prediction")

    # 3. Создание Pipeline
    pipeline = create_pipeline()

    # 4. Пять вариантов гиперпараметра C
    param_grid = {
        "classifier__C": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ]
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    # 5. Автоматический подбор гиперпараметров
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=2,
        refit=True,
        verbose=1,
        error_score="raise",
    )

    print("\nЗапуск GridSearchCV...")
    grid_search.fit(X_train, y_train)

    results = pd.DataFrame(grid_search.cv_results_)

    results_path = REPORT_DIR / "cv_results.csv"
    results.to_csv(results_path, index=False)

    # 6. Логирование пяти CV-экспериментов
    print("\nЗапись экспериментов в MLflow...")

    for index, row in results.iterrows():
        params = grid_search.cv_results_["params"][index]
        c_value = params["classifier__C"]

        with mlflow.start_run(run_name=f"LogReg_C_{c_value}"):
            mlflow.log_param("algorithm", "LogisticRegression")
            mlflow.log_param("C", float(c_value))
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("cv_folds", 3)

            mlflow.log_metric(
                "cv_roc_auc",
                float(row["mean_test_score"]),
            )
            mlflow.log_metric(
                "cv_roc_auc_std",
                float(row["std_test_score"]),
            )

            mlflow.log_metric(
                "cv_rank",
                int(row["rank_test_score"]),
            )

        print(
            f"Experiment {index + 1}: "
            f"C={c_value}, "
            f"CV ROC-AUC={row['mean_test_score']:.4f}"
        )

    # 7. Лучшая модель уже обучена GridSearchCV
    best_model = grid_search.best_estimator_

    print("\nЛучшие параметры:", grid_search.best_params_)
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    # 8. Финальная оценка на тестовых данных
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_proba,
    )

    print("\nМетрики на тестовой выборке:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 9. Сохранение ROC-кривой
    roc_path = REPORT_DIR / "roc_curve.png"

    save_roc_curve(
        y_test,
        y_proba,
        roc_path,
    )

    # 10. Сохранение модели для FastAPI
    model_path = MODEL_DIR / "credit_default_model.pkl"

    joblib.dump(best_model, model_path)

    # 11. Сохранение метрик для DVC
    metrics_path = Path("metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # 12. Финальный MLflow Run
    with mlflow.start_run(run_name="Best_Logistic_Regression"):
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param(
            "best_C",
            float(grid_search.best_params_["classifier__C"]),
        )
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric(
            "cv_roc_auc",
            float(grid_search.best_score_),
        )

        for name, value in metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        # Сохранение артефактов
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(metrics_path))

        # Пример входных данных
        input_example = X_test.head(3).copy()

        # Пример вероятностей предсказания
        example_predictions = best_model.predict_proba(input_example)

        # Автоматическое определение схемы модели
        signature = infer_signature(
            input_example,
            example_predictions,
        )

        # Сохранение модели в MLflow
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    print("\nОбучение завершено!")
    print(f"Модель сохранена: {model_path}")
    print(f"ROC-кривая: {roc_path}")
    print(f"Метрики: {metrics_path}")


if __name__ == "__main__":
    train_model()
