from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import CreditFeatureEngineer


def create_pipeline():
    """Создать полный ML Pipeline для кредитного скоринга."""

    # Числовые признаки
    numeric_features = [
        "LIMIT_BAL",
        "AGE",
        *[f"BILL_AMT{i}" for i in range(1, 7)],
        *[f"PAY_AMT{i}" for i in range(1, 7)],
        "AVG_BILL_AMT",
        "AVG_PAY_AMT",
        "MAX_DELAY",
        "NUM_LATE_PAYMENTS",
        "CREDIT_UTILIZATION",
        "PAYMENT_TO_BILL_RATIO",
    ]

    # Категориальные признаки
    categorical_features = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "AGE_GROUP",
    ]

    # Обработка числовых признаков
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Обработка категориальных признаков
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Объединение преобразований
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    # Полный Pipeline
    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                CreditFeatureEngineer(),
            ),
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline
