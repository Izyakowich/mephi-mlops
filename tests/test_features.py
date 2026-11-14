import numpy as np
import pandas as pd

from src.features.build_features import CreditFeatureEngineer


def test_feature_engineering():
    """Проверка создания новых признаков."""

    df = pd.DataFrame(
        {
            "LIMIT_BAL": [100000, 200000],
            "AGE": [24, 40],
            "PAY_0": [2, 0],
            "PAY_2": [1, 0],
            "PAY_3": [0, 0],
            "PAY_4": [0, 0],
            "PAY_5": [0, 0],
            "PAY_6": [0, 0],
            **{f"BILL_AMT{i}": [10000, 20000] for i in range(1, 7)},
            **{f"PAY_AMT{i}": [1000, 2000] for i in range(1, 7)},
        }
    )

    transformer = CreditFeatureEngineer()
    result = transformer.fit_transform(df)

    expected_features = [
        "AVG_BILL_AMT",
        "AVG_PAY_AMT",
        "MAX_DELAY",
        "NUM_LATE_PAYMENTS",
        "CREDIT_UTILIZATION",
        "PAYMENT_TO_BILL_RATIO",
        "AGE_GROUP",
    ]

    for feature in expected_features:
        assert feature in result.columns

    assert result.loc[0, "MAX_DELAY"] == 2
    assert result.loc[0, "NUM_LATE_PAYMENTS"] == 2
    assert result.loc[0, "AGE_GROUP"] == "18-25"

    assert np.isclose(result.loc[0, "CREDIT_UTILIZATION"], 0.1)

    assert np.isclose(result.loc[0, "PAYMENT_TO_BILL_RATIO"], 0.1)


def test_input_not_modified():
    """Исходный DataFrame не должен изменяться."""

    df = pd.read_csv("tests/fixtures/credit_sample.csv").head(10)

    df = df.drop(columns=["ID", "default.payment.next.month"])

    original = df.copy(deep=True)

    transformer = CreditFeatureEngineer()
    transformer.fit_transform(df)

    pd.testing.assert_frame_equal(df, original)
