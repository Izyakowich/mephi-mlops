import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Создание дополнительных признаков кредитного риска."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
        pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
        delay_cols = [
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ]

        # Средняя задолженность
        X["AVG_BILL_AMT"] = X[bill_cols].mean(axis=1)

        # Средний платеж
        X["AVG_PAY_AMT"] = X[pay_amt_cols].mean(axis=1)

        # Максимальная просрочка
        X["MAX_DELAY"] = X[delay_cols].max(axis=1)

        # Количество месяцев с просрочками
        X["NUM_LATE_PAYMENTS"] = (X[delay_cols] > 0).sum(axis=1)

        # Использование кредитного лимита
        X["CREDIT_UTILIZATION"] = X["AVG_BILL_AMT"] / X["LIMIT_BAL"].replace(0, np.nan)

        # Отношение платежей к задолженности
        total_payments = X[pay_amt_cols].sum(axis=1)
        total_bills = X[bill_cols].sum(axis=1)

        X["PAYMENT_TO_BILL_RATIO"] = total_payments / total_bills.where(
            total_bills > 0, np.nan
        )

        # Возрастные категории
        X["AGE_GROUP"] = pd.cut(
            X["AGE"],
            bins=[17, 25, 35, 45, 60, 100],
            labels=[
                "18-25",
                "26-35",
                "36-45",
                "46-60",
                "61+",
            ],
        ).astype("object")

        return X
