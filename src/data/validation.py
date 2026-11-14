import pandas as pd
import pandera.pandas as pa
from pandera import Check


TARGET = "default.payment.next.month"

PAY_COLUMNS = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]


def create_schema():
    """Создать правила проверки кредитного датасета."""

    columns = {
        "ID": pa.Column(int, Check.ge(1), unique=True),
        "LIMIT_BAL": pa.Column(float, Check.gt(0)),
        "SEX": pa.Column(int, Check.isin([1, 2])),
        "EDUCATION": pa.Column(int, Check.between(0, 6)),
        "MARRIAGE": pa.Column(int, Check.between(0, 3)),
        "AGE": pa.Column(int, Check.between(18, 100)),
        TARGET: pa.Column(int, Check.isin([0, 1])),
    }

    for col in PAY_COLUMNS:
        columns[col] = pa.Column(int, Check.between(-2, 8))

    for col in BILL_COLUMNS:
        columns[col] = pa.Column(float)

    for col in PAY_AMT_COLUMNS:
        columns[col] = pa.Column(float, Check.ge(0))

    return pa.DataFrameSchema(
        columns,
        strict=True,
        coerce=True,
    )


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Проверить данные и вернуть валидированный DataFrame."""
    schema = create_schema()
    return schema.validate(df, lazy=True)


if __name__ == "__main__":
    data = pd.read_csv("data/raw/UCI_Credit_Card.csv")
    validated = validate_data(data)

    print("Валидация данных пройдена!")
    print(f"Количество строк: {len(validated)}")
    print(f"Количество столбцов: {len(validated.columns)}")
