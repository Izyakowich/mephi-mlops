from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validation import TARGET, validate_data


RAW_PATH = Path("data/raw/UCI_Credit_Card.csv")
OUTPUT_DIR = Path("data/processed")


def prepare_data(
    input_path=RAW_PATH,
    output_dir=OUTPUT_DIR,
):
    """Валидация, очистка и разделение датасета."""

    print("Загрузка датасета...")
    df = pd.read_csv(input_path)

    print(f"Размер исходных данных: {df.shape}")

    print("Проверка качества данных...")
    df = validate_data(df)

    print("Удаление технического столбца ID...")
    df = df.drop(columns=["ID"])

    print("Разделение на train и test...")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Подготовка данных завершена!")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")

    print(
        "Доля дефолтов в train:",
        round(train_df[TARGET].mean(), 4),
    )
    print(
        "Доля дефолтов в test:",
        round(test_df[TARGET].mean(), 4),
    )

    return train_df, test_df


if __name__ == "__main__":
    prepare_data()
