# Credit Scoring MLOps Pipeline

## Описание проекта

Итоговый проект по дисциплине
«Автоматизация процессов разработки и тестирования моделей
машинного обучения».

**Цель:** разработка воспроизводимого end-to-end MLOps-пайплайна
для прогнозирования вероятности дефолта клиента (PD-модель).

Проект охватывает:

- подготовку и валидацию данных;
- Feature Engineering и EDA;
- обучение и настройку модели;
- логирование экспериментов через MLflow;
- версионирование данных и моделей с DVC;
- автоматическое тестирование через pytest и GitHub Actions;
- развёртывание FastAPI в Docker;
- мониторинг дрифта данных с помощью PSI.

## Датасет

Default of Credit Card Clients Dataset (UCI).

Источник:
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

Датасет содержит 30 000 записей о клиентах кредитных карт
и 25 исходных столбцов.

Целевая переменная: `default.payment.next.month`.

- 0 - отсутствие дефолта в следующем месяце;
- 1 - наличие дефолта в следующем месяце.

В исходном датасете доля дефолтов составляет 22,12%.

## Технологии

| Категория         | Инструменты         |
| ----------------- | ------------------- |
| Язык              | Python 3.11         |
| Обработка данных  | Pandas, NumPy       |
| Машинное обучение | Scikit-learn        |
| Валидация         | Pandera             |
| Эксперименты      | MLflow              |
| Версионирование   | Git, DVC            |
| Тестирование      | pytest              |
| Качество кода     | Black, Flake8       |
| CI                | GitHub Actions      |
| REST API          | FastAPI, Pydantic   |
| Контейнеризация   | Docker              |
| Мониторинг        | PSI                 |
| Визуализация      | Matplotlib, Seaborn |

## Архитектура проекта

```text
credit-scoring-model/
├── .github/workflows/      # GitHub Actions
├── data/
│   ├── raw/                # Исходные данные
│   └── processed/          # Train/test выборки
├── models/                 # Обученная модель
├── notebooks/
│   └── 01_eda.ipynb        # Разведочный анализ
├── reports/
│   ├── eda/                # Графики EDA
│   ├── roc_curve.png
│   └── drift_report.json
├── scripts/
│   └── run_docker.sh
├── src/
│   ├── api/app.py
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── validation.py
│   ├── features/build_features.py
│   ├── models/
│   │   ├── pipeline.py
│   │   └── train.py
│   └── monitoring/drift.py
├── tests/
│   ├── fixtures/
│   ├── test_validation.py
│   ├── test_features.py
│   └── test_drift.py
├── Dockerfile
├── dvc.yaml
├── metrics.json
├── requirements.txt
└── README.md
```

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/Izyakowich/mephi-mlops.git
cd mephi-mlops
```

### 2. Виртуальное окружение

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

```bash
python -m pip install -r requirements.txt
```

### 4. Получение данных

Исходный датасет версионируется с помощью DVC.

Если настроено удалённое DVC-хранилище:

```bash
dvc pull
```

Иначе необходимо поместить исходный CSV по пути:

`data/raw/UCI_Credit_Card.csv`

## Подготовка данных

```bash
python -m src.data.validation
python -m src.data.make_dataset
```

Используется стратифицированное разделение 80/20:

- Train: 24 000 наблюдений.
- Test: 6 000 наблюдений.

## Feature Engineering

Создаются дополнительные признаки:

- AVG_BILL_AMT - средняя задолженность;
- AVG_PAY_AMT - средний платёж;
- MAX_DELAY - максимальная просрочка;
- NUM_LATE_PAYMENTS - число месяцев с просрочкой;
- CREDIT_UTILIZATION - использование кредитного лимита;
- PAYMENT_TO_BILL_RATIO - отношение платежей к задолженности;
- AGE_GROUP - возрастная категория.

Feature Engineering включён в Scikit-learn Pipeline,
что обеспечивает одинаковую обработку при обучении
и инференсе.

## Обучение модели

Используется LogisticRegression с автоматическим
подбором параметра C через GridSearchCV.

Проверяются значения:

`0.01, 0.1, 1.0, 10.0, 100.0`

Кросс-валидация: StratifiedKFold, 3 разбиения.

Запуск обучения:

```bash
python -m src.models.train
```

### Результаты на тестовой выборке

| Метрика    | Значение |
| ---------- | -------: |
| ROC-AUC    |   0.7615 |
| Precision  |   0.4613 |
| Recall     |   0.5931 |
| F1-Score   |   0.5190 |
| Лучший C   |      0.1 |
| CV ROC-AUC |   0.7726 |

Результаты получены на отложенной тестовой выборке.

## MLflow Tracking

Проводятся пять экспериментов с разными параметрами
LogisticRegression.

MLflow записывает:

- параметры модели;
- метрики кросс-валидации;
- тестовые метрики лучшей модели;
- ROC-кривую;
- результаты GridSearchCV;
- обученную модель со схемой входных данных.

Запуск интерфейса:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Интерфейс: http://127.0.0.1:5000

## DVC Pipeline

Пайплайн включает две стадии:

1. prepare - валидация и подготовка данных.
2. train - обучение, оценка и сохранение модели.

Выполнение:

```bash
dvc repro
```

Проверка состояния:

```bash
dvc status
```

Просмотр метрик:

```bash
dvc metrics show
```

## Тестирование

Для автоматических проверок используется pytest.

```bash
python -m pytest tests/ -v
```

Проверка форматирования:

```bash
python -m black --check src tests
```

Проверка качества кода:

```bash
python -m flake8 src tests
```

GitHub Actions запускает тесты, линтинг,
проверку форматирования и валидацию тестового CSV
при push и pull request.

## REST API

FastAPI предоставляет следующие endpoints:

- GET `/` - информация об API.
- GET `/health` - проверка доступности модели.
- POST `/predict` - прогноз вероятности дефолта.

Локальный запуск:

```bash
python -m uvicorn src.api.app:app --reload --port 8000
```

Swagger UI: http://127.0.0.1:8000/docs

Пример ответа `/predict`:

```json
{
  "default_prediction": 0,
  "default_probability": 0.27
}
```

Числа приведены для иллюстрации формата ответа.

## Docker

Сборка образа:

```bash
docker build -t credit-scoring-api:latest .
```

Запуск:

```bash
docker run --rm -p 8000:8000 credit-scoring-api:latest
```

Также доступен скрипт:

```bash
./scripts/run_docker.sh
```

## Мониторинг дрифта

Для оценки изменений распределения признаков
используется Population Stability Index (PSI).

Мониторятся:

- LIMIT_BAL;
- AGE;
- BILL_AMT1;
- PAY_AMT1.

Ориентировочные пороги:

- PSI < 0.1 - стабильное распределение;
- 0.1 <= PSI < 0.25 - предупреждение;
- PSI >= 0.25 - значительное изменение.

Запуск при работающем API:

```bash
python -m src.monitoring.drift --sample-size 100
```

Отчёт сохраняется в `reports/drift_report.json`.

## EDA

Разведочный анализ находится в:

`notebooks/01_eda.ipynb`

Включает:

- анализ пропусков и дубликатов;
- распределение целевой переменной;
- анализ возраста;
- анализ кредитных лимитов;
- сравнение дефолтов по возрастным группам;
- корреляционную матрицу.

Графики сохраняются в `reports/eda/`.

## Итоги

Разработан MLOps-пайплайн для предсказания
вероятности дефолта клиента.

Реализованы валидация, обучение, кросс-валидация,
логирование экспериментов, версионирование,
автоматические тесты, REST API, Docker
и мониторинг дрифта.

Проект предназначен для учебной демонстрации.
Модель не должна использоваться для реальных
кредитных решений без дополнительной проверки,
калибровки вероятностей и оценки рисков.
