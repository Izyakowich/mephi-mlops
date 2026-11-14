
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLBACKEND=Agg

# Установка Python-зависимостей
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Исходный код приложения
COPY src/ ./src/

# Обученная модель
RUN mkdir -p models

COPY models/credit_default_model.pkl \
    models/credit_default_model.pkl

# Порт FastAPI
EXPOSE 8000

# Запуск API
CMD ["uvicorn", "src.api.app:app", \
    "--host", "0.0.0.0", "--port", "8000"]
