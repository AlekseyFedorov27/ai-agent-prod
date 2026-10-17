FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    FASTEMBED_CACHE_PATH=/home/appuser/.cache/fastembed

# curl нужен для healthcheck; build-essential на случай пакетов без готовых wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Приложение работает не от root
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Каталоги, которые монтируются как тома, создаём заранее с нужным владельцем:
# при первом монтировании именованный том наследует эти права.
RUN mkdir -p /app/uploads /home/appuser/.cache/fastembed \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
