FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-app.txt .

RUN python -m pip install --no-cache-dir -r requirements-app.txt

RUN useradd --create-home --uid 10001 appuser

COPY app/ ./app/
COPY artifacts/mobilenet_v2_finetuned.keras ./artifacts/mobilenet_v2_finetuned.keras
COPY reports/mobilenet_v2_finetuned_metrics.json ./reports/mobilenet_v2_finetuned_metrics.json

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]