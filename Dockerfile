FROM python:3.12-slim

WORKDIR /app

COPY requirements-app.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -r requirements-app.txt && pip install --no-cache-dir --no-deps -e .

COPY app/ ./app/
COPY data/processed/hotels.csv ./data/processed/hotels.csv
COPY data/processed/hotel_aspect_scores.json ./data/processed/hotel_aspect_scores.json

# Cloud Run injects $PORT and expects the container to listen on it.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
