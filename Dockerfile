FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor.py .

ENV MODEL_DIR=/app/model

EXPOSE 8080

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${AIP_HTTP_PORT:-8080} --workers 1 --threads 8 --timeout 0 --access-logfile - --error-logfile - predictor:app"]
