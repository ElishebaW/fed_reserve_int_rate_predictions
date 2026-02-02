FROM python:3.11-slim

# Keep Python output unbuffered, avoid .pyc files, and use headless backend for matplotlib
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    KAGGLE_CONFIG_DIR=/root/.kaggle

WORKDIR /app

# System libs needed by manylinux matplotlib wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default action: run the training script (override with `docker run ... <cmd>` if needed)
CMD ["python", "train_interest_rate_model.py"]
