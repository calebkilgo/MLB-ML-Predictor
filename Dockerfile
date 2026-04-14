FROM python:3.11-slim

# System deps needed by lightgbm and a few other wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install Python deps first (caches across code changes)
COPY pyproject.toml ./
COPY README.md ./README.md
RUN pip install --upgrade pip && \
    pip install . && \
    pip install python-multipart

# Copy the rest of the project
COPY . .

# Fly.io will mount a volume at /app/data so this directory exists
# and persists across deploys. We still create it here so the first
# cold start doesn't fail before the volume is attached.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/access \
             /app/data/predictions /app/models /app/reports

EXPOSE 8080

# Entry: first-boot ETL + training if needed, then uvicorn.
# The ETL scripts are idempotent — if parquets already exist, they skip.
CMD bash -c '\
    set -e; \
    if [ ! -f /app/data/processed/features_v2.parquet ] || [ ! -f /app/models/clf_v2.pkl ]; then \
        echo "[bootstrap] first boot: running full ETL + training"; \
        python -m src.etl.build_dataset && \
        python -m src.etl.starter_logs && \
        python -m src.features.rolling_v2 && \
        python -m src.models.train_v2 ; \
    else \
        echo "[bootstrap] features and model already on volume, skipping ETL"; \
    fi; \
    exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}'