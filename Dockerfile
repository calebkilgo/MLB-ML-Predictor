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
    pip install python-multipart xgboost scipy

# Copy the rest of the project
COPY . .

# Fly.io will mount a volume at /app/data so this directory exists
# and persists across deploys. We still create it here so the first
# cold start doesn't fail before the volume is attached.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/access \
             /app/data/predictions /app/models /app/reports

EXPOSE 8080

# Model version stamp — bump this string whenever the feature set or
# model architecture changes so Railway automatically retrains on next deploy.
# The bootstrap writes this stamp to /app/models/.model_version after a
# successful train; if it is absent or mismatched, a retrain is triggered.
ENV MODEL_VERSION="v2-ensemble-2025-04"

# Entry: version-gated ETL + training, then uvicorn.
# - Full ETL runs only when neither features nor raw data exist (first boot).
# - Model retrain runs whenever MODEL_VERSION stamp is stale — this lets
#   you force a retrain by bumping MODEL_VERSION without deleting volumes.
CMD bash -c '\
    set -e; \
    STAMP=/app/models/.model_version; \
    CURRENT=$(cat "$STAMP" 2>/dev/null || echo ""); \
    NEED_ETL=0; NEED_TRAIN=0; \
    if [ ! -f /app/data/processed/features_v2.parquet ]; then \
        NEED_ETL=1; NEED_TRAIN=1; \
    fi; \
    if [ "$CURRENT" != "$MODEL_VERSION" ] || [ ! -f /app/models/clf_v2.pkl ]; then \
        NEED_TRAIN=1; \
    fi; \
    if [ "$NEED_ETL" = "1" ]; then \
        echo "[bootstrap] first boot — running full ETL"; \
        python -m src.etl.build_dataset && \
        python -m src.etl.starter_logs; \
    fi; \
    if [ "$NEED_TRAIN" = "1" ]; then \
        echo "[bootstrap] rebuilding features + training ensemble (version $MODEL_VERSION)"; \
        python -m src.features.rolling_v2 && \
        python -m src.models.train_v2 && \
        echo "$MODEL_VERSION" > "$STAMP"; \
    else \
        echo "[bootstrap] model is current ($MODEL_VERSION), skipping retrain"; \
    fi; \
    exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}'