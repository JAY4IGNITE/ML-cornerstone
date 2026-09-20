# Backend + ML pipeline image. Multi-purpose: trains on synthetic data at first
# run (if no artifacts are mounted) and serves the FastAPI app.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal; scikit-learn/xgboost wheels are prebuilt.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching).
COPY requirements.txt requirements-extras.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-extras.txt

# Copy source and install the package.
COPY loan_risk ./loan_risk
COPY backend ./backend
COPY config ./config
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN pip install -e . && chmod +x docker/entrypoint.sh

# Non-root user for safety.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
ENTRYPOINT ["./docker/entrypoint.sh"]
