#!/usr/bin/env bash
# Container entrypoint: ensure a trained model exists (train on the SYNTHETIC
# fixture if artifacts are absent), then serve the API. Mount real artifacts or
# real data + set LOAN_RISK config to serve a real-data model instead.
set -euo pipefail

ARTIFACT="${LOAN_RISK_ARTIFACTS:-artifacts}/model.joblib"

if [ ! -f "$ARTIFACT" ]; then
  echo "[entrypoint] No model artifact found — training on synthetic data..."
  python -m loan_risk.pipeline.run
else
  echo "[entrypoint] Found existing model artifact — skipping training."
fi

echo "[entrypoint] Starting API on 0.0.0.0:8000"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
