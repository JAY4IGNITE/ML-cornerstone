#!/usr/bin/env bash
# Container entrypoint: ensure a trained model exists, then serve the API.
#
# First-run training logic:
#   * If a model artifact is already present, skip training and just serve.
#   * If there is no artifact AND no real-data CSV on disk, force training on the
#     SYNTHETIC fixture. config/config.yaml ships `dataset.source: real`, and the
#     pipeline raises FileNotFoundError for `source: real` with no CSV — which
#     would crash a fresh clone. We avoid editing config.yaml: instead we write a
#     synthetic override config to a writable temp path and point LOAN_RISK_CONFIG
#     at it (the config loader honors that env var).
#   * If there is no artifact but a real-data CSV IS present (mounted data), train
#     on the configured source as-is.
# Mount real artifacts or real data to serve a real-data model instead.
set -euo pipefail

ARTIFACT="${LOAN_RISK_ARTIFACTS:-artifacts}/model.joblib"
CONFIG_SRC="${LOAN_RISK_CONFIG:-config/config.yaml}"
# Default location the real Kaggle file lands in (config paths.data_raw +
# dataset.primary_file). Override by mounting data/ or setting LOAN_RISK_CONFIG.
REAL_CSV="${LOAN_RISK_REAL_CSV:-data/raw/application_train.csv}"

if [ ! -f "$ARTIFACT" ]; then
  if [ ! -f "$REAL_CSV" ]; then
    echo "[entrypoint] No model artifact and no real-data CSV found — training on the SYNTHETIC fixture."
    OVERRIDE="/tmp/loan_risk_synthetic_config.yaml"
    # Copy the shipped config with dataset.source forced to synthetic. Relative
    # paths inside stay resolved against the repo root, not the config location,
    # so a /tmp override still writes data/artifacts under /app.
    python - "$CONFIG_SRC" "$OVERRIDE" <<'PY'
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
cfg["dataset"]["source"] = "synthetic"
with open(dst, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
PY
    export LOAN_RISK_CONFIG="$OVERRIDE"
    python -m loan_risk.pipeline.run
  else
    echo "[entrypoint] No model artifact found — real-data CSV present, training on the configured dataset source."
    python -m loan_risk.pipeline.run
  fi
else
  echo "[entrypoint] Found existing model artifact — skipping training."
fi

echo "[entrypoint] Starting API on 0.0.0.0:8000"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
