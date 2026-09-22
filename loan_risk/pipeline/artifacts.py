"""Artifact persistence — the contract between training and serving.

Saves to paths.artifacts:
  * model.joblib          -> the fitted (calibrated) end-to-end pipeline
  * metadata.json         -> model name/version, calibration, training info,
                             selection metric, dataset manifest reference
  * feature_schema.json   -> the exact inputs the model expects (from schema.py),
                             so serving-layer validation + the Streamlit form
                             derive from one machine-readable source
  * metrics.json          -> full evaluation results (written by run.py)

The serving layer (ModelService) loads model.joblib + metadata.json +
feature_schema.json once, cached; it NEVER retrains.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from ..config import Config
from ..schema import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
FEATURE_SCHEMA_FILE = "feature_schema.json"
METRICS_FILE = "metrics.json"


def build_feature_schema() -> dict[str, Any]:
    """Machine-readable input contract for the API + frontend form."""
    return {
        "numeric_features": [
            {"name": f.name, "label": f.label, "unit": f.unit,
             "meaning": f.meaning, "min": f.min_value, "max": f.max_value,
             "example": f.example, "missing_behavior": f.missing_behavior}
            for f in NUMERIC_FEATURES
        ],
        "categorical_features": [
            {"name": f.name, "label": f.label, "meaning": f.meaning,
             "categories": list(f.categories), "example": f.example,
             "missing_behavior": f.missing_behavior}
            for f in CATEGORICAL_FEATURES
        ],
        "engineered_features": [
            {"name": e.name, "formula": e.formula, "inputs": list(e.inputs),
             "unit": e.unit, "interpretation": e.interpretation,
             "leakage_assessment": e.leakage_assessment}
            for e in ENGINEERED_FEATURES
        ],
    }


def save_model(estimator, cfg: Config) -> Path:
    out = cfg.path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    path = out / MODEL_FILE
    joblib.dump(estimator, path)
    return path


def save_json(obj: dict[str, Any], filename: str, cfg: Config) -> Path:
    out = cfg.path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def save_feature_schema(cfg: Config) -> Path:
    return save_json(build_feature_schema(), FEATURE_SCHEMA_FILE, cfg)


def artifacts_exist(cfg: Config) -> bool:
    out = cfg.path("artifacts")
    return (out / MODEL_FILE).exists() and (out / METADATA_FILE).exists()


def load_model(cfg: Config):
    path = cfg.path("artifacts") / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train first: "
            f"`python -m loan_risk.pipeline.run`."
        )
    return joblib.load(path)


def load_json(filename: str, cfg: Config) -> dict[str, Any]:
    path = cfg.path("artifacts") / filename
    if not path.exists():
        raise FileNotFoundError(f"Artifact {filename} not found at {path}.")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_metadata(cfg: Config) -> dict[str, Any]:
    return load_json(METADATA_FILE, cfg)


def load_feature_schema(cfg: Config) -> dict[str, Any]:
    return load_json(FEATURE_SCHEMA_FILE, cfg)
