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
import hashlib
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


def file_sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def verify_bundle(cfg: Config, metadata: dict[str, Any]) -> None:
    """Reject mixed/corrupt runs before loading pickle data. Local artifacts
    must still be trusted: checksums detect accidental changes, not attackers."""
    root = cfg.path("artifacts").resolve()
    hashes = metadata.get("file_hashes", {})
    if hashes and not {MODEL_FILE, METRICS_FILE, FEATURE_SCHEMA_FILE} <= hashes.keys():
        raise ValueError("Artifact bundle lacks required checksums.")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Artifact bundle is incomplete or has an invalid path.")
        if file_sha256(path) != expected:
            raise ValueError("Artifact checksum mismatch; retrain or restore the complete bundle.")


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
    joblib.dump(estimator, path, compress=3)
    return path


def save_json(obj: dict[str, Any], filename: str, cfg: Config) -> Path:
    out = cfg.path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str, allow_nan=False)
    return path


def save_feature_schema(cfg: Config) -> Path:
    return save_json(build_feature_schema(), FEATURE_SCHEMA_FILE, cfg)


def artifacts_exist(cfg: Config) -> bool:
    out = cfg.path("artifacts")
    return (out / MODEL_FILE).exists() and (out / METADATA_FILE).exists()


def load_model(cfg: Config, model_key: str | None = None):
    out = cfg.path("artifacts")
    if model_key:
        if model_key not in load_metadata(cfg).get("candidate_details", {}):
            raise ValueError("Requested model is not registered in this training run.")
        candidate_path = out / "models" / f"{model_key}.joblib"
        if candidate_path.exists():
            return joblib.load(candidate_path)
        raise FileNotFoundError("Registered candidate model is missing.")
    path = out / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train first: "
            f"`python -m loan_risk.pipeline.run`."
        )
    return joblib.load(path)


def load_all_models(cfg: Config) -> dict[str, Any]:
    """Load only candidates registered and hashed in the same training run."""
    models_dir = cfg.path("artifacts") / "models"
    result: dict[str, Any] = {}
    metadata = load_metadata(cfg)
    verify_bundle(cfg, metadata)
    for key in metadata.get("candidate_details", {}):
        relative = f"models/{key}.joblib"
        if relative not in metadata.get("file_hashes", {}):
            raise ValueError("Candidate model lacks run-integrity evidence.")
        result[key] = joblib.load(models_dir / f"{key}.joblib")
    return result


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
