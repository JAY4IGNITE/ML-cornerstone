"""Service layer between FastAPI and the ML artifacts.

Loads the fitted pipeline + metadata + feature schema ONCE at startup and never
retrains (05_API_CONTRACT.md). Converts a validated ApplicantInput into the
exact canonical model frame the training pipeline expects — including the
missingness indicators the API caller does not supply directly
(DAYS_EMPLOYED_MISSING for pensioners/unemployed, and one flag per EXT_SOURCE
credit score) — so train/serve stay consistent.

Persistence seam: `persist_prediction` is a no-op stub returning None now
(stateless), with a clear signature for a future Postgres/Supabase layer.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

import pandas as pd

from loan_risk.config import Config, load_config
from loan_risk.data.ingestion import EMPLOYMENT_MISSING_FLAG, EXT_SOURCE_MISSING_FLAGS
from loan_risk.pipeline import artifacts
from loan_risk.pipeline.explain import local_explanation
from loan_risk.pipeline.risk_score import (
    probability_to_band,
    probability_to_score,
    score_definition,
)
from loan_risk.schema import categorical_names, numeric_names


def _safe_reference(value: Any) -> Optional[str]:
    """Reduce a stored artifact reference to a bare filename before it leaves
    the API. run.py already writes this relative, but if an older/absolute path
    ever reached metadata we must not expose a host filesystem path to clients
    (05_API_CONTRACT.md: no internal path leakage). Splitting on both separators
    handles Windows- and POSIX-style values alike."""
    if not value:
        return None
    text = str(value).replace("\\", "/")
    return text.rsplit("/", 1)[-1] or None


LIMITATIONS_BASE = [
    "Estimate of default probability, not a guaranteed outcome.",
    "Trained on an application-level feature subset; does not use full credit "
    "bureau history.",
    "Requires human oversight; not an autonomous lending decision.",
]


class ModelService:
    """Holds the loaded model and answers predictions. One instance per process."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or load_config()
        self.model = None
        self.metadata: dict[str, Any] = {}
        self.feature_schema: dict[str, Any] = {}
        self._load_error: Optional[str] = None
        self._try_load()

    # -- lifecycle ------------------------------------------------------
    def _try_load(self) -> None:
        # A missing artifact is the common case (train-on-first-run); a corrupt
        # or unreadable one must also degrade to "model unavailable" rather than
        # crash the whole process at import/startup. Either way the API stays up
        # and /health reports the model as not ready.
        try:
            self.model = artifacts.load_model(self.cfg)
            self.metadata = artifacts.load_metadata(self.cfg)
            self.feature_schema = artifacts.load_feature_schema(self.cfg)
            self._load_error = None
        except FileNotFoundError as exc:
            self.model = None
            self._load_error = str(exc)
        except Exception as exc:  # noqa: BLE001 — deliberate: never crash on load
            self.model = None
            self._load_error = f"Model artifact could not be loaded: {type(exc).__name__}."

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # -- input transformation ------------------------------------------
    def to_model_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Build the single-row canonical frame the pipeline consumes."""
        row: dict[str, Any] = {}
        for name in numeric_names():
            row[name] = payload.get(name, None)
        # indicators the caller doesn't send, mirroring ingestion.standardize so
        # train/serve produce identical columns:
        #  - employment missing => pensioner/unemployed (EMPLOYMENT_YEARS omitted)
        emp = payload.get("EMPLOYMENT_YEARS", None)
        row[EMPLOYMENT_MISSING_FLAG] = 1 if emp is None else 0
        #  - one flag per external credit score; missing => the caller omitted it
        for col, flag in EXT_SOURCE_MISSING_FLAGS.items():
            row[flag] = 1 if payload.get(col, None) is None else 0
        for name in categorical_names():
            row[name] = payload.get(name, None)
        return pd.DataFrame([row])

    # -- prediction -----------------------------------------------------
    def predict(self, payload: dict[str, Any], *, with_explanation: bool = True) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("Model not loaded.")
        X = self.to_model_frame(payload)
        prob = float(self.model.predict_proba(X)[:, 1][0])
        score = probability_to_score(prob)
        band = probability_to_band(prob, self.cfg)

        explanation = None
        explanation_available = False
        if with_explanation:
            try:
                explanation = local_explanation(self.model, X, self.cfg)
                explanation_available = bool(explanation.get("contributions"))
            except Exception:
                explanation = None
                explanation_available = False

        result = {
            "default_probability": prob,
            "risk_score": score,
            "risk_band": band,
            "model_version": self.metadata.get("model_version", "unknown"),
            "explanation_available": explanation_available,
            "explanation": explanation,
            "limitations": self._limitations(),
            "disclaimer": score_definition(self.cfg)["disclaimer"],
        }
        pid = self.persist_prediction(payload, result)
        if pid:
            result["prediction_id"] = pid
        return result

    def _limitations(self) -> list[str]:
        lims = list(LIMITATIONS_BASE)
        if self.metadata.get("synthetic"):
            lims.insert(0, "MODEL TRAINED ON SYNTHETIC DATA — outputs are for "
                           "demonstration only and are NOT real risk estimates.")
        return lims

    # -- persistence seam (stateless now) -------------------------------
    def persist_prediction(self, payload: dict[str, Any], result: dict[str, Any]) -> Optional[str]:
        """No-op today (stateless). A future Postgres/Supabase implementation
        would insert (payload, result) and return the stored row id. We do NOT
        log raw applicant PII here (05_API_CONTRACT.md)."""
        return None

    # -- read-only reports (for the dashboard; never fabricated) --------
    def get_metrics(self) -> Optional[dict[str, Any]]:
        try:
            return artifacts.load_json(artifacts.METRICS_FILE, self.cfg)
        except FileNotFoundError:
            return None

    def get_feature_schema(self) -> dict[str, Any]:
        return self.feature_schema

    def get_manifest(self) -> Optional[dict[str, Any]]:
        path = self.cfg.path("manifest")
        if not path.exists():
            return None
        import json
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def get_validation_report(self) -> Optional[dict[str, Any]]:
        path = self.cfg.path("reports") / "validation_report.json"
        if not path.exists():
            return None
        import json
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # -- model info -----------------------------------------------------
    def model_info(self) -> dict[str, Any]:
        m = self.metadata
        return {
            "model_name": m.get("model_name", "unknown"),
            "model_version": m.get("model_version", "unknown"),
            "model_key": m.get("model_key"),
            "trained_at": m.get("trained_at"),
            "dataset_source": m.get("dataset_source", "unknown"),
            "synthetic": bool(m.get("synthetic", False)),
            "synthetic_warning": m.get("synthetic_warning"),
            "manifest_reference": _safe_reference(m.get("manifest_reference")),
            "feature_schema_version": m.get("feature_schema_version", "unknown"),
            "calibration": m.get("calibration", {}),
            "selection_metric": m.get("selection_metric"),
            "responsible_use_note": m.get(
                "responsible_use_note",
                "Analytical estimate; requires human oversight.",
            ),
        }
