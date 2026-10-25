"""Service layer between the Streamlit UI and the ML artifacts.

Loads the fitted pipeline + metadata + feature schema ONCE (cached per process)
and never retrains. Converts a validated ApplicantInput into the
exact canonical model frame the training pipeline expects — including the
missingness indicators the API caller does not supply directly
(DAYS_EMPLOYED_MISSING for pensioners/unemployed, and one flag per EXT_SOURCE
credit score) — so train/serve stay consistent.

Persistence seam: `persist_prediction` is a no-op stub returning None now
(stateless), with a clear signature for a future Postgres/Supabase layer.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from loan_risk.config import Config, load_config
from loan_risk.data.ingestion import EMPLOYMENT_MISSING_FLAG, EXT_SOURCE_MISSING_FLAGS
from loan_risk.pipeline import artifacts
from loan_risk.pipeline.explain import local_explanation
from loan_risk.pipeline.risk_score import (
    probability_to_band,
    probability_to_decision,
    probability_to_score,
    score_definition,
)
from loan_risk.schema import categorical_names, numeric_names
from loan_risk.serving.validation import validate_payload, OPTIONAL_FIELDS

logger = logging.getLogger(__name__)


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
    "Requires human oversight; designed to assist credit underwriting analysts.",
]

MODEL_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest (Ensemble Classifier)",
    "decision_tree": "Decision Tree (Interpretable Tree)",
    "logistic_regression": "Logistic Regression (Linear Baseline)",
}


class ModelService:
    """Holds the loaded models and answers predictions. One instance per process."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or load_config()
        self.model = None
        self.models: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.feature_schema: dict[str, Any] = {}
        self._load_error: Optional[str] = None
        self._try_load()

    # -- lifecycle ------------------------------------------------------
    def _try_load(self) -> None:
        try:
            self.metadata = artifacts.load_metadata(self.cfg)
            artifacts.verify_bundle(self.cfg, self.metadata)
            self.model = artifacts.load_model(self.cfg)
            self.models = artifacts.load_all_models(self.cfg)
            selected = self.metadata.get("model_key")
            if not selected:
                raise ValueError("Artifact metadata is missing the selected model identity.")
            self.models[selected] = self.model
            self.scoring_cfg = Config(self.cfg.raw | {
                "risk_score": self.metadata.get("risk_score_config", self.cfg["risk_score"])
            }, self.cfg.root)
            self.feature_schema = artifacts.load_feature_schema(self.cfg)
            self._metrics = artifacts.load_json(artifacts.METRICS_FILE, self.cfg)
            self._manifest = (artifacts.load_json("dataset_manifest.json", self.cfg)
                              if "dataset_manifest.json" in self.metadata.get("file_hashes", {}) else None)
            self._validation_report = (artifacts.load_json("validation_report.json", self.cfg)
                                       if "validation_report.json" in self.metadata.get("file_hashes", {}) else None)
            if (self.metadata.get("file_hashes") and
                    self._metrics.get("model_version") != self.metadata.get("model_version")):
                raise ValueError("Model and evaluation report versions disagree.")
            self._load_error = None
        except FileNotFoundError as exc:
            self.model = None
            self.models = {}
            self._load_error = "Required model artifacts are missing. Run training first."
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model artifacts could not be loaded")
            self.model = None
            self.models = {}
            self._load_error = f"Model artifact could not be loaded: {type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None or bool(self.models)

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def get_available_models(self) -> dict[str, str]:
        """Return dict of {model_key: display_name}."""
        return {key: MODEL_DISPLAY_NAMES.get(key, key.replace("_", " ").title())
                for key in self.models}

    # -- input transformation ------------------------------------------
    def to_model_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Build the single-row canonical frame the pipeline consumes."""
        row: dict[str, Any] = {}
        for name in numeric_names():
            row[name] = payload.get(name, None)
        emp = payload.get("EMPLOYMENT_YEARS", None)
        row[EMPLOYMENT_MISSING_FLAG] = 1 if emp is None else 0
        for col, flag in EXT_SOURCE_MISSING_FLAGS.items():
            row[flag] = 1 if payload.get(col, None) is None else 0
        for name in categorical_names():
            row[name] = payload.get(name, None)
        # Match training missing-value semantics: SimpleImputer expects NaN,
        # not Python None in object columns or all-missing numeric columns.
        frame = pd.DataFrame([row])
        for name in numeric_names():
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        for name in categorical_names():
            frame[name] = frame[name].where(frame[name].notna(), np.nan)
        return frame

    @staticmethod
    def _probability(model, X) -> float:
        classes = list(model.classes_)
        if classes != [0, 1]:
            raise ValueError("Model classes must be [0, 1].")
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        if (probabilities.shape != (1, 2) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1)):
            raise ValueError("Model returned an invalid probability distribution.")
        return float(probabilities[0, 1])

    def _threshold(self, key: str) -> tuple[float, str]:
        details = self.metadata.get("candidate_details", {}).get(key, {})
        threshold = details.get("decision_threshold")
        if threshold is None and key == self.metadata.get("model_key"):
            threshold = self.metadata.get("decision_threshold")
        if threshold is not None:
            return float(threshold), "held_out_validation_f1"
        return float(self.cfg.get("risk_score", {}).get("threshold", 0.5)), "configured_unvalidated"

    # -- prediction -----------------------------------------------------
    def predict(
        self,
        payload: dict[str, Any],
        model_key: str | None = None,
        *,
        with_explanation: bool = True,
    ) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("Model not loaded.")

        chosen_key = model_key or self.metadata.get("model_key")
        if chosen_key not in self.models:
            raise ValueError(f"Requested model is unavailable: {chosen_key!r}.")
        target_model = self.models[chosen_key]
        payload, errors = validate_payload(payload)
        if errors:
            raise ValueError("Invalid applicant input: " + "; ".join(
                f"{e['field']}: {e['message']}" for e in errors))

        X = self.to_model_frame(payload)
        prob = self._probability(target_model, X)
        score = probability_to_score(prob)
        scoring_cfg = getattr(self, "scoring_cfg", self.cfg)
        band = probability_to_band(prob, scoring_cfg)

        thr, threshold_basis = self._threshold(chosen_key)
        decision = probability_to_decision(prob, threshold=thr)

        explanation = None
        explanation_available = False
        if with_explanation:
            try:
                explanation = local_explanation(target_model, X, self.cfg)
                explanation_available = bool(explanation.get("contributions"))
            except Exception:
                logger.exception("Local explanation failed; returning prediction without it")
                explanation = None
                explanation_available = False

        # Multi-model consensus comparison
        all_preds = {}
        comparison_errors = []
        for m_key, m_estimator in self.models.items():
            try:
                p = prob if m_key == chosen_key else self._probability(m_estimator, X)
                model_threshold, model_threshold_basis = self._threshold(m_key)
                all_preds[m_key] = {
                    "model_key": m_key,
                    "display_name": MODEL_DISPLAY_NAMES.get(m_key, m_key.title()),
                    "default_probability": p,
                    "risk_score": probability_to_score(p),
                    "risk_band": probability_to_band(p, scoring_cfg),
                    "decision": probability_to_decision(p, threshold=model_threshold),
                    "threshold_basis": model_threshold_basis,
                }
            except Exception:
                logger.exception("Comparison prediction failed for %s", m_key)
                comparison_errors.append(m_key)

        missing = sorted(name for name in OPTIONAL_FIELDS if payload.get(name) is None)
        limitations = self._limitations()
        if missing:
            limitations.append("Missing inputs imputed by the model: " + ", ".join(missing))
        outside = [name for name, bounds in self.metadata.get("input_profile", {}).items()
                   if payload.get(name) is not None
                   and not bounds["min"] <= payload[name] <= bounds["max"]]
        if outside:
            limitations.append("Inputs outside the observed training range; estimate may extrapolate: "
                               + ", ".join(sorted(outside)))
        if threshold_basis == "configured_unvalidated":
            limitations.append("The configured threshold has not been validated for this artifact.")

        result = {
            "model_key": chosen_key,
            "model_name": MODEL_DISPLAY_NAMES.get(chosen_key, chosen_key.title()),
            "default_probability": prob,
            "risk_score": score,
            "risk_band": band,
            "predicted_target": decision["predicted_target"],
            "target_label": decision["target_label"],
            "decision": decision["recommendation"],
            "decision_color": decision["badge_color"],
            "decision_action": decision["action"],
            "decision_threshold": decision["threshold"],
            "threshold_basis": threshold_basis,
            "model_version": self.metadata.get("model_version", "0.1.0"),
            "trained_at": self.metadata.get("trained_at"),
            "assessed_at": datetime.now(timezone.utc).isoformat(),
            "dataset_source": self.metadata.get("dataset_source", "unknown"),
            "synthetic": self.metadata.get("synthetic"),
            "synthetic_warning": self.metadata.get("synthetic_warning"),
            "explanation_available": explanation_available,
            "explanation": explanation,
            "all_model_predictions": all_preds,
            "comparison_unavailable": comparison_errors,
            "missing_inputs": missing,
            "outside_training_range": sorted(outside),
            "limitations": limitations,
            "disclaimer": score_definition(scoring_cfg)["disclaimer"],
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
        if not self.metadata.get("file_hashes"):
            lims.append("Legacy artifact: training-run integrity has not been verified. Retrain before relying on results.")
        lims.append("External scores must use the original dataset definitions; estimates are not validated for a different population or currency scale.")
        return lims

    # -- persistence seam (stateless now) -------------------------------
    def persist_prediction(self, payload: dict[str, Any], result: dict[str, Any]) -> Optional[str]:
        """No-op today (stateless). A future Postgres/Supabase implementation
        would insert (payload, result) and return the stored row id. We do NOT
        log raw applicant PII here (05_API_CONTRACT.md)."""
        return None

    # -- read-only reports (for the dashboard; never fabricated) --------
    def get_metrics(self) -> Optional[dict[str, Any]]:
        # Snapshot loaded with the model; a later training run cannot swap its report.
        return getattr(self, "_metrics", None)

    def get_feature_schema(self) -> dict[str, Any]:
        return self.feature_schema

    def get_manifest(self) -> Optional[dict[str, Any]]:
        if "dataset_manifest.json" in self.metadata.get("file_hashes", {}):
            return self._manifest
        path = self.cfg.path("manifest")
        if not path.exists():
            return None
        import json
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def get_validation_report(self) -> Optional[dict[str, Any]]:
        if "validation_report.json" in self.metadata.get("file_hashes", {}):
            return self._validation_report
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
            "selection_basis": m.get("selection_basis"),
            "responsible_use_note": m.get(
                "responsible_use_note",
                "Analytical estimate; requires human oversight.",
            ),
        }
