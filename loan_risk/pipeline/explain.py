"""Explainability (03/06 requirements).

Two levels:
  * GLOBAL: feature importance for the final model, computed from the model's
    own mechanism (LogReg coefficients, tree feature_importances_) over the
    POST-TRANSFORM feature names, so names survive preprocessing.
  * LOCAL: per-prediction contributions. Uses SHAP when installed and the model
    is supported; otherwise falls back to a transparent linear/tree-based
    approximation. Every explanation states which method produced it.

Honesty rules (03_ML_REQUIREMENTS.md):
  * importance is described as association, NOT causation;
  * we never invent adverse-action reasons — a fallback explanation is labeled
    as an approximation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import Config
from .preprocess import get_output_feature_names

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:  # pragma: no cover
    _HAS_SHAP = False


def shap_available() -> bool:
    return _HAS_SHAP


def _unwrap(estimator):
    """Return (preprocessor Pipeline, final model) from a fitted estimator that
    may be a plain Pipeline or a CalibratedClassifierCV wrapping one."""
    if isinstance(estimator, Pipeline):
        pre = Pipeline(estimator.steps[:-1])
        model = estimator.steps[-1][1]
        return pre, model
    # CalibratedClassifierCV: use the first calibrated sub-estimator's pipeline
    calibrators = getattr(estimator, "calibrated_classifiers_", None)
    if calibrators:
        inner = getattr(calibrators[0], "estimator", None)
        if isinstance(inner, Pipeline):
            pre = Pipeline(inner.steps[:-1])
            model = inner.steps[-1][1]
            return pre, model
    return None, None


def global_importance(estimator, cfg: Config, top_k: int | None = None) -> dict[str, Any]:
    """Global feature importance from the model's native mechanism."""
    top_k = top_k or int(cfg["explainability"]["top_k_features"])
    pre, model = _unwrap(estimator)
    if pre is None or model is None:
        return {"method": "unavailable",
                "note": "Could not access underlying model for importance.",
                "features": []}

    try:
        names = get_output_feature_names(pre)
    except Exception as exc:  # pragma: no cover
        return {"method": "unavailable", "note": f"feature names error: {exc}",
                "features": []}

    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        method = "tree_feature_importances"
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        method = "logistic_regression_abs_coefficient"
    else:
        return {"method": "unavailable",
                "note": "Model exposes neither importances nor coefficients.",
                "features": []}

    n = min(len(names), len(vals))
    order = np.argsort(vals[:n])[::-1][:top_k]
    return {
        "method": method,
        "interpretation": "Association with predicted risk, not causation.",
        "features": [{"feature": str(names[i]), "importance": float(vals[i])}
                     for i in order],
    }


def local_explanation(estimator, X_row: pd.DataFrame, cfg: Config,
                      top_k: int | None = None) -> dict[str, Any]:
    """Per-prediction contributions for a single applicant row (1-row DF)."""
    top_k = top_k or int(cfg["explainability"]["top_k_features"])
    pre, model = _unwrap(estimator)
    if pre is None or model is None:
        return {"method": "unavailable", "contributions": [],
                "note": "Underlying model not accessible for local explanation."}

    try:
        names = get_output_feature_names(pre)
        X_trans = pre.transform(X_row)
        X_trans = np.asarray(X_trans, dtype=float)
    except Exception as exc:  # pragma: no cover
        return {"method": "unavailable", "contributions": [],
                "note": f"transform error: {exc}"}

    # --- SHAP path ---
    if _HAS_SHAP and hasattr(model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_trans)
            row = _extract_shap_row(sv, n_features=len(names))
            if row is not None:
                contribs = _rank_contributions(names, row, top_k)
                return {"method": "shap_tree", "contributions": contribs,
                        "interpretation": "SHAP values: signed contribution to the "
                                          "model's risk output for this applicant."}
        except Exception:  # fall through to native
            pass

    # --- Native linear path ---
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float).ravel()
        contrib = coef[:len(names)] * X_trans.ravel()[:len(names)]
        contribs = _rank_contributions(names, contrib, top_k)
        return {"method": "linear_coefficient_times_value", "contributions": contribs,
                "interpretation": "Approximate signed contribution "
                                  "(coefficient x standardized value)."}

    # --- Tree fallback without SHAP: use global importances (unsigned) ---
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        contribs = _rank_contributions(names, vals[:len(names)], top_k, signed=False)
        return {"method": "tree_importance_fallback", "contributions": contribs,
                "note": "SHAP not installed; showing model global importances as "
                        "an APPROXIMATION (not a per-applicant attribution).",
                "interpretation": "Approximate association, not causation."}

    return {"method": "unavailable", "contributions": [],
            "note": "No supported explanation mechanism for this model."}


def _extract_shap_row(sv, n_features: int) -> np.ndarray | None:
    """Normalize SHAP output (which varies by version/model) to a 1-D vector of
    per-feature contributions for a single row, positive class.

    Handles:
      * list[array] (older API): index 1 = positive class
      * ndarray (n_samples, n_features)                 -> row 0
      * ndarray (n_samples, n_features, n_classes)      -> row 0, class 1
    """
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    arr = np.asarray(sv, dtype=float)
    if arr.ndim == 3:                       # (samples, features, classes)
        cls = 1 if arr.shape[2] > 1 else 0
        arr = arr[0, :, cls]
    elif arr.ndim == 2:                     # (samples, features)
        arr = arr[0]
    arr = arr.ravel()
    if arr.size != n_features:
        return None
    return arr


def _rank_contributions(names, values, top_k: int, signed: bool = True) -> list[dict]:
    values = np.asarray(values, dtype=float)
    n = min(len(names), len(values))
    order = np.argsort(np.abs(values[:n]))[::-1][:top_k]
    out = []
    for i in order:
        out.append({
            "feature": str(names[i]),
            "contribution": float(values[i]),
            "direction": ("increases_risk" if values[i] > 0 else "decreases_risk")
            if signed else "n/a",
        })
    return out
