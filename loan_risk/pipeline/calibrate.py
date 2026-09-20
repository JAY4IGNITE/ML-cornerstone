"""Probability calibration (03_ML_REQUIREMENTS.md).

Wraps the selected fitted pipeline in CalibratedClassifierCV using cross-
validation on the TRAIN+VAL data only (the test set stays untouched until final
evaluation). We report Brier score before vs after so the calibration claim is
evidence-backed, never asserted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from ..config import Config


def calibrate_pipeline(fitted_pipeline, X_fit, y_fit, cfg: Config):
    """Return a calibrated estimator trained via CV on the provided data.

    ``fitted_pipeline`` is cloned (unfitted) so calibration re-fits base models
    within CV folds — no leakage from a model already fit on all of X_fit.
    """
    method = cfg["calibration"]["method"]
    cv = int(cfg["calibration"]["cv"])
    base = clone(fitted_pipeline)
    calibrated = CalibratedClassifierCV(base, method=method, cv=cv)
    calibrated.fit(X_fit, y_fit)
    return calibrated


def calibration_effect(y_true, prob_before, prob_after) -> dict[str, Any]:
    """Brier score improvement from calibration (lower is better)."""
    b_before = float(brier_score_loss(np.asarray(y_true).astype(int), prob_before))
    b_after = float(brier_score_loss(np.asarray(y_true).astype(int), prob_after))
    return {
        "brier_before": b_before,
        "brier_after": b_after,
        "improvement": b_before - b_after,
        "improved": b_after < b_before,
        "method": None,  # filled by caller
    }
