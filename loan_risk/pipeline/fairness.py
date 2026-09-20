"""Fairness slicing diagnostics (RESPONSIBLE_USE / 03_ML_REQUIREMENTS).

Computes held-out performance and error rates sliced by sensitive attributes
(``CODE_GENDER`` and age bands) so disparities are surfaced rather than hidden.
``CODE_GENDER`` ranks among the model's most important features, which makes
this diagnostic mandatory reading — not optional polish.

IMPORTANT — this is a DIAGNOSTIC, not a fairness certification. It reports raw
group metrics and simple disparity ratios on one held-out split. It does not:
  * establish legal compliance (disparate impact / ECOA / GDPR) — that requires
    domain and legal review;
  * mitigate any disparity it finds;
  * account for legitimate risk differences vs. proxy discrimination.
Read it as "here is where the model behaves differently across groups," and
treat any disparity as a flag for human review, per RESPONSIBLE_USE.md.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from .evaluate import compute_metrics

# Age bands (years) for slicing. Open-ended top band.
_AGE_BANDS = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 200)]
_MIN_GROUP = 100  # groups smaller than this are reported but flagged low-support


def _age_band_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}" if hi < 200 else f"{lo}+"


def _slice_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    m = compute_metrics(y_true, y_prob, threshold=threshold)
    cm = m["confusion_matrix"]
    # selection rate = fraction flagged positive at the operating threshold
    flagged = cm["tp"] + cm["fp"]
    n = m["n"]
    fnr = cm["fn"] / (cm["fn"] + cm["tp"]) if (cm["fn"] + cm["tp"]) else None
    fpr = cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) else None
    return {
        "n": n,
        "low_support": n < _MIN_GROUP,
        "positive_rate": m["positive_rate"],
        "selection_rate": (flagged / n) if n else None,
        "roc_auc": m["roc_auc"] if isinstance(m["roc_auc"], float) else None,
        "recall_tpr": m["recall"],
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "precision": m["precision"],
    }


def _disparity(groups: dict[str, dict[str, Any]], key: str) -> dict[str, Any] | None:
    """max/min ratio and gap across groups for a rate metric (ignores low-support
    and null groups). Ratio near 1.0 = parity."""
    vals = {g: s[key] for g, s in groups.items()
            if s.get(key) is not None and not s.get("low_support")}
    if len(vals) < 2:
        return None
    hi_g, hi = max(vals.items(), key=lambda kv: kv[1])
    lo_g, lo = min(vals.items(), key=lambda kv: kv[1])
    return {
        "metric": key,
        "max_group": hi_g, "max_value": hi,
        "min_group": lo_g, "min_value": lo,
        "gap": hi - lo,
        "ratio_min_over_max": (lo / hi) if hi else None,
    }


def fairness_report(
    X_test: pd.DataFrame, y_test, y_prob, cfg: Config, *, threshold: float,
) -> dict[str, Any]:
    """Sliced performance by CODE_GENDER and age band at the operating threshold."""
    y_test = np.asarray(y_test).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    X_test = X_test.reset_index(drop=True)

    report: dict[str, Any] = {
        "operating_threshold": float(threshold),
        "disclaimer": (
            "Diagnostic only. Raw group metrics + simple disparity ratios on one "
            "held-out split. NOT a fairness certification, legal-compliance "
            "assessment, or mitigation. Disparities are flags for human review."
        ),
        "slices": {},
        "disparities": {},
    }

    # --- CODE_GENDER slices ---
    if "CODE_GENDER" in X_test.columns:
        gcol = X_test["CODE_GENDER"].astype("object")
        gender_groups: dict[str, dict[str, Any]] = {}
        for val in sorted(gcol.dropna().unique()):
            mask = (gcol == val).to_numpy()
            if mask.sum() == 0:
                continue
            gender_groups[str(val)] = _slice_metrics(y_test[mask], y_prob[mask], threshold)
        report["slices"]["CODE_GENDER"] = gender_groups
        report["disparities"]["CODE_GENDER"] = {
            k: _disparity(gender_groups, k)
            for k in ("selection_rate", "false_negative_rate",
                      "false_positive_rate", "roc_auc")
        }

    # --- AGE_YEARS band slices ---
    if "AGE_YEARS" in X_test.columns:
        age = pd.to_numeric(X_test["AGE_YEARS"], errors="coerce").to_numpy()
        age_groups: dict[str, dict[str, Any]] = {}
        for lo, hi in _AGE_BANDS:
            mask = (age >= lo) & (age < hi)
            if mask.sum() == 0:
                continue
            age_groups[_age_band_label(lo, hi)] = _slice_metrics(
                y_test[mask], y_prob[mask], threshold)
        report["slices"]["AGE_BAND"] = age_groups
        report["disparities"]["AGE_BAND"] = {
            k: _disparity(age_groups, k)
            for k in ("selection_rate", "false_negative_rate",
                      "false_positive_rate", "roc_auc")
        }

    return report
