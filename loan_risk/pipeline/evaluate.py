"""Evaluation metrics + threshold analysis (03/05 requirements).

Computes the full required metric set (accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, confusion matrix, Brier score) plus a calibration curve and a
threshold sweep. Never selects a model on accuracy alone — the orchestrator
uses the configured selection_metric (default ROC-AUC).

All numbers are computed from real predictions on held-out data. Nothing is
fabricated; if a metric cannot be computed it is reported as null with a reason.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, *, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    def _safe(fn_, **kw):
        try:
            return float(fn_(y_true, y_prob if kw.pop("proba", False) else y_pred, **kw))
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc)}

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe(roc_auc_score, proba=True),
        "pr_auc": _safe(average_precision_score, proba=True),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def threshold_analysis(y_true, y_prob, thresholds=None) -> list[dict[str, float]]:
    """Sweep decision thresholds and report the operating trade-offs."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "flagged_rate": float(y_pred.mean()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })
    return rows


def calibration_curve_points(y_true, y_prob, n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability-curve points: mean predicted vs observed frequency per bin."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    points = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        points.append({
            "bin_lower": float(bins[b]),
            "bin_upper": float(bins[b + 1]),
            "mean_predicted": float(y_prob[mask].mean()),
            "observed_frequency": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return points


def best_f1_threshold(y_true, y_prob) -> dict[str, float]:
    """Threshold maximizing F1 on the given set (reported, not auto-applied)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    # precision_recall_curve returns one extra prec/rec point without a threshold
    best_i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return {
        "threshold": float(thr[best_i]) if len(thr) else 0.5,
        "f1": float(f1[best_i]) if len(thr) else 0.0,
        "precision": float(prec[best_i]) if len(thr) else 0.0,
        "recall": float(rec[best_i]) if len(thr) else 0.0,
    }
