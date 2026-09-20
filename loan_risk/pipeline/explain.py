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

import logging
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import Config
from .preprocess import get_output_feature_names

logger = logging.getLogger(__name__)

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:  # pragma: no cover
    _HAS_SHAP = False


def shap_available() -> bool:
    return _HAS_SHAP


def _iter_base_pipelines(estimator) -> Iterator[tuple[Pipeline, Any]]:
    """Yield each fitted ``(preprocessor, final_model)`` pair backing ``estimator``.

    A plain Pipeline yields one pair. A ``CalibratedClassifierCV`` yields one
    pair per calibration fold (``calibrated_classifiers_``): the SERVED model
    averages the probability outputs of ALL folds, so a global importance that
    reflected only fold 0 (as this module previously did) would misrepresent the
    deployed ensemble. Callers that need a single representative model can take
    the first pair.
    """
    if isinstance(estimator, Pipeline):
        yield Pipeline(estimator.steps[:-1]), estimator.steps[-1][1]
        return
    calibrators = getattr(estimator, "calibrated_classifiers_", None)
    if calibrators:
        for cal in calibrators:
            inner = getattr(cal, "estimator", None)
            if isinstance(inner, Pipeline):
                yield Pipeline(inner.steps[:-1]), inner.steps[-1][1]


def _unwrap(estimator):
    """Return the FIRST ``(preprocessor, final_model)`` pair (or ``(None, None)``).

    Used by the local, per-applicant explanation, where running SHAP across
    every calibration fold would be expensive; fold 0 is a documented
    approximation of the averaged ensemble.
    """
    for pre, model in _iter_base_pipelines(estimator):
        return pre, model
    return None, None


def _model_importances(model) -> tuple[np.ndarray, str] | None:
    """Native importance/coefficient vector for a model, with its method label."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float), "tree_feature_importances"
    if hasattr(model, "coef_"):
        return (np.abs(np.asarray(model.coef_, dtype=float).ravel()),
                "logistic_regression_abs_coefficient")
    return None


def global_importance(estimator, cfg: Config, top_k: int | None = None) -> dict[str, Any]:
    """Global feature importance from the model's native mechanism.

    For a calibrated model this AVERAGES importances across all calibration
    folds (keyed by post-transform feature name, so folds whose one-hot columns
    differ still combine correctly), matching the averaged ensemble that is
    actually served.
    """
    top_k = top_k or int(cfg["explainability"]["top_k_features"])

    agg: dict[str, float] = {}
    seen: dict[str, int] = {}  # folds in which each feature actually appeared
    method: str | None = None
    folds = 0
    for pre, model in _iter_base_pipelines(estimator):
        try:
            names = get_output_feature_names(pre)
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping a fold in global_importance: feature-name error: %s", exc)
            continue
        got = _model_importances(model)
        if got is None:
            continue
        vals, method = got
        if len(names) != len(vals):
            # Names and importances must align 1:1. A mismatch is a real defect
            # (not something to silently truncate past): skip the fold loudly.
            logger.warning(
                "global_importance: feature-name/importance length mismatch "
                "(%d names vs %d values); skipping this fold.", len(names), len(vals))
            continue
        for name, val in zip(names, vals):
            agg[name] = agg.get(name, 0.0) + float(val)
            seen[name] = seen.get(name, 0) + 1
        folds += 1

    if not agg or method is None:
        return {"method": "unavailable",
                "note": "Could not access underlying model(s) for importance.",
                "features": []}

    # Average each feature over the folds it actually appeared in. One-hot
    # columns for rare categories can be absent from a fold whose training
    # subset didn't contain that category; dividing by the global fold count
    # would understate them, so we divide by per-feature presence instead.
    mean_imp = {name: agg[name] / seen[name] for name in agg}
    ranked = sorted(mean_imp.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    interpretation = "Association with predicted risk, not causation."
    if folds > 1:
        interpretation += f" Averaged across {folds} calibration folds."
    return {
        "method": method,
        "folds_averaged": folds,
        "interpretation": interpretation,
        "features": [{"feature": str(name), "importance": imp}
                     for name, imp in ranked],
    }


def permutation_global_importance(
    estimator, X: pd.DataFrame, y, cfg: Config, *,
    top_k: int | None = None, n_repeats: int = 5, max_samples: int = 5000,
) -> dict[str, Any]:
    """Model-agnostic global importance via permutation on the held-out set.

    Unlike tree Gini importance (which is training-data, impurity-based, and
    biased toward high-cardinality splits), permutation importance measures the
    drop in a scoring metric when a feature's values are shuffled on data the
    model did not fit. Because it permutes the ORIGINAL input columns (before
    one-hot encoding), a categorical's importance is aggregated across all its
    dummy columns automatically — the "aggregate by original feature" view the
    fragmented Gini importance cannot give.

    Computed on a capped random sample for cost. Scored by ROC-AUC (ranking
    quality), which does not depend on the decision threshold.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    top_k = top_k or int(cfg["explainability"]["top_k_features"])
    X = X.reset_index(drop=True)
    y = np.asarray(y).ravel()
    if len(X) > max_samples:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]

    try:
        def _auc(est, Xs, ys):
            return roc_auc_score(ys, est.predict_proba(Xs)[:, 1])

        result = permutation_importance(
            estimator, X, y, scoring=_auc, n_repeats=n_repeats,
            random_state=cfg.seed, n_jobs=1,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("permutation importance failed: %s", exc)
        return {"method": "unavailable", "note": f"permutation importance error: {exc}",
                "features": []}

    order = np.argsort(result.importances_mean)[::-1][:top_k]
    cols = list(X.columns)
    return {
        "method": "permutation_importance_roc_auc",
        "scoring": "roc_auc",
        "n_repeats": n_repeats,
        "n_samples": int(len(X)),
        "interpretation": ("Mean drop in held-out ROC-AUC when the feature is "
                           "randomly shuffled. Association with predictive value, "
                           "not causation. Aggregated per original input feature."),
        "features": [
            {"feature": str(cols[i]),
             "importance": float(result.importances_mean[i]),
             "std": float(result.importances_std[i])}
            for i in order
        ],
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
        row = X_trans.ravel()
        _warn_if_len_mismatch("linear_coefficient", names, coef, row)
        contrib = coef[:len(names)] * row[:len(names)]
        contribs = _rank_contributions(names, contrib, top_k)
        return {"method": "linear_coefficient_times_value", "contributions": contribs,
                "interpretation": "Approximate signed contribution "
                                  "(coefficient x standardized value)."}

    # --- Tree fallback without SHAP: use global importances (unsigned) ---
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        _warn_if_len_mismatch("tree_importance_fallback", names, vals)
        contribs = _rank_contributions(names, vals[:len(names)], top_k, signed=False)
        return {"method": "tree_importance_fallback", "contributions": contribs,
                "note": "SHAP not installed; showing model global importances as "
                        "an APPROXIMATION (not a per-applicant attribution).",
                "interpretation": "Approximate association, not causation."}

    return {"method": "unavailable", "contributions": [],
            "note": "No supported explanation mechanism for this model."}


def _warn_if_len_mismatch(where: str, names, *arrays) -> None:
    """Log (do not crash) when explanation arrays don't align with feature names.

    Local explanations are served per-request, so we degrade gracefully rather
    than raise — but a mismatch means contributions could be mislabeled, which
    must not pass silently. In the current single-ColumnTransformer design the
    lengths always match; this guards against future drift.
    """
    for arr in arrays:
        if len(arr) != len(names):
            logger.warning(
                "local_explanation[%s]: length mismatch (%d names vs %d values); "
                "contributions truncated to the shorter length and may be misaligned.",
                where, len(names), len(arr))


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
