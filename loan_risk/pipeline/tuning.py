"""Model selection and hyperparameter tuning (03_ML_REQUIREMENTS).

Two upgrades over a single train/validation split:

  * ``cross_val_select`` scores each model with stratified k-fold CV on the
    training data and selects by MEAN CV score, reporting the standard deviation
    so selection is not a one-split fluke. The whole sklearn Pipeline (including
    preprocessing) is cloned and refit inside every fold, so no fold's statistics
    leak into another — the same leakage discipline as the rest of the pipeline.

  * ``tune_model`` runs a RandomizedSearchCV over configured hyperparameter
    distributions for the heavier models. To keep the search tractable on the
    full ~307k-row dataset it may fit on a stratified subsample (``max_search_
    samples``); the winning hyperparameters are then refit on the full data by
    the caller. Search is opt-in via ``tuning.enabled``.

All fitting here happens on TRAIN(+VAL) data only; TEST is never touched.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from ..config import Config
from .models import ModelSpec, build_pipeline

# scoring names sklearn understands for our selection metrics
_SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}


def _scoring_name(cfg: Config) -> str:
    metric = cfg["evaluation"]["selection_metric"]
    return _SCORING.get(metric, "roc_auc")


def cross_val_select(
    specs: list[ModelSpec], X: pd.DataFrame, y, cfg: Config,
) -> tuple[str, dict[str, Any]]:
    """Score each model with stratified k-fold CV; select by mean score.

    Returns ``(best_key, per_model_cv)`` where per_model_cv[key] has
    ``mean``/``std``/``scores``/``scoring``.
    """
    folds = int(cfg["selection"]["cv_folds"])
    scoring = _scoring_name(cfg)
    y = np.asarray(y).ravel()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)

    per_model: dict[str, Any] = {}
    for spec in specs:
        pipe = build_pipeline(spec, cfg)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring, n_jobs=1)
        per_model[spec.key] = {
            "display_name": spec.display_name,
            "scoring": scoring,
            "cv_folds": folds,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": [float(s) for s in scores],
        }

    best_key = max(per_model, key=lambda k: per_model[k]["mean"])
    return best_key, per_model


def _subsample(X: pd.DataFrame, y: np.ndarray, n: int, seed: int):
    if len(X) <= n:
        return X, y
    # stratified subsample preserving the positive rate
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = max(1, round(n * len(pos_idx) / len(y)))
    n_neg = n - n_pos
    pick = np.concatenate([
        rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
        rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
    ])
    rng.shuffle(pick)
    return X.iloc[pick].reset_index(drop=True), y[pick]


def tune_model(
    spec: ModelSpec, X: pd.DataFrame, y, cfg: Config,
) -> dict[str, Any] | None:
    """RandomizedSearchCV for one model. Returns best params + CV score, or None
    if tuning is disabled or this model has no configured distribution."""
    tcfg = cfg.get("tuning") or {}
    if not tcfg.get("enabled"):
        return None
    if spec.key not in (tcfg.get("models") or []):
        return None
    dists = (tcfg.get("param_distributions") or {}).get(spec.key)
    if not dists:
        return None

    y = np.asarray(y).ravel()
    Xs, ys = _subsample(X, y, int(tcfg.get("max_search_samples", 40000)), cfg.seed)
    pipe = build_pipeline(spec, cfg)
    search = RandomizedSearchCV(
        pipe, param_distributions=dists,
        n_iter=int(tcfg.get("n_iter", 8)),
        cv=StratifiedKFold(n_splits=int(tcfg.get("cv_folds", 3)), shuffle=True,
                           random_state=cfg.seed),
        scoring=_scoring_name(cfg), random_state=cfg.seed, n_jobs=1, refit=False,
    )
    search.fit(Xs, ys)
    return {
        "best_params": dict(search.best_params_),
        "best_cv_score": float(search.best_score_),
        "scoring": _scoring_name(cfg),
        "n_iter": int(tcfg.get("n_iter", 8)),
        "search_samples": int(len(Xs)),
        "note": ("Hyperparameters selected by RandomizedSearchCV on a stratified "
                 "subsample; refit on the full training data."),
    }


def apply_params(spec: ModelSpec, best_params: dict[str, Any]) -> ModelSpec:
    """Return a copy of ``spec`` with tuned hyperparameters applied to a cloned
    estimator (keys are ``model__<param>`` from the pipeline search space)."""
    est = clone(spec.estimator)
    stripped = {k.split("__", 1)[1]: v for k, v in best_params.items()
                if k.startswith("model__")}
    est.set_params(**stripped)
    return ModelSpec(spec.key, spec.display_name, est, spec.scale)
