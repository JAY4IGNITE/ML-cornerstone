"""Model factory (03_ML_REQUIREMENTS.md — minimum 4 models).

Each model is paired with a preprocessing pipeline (linear models get scaling,
trees do not) into one sklearn Pipeline so preprocessing is fit only on train
and applied identically at inference.

XGBoost is optional: if the package is not installed it is skipped and the
skip is reported honestly (never silently claimed as trained).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ..config import Config
from .preprocess import build_preprocessor

try:  # optional dependency
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_XGB = False


def xgboost_available() -> bool:
    return _HAS_XGB


@dataclass
class ModelSpec:
    key: str
    display_name: str
    estimator: Any
    scale: bool  # whether preprocessing should scale numeric features


def _logreg(cfg: Config) -> ModelSpec:
    p = cfg["models"]["logistic_regression"]
    est = LogisticRegression(
        C=float(p["C"]), max_iter=int(p["max_iter"]),
        class_weight=p.get("class_weight"), random_state=cfg.seed,
    )
    # Scaling matters for the linear model's convergence/conditioning; trees are
    # scale-invariant and never scale. `scale_for_linear` makes the linear case
    # configurable (default True) instead of hardcoding it.
    scale = bool(cfg["preprocessing"].get("scale_for_linear", True))
    return ModelSpec("logistic_regression", "Logistic Regression", est, scale=scale)


def _tree(cfg: Config) -> ModelSpec:
    p = cfg["models"]["decision_tree"]
    est = DecisionTreeClassifier(
        max_depth=int(p["max_depth"]), min_samples_leaf=int(p["min_samples_leaf"]),
        class_weight=p.get("class_weight"), random_state=cfg.seed,
    )
    return ModelSpec("decision_tree", "Decision Tree", est, scale=False)


def _forest(cfg: Config) -> ModelSpec:
    p = cfg["models"]["random_forest"]
    est = RandomForestClassifier(
        n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
        min_samples_leaf=int(p["min_samples_leaf"]), n_jobs=int(p.get("n_jobs", -1)),
        class_weight=p.get("class_weight"), random_state=cfg.seed,
    )
    return ModelSpec("random_forest", "Random Forest", est, scale=False)


def _xgb(cfg: Config, scale_pos_weight: float) -> ModelSpec:
    p = cfg["models"]["xgboost"]
    est = XGBClassifier(
        n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
        learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]),
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        tree_method="hist", random_state=cfg.seed, n_jobs=-1,
    )
    return ModelSpec("xgboost", "XGBoost", est, scale=False)


def build_model_specs(cfg: Config, *, scale_pos_weight: float = 1.0) -> tuple[list[ModelSpec], list[str]]:
    """Return (enabled model specs, skipped-model notes)."""
    specs: list[ModelSpec] = []
    skipped: list[str] = []
    m = cfg["models"]
    if m["logistic_regression"]["enabled"]:
        specs.append(_logreg(cfg))
    if m["decision_tree"]["enabled"]:
        specs.append(_tree(cfg))
    if m["random_forest"]["enabled"]:
        specs.append(_forest(cfg))
    if m["xgboost"]["enabled"]:
        if _HAS_XGB:
            specs.append(_xgb(cfg, scale_pos_weight))
        else:
            skipped.append("xgboost: package not installed (skipped, not trained). "
                           "Install with `pip install -r requirements-extras.txt`.")
    return specs, skipped


def build_pipeline(spec: ModelSpec, cfg: Config) -> Pipeline:
    """Preprocessing + estimator as one fit/predict unit."""
    pre = build_preprocessor(cfg, scale=spec.scale)
    return Pipeline([("preprocess", pre), ("model", spec.estimator)])
