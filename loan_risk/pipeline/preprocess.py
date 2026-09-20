"""Preprocessing — sklearn Pipeline + ColumnTransformer (02/03 requirements).

Design guarantees:
  * Feature engineering runs FIRST (stateless), then the ColumnTransformer.
  * All fitting happens on TRAIN only (the caller fits the full pipeline on
    X_train) — no test statistics ever leak (Non-Negotiable Principle).
  * Numeric: median imputation (+ scaling for linear models when configured).
  * Categorical: explicit unknown handling + one-hot; unseen categories at
    inference are ignored (handle_unknown='ignore'), never crash.
  * Feature names are preserved via get_feature_names_out for explainability.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import Config
from ..schema import categorical_names, engineered_names, numeric_names
from ..data.ingestion import EMPLOYMENT_MISSING_FLAG, EXT_SOURCE_MISSING_FLAGS
from .features import FeatureEngineer


def numeric_feature_columns() -> list[str]:
    """Numeric columns entering the ColumnTransformer: raw numeric + the
    missingness indicators (employment + external scores) + engineered
    features."""
    return (
        numeric_names()
        + [EMPLOYMENT_MISSING_FLAG]
        + list(EXT_SOURCE_MISSING_FLAGS.values())
        + engineered_names()
    )


def build_column_transformer(cfg: Config, *, scale: bool) -> ColumnTransformer:
    num_cols = numeric_feature_columns()
    cat_cols = categorical_names()
    unknown_token = cfg["preprocessing"]["categorical_unknown_token"]

    numeric_steps: list = [
        ("impute", SimpleImputer(strategy=cfg["preprocessing"]["numeric_imputation"])),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=unknown_token)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor(cfg: Config, *, scale: bool) -> Pipeline:
    """Full stateless-engineering + fitted-transform preprocessing pipeline.

    ``scale`` should be True for linear models (LogReg) and False for trees.
    """
    return Pipeline([
        ("engineer", FeatureEngineer()),
        ("columns", build_column_transformer(cfg, scale=scale)),
    ])


def _find_column_transformer(obj) -> ColumnTransformer | None:
    """Locate the fitted ColumnTransformer wherever it sits in a (possibly
    nested) Pipeline — it may be under a 'preprocess' step or at top level."""
    if isinstance(obj, ColumnTransformer):
        return obj
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            found = _find_column_transformer(step)
            if found is not None:
                return found
    return None


def get_output_feature_names(fitted_preprocessor: Pipeline) -> list[str]:
    """Recover post-transform feature names for explainability / consistency."""
    ct = _find_column_transformer(fitted_preprocessor)
    if ct is None:
        raise ValueError("No fitted ColumnTransformer found in preprocessor.")
    return list(ct.get_feature_names_out())
