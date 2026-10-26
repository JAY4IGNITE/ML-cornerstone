"""Feature engineering — implemented once, used identically in training and
serving (Execution Rule #9). Every engineered feature is defined in
schema.ENGINEERED_FEATURES with a documented formula + leakage assessment.

All features here are point-in-time: they use only application-level fields
known at prediction time, so they introduce no leakage.

This is a stateless sklearn transformer (no fitting on data) so it slots into a
Pipeline BEFORE the ColumnTransformer and can never leak test statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..schema import engineered_names


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered columns added.

    Ratios guard against division-by-zero by returning NaN (later imputed),
    never inf — consistent with the 'no infinite values' validation rule.
    """
    out = df.copy()

    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return num / den

    def _col(name: str) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series(np.nan, index=out.index)

    out["CREDIT_INCOME_RATIO"] = safe_div(_col("AMT_CREDIT"), _col("AMT_INCOME_TOTAL"))
    out["ANNUITY_INCOME_RATIO"] = safe_div(_col("AMT_ANNUITY"), _col("AMT_INCOME_TOTAL"))
    out["CREDIT_ANNUITY_RATIO"] = safe_div(_col("AMT_CREDIT"), _col("AMT_ANNUITY"))
    out["CREDIT_GOODS_RATIO"] = safe_div(_col("AMT_CREDIT"), _col("AMT_GOODS_PRICE"))
    out["EMPLOYMENT_AGE_RATIO"] = safe_div(_col("EMPLOYMENT_YEARS"), _col("AGE_YEARS"))
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out.columns]
    if ext_cols:
        ext_frame = out[ext_cols]
        out["EXT_SOURCE_MEAN"] = ext_frame.mean(axis=1, skipna=True)
        out["EXT_SOURCE_MIN"] = ext_frame.min(axis=1, skipna=True)
        out["EXT_SOURCE_MAX"] = ext_frame.max(axis=1, skipna=True)
        out["EXT_SOURCE_STD"] = ext_frame.std(axis=1, skipna=True)
    else:
        out["EXT_SOURCE_MEAN"] = np.nan
        out["EXT_SOURCE_MIN"] = np.nan
        out["EXT_SOURCE_MAX"] = np.nan
        out["EXT_SOURCE_STD"] = np.nan
    out["INCOME_PER_FAM_MEMBER"] = safe_div(_col("AMT_INCOME_TOTAL"), _col("CNT_FAM_MEMBERS"))
    out["PAYMENT_RATE"] = safe_div(_col("AMT_ANNUITY"), _col("AMT_CREDIT"))
    out["GOODS_CREDIT_DIFF"] = _col("AMT_CREDIT") - _col("AMT_GOODS_PRICE")
    out["AGE_EMPLOYED_DIFF"] = _col("AGE_YEARS") - _col("EMPLOYMENT_YEARS")
    out["CHILDREN_RATIO"] = safe_div(_col("CNT_CHILDREN"), _col("CNT_FAM_MEMBERS"))
    out["INCOME_MINUS_ANNUITY"] = _col("AMT_INCOME_TOTAL") - _col("AMT_ANNUITY")

    # replace any residual inf (belt-and-suspenders) with NaN
    eng = engineered_names()
    out[eng] = out[eng].replace([np.inf, -np.inf], np.nan)
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer wrapper so engineering lives inside the fitted
    Pipeline and is applied identically at inference."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn signature
        return self

    def transform(self, X):  # noqa: N803
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame.")
        return add_engineered_features(X)

    def get_feature_names_out(self, input_features=None):  # pragma: no cover
        base = list(input_features) if input_features is not None else []
        return np.array(base + engineered_names())
