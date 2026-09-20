"""Ingestion: load the raw application file and standardize it to the canonical
model frame the rest of the pipeline (and the API) operate on.

The RAW Home Credit file stores age/employment as negative day-counts
(``DAYS_BIRTH``, ``DAYS_EMPLOYED``) relative to the application date, and uses
the documented ``365243`` sentinel in ``DAYS_EMPLOYED`` for pensioners /
unemployed. We convert these to human-friendly ``AGE_YEARS`` /
``EMPLOYMENT_YEARS`` exactly once here, so training data and API inputs share
identical columns (Execution Rule #9: consistent train/inference pipelines).

This same standardization runs for both the synthetic fixture and real Kaggle
data — the synthetic generator emits the raw column layout on purpose.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..schema import categorical_names, numeric_names

# Columns expected in the raw application file (subset we use). Column names
# match the public Home Credit schema exactly; we never invent columns.
RAW_ID = "SK_ID_CURR"
RAW_TARGET = "TARGET"

RAW_NUMERIC_DIRECT = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CNT_FAM_MEMBERS", "CNT_CHILDREN", "EXT_SOURCE_1", "EXT_SOURCE_2",
    "EXT_SOURCE_3", "REGION_POPULATION_RELATIVE",
]
RAW_DAYS = ["DAYS_BIRTH", "DAYS_EMPLOYED"]  # converted to AGE_YEARS / EMPLOYMENT_YEARS
RAW_CATEGORICAL = list(categorical_names())

# Full ordered list of raw columns (excluding target/id) validation checks for.
RAW_FEATURE_COLUMNS = RAW_NUMERIC_DIRECT + RAW_DAYS + RAW_CATEGORICAL
DAYS_EMPLOYED_SENTINEL = 365243
# Added during standardization to flag the pensioner/unemployed sentinel.
EMPLOYMENT_MISSING_FLAG = "DAYS_EMPLOYED_MISSING"
# External-score columns whose missingness is itself informative on the real
# Home Credit data (EXT_SOURCE_1/3 are absent for a large share of rows). We
# create an explicit 0/1 indicator per source at standardization AND at serving
# so the model can distinguish "score missing" from "score equals the median".
EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
EXT_SOURCE_MISSING_FLAGS = {c: f"{c}_MISSING" for c in EXT_SOURCE_COLUMNS}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a raw application CSV as strings-then-typed (no silent coercion)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Application file not found: {path}. Generate the synthetic fixture "
            f"(`python -m loan_risk.data.synthetic`) or download real data "
            f"(`python -m loan_risk.data.download`)."
        )
    return pd.read_csv(path)


def resolve_source_path(cfg: Config) -> Path:
    """Pick the raw file to use based on config `dataset.source`."""
    ds = cfg["dataset"]
    if ds["source"] == "real":
        return cfg.path("data_raw") / ds["primary_file"]
    if ds["source"] == "synthetic":
        return cfg.path("data_synthetic") / ds["synthetic_file"]
    raise ValueError(f"Unknown dataset.source: {ds['source']!r}")


def standardize(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Map raw columns to the canonical model frame.

    Produces exactly: id, target (if present), numeric_names(), categorical_names(),
    plus the EMPLOYMENT_MISSING_FLAG indicator. Raw DAYS_* are dropped after
    conversion. Does not impute or encode — that is the preprocessing stage.
    """
    sentinel = (
        int(cfg["preprocessing"]["days_employed_anomaly"])
        if cfg is not None else DAYS_EMPLOYED_SENTINEL
    )
    out = pd.DataFrame(index=df.index)

    if RAW_ID in df.columns:
        out[RAW_ID] = df[RAW_ID].values
    if RAW_TARGET in df.columns:
        out[RAW_TARGET] = df[RAW_TARGET].values

    # direct numeric passthrough
    for col in RAW_NUMERIC_DIRECT:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    # external-score missingness indicators (computed from the coerced values,
    # so a non-numeric/absent EXT_SOURCE becomes a flagged missing value)
    for col, flag in EXT_SOURCE_MISSING_FLAGS.items():
        out[flag] = out[col].isna().astype(int)

    # DAYS_BIRTH (negative) -> AGE_YEARS (positive)
    out["AGE_YEARS"] = (-pd.to_numeric(df["DAYS_BIRTH"], errors="coerce")) / 365.25

    # DAYS_EMPLOYED (negative) -> EMPLOYMENT_YEARS; 365243 sentinel OR genuine
    # NaN -> NaN + flag. We flag BOTH the pensioner/unemployed sentinel and a
    # truly-missing value (mirroring the EXT_SOURCE .isna() flags above), so the
    # model can always distinguish "employment unknown" from a real, later-imputed
    # value — and so training matches serving, where an omitted EMPLOYMENT_YEARS
    # also sets the flag.
    days_emp = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce")
    is_missing = (days_emp == sentinel) | days_emp.isna()
    out[EMPLOYMENT_MISSING_FLAG] = is_missing.astype(int)
    emp_years = (-days_emp) / 365.25
    emp_years = emp_years.where(~is_missing, other=np.nan)
    out["EMPLOYMENT_YEARS"] = emp_years

    # categoricals passthrough as string (NaN preserved)
    for col in RAW_CATEGORICAL:
        out[col] = df[col].astype("object")

    # column order: keep numeric schema order, then the missingness indicators,
    # then categoricals
    ordered = [c for c in (RAW_ID, RAW_TARGET) if c in out.columns]
    ordered += (
        numeric_names()
        + [EMPLOYMENT_MISSING_FLAG]
        + list(EXT_SOURCE_MISSING_FLAGS.values())
        + categorical_names()
    )
    return out[ordered]


def load_standardized(cfg: Config) -> pd.DataFrame:
    """Load the configured raw source and standardize in one call."""
    raw = load_raw(resolve_source_path(cfg))
    return standardize(raw, cfg)
