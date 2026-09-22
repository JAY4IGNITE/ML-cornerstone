"""Feature schema — the single contract shared by the whole system.

Everything downstream derives from this module:
  * synthetic data generation (columns, types, ranges, categories)
  * data validation (required columns, dtypes, bounds, allowed values)
  * preprocessing (numeric vs categorical routing)
  * feature engineering (engineered feature definitions + leakage notes)
  * serving-layer Pydantic input models (bounds + categorical enums)
  * the Streamlit form fields (labels, units, options)

Scope decision (see project memory / DATA_DICTIONARY.md):
We serve an *application-level* subset of the real Home Credit
`application_train.csv` schema — the fields a single applicant form can
realistically supply at prediction time. Supporting tables (bureau,
previous_application, ...) are the documented "advanced/offline" path and are
NOT required for a single real-time prediction. This keeps train/inference
consistent (Execution Rule #9) and avoids using post-outcome history that
would not be available at decision time (leakage — 04_DATASET_PROTOCOL.md).

Every column name and its meaning matches the public Home Credit schema. When
real data is downloaded, `validate.py` confirms these columns exist and match;
it never invents columns (Execution Rule #5, #13 / Non-Negotiable Principles).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one input column. Mirrors DATA_DICTIONARY.md rows."""

    name: str
    kind: Literal["numeric", "categorical"]
    meaning: str
    unit: str
    source: str  # e.g. "application_train.csv" or "engineered"
    # numeric bounds (inclusive) used for validation + synthetic generation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    # categorical allowed values (the canonical Home Credit categories)
    categories: tuple[str, ...] = field(default_factory=tuple)
    # how missing values are handled at inference
    missing_behavior: str = ""
    # a human-facing label + optional example for the frontend form
    label: str = ""
    example: Optional[float | str] = None

    def __post_init__(self) -> None:
        if self.kind == "categorical" and not self.categories:
            raise ValueError(f"categorical feature {self.name} needs categories")


# ---------------------------------------------------------------------------
# RAW application-level features (subset of application_train.csv)
# ---------------------------------------------------------------------------
# NOTE ON DAYS_* COLUMNS: In the RAW dataset, DAYS_BIRTH / DAYS_EMPLOYED are
# negative day-counts from the application date. Ingestion converts them ONCE to
# the canonical AGE_YEARS / EMPLOYMENT_YEARS columns the model trains on, so the
# API receives those same human-friendly columns directly — no serving-time
# conversion, and train/serve operate on identical columns.

NUMERIC_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="AMT_INCOME_TOTAL", kind="numeric",
        meaning="Total declared income of the client",
        unit="currency", source="application_train.csv",
        min_value=0.0, max_value=5_000_000.0,
        missing_behavior="median imputation (should rarely be missing)",
        label="Annual income", example=180000,
    ),
    FeatureSpec(
        name="AMT_CREDIT", kind="numeric",
        meaning="Credit amount of the loan",
        unit="currency", source="application_train.csv",
        min_value=0.0, max_value=5_000_000.0,
        missing_behavior="median imputation",
        label="Loan credit amount", example=600000,
    ),
    FeatureSpec(
        name="AMT_ANNUITY", kind="numeric",
        meaning="Loan annuity (periodic payment)",
        unit="currency", source="application_train.csv",
        min_value=0.0, max_value=500_000.0,
        missing_behavior="median imputation",
        label="Loan annuity", example=27000,
    ),
    FeatureSpec(
        name="AMT_GOODS_PRICE", kind="numeric",
        meaning="Price of the goods for which the loan is granted",
        unit="currency", source="application_train.csv",
        min_value=0.0, max_value=5_000_000.0,
        missing_behavior="median imputation",
        label="Goods price", example=540000,
    ),
    FeatureSpec(
        name="AGE_YEARS", kind="numeric",
        meaning=("Client age in years. This IS the canonical model column; the "
                 "raw file's DAYS_BIRTH (negative day-count) is converted to it "
                 "once during ingestion, so training and serving share it."),
        unit="years", source="application_train.csv (derived from DAYS_BIRTH)",
        min_value=18.0, max_value=100.0,
        missing_behavior="required; validated on input",
        label="Age", example=35,
    ),
    FeatureSpec(
        name="EMPLOYMENT_YEARS", kind="numeric",
        meaning=("Years employed at current job. Canonical model column derived "
                 "once at ingestion from DAYS_EMPLOYED; the documented 365243 "
                 "sentinel (pensioners/unemployed) becomes missing with a "
                 "DAYS_EMPLOYED_MISSING indicator. Omit at serving for "
                 "pensioners/unemployed -> flag set, value imputed."),
        unit="years", source="application_train.csv (derived from DAYS_EMPLOYED)",
        min_value=0.0, max_value=50.0,
        missing_behavior="median imputation + DAYS_EMPLOYED_MISSING flag",
        label="Years employed", example=5,
    ),
    FeatureSpec(
        name="CNT_FAM_MEMBERS", kind="numeric",
        meaning="Number of family members",
        unit="count", source="application_train.csv",
        min_value=1.0, max_value=20.0,
        missing_behavior="median imputation",
        label="Family members", example=2,
    ),
    FeatureSpec(
        name="CNT_CHILDREN", kind="numeric",
        meaning="Number of children the client has",
        unit="count", source="application_train.csv",
        min_value=0.0, max_value=20.0,
        missing_behavior="median imputation",
        label="Children", example=0,
    ),
    FeatureSpec(
        name="EXT_SOURCE_1", kind="numeric",
        meaning="Normalized score from external data source 1",
        unit="score 0-1", source="application_train.csv",
        min_value=0.0, max_value=1.0,
        missing_behavior="frequently missing; median imputation + indicator",
        label="External score 1", example=0.5,
    ),
    FeatureSpec(
        name="EXT_SOURCE_2", kind="numeric",
        meaning="Normalized score from external data source 2",
        unit="score 0-1", source="application_train.csv",
        min_value=0.0, max_value=1.0,
        missing_behavior="median imputation + indicator",
        label="External score 2", example=0.6,
    ),
    FeatureSpec(
        name="EXT_SOURCE_3", kind="numeric",
        meaning="Normalized score from external data source 3",
        unit="score 0-1", source="application_train.csv",
        min_value=0.0, max_value=1.0,
        missing_behavior="frequently missing; median imputation + indicator",
        label="External score 3", example=0.5,
    ),
    FeatureSpec(
        name="REGION_POPULATION_RELATIVE", kind="numeric",
        meaning="Normalized population of the region where the client lives",
        unit="ratio", source="application_train.csv",
        min_value=0.0, max_value=0.15,
        missing_behavior="median imputation",
        label="Region population (relative)", example=0.02,
    ),
)

CATEGORICAL_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="NAME_CONTRACT_TYPE", kind="categorical",
        meaning="Loan type: cash or revolving",
        unit="category", source="application_train.csv",
        categories=("Cash loans", "Revolving loans"),
        missing_behavior="unknown token",
        label="Contract type", example="Cash loans",
    ),
    FeatureSpec(
        name="CODE_GENDER", kind="categorical",
        meaning="Gender of the client",
        unit="category", source="application_train.csv",
        categories=("M", "F", "XNA"),
        missing_behavior="unknown token",
        label="Gender", example="F",
    ),
    FeatureSpec(
        name="FLAG_OWN_CAR", kind="categorical",
        meaning="Whether the client owns a car",
        unit="category", source="application_train.csv",
        categories=("Y", "N"),
        missing_behavior="unknown token",
        label="Owns a car", example="N",
    ),
    FeatureSpec(
        name="FLAG_OWN_REALTY", kind="categorical",
        meaning="Whether the client owns real estate",
        unit="category", source="application_train.csv",
        categories=("Y", "N"),
        missing_behavior="unknown token",
        label="Owns real estate", example="Y",
    ),
    FeatureSpec(
        name="NAME_INCOME_TYPE", kind="categorical",
        meaning="Client's income source",
        unit="category", source="application_train.csv",
        categories=("Working", "State servant", "Commercial associate",
                    "Pensioner", "Unemployed", "Student", "Businessman",
                    "Maternity leave"),
        missing_behavior="unknown token",
        label="Income type", example="Working",
    ),
    FeatureSpec(
        name="NAME_EDUCATION_TYPE", kind="categorical",
        meaning="Highest level of education achieved",
        unit="category", source="application_train.csv",
        categories=("Lower secondary", "Secondary / secondary special",
                    "Incomplete higher", "Higher education",
                    "Academic degree"),
        missing_behavior="unknown token",
        label="Education", example="Higher education",
    ),
    FeatureSpec(
        name="NAME_FAMILY_STATUS", kind="categorical",
        meaning="Family status of the client",
        unit="category", source="application_train.csv",
        categories=("Single / not married", "Married", "Civil marriage",
                    "Widow", "Separated", "Unknown"),
        missing_behavior="unknown token",
        label="Family status", example="Married",
    ),
    FeatureSpec(
        name="NAME_HOUSING_TYPE", kind="categorical",
        meaning="Housing situation of the client",
        unit="category", source="application_train.csv",
        categories=("House / apartment", "Rented apartment",
                    "With parents", "Municipal apartment",
                    "Office apartment", "Co-op apartment"),
        missing_behavior="unknown token",
        label="Housing type", example="House / apartment",
    ),
    FeatureSpec(
        name="OCCUPATION_TYPE", kind="categorical",
        meaning="Occupation of the client (often missing in real data)",
        unit="category", source="application_train.csv",
        categories=("Laborers", "Core staff", "Accountants", "Managers",
                    "Drivers", "Sales staff", "Cleaning staff",
                    "Cooking staff", "Private service staff",
                    "Medicine staff", "Security staff", "High skill tech staff",
                    "Waiters/barmen staff", "Low-skill Laborers",
                    "Realty agents", "Secretaries", "IT staff", "HR staff"),
        missing_behavior="unknown token (missingness is itself informative)",
        label="Occupation", example="Core staff",
    ),
)


# ---------------------------------------------------------------------------
# ENGINEERED features. Each is point-in-time (uses only application-level
# fields available at prediction time) => no leakage. Formulas are documented
# in DATA_DICTIONARY.md and implemented once in features.py.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineeredFeature:
    name: str
    formula: str
    inputs: tuple[str, ...]
    unit: str
    interpretation: str
    leakage_assessment: str


ENGINEERED_FEATURES: tuple[EngineeredFeature, ...] = (
    EngineeredFeature(
        name="CREDIT_INCOME_RATIO",
        formula="AMT_CREDIT / AMT_INCOME_TOTAL",
        inputs=("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        unit="ratio",
        interpretation="Loan size relative to income; higher = more leverage.",
        leakage_assessment="Safe: both known at application time.",
    ),
    EngineeredFeature(
        name="ANNUITY_INCOME_RATIO",
        formula="AMT_ANNUITY / AMT_INCOME_TOTAL",
        inputs=("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        unit="ratio",
        interpretation="Payment burden relative to income (affordability).",
        leakage_assessment="Safe: both known at application time.",
    ),
    EngineeredFeature(
        name="CREDIT_ANNUITY_RATIO",
        formula="AMT_CREDIT / AMT_ANNUITY",
        inputs=("AMT_CREDIT", "AMT_ANNUITY"),
        unit="ratio (~term)",
        interpretation="Approximate loan term in payment periods.",
        leakage_assessment="Safe: both known at application time.",
    ),
    EngineeredFeature(
        name="CREDIT_GOODS_RATIO",
        formula="AMT_CREDIT / AMT_GOODS_PRICE",
        inputs=("AMT_CREDIT", "AMT_GOODS_PRICE"),
        unit="ratio",
        interpretation="Loan vs goods value; >1 implies borrowing beyond price.",
        leakage_assessment="Safe: both known at application time.",
    ),
    EngineeredFeature(
        name="EMPLOYMENT_AGE_RATIO",
        formula="EMPLOYMENT_YEARS / AGE_YEARS",
        inputs=("EMPLOYMENT_YEARS", "AGE_YEARS"),
        unit="ratio",
        interpretation="Share of life in current employment; stability proxy.",
        leakage_assessment="Safe: derived from applicant attributes.",
    ),
    EngineeredFeature(
        name="EXT_SOURCE_MEAN",
        formula="mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) ignoring NaN",
        inputs=("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"),
        unit="score 0-1",
        interpretation="Aggregate external creditworthiness signal.",
        leakage_assessment="Safe: external scores are inputs, not outcomes.",
    ),
    EngineeredFeature(
        name="INCOME_PER_FAM_MEMBER",
        formula="AMT_INCOME_TOTAL / CNT_FAM_MEMBERS",
        inputs=("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        unit="currency",
        interpretation="Per-capita household income.",
        leakage_assessment="Safe: both known at application time.",
    ),
)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
ALL_RAW_FEATURES: tuple[FeatureSpec, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def numeric_names() -> list[str]:
    return [f.name for f in NUMERIC_FEATURES]


def categorical_names() -> list[str]:
    return [f.name for f in CATEGORICAL_FEATURES]


def engineered_names() -> list[str]:
    return [f.name for f in ENGINEERED_FEATURES]


def raw_input_names() -> list[str]:
    """All raw columns an API caller supplies (order-independent)."""
    return numeric_names() + categorical_names()


def spec_by_name(name: str) -> FeatureSpec:
    for f in ALL_RAW_FEATURES:
        if f.name == name:
            return f
    raise KeyError(name)


def categories_for(name: str) -> tuple[str, ...]:
    return spec_by_name(name).categories
