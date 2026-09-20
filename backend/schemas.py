"""Pydantic request/response schemas (05_API_CONTRACT.md).

The applicant request model is built DYNAMICALLY from loan_risk.schema so the
API validation, the ML feature schema and the frontend form never drift apart.
Numeric bounds and categorical enums come straight from FeatureSpec.

Inputs are the canonical model columns directly: training derives AGE_YEARS /
EMPLOYMENT_YEARS from the raw DAYS_* fields once in ingestion.standardize, and
the API caller supplies those human-friendly columns as-is, so train and serve
share one schema. EMPLOYMENT_YEARS is optional — omitting it marks the applicant
as a pensioner/unemployed (the DAYS_EMPLOYED_MISSING indicator is set and the
value imputed), matching the 365243 sentinel handling on the training side.
EXT_SOURCE_* and OCCUPATION_TYPE are also optional (often missing in the real
data); each omitted external score sets its own missingness flag and is imputed.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, create_model, field_validator

from loan_risk.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    categories_for,
)

# Fields that may be omitted (heavily missing in real data / imputed downstream).
# EMPLOYMENT_YEARS is optional because pensioners/unemployed have no value on the
# training side either (the 365243 sentinel maps to NaN + DAYS_EMPLOYED_MISSING);
# omitting it at serving reproduces exactly that state.
OPTIONAL_FIELDS = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                   "OCCUPATION_TYPE", "AMT_GOODS_PRICE", "AMT_ANNUITY",
                   "EMPLOYMENT_YEARS"}


def _build_applicant_model() -> type[BaseModel]:
    fields: dict[str, tuple[type, Any]] = {}
    for f in NUMERIC_FEATURES:
        optional = f.name in OPTIONAL_FIELDS
        field_info = Field(
            default=None if optional else ...,
            ge=f.min_value, le=f.max_value,
            description=f"{f.meaning} ({f.unit})",
            json_schema_extra={"example": f.example},
        )
        typ: Any = Optional[float] if optional else float
        fields[f.name] = (typ, field_info)

    for f in CATEGORICAL_FEATURES:
        optional = f.name in OPTIONAL_FIELDS
        field_info = Field(
            default=None if optional else ...,
            description=f"{f.meaning}. Allowed: {list(f.categories)}",
            json_schema_extra={"example": f.example},
        )
        typ = Optional[str] if optional else str
        fields[f.name] = (typ, field_info)

    model = create_model("ApplicantInput", __base__=BaseModel, **fields)

    # attach categorical validators
    def _make_validator(col: str):
        allowed = set(categories_for(col))

        def _v(cls, value):  # noqa: N805
            if value is None:
                return value
            if value not in allowed:
                raise ValueError(
                    f"{col} must be one of {sorted(allowed)}; got {value!r}"
                )
            return value
        return _v

    validators = {}
    for f in CATEGORICAL_FEATURES:
        validators[f"_validate_{f.name}"] = field_validator(f.name)(_make_validator(f.name))
    # rebuild with validators via subclassing
    return type("ApplicantInput", (model,), validators)


ApplicantInput = _build_applicant_model()


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_available: bool
    api_version: str
    detail: str


class CalibrationInfo(BaseModel):
    method: Optional[str] = None
    brier_before: Optional[float] = None
    brier_after: Optional[float] = None
    improved: Optional[bool] = None
    status: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_key: Optional[str] = None
    trained_at: Optional[str] = None
    dataset_source: str
    synthetic: bool
    synthetic_warning: Optional[str] = None
    manifest_reference: Optional[str] = None
    feature_schema_version: str
    calibration: CalibrationInfo
    selection_metric: Optional[str] = None
    responsible_use_note: str

    model_config = {"protected_namespaces": ()}


class ValidationResponse(BaseModel):
    valid: bool
    normalized_fields: Optional[dict[str, Any]] = None
    errors: list[dict[str, Any]] = Field(default_factory=list)


class Contribution(BaseModel):
    feature: str
    contribution: float
    direction: str


class ExplanationBlock(BaseModel):
    method: str
    interpretation: Optional[str] = None
    note: Optional[str] = None
    contributions: list[Contribution] = Field(default_factory=list)


class PredictResponse(BaseModel):
    prediction_id: Optional[str] = None
    default_probability: float = Field(ge=0.0, le=1.0)
    risk_score: int = Field(ge=0, le=100)
    risk_band: str
    model_version: str
    explanation_available: bool
    explanation: Optional[ExplanationBlock] = None
    limitations: list[str]
    disclaimer: str

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """Consistent error schema — no stack traces or internal paths exposed."""
    error: str
    detail: str
    fields: Optional[list[dict[str, Any]]] = None
