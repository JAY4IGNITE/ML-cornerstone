"""Applicant input model + validation (single source of truth for the form).

The applicant request model is built DYNAMICALLY from loan_risk.schema so the
input validation, the ML feature schema and the Streamlit form never drift
apart. Numeric bounds and categorical enums come straight from FeatureSpec.

Inputs are the canonical model columns directly: training derives AGE_YEARS /
EMPLOYMENT_YEARS from the raw DAYS_* fields once in ingestion.standardize, and
the caller supplies those human-friendly columns as-is, so train and serve
share one schema. EMPLOYMENT_YEARS is optional — omitting it marks the applicant
as a pensioner/unemployed (the DAYS_EMPLOYED_MISSING indicator is set and the
value imputed), matching the 365243 sentinel handling on the training side.
EXT_SOURCE_* and OCCUPATION_TYPE are also optional (often missing in the real
data); each omitted external score sets its own missingness flag and is imputed.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError, create_model, field_validator

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


def validate_payload(payload: dict[str, Any]) -> tuple[Optional[dict[str, Any]], list[dict[str, str]]]:
    """Validate a raw form payload against ApplicantInput.

    Returns (normalized_fields, errors). On success `errors` is empty and
    normalized_fields is the coerced/validated dict ready for ModelService.predict.
    On failure normalized_fields is None and each error is {field, message}.
    Mirrors the old API validate-input endpoint so the UI can surface
    field-level problems the same way the pipeline expects them.
    """
    try:
        model = ApplicantInput(**payload)
    except ValidationError as exc:
        errors = [
            {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
            for e in exc.errors()
        ]
        return None, errors
    return model.model_dump(), []
