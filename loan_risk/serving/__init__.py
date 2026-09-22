"""Serving layer: model loading, single-applicant prediction, input validation.

Framework-agnostic. Consumed by the Streamlit UI (streamlit_app.py). Depends only
on the `loan_risk` package + pandas/pydantic — no web framework.
"""
from .service import ModelService
from .validation import ApplicantInput, OPTIONAL_FIELDS, validate_payload

__all__ = ["ModelService", "ApplicantInput", "OPTIONAL_FIELDS", "validate_payload"]
