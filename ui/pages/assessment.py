"""Risk Assessment page — the applicant form.

Widgets are generated from the service's feature schema (never hardcoded) so the
form, the input validation and the ML feature contract cannot drift apart.
Numeric bounds/examples and categorical options all come from get_feature_schema();
validate_payload() is the authoritative final check before predict().

Optional fields (external scores, occupation, and a few financials) may be left
blank — omitting them reproduces the exact missingness state the pipeline imputes
on the training side. On a successful assessment the result is stored in session
state for the Prediction Result and Explainability pages.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from loan_risk.serving import OPTIONAL_FIELDS, validate_payload
from ui.helpers import (
    get_service,
    page_header,
    pct,
    require_model,
    risk_band_badge,
    stat_cards,
    synthetic_warning,
)
from ui.state import save_prediction


def render() -> None:
    service = get_service()
    require_model(service)  # standard "model or data not ready" gate
    schema = service.get_feature_schema() or {}
    mi = service.model_info()

    page_header(
        "Loan Risk Assessment",
        "Enter applicant details to estimate default probability. Optional fields "
        "(external scores, occupation) may be left blank and will be imputed.",
    )

    synthetic_warning(
        mi.get("synthetic", False),
        mi.get("synthetic_warning"),
        title="Synthetic data",
        fallback="This model was trained on synthetic data. Results are for "
        "demonstration only and must not inform real lending.",
    )

    numeric = schema.get("numeric_features") or []
    categorical = schema.get("categorical_features") or []
    # Partition numerics by name so the form groups them like the old UI.
    financials = [f for f in numeric if str(f.get("name", "")).startswith("AMT_")]
    external = [f for f in numeric if str(f.get("name", "")).startswith("EXT_")]
    profile_num = [f for f in numeric if f not in financials and f not in external]

    values: dict[str, Any] = {}
    with st.form("assessment_form"):
        st.subheader("Financials")
        for f in financials:
            values[f["name"]] = _numeric_input(f)

        st.subheader("Applicant profile")
        for f in profile_num:
            values[f["name"]] = _numeric_input(f)
        for f in categorical:
            values[f["name"]] = _categorical_input(f)

        st.subheader("External credit scores")
        st.caption("Optional — omit any you don't have; each is imputed with a "
                   "missingness flag, matching the training data.")
        for f in external:
            values[f["name"]] = _numeric_input(f)

        submitted = st.form_submit_button("Assess risk")

    st.info("This produces an analytical estimate, not a lending decision.")

    if submitted:
        _handle_submit(service, values)


def _numeric_input(f: dict[str, Any]) -> float | None:
    """A number_input for one numeric feature. Optional features start empty
    (value=None) so an omitted field stays omitted in the payload."""
    name = f["name"]
    optional = name in OPTIONAL_FIELDS
    label = (f.get("label") or name) + (" (optional)" if optional else "")

    lo, hi = f.get("min"), f.get("max")
    kwargs: dict[str, Any] = {}
    if lo is not None:
        kwargs["min_value"] = float(lo)
    if hi is not None:
        kwargs["max_value"] = float(hi)

    if optional:
        default: float | None = None
    else:
        ex = f.get("example")
        default = float(ex) if ex is not None else (float(lo) if lo is not None else 0.0)

    help_text = f.get("meaning") or ""
    if lo is not None and hi is not None:
        help_text = f"{help_text} — range {lo}–{hi} {f.get('unit', '')}".strip()

    return st.number_input(label, value=default, help=help_text or None, **kwargs)


def _categorical_input(f: dict[str, Any]) -> str | None:
    """A selectbox for one categorical feature. Optional ones get a leading '—'
    blank option that maps back to an omitted (None) value."""
    name = f["name"]
    optional = name in OPTIONAL_FIELDS
    label = (f.get("label") or name) + (" (optional)" if optional else "")
    cats = [str(c) for c in (f.get("categories") or [])]
    help_text = f.get("meaning") or None

    if optional:
        choice = st.selectbox(label, ["—"] + cats, index=0, help=help_text)
        return None if choice == "—" else choice

    ex = f.get("example")
    idx = cats.index(str(ex)) if ex is not None and str(ex) in cats else 0
    return st.selectbox(label, cats, index=idx, help=help_text)


def _handle_submit(service: Any, values: dict[str, Any]) -> None:
    # Drop omitted optionals (None / blank sentinel) so the pipeline's imputation
    # + missingness flags kick in exactly as they do in training.
    payload = {
        name: val for name, val in values.items()
        if val is not None and not (isinstance(val, str) and val == "—")
    }

    normalized, errors = validate_payload(payload)
    if errors:
        st.error("Please fix the following before assessing:")
        for e in errors:
            st.error(f"• **{e['field']}**: {e['message']}")
        return

    result = service.predict(normalized)
    save_prediction(normalized, result)
    st.success(
        "Assessment complete — see the summary below or open **Prediction "
        "Result** in the sidebar for the full breakdown."
    )
    stat_cards([
        ("Default probability", pct(result["default_probability"], 1), None),
        ("Risk score", f"{result['risk_score']}/100", None),
        ("Risk band", result["risk_band"], None),
    ])
    st.markdown(risk_band_badge(result["risk_band"]), unsafe_allow_html=True)
