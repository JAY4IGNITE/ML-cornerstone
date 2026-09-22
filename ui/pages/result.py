"""Prediction Result page — the most recent assessment.

Reads the last prediction from session state (set by the Assessment page) and
renders it. Everything shown — probability, score, band, limitations, disclaimer
— comes straight from the service's predict() output; the disclaimer and
limitation text are surfaced verbatim, never rewritten in the UI.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui.helpers import (
    get_service,
    page_header,
    pct,
    risk_band_badge,
    stat_cards,
    synthetic_warning,
)
from ui.state import get_prediction


def render() -> None:
    service = get_service()
    mi = service.model_info() if service.available else {}
    pred = get_prediction()

    page_header("Prediction Result")

    if pred is None:
        # Empty state — nothing assessed in this session yet.
        st.info(
            "**No assessment yet** — Submit an applicant on the Risk Assessment "
            "page to see a result here."
        )
        return

    result: dict[str, Any] = pred["result"]

    st.markdown(risk_band_badge(result["risk_band"]), unsafe_allow_html=True)

    synthetic_warning(
        mi.get("synthetic", False),
        mi.get("synthetic_warning"),
        title="Synthetic data",
        fallback="This model was trained on synthetic data. Results are for "
        "demonstration only and must not inform real lending.",
    )

    prob = result["default_probability"]
    stat_cards([
        ("Default probability", pct(prob, 1), "Estimated P(default)"),
        ("Risk score", f"{result['risk_score']}/100", "round(probability × 100)"),
        ("Risk band", result["risk_band"], "Configured probability band"),
    ])

    # Probability bar (value clamped to [0,1] for the widget; label shows the pct).
    st.subheader("Default probability")
    st.progress(min(max(float(prob), 0.0), 1.0), text=pct(prob, 1))
    st.caption(f"Model version {result['model_version']}.")
    if result.get("explanation_available"):
        st.markdown(
            "A per-applicant explanation is available — see the **Explainability** page."
        )
    else:
        st.markdown("No per-applicant explanation is available for this model.")

    # Limitations — verbatim from the service.
    st.subheader("Limitations")
    for lim in result.get("limitations", []):
        st.markdown(f"- {lim}")

    # Disclaimer — API-supplied text, shown verbatim (not rewritten in the UI).
    st.warning(f"**Not a lending decision** — {result.get('disclaimer', '')}")

    st.caption("Start a new assessment from the **Risk Assessment** page in the sidebar.")
