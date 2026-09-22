"""Responsible Use page — scope, limitations and appropriate use.

The model's own responsible_use_note (when present) leads, verbatim. The rest
is fixed guidance that must read identically on every deployment, so it is
static verbatim content rather than anything derived from the model. This page
also renders without a loaded model — the guidance stands on its own.
"""
from __future__ import annotations

import streamlit as st

from ui.helpers import get_service, page_header, synthetic_warning


def render() -> None:
    service = get_service()
    # No require_model gate: the guidance must show even with no model loaded.
    mi = service.model_info() if service.available else {}

    page_header(
        "Responsible Use",
        "Scope, limitations and appropriate use of these risk estimates.",
    )

    synthetic_warning(
        mi.get("synthetic", False),
        mi.get("synthetic_warning"),
        title="Synthetic data",
        fallback="This model was trained on synthetic data. It is a "
        "demonstration and must not inform real lending.",
    )

    # API-supplied note first, verbatim, when the model provides one.
    if mi.get("responsible_use_note"):
        st.info(mi["responsible_use_note"])

    _static_guidance()


def _static_guidance() -> None:
    """Fixed, verbatim responsible-use content — identical on every deploy."""
    st.subheader("What this tool is")
    st.markdown(
        "- An analytical aid that estimates the probability of loan default "
        "from applicant features.\n"
        "- A way to explore how the model relates inputs to predicted risk "
        "(global and per-applicant)."
    )

    st.subheader("What this tool is NOT")
    st.markdown(
        "- Not an automated lending decision system. It does not approve or "
        "deny anyone.\n"
        "- Not a fairness- or compliance-certified system. Subgroup "
        "performance is not guaranteed.\n"
        "- Not a source of adverse-action reasons. Feature contributions are "
        "approximate attributions."
    )

    st.subheader("Appropriate use")
    st.markdown(
        "- Keep a human in the loop for any decision that affects a person.\n"
        "- Treat estimates as one input among many, alongside documented "
        "policy and judgment.\n"
        "- Re-validate on real, representative, current data before any "
        "real-world consideration."
    )

    st.subheader("Limitations")
    st.markdown(
        "- Trained and evaluated on a fixed dataset; performance may not "
        "transfer to other populations.\n"
        "- Metrics describe association, not causation.\n"
        "- Missing values are imputed; individual estimates can be sensitive "
        "to those imputations."
    )

    # Closing amber banner; matches the helper's "**title** — message" style.
    st.warning(
        "**Not a lending decision** — Outputs are analytical estimates for "
        "review, not autonomous lending decisions. Do not use them to take "
        "adverse action against any individual."
    )
