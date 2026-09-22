"""Loan Default Risk — Streamlit app entry point.

Run locally:
    streamlit run streamlit_app.py

This replaces the former React frontend + FastAPI backend. The UI talks to the
model directly through loan_risk.serving.ModelService (loaded once, cached).
Outputs are analytical estimates, NOT autonomous lending decisions.
"""
from __future__ import annotations

import streamlit as st

from ui.helpers import get_service

st.set_page_config(
    page_title="Loan Default Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _render_chrome() -> None:
    """Persistent header + status pill shown on every page (parity with the old
    Layout header: title, subtitle, model-ready/not-ready pill, synthetic chip)."""
    service = get_service()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("### 📊 Loan Default Risk")
        st.caption("Analytical assessment · not a lending decision")
    with right:
        chips = []
        if service.available and service.metadata.get("synthetic"):
            chips.append(":orange-badge[⚠ Synthetic data]")
        chips.append(
            ":green-badge[● Model ready]" if service.available
            else ":red-badge[● Model not ready]"
        )
        st.markdown(
            "<div style='text-align:right;padding-top:0.6rem'>" + " ".join(chips) + "</div>",
            unsafe_allow_html=True,
        )
    st.divider()


_render_chrome()

# Function-based pages; url_path is set explicitly because every page module's
# entry function is named render() (otherwise their identifiers would collide).
from ui.pages import (  # noqa: E402
    assessment,
    dataset_quality,
    explainability,
    model_info,
    overview,
    performance,
    responsible_use,
    result,
)

nav = st.navigation([
    st.Page(overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(assessment.render, title="Risk Assessment", icon="📝", url_path="assessment"),
    st.Page(result.render, title="Prediction Result", icon="🎯", url_path="result"),
    st.Page(explainability.render, title="Explainability", icon="🔍", url_path="explainability"),
    st.Page(performance.render, title="Model Performance", icon="📈", url_path="performance"),
    st.Page(dataset_quality.render, title="Dataset Quality", icon="🧪", url_path="dataset"),
    st.Page(model_info.render, title="Model Information", icon="ℹ️", url_path="model"),
    st.Page(responsible_use.render, title="Responsible Use", icon="⚖️", url_path="responsible-use"),
])
nav.run()
