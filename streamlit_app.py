"""Loan Default Risk — Streamlit app entry point.

Run locally:
    streamlit run streamlit_app.py

This replaces the former React frontend + FastAPI backend. The UI talks to the
model directly through loan_risk.serving.ModelService (loaded once, cached).
Outputs are analytical estimates, NOT autonomous lending decisions.
"""
from __future__ import annotations

import streamlit as st

from ui.helpers import get_service, inject_base_styles

st.set_page_config(
    page_title="Loan Default Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _render_chrome() -> None:
    """Persistent header + status pill shown on every page (parity with the old
    Layout header: title, subtitle, model-ready/not-ready pill, synthetic chip)."""
    inject_base_styles()
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
# Pages are grouped into sidebar sections via the st.navigation({section: [...]})
# mapping so the 13 pages read as four coherent groups, not one long list.
from ui.pages import (  # noqa: E402
    analytics,
    assessment,
    dataset_quality,
    explainability,
    fairness,
    feature_reference,
    methodology,
    model_info,
    model_selection,
    overview,
    performance,
    responsible_use,
    result,
)

nav = st.navigation({
    "Overview": [
        st.Page(overview.render, title="Overview", url_path="overview",
                icon="🏠", default=True),
        st.Page(analytics.render, title="Analytics Dashboard",
                url_path="analytics", icon="📈"),
    ],
    "Assessment": [
        st.Page(assessment.render, title="Risk Assessment",
                url_path="assessment", icon="📝"),
        st.Page(result.render, title="Prediction Result",
                url_path="result", icon="🧾"),
        st.Page(explainability.render, title="Explainability",
                url_path="explainability", icon="🔍"),
    ],
    "Model & Methodology": [
        st.Page(model_selection.render, title="Model Selection",
                url_path="model-selection", icon="🏆"),
        st.Page(performance.render, title="Model Performance",
                url_path="performance", icon="📊"),
        st.Page(methodology.render, title="Methodology & Formulas",
                url_path="methodology", icon="🧮"),
        st.Page(feature_reference.render, title="Feature Reference",
                url_path="features", icon="📖"),
        st.Page(model_info.render, title="Model Information",
                url_path="model", icon="ℹ️"),
    ],
    "Governance": [
        st.Page(fairness.render, title="Fairness Diagnostics",
                url_path="fairness", icon="⚖️"),
        st.Page(dataset_quality.render, title="Dataset Quality",
                url_path="dataset", icon="✅"),
        st.Page(responsible_use.render, title="Responsible Use",
                url_path="responsible-use", icon="🛡️"),
    ],
})
nav.run()

