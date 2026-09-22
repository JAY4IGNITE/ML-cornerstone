"""Overview page — the dashboard landing.

A headline summary of the deployed model (name, version, a couple of test-set
KPIs) plus signposts to the other pages. Degrades gracefully when the model or
its evaluation report is absent: it shows a "not ready" note and "—" tiles
rather than halting, so the landing page always renders.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui.helpers import (
    get_service,
    metric_value,
    page_header,
    pct,
    stat_cards,
    synthetic_warning,
)

# Signposts to the other pages (title, one-line description). Rendered as a grid
# of bordered cards; kept as data so the layout stays a simple loop.
_EXPLORE = [
    ("Risk Assessment", "Enter applicant details to estimate default probability."),
    ("Model Performance", "Validation comparison, test metrics, calibration and thresholds."),
    ("Explainability", "Global drivers and per-applicant contribution breakdowns."),
    ("Dataset Quality", "Validation checks and the dataset manifest with known limits."),
    ("Model Information", "Selected model, version, training details and calibration."),
    ("Responsible Use", "Scope, limitations and appropriate use of these estimates."),
]


def render() -> None:
    service = get_service()
    mi = service.model_info() if service.available else {}
    metrics = service.get_metrics() if service.available else None

    page_header(
        "Loan Default-Risk Dashboard",
        "An analytical tool for estimating and understanding loan default risk. "
        "Outputs are probability estimates for review, not automated lending "
        "decisions.",
    )

    # Status line — Overview never halts; it just reflects readiness.
    if service.available:
        st.caption(":green-badge[● Model ready]")
    else:
        st.caption(":red-badge[● Model not ready]")

    synthetic_warning(
        mi.get("synthetic", False),
        mi.get("synthetic_warning"),
        title="Synthetic data",
        fallback="This model was trained on synthetic data. Results are for "
        "demonstration only and must not inform real lending.",
    )

    _kpis(mi, metrics)

    st.subheader("Explore")
    # Two rows of three bordered cards.
    for start in range(0, len(_EXPLORE), 3):
        cols = st.columns(3)
        for col, (title, desc) in zip(cols, _EXPLORE[start:start + 3]):
            with col, st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)


def _kpis(mi: dict[str, Any], metrics: dict[str, Any] | None) -> None:
    """Headline tiles. When there is no evaluation report the metric tiles show
    "—" with a caption explaining how to populate them, rather than fabricating."""
    test = metrics.get("final_test_metrics", {}) if metrics else {}
    stat_cards([
        ("Selected model", mi.get("model_name", "—"), "Best model by selection metric"),
        ("Model version", mi.get("model_version", "—"),
         f"Schema {mi.get('feature_schema_version', '—')}"),
        ("Test ROC-AUC", metric_value(test.get("roc_auc")) if metrics else "—",
         "Held-out test set" if metrics else "Train the pipeline to populate"),
        ("Positive rate", pct(test.get("positive_rate")) if metrics else "—",
         "Default rate in test set" if metrics else "Not available yet"),
    ])
