"""Explainability page — model-level feature importance (global) and a
per-applicant explanation for the most recent assessment (local).

Global importance comes from the service's metrics.json; the local section reads
the last prediction stored in session_state. Both are rendered exactly as the
service returned them — associations with predicted risk, never causal claims.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ui.helpers import (
    get_service,
    num,
    page_header,
    pct,
    risk_band_badge,
    synthetic_warning,
    TRAIN_HINT,
)
from ui.state import get_prediction


def render() -> None:
    service = get_service()
    page_header(
        "Explainability",
        "How the model relates features to predicted default risk — at the "
        "model level and for your most recent applicant.",
    )

    _global_importance(service)
    _applicant_explanation()


def _global_importance(service: Any) -> None:
    st.subheader("Global feature importance")
    st.caption("Model-level drivers across the evaluation data")

    metrics = service.get_metrics() if service.available else None
    if metrics is None:
        # No model/report — skip Section A but let Section B render its own state.
        st.warning("**Model or data not ready** — " + TRAIN_HINT)
        return

    synthetic_warning(
        metrics.get("synthetic", False),
        metrics.get("synthetic_warning"),
        title="Synthetic data",
        fallback="These importances were computed on synthetic data and do not "
        "reflect a real lending population.",
    )

    gi: dict[str, Any] = metrics.get("global_importance", {})
    st.markdown(f"Method: `{gi.get('method')}`")
    if gi.get("interpretation"):
        st.markdown(gi["interpretation"])
    if gi.get("note"):
        st.markdown(gi["note"])

    features = gi.get("features")
    if features:
        # Horizontal bar chart: importance by feature (largest reads at a glance).
        chart_df = pd.DataFrame(
            {"importance": [f.get("importance") for f in features]},
            index=[f.get("feature") for f in features],
        )
        st.bar_chart(chart_df, horizontal=True)

        # Numeric companion table (importance to 4dp; include std when present).
        has_std = any(f.get("std") is not None for f in features)
        rows: list[dict[str, Any]] = []
        for f in features:
            row = {"Feature": f.get("feature"), "Importance": num(f.get("importance"), 4)}
            if has_std:
                row["Std"] = num(f.get("std"), 4)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No importance values are available.")

    st.caption(
        "These values show association with predicted risk, NOT causation. A "
        "high-importance feature is one the model relies on, not a proven cause "
        "of default."
    )


def _applicant_explanation() -> None:
    st.subheader("Applicant-level explanation")

    pred = get_prediction()
    if pred is None:
        st.info(
            "**No applicant assessed yet** — Submit an applicant on the Risk "
            "Assessment page to see which features drove that specific estimate."
        )
        return

    result: dict[str, Any] = pred["result"]
    expl = result.get("explanation")

    st.markdown(
        f"Estimated default probability **{pct(result['default_probability'], 1)}** "
        f"· model version {result['model_version']}."
    )
    st.markdown(risk_band_badge(result["risk_band"]), unsafe_allow_html=True)

    if not result["explanation_available"]:
        st.info(
            "**No per-applicant explanation** — This model does not produce "
            "per-applicant feature contributions. See the global feature "
            "importance above for model-level drivers."
        )
    elif expl and expl.get("contributions"):
        st.markdown(f"Method: `{expl.get('method')}`")
        if expl.get("interpretation"):
            st.markdown(expl["interpretation"])
        if expl.get("note"):
            st.markdown(expl["note"])

        # Signed contributions: positive increases risk, negative decreases it.
        rows = [{
            "Feature": c.get("feature"),
            "Contribution": f"{float(c.get('contribution', 0.0)):+.4f}",
            "Direction": c.get("direction"),
        } for c in expl["contributions"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Positive contributions increase estimated risk; negative decrease it.")
    else:
        # explanation_available was True but contributions came back empty.
        st.info(
            "**No contributions returned** — An explanation was expected for "
            "this applicant but no feature contributions were returned."
        )

    # Always shown: these attributions are not a lending justification.
    st.warning(
        "**Approximate — not adverse-action reasons** — Feature contributions "
        "are approximate attributions for a single estimate. They must not be "
        "used as adverse-action reasons or as a substitute for a documented, "
        "human lending decision."
    )
