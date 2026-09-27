"""Methodology & Formulas page — every metric the dashboard uses, defined once.

This page is pure reference: it *describes* the formulas behind the numbers
shown elsewhere and never computes on data. Metric definitions come from the
shared ``ui.formulas`` registry so a term reads identically on every page; the
risk-score bands and engineered-feature formulas are read from configuration and
the model's own feature schema, so nothing here can drift from what the pipeline
actually does.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui.formulas import CONFUSION_LEGEND, formula, render_metric_definitions
from ui.helpers import get_service, page_header


def render() -> None:
    service = get_service()
    page_header(
        "Methodology & Formulas",
        "The exact definitions behind every metric, score and engineered feature "
        "in this dashboard — so each number is reproducible, not a black box.",
    )

    tabs = st.tabs([
        "Ranking & thresholds",
        "Probability & calibration",
        "Selection & training",
        "Risk score",
        "Engineered features",
        "Fairness",
    ])
    with tabs[0]:
        _ranking()
    with tabs[1]:
        _calibration()
    with tabs[2]:
        _selection()
    with tabs[3]:
        _risk_score(service)
    with tabs[4]:
        _engineered(service)
    with tabs[5]:
        _fairness()


def _ranking() -> None:
    st.markdown(
        "Threshold-free ranking quality first, then the confusion-matrix rates "
        "that depend on the operating cut-off."
    )
    render_metric_definitions(["roc_auc", "pr_auc"])
    st.divider()
    render_metric_definitions(["precision", "recall", "f1", "accuracy", "fpr", "fnr"])
    st.caption(CONFUSION_LEGEND)


def _calibration() -> None:
    st.markdown(
        "Ranking tells you the *order* of risk; calibration asks whether a "
        "predicted 20% actually defaults 20% of the time. We measure it with the "
        "Brier score and correct it with isotonic regression fit by cross-validation."
    )
    render_metric_definitions(["brier"])
    st.divider()
    st.markdown("**Reliability (calibration) curve**")
    formula(
        r"\text{observed}(b) = \frac{1}{|b|}\sum_{i \in b} y_i "
        r"\quad\text{vs}\quad "
        r"\text{predicted}(b) = \frac{1}{|b|}\sum_{i \in b} \hat p_i",
        "Group predictions into probability bins b; a well-calibrated model sits "
        "on the diagonal where observed frequency equals mean predicted "
        "probability. See the before/after curves on Model Performance.",
    )


def _selection() -> None:
    st.markdown(
        "How the deployed model was picked and how the class imbalance is handled "
        "during training. Full evidence is on the Model Selection page."
    )
    render_metric_definitions(["cv_mean", "scale_pos_weight"])


def _risk_score(service: Any) -> None:
    render_metric_definitions(["risk_score"])
    st.divider()
    st.markdown("**Risk bands** (applied to the probability, from configuration)")
    try:
        bands = service.cfg["risk_score"]["bands"]
    except Exception:  # noqa: BLE001 — config is optional here; degrade gracefully
        bands = []
    if bands:
        import pandas as pd
        rows = [{
            "Band": b.get("name"),
            "Probability range": f"{b.get('min_prob'):.2f} – {b.get('max_prob'):.2f}",
        } for b in bands]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Bands are cut points on the probability and carry no extra information "
        "beyond it — they exist to make the estimate easier to read, not to "
        "replace the underlying probability."
    )


def _engineered(service: Any) -> None:
    st.markdown(
        "Ratios derived once, at both training and serving time, from fields "
        "available at application — so there is no train/serve skew and no "
        "post-outcome leakage."
    )
    schema = service.get_feature_schema() or {}
    engineered = schema.get("engineered_features") or []
    if not engineered:
        st.info("No engineered-feature definitions are recorded.")
        return
    for e in engineered:
        st.markdown(f"**{e.get('name')}**")
        st.code(e.get("formula", ""), language="text")
        reading = e.get("interpretation", "")
        leak = e.get("leakage_assessment", "")
        st.caption(" · ".join(x for x in (reading, leak) if x))


def _fairness() -> None:
    st.markdown(
        "Diagnostic slicing only — raw group rates and simple disparity ratios, "
        "never a fairness certification. Full results are on Fairness Diagnostics."
    )
    render_metric_definitions(["group_rate", "disparity_ratio"])
    st.caption(
        "The informal four-fifths rule flags a disparity ratio below 0.80 for "
        "human review; it is a screening heuristic, not a legal standard."
    )


