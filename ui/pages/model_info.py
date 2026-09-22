"""Model Information page — the selected model, its training and calibration.

Read straight from the service's model_info() and feature schema. Nothing is
recomputed or fabricated in the UI; we surface only what the trained artifacts
recorded, so the page stays an honest description of the deployed model.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ui.helpers import (
    get_service,
    num,
    page_header,
    require_model,
    stat_cards,
    synthetic_warning,
)


def render() -> None:
    service = get_service()
    require_model(service)  # standard "model or data not ready" gate
    mi = service.model_info()

    page_header(
        "Model Information",
        "Details of the selected model, its training and calibration.",
    )

    synthetic_warning(
        mi.get("synthetic", False),
        mi.get("synthetic_warning"),
        title="Synthetic data",
        fallback="This model was trained on synthetic data and is for "
        "demonstration only.",
    )

    stat_cards([
        ("Model", mi.get("model_name", "—"), mi.get("model_key")),
        ("Version", mi.get("model_version", "—"),
         f"Schema {mi.get('feature_schema_version', '—')}"),
        ("Selected by", mi.get("selection_metric", "—"), mi.get("selection_basis")),
    ])

    _training(mi)
    _calibration(mi)
    _feature_schema(service.get_feature_schema() or {})
    _responsible_use(mi)


def _training(mi: dict[str, Any]) -> None:
    st.subheader("Training")
    # Field/Value table; "—" for anything the metadata did not record.
    rows = [
        {"Field": "Trained at", "Value": mi.get("trained_at") or "—"},
        {"Field": "Dataset source", "Value": mi.get("dataset_source") or "—"},
        {"Field": "Manifest reference", "Value": mi.get("manifest_reference") or "—"},
        {"Field": "Feature schema version", "Value": mi.get("feature_schema_version") or "—"},
    ]
    st.table(pd.DataFrame(rows))


def _calibration(mi: dict[str, Any]) -> None:
    st.subheader("Calibration")
    cal = mi.get("calibration") or {}

    left, right = st.columns(2)
    with left:
        st.metric("Brier before", num(cal.get("brier_before"), 4))
    with right:
        st.metric("Brier after", num(cal.get("brier_after"), 4))

    st.caption(
        f"Method: {cal.get('method', '—')} · "
        f"{'Improved' if cal.get('improved') else 'No improvement'} · "
        f"status {cal.get('status', '—')}"
    )
    st.markdown(
        "Calibration adjusts raw scores so predicted probabilities better "
        "match observed frequencies."
    )


def _feature_schema(schema: dict[str, Any]) -> None:
    st.subheader("Feature schema")

    numeric = schema.get("numeric_features") or []
    categorical = schema.get("categorical_features") or []
    engineered = schema.get("engineered_features") or []

    with st.expander("Numeric features"):
        if numeric:
            df = pd.DataFrame([{
                "name": f.get("name"),
                "label": f.get("label"),
                "unit": f.get("unit"),
                "min": f.get("min"),
                "max": f.get("max"),
                "missing_behavior": f.get("missing_behavior"),
            } for f in numeric])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No numeric features are recorded.")

    with st.expander("Categorical features"):
        if categorical:
            df = pd.DataFrame([{
                "name": f.get("name"),
                "label": f.get("label"),
                # Categories arrive as a list; join for a single readable cell.
                "categories": ", ".join(str(c) for c in (f.get("categories") or [])),
                "missing_behavior": f.get("missing_behavior"),
            } for f in categorical])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No categorical features are recorded.")

    with st.expander("Engineered features"):
        if engineered:
            df = pd.DataFrame([{
                "name": f.get("name"),
                "formula": f.get("formula"),
                "unit": f.get("unit"),
                "interpretation": f.get("interpretation"),
                "leakage_assessment": f.get("leakage_assessment"),
            } for f in engineered])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No engineered features are recorded.")


def _responsible_use(mi: dict[str, Any]) -> None:
    # Surface the model's own responsible-use note verbatim; omit when absent.
    note = mi.get("responsible_use_note")
    if note:
        st.subheader("Responsible use")
        st.info(note)
