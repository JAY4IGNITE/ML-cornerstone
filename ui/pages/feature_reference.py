"""Feature Reference page — every model input, explained.

A complete, honest data dictionary for the deployed model: each numeric and
categorical field an applicant supplies, each ratio the pipeline engineers, and
the missingness indicators the model adds behind the scenes. Everything is read
from the model's own ``feature_schema`` (the machine-readable contract saved at
training) with a fallback to :func:`build_feature_schema`, so this page can
never describe a field the model does not actually use.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui.helpers import get_service, page_header


def _schema(service: Any) -> dict[str, Any]:
    """Prefer the trained model's saved schema; fall back to the source-of-truth
    builder so the reference still renders before a model exists."""
    schema = service.get_feature_schema() if service.available else {}
    if schema:
        return schema
    try:
        from loan_risk.pipeline.artifacts import build_feature_schema
        return build_feature_schema()
    except Exception:  # noqa: BLE001 — reference is best-effort without a model
        return {}


def _match(query: str, *fields: Any) -> bool:
    """Case-insensitive substring filter across a feature's text fields."""
    if not query:
        return True
    q = query.lower()
    return any(q in str(f).lower() for f in fields if f)


def render() -> None:
    service = get_service()
    page_header(
        "Feature Reference",
        "Every input the model uses, what it means, and how missing values are "
        "handled — the complete data dictionary for the deployed pipeline.",
    )

    schema = _schema(service)
    numeric = schema.get("numeric_features") or []
    categorical = schema.get("categorical_features") or []
    engineered = schema.get("engineered_features") or []

    if not (numeric or categorical or engineered):
        st.info("No feature schema is available yet. Train the pipeline to "
                "populate the model's input contract.")
        return

    _summary(numeric, categorical, engineered)
    query = st.text_input("Filter features", placeholder="e.g. income, EXT_SOURCE, ratio")

    tabs = st.tabs([
        f"Numeric ({len(numeric)})",
        f"Categorical ({len(categorical)})",
        f"Engineered ({len(engineered)})",
        "Missingness indicators",
    ])
    with tabs[0]:
        _numeric(numeric, query)
    with tabs[1]:
        _categorical(categorical, query)
    with tabs[2]:
        _engineered(engineered, query)
    with tabs[3]:
        _missingness(query)


def _summary(numeric: list, categorical: list, engineered: list) -> None:
    from ui.helpers import stat_cards
    stat_cards([
        ("Applicant inputs", len(numeric) + len(categorical),
         "Fields supplied per applicant"),
        ("Numeric", len(numeric), "Continuous / count fields"),
        ("Categorical", len(categorical), "One-hot encoded"),
        ("Engineered", len(engineered), "Derived ratios"),
    ])


def _grid(items: list[dict[str, Any]], render_card) -> None:
    """Lay cards out two-per-row; skip an empty (filtered-out) list cleanly."""
    if not items:
        st.caption("No features match the filter.")
        return
    for start in range(0, len(items), 2):
        cols = st.columns(2)
        for col, item in zip(cols, items[start:start + 2]):
            with col, st.container(border=True):
                render_card(item)


def _numeric(features: list[dict[str, Any]], query: str) -> None:
    shown = [f for f in features
             if _match(query, f.get("name"), f.get("label"), f.get("meaning"))]

    def card(f: dict[str, Any]) -> None:
        st.markdown(f"**{f.get('label') or f.get('name')}**  \n`{f.get('name')}`")
        st.caption(f.get("meaning", ""))
        bits = []
        if f.get("unit"):
            bits.append(f"Unit: {f['unit']}")
        if f.get("min") is not None or f.get("max") is not None:
            bits.append(f"Range: {f.get('min')} – {f.get('max')}")
        if f.get("example") is not None:
            bits.append(f"Example: {f['example']}")
        st.caption(" · ".join(bits))
        if f.get("missing_behavior"):
            st.caption(f"Missing → {f['missing_behavior']}")

    _grid(shown, card)


def _categorical(features: list[dict[str, Any]], query: str) -> None:
    shown = [f for f in features
             if _match(query, f.get("name"), f.get("label"), f.get("meaning"),
                       " ".join(map(str, f.get("categories") or [])))]

    def card(f: dict[str, Any]) -> None:
        st.markdown(f"**{f.get('label') or f.get('name')}**  \n`{f.get('name')}`")
        st.caption(f.get("meaning", ""))
        cats = f.get("categories") or []
        if cats:
            st.caption("Values: " + ", ".join(str(c) for c in cats))
        if f.get("missing_behavior"):
            st.caption(f"Missing → {f['missing_behavior']}")

    _grid(shown, card)


def _engineered(features: list[dict[str, Any]], query: str) -> None:
    shown = [f for f in features
             if _match(query, f.get("name"), f.get("formula"),
                       f.get("interpretation"))]

    def card(e: dict[str, Any]) -> None:
        st.markdown(f"**{e.get('name')}**")
        st.code(e.get("formula", ""), language="text")
        if e.get("interpretation"):
            st.caption(e["interpretation"])
        if e.get("leakage_assessment"):
            st.caption(f"Leakage: {e['leakage_assessment']}")

    _grid(shown, card)
    st.caption(
        "Every engineered ratio uses only application-time fields, so it is safe "
        "from leakage and identical at training and serving."
    )


def _missingness(query: str) -> None:
    st.markdown(
        "The model does not just impute missing values — it also records *that* a "
        "value was missing, because missingness is itself informative (e.g. a "
        "pensioner has no employment length, and external scores are often "
        "absent). Each indicator below is added automatically; callers never "
        "supply it."
    )
    # Read the real indicator names from ingestion so this can't drift; fall back
    # to the documented names if that import is unavailable.
    indicators: list[dict[str, str]] = []
    try:
        from loan_risk.data.ingestion import (
            EMPLOYMENT_MISSING_FLAG,
            EXT_SOURCE_MISSING_FLAGS,
        )
        indicators.append({
            "name": EMPLOYMENT_MISSING_FLAG,
            "meaning": "Set when employment length is absent — pensioners and the "
            "unemployed — matching the raw 365243-day sentinel.",
        })
        for src, flag in EXT_SOURCE_MISSING_FLAGS.items():
            indicators.append({
                "name": flag,
                "meaning": f"Set when {src} (an external credit score) was not "
                "provided; these scores are frequently missing in real data.",
            })
    except Exception:  # noqa: BLE001
        indicators = [
            {"name": "DAYS_EMPLOYED_MISSING",
             "meaning": "Set for pensioners/unemployed with no employment length."},
            {"name": "EXT_SOURCE_1_MISSING / _2_MISSING / _3_MISSING",
             "meaning": "Set when an external credit score was not provided."},
        ]

    shown = [i for i in indicators if _match(query, i["name"], i["meaning"])]

    def card(i: dict[str, str]) -> None:
        st.markdown(f"`{i['name']}`")
        st.caption(i["meaning"])

    _grid(shown, card)


