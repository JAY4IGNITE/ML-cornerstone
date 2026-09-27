"""Explainability page — model-level feature importance (global) and a
per-applicant explanation for the most recent assessment (local).

Global importance comes from the service's metrics.json; the local section reads
the last prediction stored in session_state. Both are rendered exactly as the
service returned them — associations with predicted risk, never causal claims.
"""
from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from ui import charts
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

    st.markdown(
        "Two complementary views: **tree importance** is how often the model "
        "splits on a feature (cheap, but biased toward high-cardinality fields); "
        "**permutation importance** is the actual ROC-AUC lost when the feature "
        "is shuffled (slower, but a direct measure of predictive value)."
    )
    tab_tree, tab_perm = st.tabs(["Tree importance", "Permutation importance"])
    with tab_tree:
        _importance_block(metrics.get("global_importance", {}), kind="tree")
    with tab_perm:
        _importance_block(metrics.get("permutation_importance", {}), kind="permutation")

    st.caption(
        "The two rankings need not agree: a feature can be split on often yet "
        "carry little unique signal (or vice-versa). Both show association with "
        "predicted risk, **NOT causation** — a high-importance feature is one the "
        "model relies on, not a proven cause of default."
    )


def _importance_block(block: dict[str, Any], *, kind: str) -> None:
    """Render one importance ranking (tree or permutation) as a sorted bar chart
    plus a numeric table. Permutation bars carry ±std whiskers when available."""
    if not block:
        st.info("This importance view is not available.")
        return
    st.markdown(f"Method: `{block.get('method')}`")
    meta: list[str] = []
    if kind == "tree" and block.get("folds_averaged"):
        meta.append(f"averaged over {block['folds_averaged']} folds")
    if kind == "permutation":
        if block.get("n_repeats") is not None:
            meta.append(f"{block['n_repeats']} shuffles")
        if block.get("n_samples") is not None:
            meta.append(f"{int(block['n_samples']):,}-row sample")
    if meta:
        st.caption(" · ".join(meta))
    if block.get("interpretation"):
        st.markdown(block["interpretation"])

    feats = block.get("features") or []
    df = pd.DataFrame([{"Feature": f.get("feature"),
                        "Importance": f.get("importance"),
                        "std": f.get("std")} for f in feats])
    df = df.dropna(subset=["Importance"]).sort_values("Importance", ascending=False)
    if df.empty:
        st.info("No importance values are available.")
        return
    order = df["Feature"].tolist()
    xtitle = "Mean ROC-AUC drop" if kind == "permutation" else "Tree split importance"
    bars = alt.Chart(df).mark_bar(color=charts.ACCENT).encode(
        x=alt.X("Importance:Q", title=xtitle),
        y=alt.Y("Feature:N", sort=order, title=None),
        tooltip=[alt.Tooltip("Feature:N"),
                 alt.Tooltip("Importance:Q", format=".4f")],
    )
    layers = bars
    if df["std"].notna().any():
        d2 = df.assign(lo=df["Importance"] - df["std"].fillna(0),
                       hi=df["Importance"] + df["std"].fillna(0))
        err = alt.Chart(d2).mark_rule(color=charts.INK, strokeWidth=1.5).encode(
            y=alt.Y("Feature:N", sort=order), x="lo:Q", x2="hi:Q")
        layers = bars + err
    charts.show(layers, height=max(200, 26 * len(df)))

    has_std = df["std"].notna().any()
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        row = {"Feature": r["Feature"], "Importance": num(r["Importance"], 4)}
        if has_std:
            row["± Std"] = num(r["std"], 4)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


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
