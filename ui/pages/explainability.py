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
    require_model,
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
        "Two complementary views: **native importance** uses model coefficients "
        "or split-impurity reductions (tree importance can favor high-cardinality fields); "
        "**permutation importance** is the actual ROC-AUC lost when the feature "
        "is shuffled (slower, but a direct measure of predictive value)."
    )
    tab_tree, tab_perm = st.tabs(["Native importance", "Permutation importance"])
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
    if not feats:
        st.info("No importance values are available.")
        return
    df = pd.DataFrame([{"Feature": f.get("feature"),
                        "Importance": f.get("importance"),
                        "std": f.get("std")} for f in feats])
    df = df.dropna(subset=["Importance"]).sort_values("Importance", ascending=False)
    if df.empty:
        st.info("No importance values are available.")
        return
    order = df["Feature"].tolist()
    xtitle = "Mean ROC-AUC drop" if kind == "permutation" else "Native feature importance"
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


from ui.pages.assessment import PRESETS
from ui.state import get_prediction, save_prediction


def _applicant_explanation() -> None:
    service = get_service()
    st.subheader("Applicant-level explanation")

    pred = get_prediction()
    if pred is None:
        require_model(service)
        st.info("No applicant assessed yet. Select an illustrative profile to inspect the available explanation:")
        cols = st.columns(3)
        for col, (pkey, pval) in zip(cols, PRESETS.items()):
            with col:
                if st.button(pval["title"], key=f"expl_btn_{pkey}", use_container_width=True):
                    res = service.predict(pval["data"])
                    save_prediction(pval["data"], res)
                    st.rerun()
                st.caption(pval["desc"])
        return

    result: dict[str, Any] = pred["result"]
    expl = result.get("explanation") or {}
    contribs = expl.get("contributions") or []

    # Applicant status bar
    dec_color = result.get("decision_color", "#059669")
    st.markdown(
        f"""
        <div style="background:{dec_color}14;border:1px solid {dec_color};border-radius:8px;padding:12px 18px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
            <div>
                <span style="font-size:0.8rem;font-weight:700;color:{dec_color};text-transform:uppercase;">Evaluated Profile</span>
                <div style="font-size:1.25rem;font-weight:800;color:{dec_color};">{result.get('decision', 'Assessment Complete')} · TARGET = {result.get('predicted_target', 0)}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.85rem;color:#64748b;">Default Probability: <b>{pct(result['default_probability'], 2)}</b> | Risk Score: <b>{result['risk_score']}/100</b></div>
                <div style="font-size:0.8rem;color:#475569;">Model: <b>{result.get('model_name', 'Primary Model')}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not result.get("explanation_available") or not contribs:
        st.info(
            "**No per-applicant explanation** — Local attribution is not available for this model. "
            "See global feature importance above."
        )
        return

    st.markdown(f"**Explanation method**: `{expl.get('method', 'unavailable')}`")
    st.caption(expl.get("interpretation", expl.get("note", "")))
    if expl.get("scope"):
        st.caption(f"Scope: {expl['scope']} · Output space: {expl.get('output_space', 'unspecified')}")
    if not any(c.get("direction") in {"increases_risk", "decreases_risk"} for c in contribs):
        st.info("Only unsigned global importance is available. It does not explain this applicant or establish a direction of effect.")
        st.dataframe(pd.DataFrame([
            {"Feature": c.get("display_label", c.get("feature")),
             "Global importance": c.get("contribution")} for c in contribs
        ]), hide_index=True, use_container_width=True)
        return
    if expl.get("summary_text"):
        st.info(f"💡 **Key Driver Summary**: {expl['summary_text']}")

    # --- Interactive SHAP Contribution Bar Chart ---
    st.markdown("##### Feature Attribution")
    st.caption(
        "Signed contributions describe the explained estimator's output relative to its baseline. "
        "They are associations, not causes, evidence of repayment history, or lending advice. "
        "A base-estimator explanation need not sum to the calibrated ensemble probability."
    )

    plot_data = []
    for c in contribs:
        val = float(c.get("contribution", 0.0))
        label = c.get("display_label", c.get("feature"))
        plot_data.append({
            "Feature": label,
            "Contribution": val,
            "Absolute": abs(val),
            "Impact Type": "Increases Risk (+)" if val > 0 else "Decreases Risk (-)",
            "Reason": c.get("reason", ""),
        })

    df_shap = pd.DataFrame(plot_data).sort_values("Absolute", ascending=False).head(12)

    # Diverging horizontal bar chart
    shap_chart = alt.Chart(df_shap).mark_bar(cornerRadius=3).encode(
        y=alt.Y("Feature:N", sort=df_shap["Feature"].tolist(), title=None),
        x=alt.X("Contribution:Q", title=f"Contribution ({expl.get('output_space', 'model output')})"),
        color=alt.Color(
            "Impact Type:N",
            scale=alt.Scale(
                domain=["Increases Risk (+)", "Decreases Risk (-)"],
                range=["#dc2626", "#059669"],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            alt.Tooltip("Feature:N", title="Factor"),
            alt.Tooltip("Contribution:Q", format="+.4f", title="Contribution"),
            alt.Tooltip("Impact Type:N", title="Direction"),
            alt.Tooltip("Reason:N", title="Rationale"),
        ],
    )
    charts.show(shap_chart, height=max(220, 28 * len(df_shap)))

    # --- Feature Breakdown Table ---
    st.markdown("##### 📋 Detailed Factor-by-Factor Analysis")
    table_rows = [{
        "Factor": c.get("display_label", c.get("feature")),
        "Contribution": f"{float(c.get('contribution', 0.0)):+.4f}",
        "Direction": {"increases_risk": "Increases model output", "decreases_risk": "Decreases model output"}.get(c.get("direction"), "No signed effect"),
        "Interpretation": c.get("reason", ""),
    } for c in contribs]
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    st.caption(
        "Only the largest available contributions are shown. Attribution does not establish "
        "fairness, causality, or the accuracy of an individual prediction."
    )
