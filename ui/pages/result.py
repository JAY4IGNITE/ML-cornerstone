"""Prediction Result page — the most recent assessment.

Presents the comprehensive evaluation for the applicant:
  * Analyst review flag, without automated loan decisions.
  * Machine Learning Classification Target (Below vs at/above the model threshold).
  * Calibrated Risk Score (0-100) and probability vs recorded decision threshold.
  * Direct summary of key positive and negative SHAP risk drivers.
  * Cross-model consensus comparison across all candidate ML models.
"""
from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from ui import charts
from ui.helpers import (
    get_service,
    page_header,
    pct,
    require_model,
    risk_band_badge,
    stat_cards,
    synthetic_warning,
)
from ui.pages.assessment import PRESETS, reset_assessment
from ui.state import get_prediction, save_prediction


def render() -> None:
    service = get_service()
    mi = service.model_info() if service.available else {}
    pred = get_prediction()

    page_header(
        "Prediction Result & Review Report",
        "Detailed analytical breakdown, ML model classification, and risk factor attribution.",
    )

    if pred is None:
        require_model(service)
        st.info("ℹ️ **No assessment in this session yet.** Click a profile below to evaluate instantly:")
        cols = st.columns(3)
        for col, (pkey, pval) in zip(cols, PRESETS.items()):
            with col:
                if st.button(pval["title"], key=f"res_btn_{pkey}", use_container_width=True):
                    res = service.predict(pval["data"])
                    save_prediction(pval["data"], res)
                    st.rerun()
                st.caption(pval["desc"])
        return

    result: dict[str, Any] = pred["result"]
    prob = float(result["default_probability"])
    score = int(result["risk_score"])
    band = str(result["risk_band"])
    target = int(result.get("predicted_target", 0))
    target_lbl = str(result.get("target_label", "Below model threshold" if target == 0 else "At or above model threshold"))
    dec = str(result.get("decision", "Assessment Complete"))
    dec_color = str(result.get("decision_color", "#059669"))
    dec_action = str(result.get("decision_action", ""))
    thr = float(result.get("decision_threshold", 0.166))
    m_name = str(result.get("model_name", "Primary Model"))

    synthetic_warning(
        result.get("synthetic", mi.get("synthetic", False)),
        result.get("synthetic_warning", mi.get("synthetic_warning")),
        title="Synthetic data",
        fallback="This model was trained on synthetic data. Results are for demonstration only.",
    )

    # 1. Headline Lending Decision Banner
    st.markdown(
        f"""
        <div style="background:{dec_color}18;border:2px solid {dec_color};border-radius:12px;padding:22px 26px;margin:12px 0 20px;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                <div>
                    <span style="font-size:0.85rem;font-weight:700;text-transform:uppercase;color:{dec_color};letter-spacing:0.06em;">
                        Analyst Review Indicator · {m_name.split('(')[0].strip()}
                    </span>
                    <h1 style="margin:4px 0 8px;color:{dec_color};font-size:2.1rem;font-weight:850;">
                        {dec}
                    </h1>
                    <p style="margin:0;font-size:1.05rem;color:#1e293b;max-width:700px;font-weight:500;">
                        {dec_action}
                    </p>
                </div>
                <div style="background:white;border:1px solid #cbd5e1;border-radius:10px;padding:12px 20px;text-align:center;min-width:170px;">
                    <div style="font-size:0.75rem;font-weight:700;color:#64748b;letter-spacing:0.05em;text-transform:uppercase;">ML Binary Output</div>
                    <div style="font-size:1.8rem;font-weight:900;color:{dec_color};margin:2px 0;">TARGET = {target}</div>
                    <div style="font-size:0.8rem;font-weight:600;color:#475569;">{target_lbl}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Key Metrics Row
    stat_cards([
        ("Default Probability", pct(prob, 2), f"Decision Boundary: {pct(thr, 1)}"),
        ("Risk Score", f"{score} / 100", "Rounded probability × 100"),
        ("Risk Band", band, "Probability Interval"),
        ("Evaluated By", m_name.split("(")[0].strip(), f"Model Version {result.get('model_version', '0.1.0')}"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Assessed: {result.get('assessed_at', 'not recorded')} · "
               f"Training source: {result.get('dataset_source', 'not recorded')} · "
               f"Threshold basis: {result.get('threshold_basis', 'not recorded')}")

    # 3. Probability vs Decision Threshold Comparison
    st.subheader("Default Probability vs Decision Threshold")
    st.caption(
        f"Estimated probability is **{pct(prob, 2)}**. This assessment used a "
        f"classification threshold of **{pct(thr, 2)}**. Predictions below this "
        "threshold are class 0; predictions at or above it are class 1. "
        "These model classes are not verified outcomes or loan decisions."
    )

    gauge_df = pd.DataFrame([
        {"Metric": "Applicant Probability", "Value": prob, "Type": "Applicant"},
        {"Metric": "Decision Threshold", "Value": thr, "Type": "Threshold"},
    ])
    gauge_chart = alt.Chart(gauge_df).mark_bar(cornerRadius=4, height=28).encode(
        x=alt.X("Value:Q", title="Probability Scale (0.0 to 1.0)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Metric:N", title=None, sort=["Applicant Probability", "Decision Threshold"]),
        color=alt.Color("Type:N", scale=alt.Scale(domain=["Applicant", "Threshold"], range=[dec_color, "#475569"]), legend=None),
        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Value:Q", format=".2%")],
    )
    charts.show(gauge_chart, height=130)

    st.divider()

    # 4. Top Risk Drivers (Embedded SHAP Summary)
    expl = result.get("explanation") or {}
    contribs = expl.get("contributions") or []
    if contribs:
        st.subheader("Model Explanation")
        st.caption(f"Method: {expl.get('method', 'unavailable')} · Output space: {expl.get('output_space', 'unspecified')}")
        st.caption(expl.get("interpretation", expl.get("note", "")))

        pos_drivers = expl.get("positive_drivers") or [c for c in contribs if c["direction"] == "increases_risk"]
        neg_drivers = expl.get("negative_drivers") or [c for c in contribs if c["direction"] == "decreases_risk"]

        c_neg, c_pos = st.columns(2)
        with c_neg:
            st.markdown("##### 🟢 Top Risk-Reducing Factors")
            if neg_drivers:
                for d in neg_drivers[:4]:
                    st.markdown(
                        f"""
                        <div style="border-left:3px solid #059669;padding:6px 12px;margin-bottom:8px;background:#f0fdf4;border-radius:0 6px 6px 0;">
                            <div style="font-weight:700;color:#166534;font-size:0.95rem;">{d.get('display_label', d.get('feature'))}</div>
                            <div style="font-size:0.85rem;color:#334155;">{d.get('reason', '')}</div>
                            <div style="font-size:0.8rem;color:#059669;font-weight:600;margin-top:2px;">Impact: {float(d.get('contribution', 0.0)):+.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No decreasing contributions available in this explanation.")

        with c_pos:
            st.markdown("##### 🔴 Top Risk-Elevating Factors")
            if pos_drivers:
                for d in pos_drivers[:4]:
                    st.markdown(
                        f"""
                        <div style="border-left:3px solid #dc2626;padding:6px 12px;margin-bottom:8px;background:#fef2f2;border-radius:0 6px 6px 0;">
                            <div style="font-weight:700;color:#991b1b;font-size:0.95rem;">{d.get('display_label', d.get('feature'))}</div>
                            <div style="font-size:0.85rem;color:#334155;">{d.get('reason', '')}</div>
                            <div style="font-size:0.8rem;color:#dc2626;font-weight:600;margin-top:2px;">Impact: {float(d.get('contribution', 0.0)):+.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No increasing contributions available in this explanation.")

        st.caption(
            "Open **Explainability** for the available attribution details and their limitations."
        )

    st.divider()

    # 5. Cross-Model Consensus Comparison Table
    all_preds = result.get("all_model_predictions") or {}
    if all_preds:
        st.subheader("Cross-Model Multi-Model Comparison")
        st.caption(
            f"Predictions from {len(all_preds)} available model(s). Agreement is not independent validation of an applicant's outcome."
        )

        rows = []
        for mkey, mres in all_preds.items():
            m_dec = mres.get("decision", {})
            rows.append({
                "ML Model": mres.get("display_name", mkey),
                "P(Default)": pct(mres.get("default_probability", 0), 2),
                "Risk Score": f"{mres.get('risk_score', 0)} / 100",
                "Risk Band": mres.get("risk_band", "—"),
                "Predicted Class": f"TARGET = {m_dec.get('predicted_target', 0)}",
                "Review Indicator": m_dec.get("recommendation", "—"),
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # 6. Action Links
    st.markdown("<br>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("📝 Assess Another Applicant", use_container_width=True):
            reset_assessment()
            from ui.pages.assessment import render as render_assessment
            st.switch_page(st.Page(render_assessment, title="Risk Assessment", url_path="assessment"))
    with c_btn2:
        st.markdown(
            "<div style='text-align:right;padding-top:8px;'>"
            "<a href='explainability' target='_self' style='font-size:1.05rem;font-weight:700;text-decoration:none;'>View Explanation Details &rarr;</a>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.info(result.get("disclaimer", "Model estimates require human review."))
    for limitation in result.get("limitations", []):
        st.caption(limitation)

