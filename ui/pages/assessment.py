"""Risk Assessment page — the applicant form.

Features:
  * Quick-test profile presets (Prime, Standard, Subprime) for instantaneous demo.
  * Model Selector (XGBoost, Random Forest, Decision Tree, Logistic Regression).
  * Feature schema driven input controls with clean labels and hints.
  * Analytical review flag, risk score, probability, and target output.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from loan_risk.serving import OPTIONAL_FIELDS, validate_payload
from ui.helpers import (
    get_service,
    page_header,
    pct,
    require_model,
    risk_band_badge,
    stat_cards,
    synthetic_warning,
)
from ui.state import clear_prediction, save_prediction

PRESETS = {
    "prime": {
        "title": "Higher external scores",
        "desc": "Illustrative inputs: higher income, smaller loan, longer employment.",
        "data": {
            "AMT_INCOME_TOTAL": 250000.0,
            "AMT_CREDIT": 300000.0,
            "AMT_ANNUITY": 15000.0,
            "AMT_GOODS_PRICE": 300000.0,
            "AGE_YEARS": 42.0,
            "EMPLOYMENT_YEARS": 12.0,
            "CNT_FAM_MEMBERS": 2.0,
            "CNT_CHILDREN": 0.0,
            "EXT_SOURCE_1": 0.85,
            "EXT_SOURCE_2": 0.88,
            "EXT_SOURCE_3": 0.82,
            "REGION_POPULATION_RELATIVE": 0.03,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "F",
            "FLAG_OWN_CAR": "Y",
            "FLAG_OWN_REALTY": "Y",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House / apartment",
            "OCCUPATION_TYPE": "Managers",
        },
    },
    "standard": {
        "title": "Mid-range external scores",
        "desc": "Illustrative inputs: mid-range external scores and 3-year employment.",
        "data": {
            "AMT_INCOME_TOTAL": 120000.0,
            "AMT_CREDIT": 450000.0,
            "AMT_ANNUITY": 22000.0,
            "AMT_GOODS_PRICE": 400000.0,
            "AGE_YEARS": 33.0,
            "EMPLOYMENT_YEARS": 3.0,
            "CNT_FAM_MEMBERS": 3.0,
            "CNT_CHILDREN": 1.0,
            "EXT_SOURCE_1": 0.45,
            "EXT_SOURCE_2": 0.50,
            "EXT_SOURCE_3": 0.48,
            "REGION_POPULATION_RELATIVE": 0.02,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "Y",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House / apartment",
            "OCCUPATION_TYPE": "Sales staff",
        },
    },
    "subprime": {
        "title": "Lower external scores",
        "desc": "Illustrative inputs: lower income, larger loan, shorter employment.",
        "data": {
            "AMT_INCOME_TOTAL": 30000.0,
            "AMT_CREDIT": 800000.0,
            "AMT_ANNUITY": 50000.0,
            "AMT_GOODS_PRICE": 750000.0,
            "AGE_YEARS": 22.0,
            "EMPLOYMENT_YEARS": 0.3,
            "CNT_FAM_MEMBERS": 4.0,
            "CNT_CHILDREN": 3.0,
            "EXT_SOURCE_1": 0.05,
            "EXT_SOURCE_2": 0.08,
            "EXT_SOURCE_3": 0.05,
            "REGION_POPULATION_RELATIVE": 0.005,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "N",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Lower secondary",
            "NAME_FAMILY_STATUS": "Single / not married",
            "NAME_HOUSING_TYPE": "With parents",
            "OCCUPATION_TYPE": "Low-skill Laborers",
        },
    },
}


def apply_preset(key: str) -> None:
    """Update widget state before the form renders, including on repeat clicks."""
    data = dict(PRESETS[key]["data"])
    st.session_state["form_preset"] = key
    st.session_state["form_data"] = data
    for name, value in data.items():
        st.session_state[f"inp_{name}"] = value


def reset_assessment() -> None:
    """Discard the previous applicant and every form widget value."""
    clear_prediction()
    for key in list(st.session_state):
        if key.startswith("inp_") or key in {"form_data", "form_preset"}:
            st.session_state.pop(key, None)


def render() -> None:
    service = get_service()
    require_model(service)
    schema = service.get_feature_schema() or {}
    mi = service.model_info()

    page_header(
        "Loan Risk Assessment",
        "Estimate payment-difficulty risk for analyst review. This tool does not approve or decline loans.",
    )

    synthetic_warning(
        mi.get("synthetic", False),
        mi.get("synthetic_warning"),
        title="Synthetic data",
        fallback="This model was trained on synthetic data. Results are for demonstration only.",
    )

    # --- Quick-test preset buttons ---
    st.markdown("##### ⚡ Quick Demonstration Presets")
    st.caption("Click a preset below to instantly populate applicant data for your evaluation:")

    cols = st.columns(3)
    for col, (pkey, pval) in zip(cols, PRESETS.items()):
        with col:
            st.button(pval["title"], key=f"btn_preset_{pkey}", use_container_width=True,
                      on_click=apply_preset, args=(pkey,))
            st.caption(pval["desc"])

    current_data = st.session_state.get("form_data", {})

    st.divider()

    # --- Model Selector ---
    avail_models = service.get_available_models()
    col_m1, col_m2 = st.columns([2, 3])
    with col_m1:
        selected_model = st.selectbox(
            "🧠 Select ML Model for Prediction",
            options=list(avail_models.keys()),
            format_func=lambda k: avail_models.get(k, k),
            index=list(avail_models).index(mi["model_key"]) if mi.get("model_key") in avail_models else 0,
            help="Choose which trained machine learning model evaluates this applicant.",
        )
    with col_m2:
        st.caption(
            "Only models available in this artifact release are listed. "
            "Sample profiles are demonstrations, not verified borrower outcomes."
        )

    numeric = schema.get("numeric_features") or []
    categorical = schema.get("categorical_features") or []
    financials = [f for f in numeric if str(f.get("name", "")).startswith("AMT_")]
    external = [f for f in numeric if str(f.get("name", "")).startswith("EXT_")]
    profile_num = [f for f in numeric if f not in financials and f not in external]

    values: dict[str, Any] = {}
    with st.form("assessment_form"):
        st.subheader("1. Financial Details")
        st.caption("Income, requested loan amount, periodic annuity payment, and goods value.")
        c1, c2 = st.columns(2)
        for i, f in enumerate(financials):
            with (c1 if i % 2 == 0 else c2):
                values[f["name"]] = _numeric_input(f, current_data.get(f["name"]))

        st.subheader("2. Applicant Demographics & Employment")
        c3, c4 = st.columns(2)
        for i, f in enumerate(profile_num):
            with (c3 if i % 2 == 0 else c4):
                values[f["name"]] = _numeric_input(f, current_data.get(f["name"]))

        c5, c6 = st.columns(2)
        for i, f in enumerate(categorical):
            with (c5 if i % 2 == 0 else c6):
                values[f["name"]] = _categorical_input(f, current_data.get(f["name"]))

        st.subheader("3. Normalized External Scores (0.0–1.0)")
        st.caption("Use the dataset's external-source scores only. Their providers and exact definitions are undisclosed; do not substitute a commercial credit score.")
        c7, c8, c9 = st.columns(3)
        for i, f in enumerate(external):
            with [c7, c8, c9][i % 3]:
                values[f["name"]] = _numeric_input(f, current_data.get(f["name"]))

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Run Risk Assessment & Predict", use_container_width=True, type="primary")

    if submitted:
        _handle_submit(service, values, selected_model)


def _numeric_input(f: dict[str, Any], preset_val: Any = None) -> float | None:
    name = f["name"]
    optional = name in OPTIONAL_FIELDS
    label = (f.get("label") or name) + (" (optional)" if optional else "")

    lo, hi = f.get("min"), f.get("max")
    kwargs: dict[str, Any] = {}
    if lo is not None:
        kwargs["min_value"] = float(lo)
    if hi is not None:
        kwargs["max_value"] = float(hi)

    if preset_val is not None:
        default = float(preset_val)
    elif optional:
        default = None
    else:
        ex = f.get("example")
        default = float(ex) if ex is not None else (float(lo) if lo is not None else 0.0)

    help_text = f.get("meaning") or ""
    if lo is not None and hi is not None:
        help_text = f"{help_text} — range {lo}–{hi} {f.get('unit', '')}".strip()

    return st.number_input(label, value=default, help=help_text or None, key=f"inp_{name}", **kwargs)


def _categorical_input(f: dict[str, Any], preset_val: Any = None) -> str | None:
    name = f["name"]
    optional = name in OPTIONAL_FIELDS
    label = (f.get("label") or name) + (" (optional)" if optional else "")
    cats = [str(c) for c in (f.get("categories") or [])]
    help_text = f.get("meaning") or None

    if preset_val is not None and str(preset_val) in cats:
        target_idx = cats.index(str(preset_val))
        if optional:
            return st.selectbox(label, ["—"] + cats, index=target_idx + 1, help=help_text, key=f"inp_{name}")
        return st.selectbox(label, cats, index=target_idx, help=help_text, key=f"inp_{name}")

    if optional:
        choice = st.selectbox(label, ["—"] + cats, index=0, help=help_text, key=f"inp_{name}")
        return None if choice == "—" else choice

    ex = f.get("example")
    idx = cats.index(str(ex)) if ex is not None and str(ex) in cats else 0
    return st.selectbox(label, cats, index=idx, help=help_text, key=f"inp_{name}")


def _handle_submit(service: Any, values: dict[str, Any], selected_model: str) -> None:
    clear_prediction()
    payload = {
        name: val for name, val in values.items()
        if val is not None and not (isinstance(val, str) and val == "—")
    }

    normalized, errors = validate_payload(payload)
    if errors:
        st.error("Please fix the following validation items before assessing:")
        for e in errors:
            st.error(f"• **{e['field']}**: {e['message']}")
        return

    try:
        result = service.predict(normalized, model_key=selected_model)
    except (ValueError, RuntimeError):
        st.error("Assessment could not be completed. Check the inputs and model availability; no new result was saved.")
        return
    save_prediction(normalized, result)

    st.success(
        f"✅ Assessment complete via **{result.get('model_name', selected_model)}**! "
        "Detailed report available on the **Prediction Result** and **Explainability** pages."
    )

    # Decision Header Card
    dec_color = result.get("decision_color", "#059669")
    st.markdown(
        f"""
        <div style="background:{dec_color}1a;border:2px solid {dec_color};border-radius:10px;padding:16px 20px;margin-bottom:18px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-size:0.85rem;font-weight:600;text-transform:uppercase;color:{dec_color};letter-spacing:0.05em;">Analyst Review Indicator</span>
                    <h2 style="margin:2px 0 6px;color:{dec_color};font-weight:800;">{result.get('decision', 'Assessment Complete')}</h2>
                    <p style="margin:0;font-size:0.95rem;color:#334155;">{result.get('decision_action', '')}</p>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.85rem;color:#64748b;font-weight:600;">TARGET PREDICTION</div>
                    <div style="font-size:1.6rem;font-weight:800;color:{dec_color};">
                        Class {result.get('predicted_target', 0)}
                    </div>
                    <div style="font-size:0.8rem;color:#64748b;">{result.get('target_label', '')}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_cards([
        ("Default Probability", pct(result["default_probability"], 1), f"Cutoff: {pct(result.get('decision_threshold', 0.166), 1)}"),
        ("Risk Score", f"{result['risk_score']} / 100", "Rounded probability × 100"),
        ("Risk Band", result["risk_band"], "Configured probability interval"),
        ("Active Model", result.get("model_name", selected_model).split("(")[0].strip(), "Model Used"),
    ])
    for limitation in result.get("limitations", []):
        st.caption(limitation)

    st.markdown(
        f"<div style='margin-top:10px;'>{risk_band_badge(result['risk_band'])} &nbsp; "
        f"<a href='result' target='_self' style='text-decoration:none;font-weight:600;'>👉 View In-Depth Analysis on Prediction Result Page</a></div>",
        unsafe_allow_html=True,
    )

