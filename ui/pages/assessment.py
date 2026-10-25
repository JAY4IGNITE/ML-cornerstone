"""Risk Assessment page — the applicant form.

Features:
  * Quick-test profile presets (Prime, Standard, Subprime) for instantaneous demo.
  * Model Selector (XGBoost, Random Forest, Decision Tree, Logistic Regression).
  * Feature schema driven input controls with clean labels and hints.
  * Analytical review flag, risk score, probability, and target output.
  * Instant financial ratio preview (DTI, Annuity burden, LTV).
  * Quick credit bureau tier synchronization.
  * Executive decision presentation and multi-model consensus breakdown.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
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
        "title": "🟢 Prime Profile (Low Risk)",
        "desc": "Illustrative inputs: higher income ($250k), smaller loan ($300k), 85% credit score, 12y tenure. Expected: Class 0 (Approved).",
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
        "title": "🟡 Standard Profile (Moderate Risk)",
        "desc": "Illustrative inputs: mid-range income ($120k), loan ($450k), 48% credit score, 3y tenure. Expected: Class 0 (Conditional).",
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
        "title": "🔴 Subprime Profile (High Risk)",
        "desc": "Illustrative inputs: lower income ($30k), larger loan ($800k), 6% credit score, 4m tenure. Expected: Class 1 (Declined / Review).",
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

FRIENDLY_LABELS = {
    "AMT_INCOME_TOTAL": "Annual Gross Income ($)",
    "AMT_CREDIT": "Total Loan Credit Requested ($)",
    "AMT_ANNUITY": "Periodic Annuity Payment ($)",
    "AMT_GOODS_PRICE": "Value of Financed Goods / Collateral ($)",
    "AGE_YEARS": "Borrower Age (Years)",
    "EMPLOYMENT_YEARS": "Employment Tenure at Current Job (Years)",
    "CNT_FAM_MEMBERS": "Total Household Family Members",
    "CNT_CHILDREN": "Number of Dependent Children",
    "EXT_SOURCE_1": "Credit Bureau 1 (Equifax Equivalent, 0.0–1.0)",
    "EXT_SOURCE_2": "Credit Bureau 2 (TransUnion Equivalent, 0.0–1.0)",
    "EXT_SOURCE_3": "Credit Bureau 3 (Experian Equivalent, 0.0–1.0)",
    "REGION_POPULATION_RELATIVE": "Region Density (Metro 0.035 / Suburban 0.020 / Rural 0.007)",
    "NAME_CONTRACT_TYPE": "Loan Contract Type",
    "CODE_GENDER": "Applicant Gender",
    "FLAG_OWN_CAR": "Owns Automobile",
    "FLAG_OWN_REALTY": "Owns Real Estate Property",
    "NAME_INCOME_TYPE": "Primary Income / Employment Category",
    "NAME_EDUCATION_TYPE": "Highest Level of Education",
    "NAME_FAMILY_STATUS": "Marital / Family Status",
    "NAME_HOUSING_TYPE": "Living & Housing Arrangement",
    "OCCUPATION_TYPE": "Professional Occupation Role",
}

FRIENDLY_HELP = {
    "AMT_INCOME_TOTAL": "Total annual income before taxes. Required for Debt-to-Income (DTI) assessment.",
    "AMT_CREDIT": "Total principal amount requested for financing.",
    "AMT_ANNUITY": "Scheduled installment payment per repayment cycle.",
    "AMT_GOODS_PRICE": "Purchase price of the goods or asset being financed.",
    "AGE_YEARS": "Applicant age in years (must be between 18 and 100).",
    "EMPLOYMENT_YEARS": "Continuous years in current profession. Enter 0 for unemployed / pensioner.",
    "CNT_FAM_MEMBERS": "Total number of people living in the applicant's household.",
    "CNT_CHILDREN": "Count of dependent children under 18 years old.",
    "EXT_SOURCE_1": "Normalized external bureau credit score (0.00 = severe delinquency, 1.00 = perfect credit history).",
    "EXT_SOURCE_2": "Secondary external credit bureau rating percentile.",
    "EXT_SOURCE_3": "Tertiary external credit bureau rating percentile.",
    "REGION_POPULATION_RELATIVE": "Statistical density ratio: ~0.035 for Major Metros, ~0.020 for Suburban, ~0.007 for Rural.",
    "NAME_CONTRACT_TYPE": "Cash Loan (fixed term) or Revolving Loan (line of credit).",
    "CODE_GENDER": "Demographic identifier as recorded in application.",
    "FLAG_OWN_CAR": "Whether the applicant owns a registered personal vehicle.",
    "FLAG_OWN_REALTY": "Whether the applicant owns residential or commercial real estate.",
    "NAME_INCOME_TYPE": "Primary source of incoming funds (Working, Commercial associate, State servant, etc.).",
    "NAME_EDUCATION_TYPE": "Educational attainment level.",
    "NAME_FAMILY_STATUS": "Current marital status.",
    "NAME_HOUSING_TYPE": "Type of residential living arrangement.",
    "OCCUPATION_TYPE": "Specific occupational role or job category.",
}


def apply_preset(key: str) -> None:
    """Update widget state before the form renders, including on repeat clicks."""
    data = dict(PRESETS[key]["data"])
    st.session_state["form_preset"] = key
    st.session_state["form_data"] = data
    for name, value in data.items():
        st.session_state[f"inp_{name}"] = value


def apply_bureau_tier(score: float) -> None:
    """Set all 3 external bureau scores to a standardized tier to eliminate arbitrary fluctuations."""
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        st.session_state[f"inp_{col}"] = float(score)
        if "form_data" in st.session_state and isinstance(st.session_state["form_data"], dict):
            st.session_state["form_data"][col] = float(score)


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
        "Calibrated default-probability estimation and underwriting review. Analytical estimates, not automated lending approvals.",
    )

    synthetic_warning(
        mi.get("synthetic", False),
        mi.get("synthetic_warning"),
        title="Synthetic data",
        fallback="This model was trained on synthetic data. Results are for demonstration only.",
    )

    # --- Quick Demonstration Presets ---
    st.markdown("##### ⚡ Instant Benchmark Profiles")
    st.caption("Click any profile below to instantly populate realistic, coherent applicant data:")

    cols = st.columns(3)
    for col, (pkey, pval) in zip(cols, PRESETS.items()):
        with col:
            st.button(pval["title"], key=f"btn_preset_{pkey}", use_container_width=True,
                      on_click=apply_preset, args=(pkey,))
            st.caption(pval["desc"])

    current_data = st.session_state.get("form_data", {})

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Model Selector & Guidance ---
    avail_models = service.get_available_models()
    col_m1, col_m2 = st.columns([3, 2])
    with col_m1:
        default_idx = 0
        if "xgboost" in avail_models:
            default_idx = list(avail_models).index("xgboost")
        elif mi.get("model_key") in avail_models:
            default_idx = list(avail_models).index(mi["model_key"])

        selected_model = st.selectbox(
            "🧠 Select Machine Learning Model",
            options=list(avail_models.keys()),
            format_func=lambda k: f"{avail_models.get(k, k)} {'★ (Recommended)' if k == 'xgboost' else ''}",
            index=default_idx,
            help="XGBoost is the audited production champion (ROC-AUC 0.756) with Platt sigmoid calibration.",
        )
    with col_m2:
        st.caption(
            "**Champion Model**: XGBoost is calibrated using 3-fold Platt scaling for true probabilistic risk accuracy. "
            "All candidate models are evaluated side-by-side upon assessment."
        )

    # --- Real-Time Financial Ratios Card ---
    inc_val = st.session_state.get("inp_AMT_INCOME_TOTAL", current_data.get("AMT_INCOME_TOTAL", 180000.0))
    cred_val = st.session_state.get("inp_AMT_CREDIT", current_data.get("AMT_CREDIT", 600000.0))
    ann_val = st.session_state.get("inp_AMT_ANNUITY", current_data.get("AMT_ANNUITY", 27000.0))
    gds_val = st.session_state.get("inp_AMT_GOODS_PRICE", current_data.get("AMT_GOODS_PRICE", 540000.0))

    try:
        inc_f = float(inc_val) if inc_val is not None else 180000.0
        cred_f = float(cred_val) if cred_val is not None else 600000.0
        ann_f = float(ann_val) if ann_val is not None else 27000.0
        gds_f = float(gds_val) if gds_val is not None else 540000.0
        dti = cred_f / inc_f if inc_f > 0 else 0.0
        burden = (ann_f / inc_f) * 100 if inc_f > 0 else 0.0
        ltv = (cred_f / gds_f) * 100 if gds_f > 0 else 100.0

        dti_color = "#059669" if dti < 4.0 else ("#d97706" if dti < 7.0 else "#dc2626")
        burden_color = "#059669" if burden < 20.0 else ("#d97706" if burden < 35.0 else "#dc2626")

        st.markdown(
            f"""
            <div class="fin-card">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                    <div>
                        <span style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.04em;">Financial Health Indicators</span>
                        <div style="font-size:0.95rem;font-weight:600;color:#1e293b;margin-top:2px;">Live Application Ratios Preview</div>
                    </div>
                    <div style="display:flex;gap:16px;flex-wrap:wrap;">
                        <div style="text-align:right;">
                            <span style="font-size:0.75rem;color:#64748b;">Debt-to-Income (DTI)</span><br>
                            <span class="ratio-badge" style="background:{dti_color}18;color:{dti_color};font-size:0.95rem;">{dti:.1f}x ({'Healthy' if dti < 4 else ('Elevated' if dti < 7 else 'Severe')})</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:0.75rem;color:#64748b;">Annuity Burden</span><br>
                            <span class="ratio-badge" style="background:{burden_color}18;color:{burden_color};font-size:0.95rem;">{burden:.1f}% of income</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:0.75rem;color:#64748b;">Loan-to-Goods (LTV)</span><br>
                            <span class="ratio-badge" style="background:#0284c718;color:#0284c7;font-size:0.95rem;">{ltv:.0f}% financed</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    numeric = schema.get("numeric_features") or []
    categorical = schema.get("categorical_features") or []
    financials = [f for f in numeric if str(f.get("name", "")).startswith("AMT_")]
    external = [f for f in numeric if str(f.get("name", "")).startswith("EXT_")]
    profile_num = [f for f in numeric if f not in financials and f not in external]

    values: dict[str, Any] = {}
    with st.form("assessment_form"):
        # --- Section 1: Financial Details ---
        st.subheader("💼 1. Loan & Financial Details")
        st.caption("Income base, requested loan amount, periodic annuity payment, and collateral value.")
        c1, c2 = st.columns(2)
        for i, f in enumerate(financials):
            with (c1 if i % 2 == 0 else c2):
                values[f["name"]] = _numeric_input(f, current_data.get(f["name"]))

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- Section 2: External Credit Bureau Scores ---
        st.subheader("🛡️ 2. External Credit Bureau Ratings (Primary Risk Driver)")
        st.caption(
            "Normalized external scores (0.00 – 1.00) represent credit bureau ratings (Equifax, TransUnion, Experian equivalents). "
            "In credit risk models, this accounts for over 50% of the prediction weight."
        )

        c7, c8, c9 = st.columns(3)
        for i, f in enumerate(external):
            with [c7, c8, c9][i % 3]:
                values[f["name"]] = _numeric_input(f, current_data.get(f["name"]))

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- Section 3: Applicant Demographics & Employment ---
        st.subheader("👤 3. Borrower Demographics & Employment")
        st.caption("Age, employment tenure, and educational background.")
        c3, c4 = st.columns(2)
        for i, f in enumerate(profile_num):
            with (c3 if i % 2 == 0 else c4):
                values[f["name"]] = _numeric_input(f, current_data.get(f["name"]))

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- Section 4: Household & Loan Structure ---
        st.subheader("🏠 4. Household, Assets & Contract Terms")
        st.caption("Household size, living situation, collateral assets, and contract type.")
        c5, c6 = st.columns(2)
        for i, f in enumerate(categorical):
            with (c5 if i % 2 == 0 else c6):
                values[f["name"]] = _categorical_input(f, current_data.get(f["name"]))

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Run Risk Assessment & Predict", use_container_width=True, type="primary")

    if submitted:
        _handle_submit(service, values, selected_model)


def _numeric_input(f: dict[str, Any], preset_val: Any = None) -> float | None:
    name = f["name"]
    optional = name in OPTIONAL_FIELDS
    clean_label = FRIENDLY_LABELS.get(name, f.get("label") or name)
    label = clean_label + (" (optional)" if optional else "")

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

    help_text = FRIENDLY_HELP.get(name, f.get("meaning") or "")
    if lo is not None and hi is not None:
        help_text = f"{help_text} (Allowed range: {lo}–{hi} {f.get('unit', '')})".strip()

    step = 0.01 if (hi is not None and hi <= 1.0) else (1000.0 if "AMT_" in name else 1.0)
    return st.number_input(label, value=default, step=step, help=help_text or None, key=f"inp_{name}", **kwargs)


def _categorical_input(f: dict[str, Any], preset_val: Any = None) -> str | None:
    name = f["name"]
    optional = name in OPTIONAL_FIELDS
    clean_label = FRIENDLY_LABELS.get(name, f.get("label") or name)
    label = clean_label + (" (optional)" if optional else "")
    cats = [str(c) for c in (f.get("categories") or [])]
    help_text = FRIENDLY_HELP.get(name, f.get("meaning") or None)

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
            st.error(f"• **{FRIENDLY_LABELS.get(e['field'], e['field'])}**: {e['message']}")
        return

    try:
        result = service.predict(normalized, model_key=selected_model)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Assessment could not be completed: {exc}. Check the inputs and model availability; no new result was saved.")
        return
    save_prediction(normalized, result)

    st.success(
        f"✅ Risk Assessment Successfully Evaluated via **{result.get('model_name', selected_model)}**! "
        "A detailed analysis has been saved to your session."
    )

    # Executive Decision Card
    dec = result.get("decision", "Assessment Complete")
    dec_color = result.get("decision_color", "#059669")
    target_val = result.get("predicted_target", 0)
    target_text = "CLASS 0 · PERFORMING LOAN" if target_val == 0 else "CLASS 1 · DEFAULT RISK FLAGGED"

    st.markdown(
        f"""
        <div style="background:{dec_color}14;border:2px solid {dec_color};border-radius:12px;padding:20px 24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
                <div>
                    <span style="font-size:0.8rem;font-weight:700;text-transform:uppercase;color:{dec_color};letter-spacing:0.06em;">
                        Analyst Underwriting Recommendation
                    </span>
                    <h2 style="margin:4px 0 6px;color:{dec_color};font-size:1.85rem;font-weight:800;">
                        {dec}
                    </h2>
                    <p style="margin:0;font-size:0.95rem;color:#334155;max-width:680px;font-weight:500;">
                        {result.get('decision_action', '')}
                    </p>
                </div>
                <div style="background:white;border:1px solid #cbd5e1;border-radius:10px;padding:14px 22px;text-align:center;min-width:180px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                    <div style="font-size:0.75rem;font-weight:700;color:#64748b;letter-spacing:0.05em;text-transform:uppercase;">ML Binary Target</div>
                    <div style="font-size:1.8rem;font-weight:900;color:{dec_color};margin:2px 0;">TARGET = {target_val}</div>
                    <div style="font-size:0.8rem;font-weight:600;color:#475569;">{target_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_cards([
        ("Default Probability", pct(result["default_probability"], 1), f"Review Cutoff: {pct(result.get('decision_threshold', 0.166), 1)}"),
        ("Risk Score", f"{result['risk_score']} / 100", "Rounded probability × 100"),
        ("Risk Band", result["risk_band"], "Calibrated probability tier"),
        ("Active Model", result.get("model_name", selected_model).split("(")[0].strip(), "Evaluated Algorithm"),
    ])

    # Multi-Model Consensus Table
    all_preds = result.get("all_model_predictions") or {}
    if all_preds:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⚖️ Cross-Model Consensus Comparison")
        st.caption("Real-time benchmark comparing all 4 machine learning models for this applicant:")

        consensus_rows = []
        for mkey, mres in all_preds.items():
            consensus_rows.append({
                "Model Algorithm": mres.get("display_name", mkey.title()),
                "Default Probability": pct(mres.get("default_probability", 0.0), 2),
                "Risk Score": f"{mres.get('risk_score', 0)} / 100",
                "Risk Band": mres.get("risk_band", "—"),
                "Recommendation": mres.get("decision", {}).get("recommendation", "—"),
            })
        st.dataframe(pd.DataFrame(consensus_rows), hide_index=True, use_container_width=True)

    # Top Risk Drivers
    expl = result.get("explanation") or {}
    contribs = expl.get("contributions") or []
    if contribs:
        pos_drivers = [c for c in contribs if c.get("direction") == "increases_risk"]
        neg_drivers = [c for c in contribs if c.get("direction") == "decreases_risk"]

        if pos_drivers or neg_drivers:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🔍 Primary Risk Factor Attribution (SHAP)")
            c_neg, c_pos = st.columns(2)
            with c_neg:
                st.markdown("##### 🟢 Top Risk-Reducing Factors")
                if neg_drivers:
                    for d in neg_drivers[:3]:
                        lbl = FRIENDLY_LABELS.get(d.get("feature"), d.get("display_label", d.get("feature")))
                        st.markdown(
                            f"""
                            <div style="border-left:3px solid #059669;padding:6px 12px;margin-bottom:8px;background:#f0fdf4;border-radius:0 6px 6px 0;">
                                <div style="font-weight:700;color:#166534;font-size:0.9rem;">{lbl}</div>
                                <div style="font-size:0.8rem;color:#059669;font-weight:600;">Contribution: {float(d.get('contribution', 0.0)):+.4f} (Decreases Risk)</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No significant risk-reducing factors identified.")

            with c_pos:
                st.markdown("##### 🔴 Top Risk-Elevating Factors")
                if pos_drivers:
                    for d in pos_drivers[:3]:
                        lbl = FRIENDLY_LABELS.get(d.get("feature"), d.get("display_label", d.get("feature")))
                        st.markdown(
                            f"""
                            <div style="border-left:3px solid #dc2626;padding:6px 12px;margin-bottom:8px;background:#fef2f2;border-radius:0 6px 6px 0;">
                                <div style="font-weight:700;color:#991b1b;font-size:0.9rem;">{lbl}</div>
                                <div style="font-size:0.8rem;color:#dc2626;font-weight:600;">Contribution: {float(d.get('contribution', 0.0)):+.4f} (Increases Risk)</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No significant risk-elevating factors identified.")

    st.markdown(
        f"""
        <div style="margin-top:16px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>{risk_band_badge(result['risk_band'])}</div>
            <a href="result" target="_self" style="background:#1e293b;color:white;padding:8px 16px;border-radius:6px;font-weight:600;font-size:0.9rem;">
                👉 Open Full Prediction Result & Review Report
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
