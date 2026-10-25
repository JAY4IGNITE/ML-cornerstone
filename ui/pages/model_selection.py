"""Model Selection page — evidence for the selected model.

Answers the single question "why this model?" from the evaluation report the
pipeline already wrote (metrics.json). Nothing is recomputed or fabricated: the
cross-validation scores, per-fold spreads and tuning results shown here are read
straight from ``per_model_cross_validation`` and ``hyperparameter_tuning``.

Selection rule (config/config.yaml → selection.cv_folds): each candidate is
scored by configured stratified cross-validation on TRAIN and ranked by the
MEAN ROC-AUC (std reported) — never a single lucky split, never raw accuracy.
"""
from __future__ import annotations

from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st

from ui import charts
from ui.formulas import render_metric_definitions
from ui.helpers import (
    get_service,
    num,
    page_header,
    pct,
    require_model,
    stat_cards,
    synthetic_warning,
    TRAIN_HINT,
)

# Fixed display order (weakest → strongest CV score) so tables read consistently.
_ORDER = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
_TUNED_NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost"}


from ui.pages.assessment import PRESETS


def render() -> None:
    service = get_service()
    require_model(service)

    metrics = service.get_metrics()
    if metrics is None:
        st.warning("**Model or data not ready** — " + TRAIN_HINT)
        st.stop()

    selected = metrics.get("selected_model", "—")
    cv: dict[str, Any] = metrics.get("per_model_cross_validation", {})

    page_header(
        "Model Selection & Prediction Analysis",
        f"Why **{selected}** was chosen — the cross-validated evidence behind "
        "the decision, plus live candidate model prediction testing.",
    )

    synthetic_warning(
        metrics.get("synthetic", False),
        metrics.get("synthetic_warning"),
        title="Synthetic data",
        fallback="These selection scores were computed on synthetic data and "
        "are illustrative only.",
    )

    _headline(cv, selected, metrics)
    _live_model_predictor(service)
    _how_selected(metrics)
    _cv_comparison(cv, selected)
    _per_fold(cv, selected)
    _why_winner(cv, selected)
    _tuning(metrics)
    _formulas()


def _live_model_predictor(service: Any) -> None:
    st.subheader("Interactive Model Prediction & Output Test")
    st.caption("Select any trained candidate model below and test how it predicts across different borrower risk profiles:")

    avail_models = service.get_available_models()
    col1, col2 = st.columns([2, 3])
    with col1:
        chosen_model = st.selectbox(
            "🧠 Select Candidate Model to Test",
            options=list(avail_models.keys()),
            format_func=lambda k: avail_models.get(k, k),
            key="ms_model_choice",
        )
    with col2:
        profile_choice = st.selectbox(
            "👤 Select Test Profile",
            options=["prime", "standard", "subprime"],
            format_func=lambda k: PRESETS[k]["title"],
            key="ms_profile_choice",
        )

    profile_data = PRESETS[profile_choice]["data"]
    res = service.predict(profile_data, model_key=chosen_model, with_explanation=False)

    c_left, c_right = st.columns([3, 2])
    with c_left:
        dec_color = res.get("decision_color", "#059669")
        st.markdown(
            f"""
            <div style="background:{dec_color}14;border:2px solid {dec_color};border-radius:10px;padding:14px 18px;margin-top:6px;">
                <div style="font-size:0.8rem;font-weight:700;color:{dec_color};text-transform:uppercase;">Model Output & Decision</div>
                <h3 style="margin:2px 0 4px;color:{dec_color};">{res.get('decision')}</h3>
                <div style="font-size:0.95rem;color:#1e293b;">{res.get('decision_action')}</div>
                <div style="margin-top:8px;font-size:0.85rem;color:#475569;">
                    <b>Binary Class:</b> TARGET = {res.get('predicted_target')} ({res.get('target_label')}) &nbsp;|&nbsp;
                    <b>P(Default):</b> {pct(res['default_probability'], 2)} &nbsp;|&nbsp;
                    <b>Risk Score:</b> {res['risk_score']} / 100 ({res['risk_band']} Risk)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_right:
        st.markdown("**Available model predictions:**")
        mini_rows = []
        for mk, mr in (res.get("all_model_predictions") or {}).items():
            is_active = (mk == chosen_model)
            prefix = "👉 " if is_active else ""
            mini_rows.append({
                "Model": f"{prefix}{mr.get('display_name', mk).split('(')[0].strip()}",
                "Prob": pct(mr.get("default_probability", 0), 1),
                "Score": f"{mr.get('risk_score', 0)}/100",
                "Decision": mr.get("decision", {}).get("recommendation", "").split("(")[0].strip(),
            })
        st.dataframe(pd.DataFrame(mini_rows), hide_index=True, use_container_width=True)

    st.divider()


def _winner_entry(cv: dict[str, Any], selected: str) -> dict[str, Any]:
    for entry in cv.values():
        if entry.get("display_name") == selected:
            return entry
    return {}


def _stats(entry: dict[str, Any]) -> dict[str, Optional[float]]:
    scores = [s for s in (entry.get("scores") or []) if isinstance(s, (int, float))]
    return {
        "mean": entry.get("mean"),
        "std": entry.get("std"),
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
    }


def _headline(cv: dict[str, Any], selected: str, metrics: dict[str, Any]) -> None:
    w = _winner_entry(cv, selected)
    folds = w.get("cv_folds")
    stat_cards([
        ("Selected model", selected, "Recorded training selection"),
        ("Selection metric", metrics.get("selection_metric", "—"), "Recorded selection criterion"),
        ("Validation", f"{folds}-fold CV" if folds else "Cross-validation",
         metrics.get("selection_basis", "Not recorded")),
        ("Winning CV mean", f"{num(w.get('mean'), 4)} ± {num(w.get('std'), 4)}",
         "Recorded score across folds"),
    ])


def _how_selected(metrics: dict[str, Any]) -> None:
    st.markdown(
        f"**Recorded selection method:** {metrics.get('selection_basis', 'not recorded')}. "
        f"**Selection metric:** {metrics.get('selection_metric', 'not recorded')}. "
        "Cross-validation scores compare candidate configurations; the held-out test "
        "report provides the separate final evaluation."
    )
    st.markdown(
        "Accuracy alone can hide missed positives in imbalanced data. Review "
        "ROC-AUC, average precision, calibration, recall and precision together. "
        "Fold variability is descriptive; it does not prove a statistically significant lead."
    )


def _cv_comparison(cv: dict[str, Any], selected: str) -> None:
    st.subheader("Cross-validated model comparison")
    rows: list[dict[str, Any]] = []
    for key in _ORDER:
        entry = cv.get(key)
        if not entry:
            continue
        mean, std = entry.get("mean"), entry.get("std") or 0.0
        rows.append({
            "Model": entry.get("display_name", key),
            "mean": mean, "std": std,
            "lower": mean - std if mean is not None else None,
            "upper": mean + std if mean is not None else None,
            "sel": entry.get("display_name") == selected,
        })
    if not rows:
        st.info("No cross-validation results are available.")
        return

    df = pd.DataFrame(rows)
    order = df.sort_values("mean", ascending=False)["Model"].tolist()
    base = alt.Chart(df)
    bars = base.mark_bar().encode(
        x=alt.X("mean:Q", title="Mean selection score",
                scale=alt.Scale(zero=False), axis=alt.Axis(format=".3f")),
        y=alt.Y("Model:N", sort=order, title=None),
        color=alt.condition("datum.sel", alt.value(charts.ACCENT), alt.value(charts.MUTED)),
        tooltip=[alt.Tooltip("Model:N"),
                 alt.Tooltip("mean:Q", format=".4f", title="CV mean"),
                 alt.Tooltip("std:Q", format=".4f", title="CV std")],
    )
    err = base.mark_rule(color=charts.INK, strokeWidth=1.5).encode(
        y=alt.Y("Model:N", sort=order), x="lower:Q", x2="upper:Q",
    )
    charts.show(bars + err, height=230)
    st.caption(
        "Bars = recorded mean selection scores; whiskers = ±1 standard deviation. "
        f"**{selected}** is the recorded selected model."
    )


def _per_fold(cv: dict[str, Any], selected: str) -> None:
    st.subheader("Per-fold stability")
    recs: list[dict[str, Any]] = []
    for key in _ORDER:
        entry = cv.get(key)
        if not entry:
            continue
        for i, score in enumerate(entry.get("scores") or [], start=1):
            recs.append({
                "Model": entry.get("display_name", key),
                "Fold": f"Fold {i}", "score": score,
                "sel": entry.get("display_name") == selected,
            })
    if not recs:
        st.info("No per-fold scores are available.")
        return

    df = pd.DataFrame(recs)
    order = df.groupby("Model")["score"].mean().sort_values(ascending=False).index.tolist()
    pts = alt.Chart(df).mark_circle(size=95, opacity=0.75).encode(
        x=alt.X("score:Q", title="Fold selection score",
                scale=alt.Scale(zero=False), axis=alt.Axis(format=".3f")),
        y=alt.Y("Model:N", sort=order, title=None),
        color=alt.condition("datum.sel", alt.value(charts.ACCENT), alt.value(charts.MUTED)),
        tooltip=["Model", "Fold", alt.Tooltip("score:Q", format=".4f")],
    )
    charts.show(pts, height=230)

    trows: list[dict[str, Any]] = []
    for key in _ORDER:
        entry = cv.get(key)
        if not entry:
            continue
        s = _stats(entry)
        trows.append({
            "Model": entry.get("display_name", key),
            "CV mean": num(s["mean"], 4), "CV std": num(s["std"], 4),
            "Min fold": num(s["min"], 4), "Max fold": num(s["max"], 4),
            "Selected": "✓" if entry.get("display_name") == selected else "",
        })
    st.dataframe(pd.DataFrame(trows), hide_index=True, use_container_width=True)
    st.caption(
        "Each dot is one fold score. The spread describes variability on these "
        "splits and is not a confidence interval or proof of generalization."
    )


def _why_winner(cv: dict[str, Any], selected: str) -> None:
    st.subheader(f"Why {selected}?")
    ranked = sorted(
        [e for e in cv.values() if isinstance(e.get("mean"), (int, float))],
        key=lambda e: e["mean"], reverse=True,
    )
    if len(ranked) >= 2:
        top, second = ranked[0], ranked[1]
        margin = top["mean"] - second["mean"]
        top_std = top.get("std") or 0.0
        st.markdown(
            f"- **Highest recorded mean score.** {top.get('display_name')} scored "
            f"**{num(top['mean'], 4)}**, ahead of {second.get('display_name')} "
            f"({num(second['mean'], 4)}) by **{num(margin, 4)}**. Its "
            f"fold standard deviation is ±{num(top_std, 4)}. This comparison "
            "does not establish statistical significance."
        )
    st.markdown(
        "Selection identifies the best candidate under the recorded criterion. "
        "It does not certify an individual prediction. See Model Performance "
        "for the measured calibration and test-set errors, and Model Information "
        "for the saved training method."
    )


def _tuning(metrics: dict[str, Any]) -> None:
    tuning: dict[str, Any] = metrics.get("hyperparameter_tuning") or {}
    if not tuning:
        return
    st.subheader("Hyperparameter tuning")
    st.caption(
        "Opt-in RandomizedSearchCV (config → tuning) for the heavier models. "
        "The winning configuration is refit on the full training data."
    )
    for key in _ORDER:
        t = tuning.get(key)
        if not t:
            continue
        name = _TUNED_NAMES.get(key, key)
        with st.expander(f"{name} — best search CV {num(t.get('best_cv_score'), 4)}"):
            params = t.get("best_params") or {}
            if params:
                prows = [{"Parameter": p.replace("model__", ""), "Value": v}
                         for p, v in params.items()]
                st.dataframe(pd.DataFrame(prows), hide_index=True, use_container_width=True)
            meta = []
            if t.get("n_iter") is not None:
                meta.append(f"{t['n_iter']} candidates")
            if t.get("search_samples") is not None:
                meta.append(f"{int(t['search_samples']):,}-row subsample")
            if meta:
                st.caption(" · ".join(meta))
            if t.get("note"):
                st.caption(t["note"])
    st.caption(
        "Note: search scores are computed on the tuning subsample and are **not** "
        "directly comparable to the full-data cross-validation scores above."
    )


def _formulas() -> None:
    with st.expander("Formulas & definitions"):
        render_metric_definitions(["cv_mean", "roc_auc", "scale_pos_weight", "accuracy"])




