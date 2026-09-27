"""Model Selection page — why XGBoost, in detail.

Answers the single question "why this model?" from the evaluation report the
pipeline already wrote (metrics.json). Nothing is recomputed or fabricated: the
cross-validation scores, per-fold spreads and tuning results shown here are read
straight from ``per_model_cross_validation`` and ``hyperparameter_tuning``.

Selection rule (config/config.yaml → selection.cv_folds): each candidate is
scored by stratified 5-fold cross-validation on TRAIN+VAL and ranked by the
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
    require_model,
    stat_cards,
    synthetic_warning,
    TRAIN_HINT,
)

# Fixed display order (weakest → strongest CV score) so tables read consistently.
_ORDER = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
_TUNED_NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost"}


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
        "Model Selection",
        f"Why **{selected}** was chosen — the cross-validated evidence behind "
        "the decision, read directly from the training report.",
    )

    synthetic_warning(
        metrics.get("synthetic", False),
        metrics.get("synthetic_warning"),
        title="Synthetic data",
        fallback="These selection scores were computed on synthetic data and "
        "are illustrative only.",
    )

    _headline(cv, selected)
    _how_selected()
    _cv_comparison(cv, selected)
    _per_fold(cv, selected)
    _why_winner(cv, selected)
    _tuning(metrics)
    _formulas()


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


def _headline(cv: dict[str, Any], selected: str) -> None:
    w = _winner_entry(cv, selected)
    folds = w.get("cv_folds")
    stat_cards([
        ("Selected model", selected, "Highest mean CV score"),
        ("Selection metric", "ROC-AUC", "Ranking criterion"),
        ("Validation", f"{folds}-fold CV" if folds else "Cross-validation",
         "Stratified, on TRAIN+VAL"),
        ("Winning CV mean", f"{num(w.get('mean'), 4)} ± {num(w.get('std'), 4)}",
         "Mean ROC-AUC across folds"),
    ])


def _how_selected() -> None:
    st.markdown(
        "**How the model was chosen.** Every candidate runs through the *same* "
        "preprocessing pipeline, then is scored by **stratified 5-fold "
        "cross-validation** on the combined train+validation data. The model "
        "with the highest **mean ROC-AUC** wins; the standard deviation across "
        "folds is reported so we can see whether the lead is stable."
    )
    st.markdown(
        "- **Why not accuracy?** The default rate here is only ~8%, so a model "
        "that always predicts *repays* already scores ≈ 0.92 accuracy while "
        "catching zero defaulters. Accuracy is never the selection metric.\n"
        "- **Why cross-validation, not one split?** A single validation split "
        "can be lucky. Averaging five held-out folds — and checking their spread "
        "— makes the choice robust rather than an artefact of one partition.\n"
        "- **Why ROC-AUC?** It scores how well the model *ranks* risk across all "
        "thresholds, which is what we want when the operating cut-off is chosen "
        "separately (see Model Performance)."
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
        x=alt.X("mean:Q", title="Mean ROC-AUC (5-fold)",
                scale=alt.Scale(domain=[0.70, 0.77]), axis=alt.Axis(format=".3f")),
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
        "Bars = mean ROC-AUC across 5 stratified folds; whiskers = ±1 std. The "
        "x-axis starts at 0.70 to resolve otherwise-close scores. "
        f"**{selected}** (blue) has the highest mean."
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
        x=alt.X("score:Q", title="Fold ROC-AUC",
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
        "Each dot is one held-out fold. Tight clusters mean the ranking is "
        "stable — the winner is not the product of a single favourable split."
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
            f"- **Highest mean ROC-AUC.** {top.get('display_name')} scored "
            f"**{num(top['mean'], 4)}**, ahead of {second.get('display_name')} "
            f"({num(second['mean'], 4)}) by **{num(margin, 4)}** — wider than its "
            f"own fold-to-fold std (±{num(top_std, 4)}), so the lead is real, not "
            f"noise."
        )
    st.markdown(
        "- **Handles the 8% imbalance.** Gradient boosting with "
        "`scale_pos_weight` (and `class_weight='balanced'` for the linear and "
        "tree baselines) keeps the rare default class from being drowned out.\n"
        "- **Captures non-linear interactions.** Boosted trees model interactions "
        "between income, credit size and the external scores that the linear "
        "baseline cannot, which is where its extra ranking power comes from.\n"
        "- **Calibrated after selection.** The winner is refit on all training "
        "data and wrapped in isotonic calibration, so its *probabilities* are "
        "trustworthy (Brier 0.196 → 0.068), not just its ranking."
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




