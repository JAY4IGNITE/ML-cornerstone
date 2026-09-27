"""Model Performance page — held-out test metrics, per-model validation,
confusion matrix, threshold analysis, and calibration.

Everything here is read straight from the service's metrics.json (the pipeline's
evaluation report). Nothing is recomputed or fabricated in the UI — we only
render what the trained artifacts already recorded.
"""
from __future__ import annotations

from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st

from ui import charts
from ui.helpers import (
    get_service,
    metric_value,
    num,
    page_header,
    pct,
    require_model,
    stat_cards,
    synthetic_warning,
    TRAIN_HINT,
)


def render() -> None:
    service = get_service()
    require_model(service)  # halts with the standard gate if the model is absent

    metrics = service.get_metrics()
    if metrics is None:
        # Model loaded but no evaluation report on disk — cannot show performance.
        st.warning("**Model or data not ready** — " + TRAIN_HINT)
        st.stop()

    page_header(
        "Model Performance",
        f"Selected model **{metrics['selected_model']}** by "
        f"**{metrics['selection_metric']}**.",
    )

    synthetic_warning(
        metrics.get("synthetic", False),
        metrics.get("synthetic_warning"),
        title="Synthetic data",
        fallback="These metrics are computed on synthetic data and are "
        "illustrative only.",
    )

    test: dict[str, Any] = metrics["final_test_metrics"]

    # ---- headline KPIs ----------------------------------------------------
    stat_cards([
        ("Test ROC-AUC", metric_value(test.get("roc_auc")), "Held-out test set"),
        ("Test PR-AUC", metric_value(test.get("pr_auc")), "Precision–recall AUC"),
        ("Brier score", num(test.get("brier_score"), 4), "Lower is better"),
        ("F1", num(test.get("f1"), 4), f"At threshold {test.get('threshold')}"),
    ])
    _pr_auc_lift(test)

    _operating_point(metrics, test)
    _validation_comparison(metrics)
    _final_test_metrics(test)
    _confusion_matrix(test)
    _threshold_analysis(metrics)
    _calibration(metrics)

    # ---- closing interpretation (verbatim; phrasing keys off synthetic) ---
    where = "synthetic data" if metrics.get("synthetic") else "the held-out test set"
    st.info(
        "**Interpretation** — These are association metrics evaluated on "
        f"{where}. They describe statistical performance on this dataset and "
        "do not establish causation or fitness for real lending decisions."
    )


def _base_rate(test: dict[str, Any]) -> Optional[float]:
    """Positive (default) rate on the test set, derived from the confusion
    matrix so it is always available and consistent with the reported counts."""
    cm = test.get("confusion_matrix") or {}
    n = test.get("n") or sum(cm.get(k, 0) for k in ("tn", "fp", "fn", "tp"))
    pos = cm.get("tp", 0) + cm.get("fn", 0)
    return pos / n if n else None


def _pr_auc_lift(test: dict[str, Any]) -> None:
    pr = test.get("pr_auc")
    base = _base_rate(test)
    if not isinstance(pr, (int, float)) or not base:
        return
    st.caption(
        f"PR-AUC {num(pr, 4)} vs a no-skill baseline of {pct(base)} (the test "
        f"default rate) — about **{pr / base:.1f}×** lift over random ranking. "
        "PR-AUC, not accuracy, is the honest headline under this imbalance."
    )


def _operating_point(metrics: dict[str, Any], test: dict[str, Any]) -> None:
    st.subheader("Operating threshold")
    sel = metrics.get("selected_threshold", {})
    base_recall = test.get("recall")          # at the raw 0.5 cut
    op_recall = sel.get("test_recall")         # at the F1-optimal cut
    op_prec = sel.get("test_precision")
    st.markdown(
        "A 0.5 cut-off is the **wrong** operating point when only ~8% of "
        "applicants default: almost no one clears 0.5, so the model flags "
        f"hardly anyone and misses most defaulters — test recall is just "
        f"**{pct(base_recall)}**. The pipeline instead picks the **F1-optimal** "
        f"threshold **{num(sel.get('value'), 4)}** on the *validation* split "
        "(never the test set), then reports it on test. That single change "
        f"lifts recall to **{pct(op_recall)}** at **{pct(op_prec)}** precision."
    )
    stat_cards([
        ("Recall at 0.5", pct(base_recall), "Defaulters caught at the raw cut"),
        ("Operating threshold", num(sel.get("value"), 4),
         "F1-optimal, chosen on validation"),
        ("Recall at operating point", pct(op_recall), "Defaulters caught"),
        ("Precision at operating point", pct(op_prec), "Flags that truly default"),
    ])
    st.caption(
        "The threshold is reported, not silently auto-applied at serving — the "
        "operating point is a policy choice for a human, not the model's to make."
    )


def _validation_comparison(metrics: dict[str, Any]) -> None:
    st.subheader("Validation comparison")
    selected = metrics["selected_model"]
    per_model: dict[str, Any] = metrics.get("per_model_validation", {})

    rows: list[dict[str, Any]] = []
    for entry in per_model.values():
        vm = entry.get("val_metrics", {})
        rows.append({
            "Model": entry.get("display_name"),
            # Selected flag lets the reader see which row won without a legend.
            "Selected": "✓" if entry.get("display_name") == selected else "",
            "Accuracy": num(vm.get("accuracy"), 4),
            "Precision": num(vm.get("precision"), 4),
            "Recall": num(vm.get("recall"), 4),
            "F1": num(vm.get("f1"), 4),
            # ROC/PR-AUC may be an {'error': ...} dict on a single-class split.
            "ROC-AUC": metric_value(vm.get("roc_auc")),
            "PR-AUC": metric_value(vm.get("pr_auc")),
            "Brier": num(vm.get("brier_score"), 4),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Selected model: **{selected}** (by {metrics['selection_metric']}).")
    else:
        st.info("No per-model validation results are available.")


def _final_test_metrics(test: dict[str, Any]) -> None:
    st.subheader("Final test metrics")
    row = {
        "Accuracy": num(test.get("accuracy"), 4),
        "Precision": num(test.get("precision"), 4),
        "Recall": num(test.get("recall"), 4),
        "F1": num(test.get("f1"), 4),
        "ROC-AUC": metric_value(test.get("roc_auc")),
        "PR-AUC": metric_value(test.get("pr_auc")),
        "Brier": num(test.get("brier_score"), 4),
        "n": test.get("n"),
    }
    st.dataframe(pd.DataFrame([row]), hide_index=True, use_container_width=True)


def _confusion_matrix(test: dict[str, Any]) -> None:
    st.subheader("Confusion matrix")
    cm = test["confusion_matrix"]
    df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0 (No default)", "Pred 1 (Default)"],
    )
    st.table(df)
    st.caption(f"Test set at threshold {test.get('threshold')}.")
    st.caption(
        "TN = correct non-default · FP = false alarm · "
        "FN = missed default · TP = caught default."
    )


def _threshold_analysis(metrics: dict[str, Any]) -> None:
    st.subheader("Threshold analysis")
    sel = metrics["selected_threshold"]
    st.caption(
        f"F1-optimal threshold {sel['value']}, selected on validation "
        f"(val F1 {sel['val_f1']}) and reported on test (test F1 "
        f"{sel['test_f1']}, precision {sel['test_precision']}, recall "
        f"{sel['test_recall']}). Not auto-applied at serving."
    )

    _threshold_chart(metrics)

    target = sel.get("value")
    rows: list[dict[str, Any]] = []
    for r in metrics.get("threshold_analysis", []):
        thr = r.get("threshold")
        # Mark the analysed row nearest the selected threshold.
        is_sel = (
            target is not None
            and thr is not None
            and abs(float(thr) - float(target)) < 1e-9
        )
        rows.append({
            "": "◀ selected" if is_sel else "",
            "Threshold": thr,
            "Precision": num(r.get("precision"), 4),
            "Recall": num(r.get("recall"), 4),
            "F1": num(r.get("f1"), 4),
            "Flagged rate": num(r.get("flagged_rate"), 4),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No threshold analysis is available.")


def _threshold_chart(metrics: dict[str, Any]) -> None:
    """Precision / recall / F1 as the decision threshold sweeps, with the
    deployed operating point marked — shows the trade-off the threshold makes."""
    ta = metrics.get("threshold_analysis", [])
    rows: list[dict[str, Any]] = []
    for r in ta:
        thr = r.get("threshold")
        if thr is None:
            continue
        for key, label in (("precision", "Precision"), ("recall", "Recall"),
                           ("f1", "F1")):
            v = r.get(key)
            if isinstance(v, (int, float)):
                rows.append({"Threshold": float(thr), "Metric": label, "Value": v})
    if not rows:
        return
    df = pd.DataFrame(rows)
    sel = (metrics.get("selected_threshold") or {}).get("value")
    lines = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Threshold:Q", title="Decision threshold", axis=alt.Axis(format=".2f")),
        y=alt.Y("Value:Q", title="Score", axis=alt.Axis(format="%")),
        color=alt.Color("Metric:N", title=None, scale=alt.Scale(
            domain=["Precision", "Recall", "F1"],
            range=[charts.ACCENT, charts.POS, charts.AMBER])),
        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Threshold:Q", format=".2f"),
                 alt.Tooltip("Value:Q", format=".1%")],
    )
    layers = lines
    if isinstance(sel, (int, float)):
        rule = alt.Chart(pd.DataFrame({"t": [sel]})).mark_rule(
            color=charts.INK, strokeDash=[4, 4]).encode(x="t:Q")
        layers = lines + rule
    charts.show(layers, height=300)
    st.caption(
        f"Dashed line = deployed operating point ({num(sel, 4)}). Lowering the "
        "threshold trades precision for recall; F1 peaks near the chosen point."
    )


def _calibration(metrics: dict[str, Any]) -> None:
    st.subheader("Calibration")
    cal = metrics["calibration"]

    left, right = st.columns(2)
    with left:
        st.metric("Brier before", num(cal.get("brier_before"), 4))
    with right:
        st.metric("Brier after", num(cal.get("brier_after"), 4))
    bb, ba = cal.get("brier_before"), cal.get("brier_after")
    if isinstance(bb, (int, float)) and isinstance(ba, (int, float)) and bb:
        st.caption(
            f"Isotonic calibration cut the Brier score from {num(bb, 4)} to "
            f"{num(ba, 4)} — a {pct(1 - ba / bb)} reduction. A well-calibrated "
            "model sits on the diagonal: a predicted 20% really defaults ~20%."
        )
    else:
        st.caption("Reliability of predicted probabilities after calibration.")

    _reliability_chart(cal)

    curve = cal.get("curve_after") or []
    if curve:
        # bin_lower/bin_upper arrive as 0..1 fractions -> show as percentages.
        table_rows = [{
            "Bin lower %": num((b.get("bin_lower") or 0) * 100, 1),
            "Bin upper %": num((b.get("bin_upper") or 0) * 100, 1),
            "Mean predicted": num(b.get("mean_predicted"), 4),
            "Observed frequency": num(b.get("observed_frequency"), 4),
            "Count": b.get("count"),
        } for b in curve]
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No reliability curve is available.")


def _reliability_chart(cal: dict[str, Any]) -> None:
    """Overlay the before/after reliability curves against the perfect-calibration
    diagonal. Both axes share one scale so the diagonal is a true 45° reference."""
    rows: list[dict[str, Any]] = []
    for label, key in (("Before", "curve_before"), ("After", "curve_after")):
        for b in cal.get(key) or []:
            mp, of = b.get("mean_predicted"), b.get("observed_frequency")
            if isinstance(mp, (int, float)) and isinstance(of, (int, float)):
                rows.append({"Mean predicted": mp, "Observed": of, "Curve": label})
    if not rows:
        return
    df = pd.DataFrame(rows)
    hi = min(1.0, float(max(df["Mean predicted"].max(), df["Observed"].max())) * 1.1)
    scale = alt.Scale(domain=[0, hi])
    diag = alt.Chart(pd.DataFrame({"v": [0, hi]})).mark_line(
        color=charts.MUTED, strokeDash=[4, 4]).encode(
        x=alt.X("v:Q", scale=scale), y=alt.Y("v:Q", scale=scale))
    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Mean predicted:Q", title="Mean predicted probability",
                scale=scale, axis=alt.Axis(format="%")),
        y=alt.Y("Observed:Q", title="Observed default frequency",
                scale=scale, axis=alt.Axis(format="%")),
        color=alt.Color("Curve:N", title=None, scale=alt.Scale(
            domain=["Before", "After"], range=[charts.MUTED, charts.ACCENT])),
        tooltip=[alt.Tooltip("Curve:N"),
                 alt.Tooltip("Mean predicted:Q", format=".1%"),
                 alt.Tooltip("Observed:Q", format=".1%")],
    )
    charts.show(diag + line, height=340)
    st.caption(
        "Dashed diagonal = perfect calibration. The calibrated (After) curve "
        "hugs the diagonal far more closely than the raw (Before) curve."
    )
