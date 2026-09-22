"""Model Performance page — held-out test metrics, per-model validation,
confusion matrix, threshold analysis, and calibration.

Everything here is read straight from the service's metrics.json (the pipeline's
evaluation report). Nothing is recomputed or fabricated in the UI — we only
render what the trained artifacts already recorded.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ui.helpers import (
    get_service,
    metric_value,
    num,
    page_header,
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


def _calibration(metrics: dict[str, Any]) -> None:
    st.subheader("Calibration")
    cal = metrics["calibration"]

    left, right = st.columns(2)
    with left:
        st.metric("Brier before", num(cal.get("brier_before"), 4))
    with right:
        st.metric("Brier after", num(cal.get("brier_after"), 4))
    st.caption("Reliability of predicted probabilities after calibration.")

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

        # Reliability curve: observed vs mean-predicted (perfect = diagonal).
        chart_df = pd.DataFrame(
            {"Observed frequency": [b.get("observed_frequency") for b in curve]},
            index=[b.get("mean_predicted") for b in curve],
        )
        chart_df.index.name = "Mean predicted"
        st.line_chart(chart_df)
    else:
        st.info("No reliability curve is available.")
