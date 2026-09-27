"""Fairness Diagnostics page — group metrics and disparity ratios.

Surfaces the ``fairness`` block the pipeline recorded: raw per-group rates on one
held-out split and simple min/max disparity ratios, sliced by gender and age
band. This is deliberately framed as a *diagnostic*, not a certification — the
pipeline's own disclaimer is shown prominently and repeated in the closing note.
Nothing is recomputed in the UI.
"""
from __future__ import annotations

import math
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from ui import charts
from ui.helpers import get_service, num, page_header, require_model, TRAIN_HINT

# Human labels for the metric keys the disparity block reports.
_METRIC_LABELS = {
    "selection_rate": "Selection rate",
    "false_negative_rate": "False-negative rate",
    "false_positive_rate": "False-positive rate",
    "roc_auc": "ROC-AUC",
}
_FOUR_FIFTHS = 0.80


def _rate(x: Any) -> str:
    """Percent with NaN/None guard (group blocks can carry NaN for tiny groups)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def render() -> None:
    service = get_service()
    require_model(service)

    metrics = service.get_metrics()
    fairness = (metrics or {}).get("fairness") if metrics else None
    if not fairness:
        st.warning("**Fairness diagnostics not available** — " + TRAIN_HINT)
        st.stop()

    page_header(
        "Fairness Diagnostics",
        "How error rates vary across applicant groups — a screening tool for "
        "human review, not a fairness guarantee.",
    )

    st.warning("**Diagnostic only** — " + fairness.get("disclaimer", ""))

    thr = fairness.get("operating_threshold")
    if thr is not None:
        st.caption(
            f"All threshold-dependent rates below use the F1-optimal operating "
            f"point **{num(thr, 4)}** (chosen on validation), not the raw 0.5 cut."
        )

    slices: dict[str, Any] = fairness.get("slices", {})
    disparities: dict[str, Any] = fairness.get("disparities", {})
    for attr in slices:
        _attribute(attr, slices[attr], disparities.get(attr, {}))

    st.info(
        "**Interpretation** — Disparities can reflect real differences in the "
        "underlying populations, sampling noise in small groups, or model bias; "
        "this page cannot tell which. Treat every flag as a prompt for human "
        "review, never as an automated decision or a compliance sign-off."
    )


_ATTR_LABELS = {"CODE_GENDER": "By gender", "AGE_BAND": "By age band"}


def _attribute(attr: str, groups: dict[str, Any], disparity: dict[str, Any]) -> None:
    st.subheader(_ATTR_LABELS.get(attr, attr))

    # Grouped bars: default rate vs the model's selection (flag) rate per group.
    # Tiny low-support groups are kept in the table but dropped from the chart so
    # a 2-person group cannot distort the axis.
    chart_rows, dropped = [], []
    for name, g in groups.items():
        if g.get("low_support"):
            dropped.append(name)
            continue
        chart_rows.append({"Group": name, "Measure": "Default rate",
                           "Rate": g.get("positive_rate")})
        chart_rows.append({"Group": name, "Measure": "Selection rate",
                           "Rate": g.get("selection_rate")})
    if chart_rows:
        long = pd.DataFrame(chart_rows).dropna(subset=["Rate"])
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("Group:N", title=None, sort=list(groups.keys())),
            xOffset="Measure:N",
            y=alt.Y("Rate:Q", title="Rate", axis=alt.Axis(format="%")),
            color=alt.Color("Measure:N", title=None,
                            scale=alt.Scale(domain=["Default rate", "Selection rate"],
                                            range=[charts.MUTED, charts.ACCENT])),
            tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("Measure:N"),
                     alt.Tooltip("Rate:Q", format=".1%")],
        )
        charts.show(chart, height=260)
        st.caption(
            "Default rate = observed positives in the group; selection rate = "
            "share the model flags at the operating threshold."
            + (f" Low-support group(s) omitted from the chart: {', '.join(dropped)}."
               if dropped else "")
        )

    rows = []
    for name, g in groups.items():
        rows.append({
            "Group": name, "n": g.get("n"),
            "Default rate": _rate(g.get("positive_rate")),
            "Selection rate": _rate(g.get("selection_rate")),
            "ROC-AUC": num(g.get("roc_auc"), 4),
            "Recall (TPR)": _rate(g.get("recall_tpr")),
            "FNR": _rate(g.get("false_negative_rate")),
            "FPR": _rate(g.get("false_positive_rate")),
            "Precision": _rate(g.get("precision")),
            "Support": "low" if g.get("low_support") else "ok",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    _disparities(disparity)


def _fmt_metric(metric: str, value: Any) -> str:
    """ROC-AUC is a score (4dp); the rest are rates (percent)."""
    if metric == "roc_auc":
        return num(value, 4)
    return _rate(value)


def _disparities(disparity: dict[str, Any]) -> None:
    if not disparity:
        return
    st.markdown("**Disparity ratios** — smallest group value ÷ largest")
    rows = []
    for metric, d in disparity.items():
        ratio = d.get("ratio_min_over_max")
        flag = ""
        if isinstance(ratio, (int, float)) and not math.isnan(ratio):
            flag = "⚠️ review" if ratio < _FOUR_FIFTHS else "✓"
        rows.append({
            "Metric": _METRIC_LABELS.get(metric, metric),
            "Lowest": f"{d.get('min_group')} ({_fmt_metric(metric, d.get('min_value'))})",
            "Highest": f"{d.get('max_group')} ({_fmt_metric(metric, d.get('max_value'))})",
            "Ratio": num(ratio, 3),
            "Four-fifths": flag,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "A ratio of 1.00 is parity. The informal four-fifths rule flags ratios "
        "below 0.80 (⚠️) for human review. For error rates (FNR/FPR) a low ratio "
        "means the burden of mistakes falls unevenly across groups."
    )


