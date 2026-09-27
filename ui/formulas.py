"""Shared analytical content — metric formulas and small render helpers.

One module holds every metric definition (name, LaTeX, plain-language reading)
so the whole dashboard explains a term identically: pages import
``render_metric_definitions([...])`` to drop a consistent "how to read this"
block beside their charts, or ``METRIC_DEFINITIONS`` to look one up.

Nothing here computes on data; it only *describes* metrics the pipeline already
recorded, keeping the UI honest (no fabricated numbers).
"""
from __future__ import annotations

from typing import Iterable

import streamlit as st

# key -> (display title, LaTeX body, plain-language interpretation)
METRIC_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "roc_auc": (
        "ROC-AUC",
        r"\text{AUC} = \Pr\!\big(\hat p(x^{+}) > \hat p(x^{-})\big)",
        "Probability the model scores a random defaulter above a random "
        "non-defaulter. 0.5 = chance, 1.0 = perfect ranking. Threshold-free, so "
        "it measures ranking quality regardless of the operating cut-off.",
    ),
    "pr_auc": (
        "PR-AUC (average precision)",
        r"\text{AP} = \sum_k \big(R_k - R_{k-1}\big)\,P_k",
        "Area under the precision–recall curve. The right summary for rare "
        "positives: its no-skill baseline equals the positive rate (≈ 0.081 "
        "here), so anything above that is lift over guessing.",
    ),
    "brier": (
        "Brier score",
        r"\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}\big(\hat p_i - y_i\big)^2",
        "Mean squared error of the predicted probabilities (lower is better). "
        "Rewards calibration — probabilities matching observed frequencies — not "
        "just correct ranking.",
    ),
    "precision": (
        "Precision",
        r"\text{Precision} = \frac{TP}{TP + FP}",
        "Of the applicants the model flags, the share that truly default. Falls "
        "as the threshold drops (more false alarms).",
    ),
    "recall": (
        "Recall (TPR / sensitivity)",
        r"\text{Recall} = \frac{TP}{TP + FN}",
        "Of the applicants who truly default, the share the model catches. "
        "Rises as the threshold drops.",
    ),
    "f1": (
        "F1 score",
        r"F_1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}"
        r"{\text{Precision}+\text{Recall}}",
        "Harmonic mean of precision and recall — the single balance point used "
        "to pick the operating threshold on the validation set.",
    ),
    "accuracy": (
        "Accuracy",
        r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}",
        "Share of all predictions that are correct. Misleading under imbalance: "
        "always predicting 'repays' already scores ≈ 0.92 here, which is why "
        "accuracy is never the selection metric.",
    ),
    "fpr": (
        "False-positive rate",
        r"\text{FPR} = \frac{FP}{FP + TN}",
        "Share of non-defaulters wrongly flagged.",
    ),
    "fnr": (
        "False-negative rate",
        r"\text{FNR} = \frac{FN}{FN + TP}",
        "Share of true defaulters the model misses (1 − recall).",
    ),
    "risk_score": (
        "Risk score",
        r"\text{score} = \operatorname{round}\!\big(100 \cdot \hat p\big)",
        "A transparent 0–100 restatement of the predicted probability. "
        "Monotonic in probability; it adds no information of its own.",
    ),
    "scale_pos_weight": (
        "Imbalance weighting",
        r"\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}}",
        "Positives are up-weighted (XGBoost) — or class_weight='balanced' for "
        "the other models — so the ~8% default class is not drowned out.",
    ),
    "cv_mean": (
        "Cross-validated selection score",
        r"\bar{s} = \frac{1}{K}\sum_{k=1}^{K} s_k,\qquad "
        r"\sigma = \sqrt{\tfrac{1}{K}\sum_{k}\big(s_k-\bar{s}\big)^2}",
        "Each model is scored on K held-out folds; selection uses the mean "
        "(reporting std for stability), never a single lucky split.",
    ),
    "point_biserial": (
        "Point-biserial correlation",
        r"r_{pb} = \frac{\bar{x}_1 - \bar{x}_0}{s_x}\,"
        r"\sqrt{\frac{n_1 n_0}{n^2}}",
        "Pearson correlation between a numeric feature and the binary target. "
        "Sign shows direction; magnitude captures linear association only.",
    ),
    "disparity_ratio": (
        "Disparity ratio",
        r"\text{ratio} = \frac{\min_g m_g}{\max_g m_g}",
        "Smallest group value over the largest. 1.0 = parity; the informal "
        "four-fifths rule flags ratios below 0.8 for review. Diagnostic only.",
    ),
    "group_rate": (
        "Group default / selection rate",
        r"\text{rate}_g = \frac{1}{n_g}\sum_{i \in g} y_i",
        "Mean of the binary target (default rate) — or of the flag decisions "
        "(selection rate) — within a group.",
    ),
}


def formula(latex: str, caption: str | None = None) -> None:
    """Render a centered formula with an optional caption underneath."""
    st.latex(latex)
    if caption:
        st.caption(caption)


def render_metric_definitions(keys: Iterable[str]) -> None:
    """Render a consistent (title, formula, plain-language reading) block for
    each requested metric key. Unknown keys are skipped silently."""
    for key in keys:
        entry = METRIC_DEFINITIONS.get(key)
        if not entry:
            continue
        title, latex, reading = entry
        st.markdown(f"**{title}**")
        st.latex(latex)
        st.caption(reading)


CONFUSION_LEGEND = (
    "TN = correct non-default · FP = false alarm · "
    "FN = missed default · TP = caught default."
)
