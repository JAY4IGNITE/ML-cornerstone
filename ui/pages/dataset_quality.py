"""Dataset Quality page — data-validation checks and the dataset manifest.

Everything here is read straight from the service's validation_report.json and
the dataset manifest.json (via ModelService). Nothing is recomputed in the UI;
we render only what the pipeline recorded, and known limitations come verbatim
from the manifest when it supplies them.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from loan_risk.serving import ModelService
from ui.helpers import (
    get_service,
    page_header,
    pct,
    require_model,
    stat_cards,
    synthetic_warning,
)

# Tri-state check status -> glanceable emoji (parity with the performance page's
# single-glyph convention, extended to pass/warn/fail).
_STATUS_EMOJI = {"pass": "✅", "warn": "⚠️", "fail": "❌"}

# Verbatim fallback shown only when the manifest records no limitations of its
# own — kept exactly as specified so the honest caveats never silently vanish.
_FALLBACK_LIMITATIONS = [
    "Synthetic or historical data may not reflect current populations.",
    "Class imbalance can bias precision/recall at a fixed threshold.",
    "Features are proxies; missing values are imputed and may distort "
    "individual estimates.",
    "No fairness certification — subgroup performance is not guaranteed.",
]


def render() -> None:
    service = get_service()
    require_model(service)  # standard "model or data not ready" gate

    manifest: dict[str, Any] = service.get_manifest() or {}
    report: dict[str, Any] = service.get_validation_report() or {}
    mi = service.model_info()

    page_header(
        "Dataset Quality",
        "Data-validation checks and the dataset manifest, including known "
        "limitations.",
    )

    synthetic_warning(
        mi.get("synthetic", False),
        mi.get("synthetic_warning"),
        title="Synthetic data",
        fallback="This dataset is synthetic. It approximates the structure of "
        "real credit data but must not be treated as a real lending "
        "population.",
    )

    _validation_summary(report, service)
    _validation_checks(report)
    _dataset_manifest(manifest)
    _known_limitations(manifest)


def _summary_counts(report: dict[str, Any], service: ModelService) -> dict[str, Any]:
    """Resolve pass/warn/fail counts defensively: the report's own summary
    block first, then a tally of its checks, then metrics.validation_summary."""
    summary = report.get("summary")
    if isinstance(summary, dict) and any(k in summary for k in ("pass", "warn", "fail")):
        return summary

    checks = report.get("checks")
    if isinstance(checks, list) and checks:
        counts = {"pass": 0, "warn": 0, "fail": 0}
        for c in checks:
            status = str((c or {}).get("status", "")).lower()
            if status in counts:
                counts[status] += 1
        return counts

    metrics = service.get_metrics() or {}
    vs = metrics.get("validation_summary")
    return vs if isinstance(vs, dict) else {}


def _validation_summary(report: dict[str, Any], service: ModelService) -> None:
    st.subheader("Validation summary")
    counts = _summary_counts(report, service)

    def _c(key: str) -> Any:
        v = counts.get(key)
        return v if v is not None else "—"

    stat_cards([
        ("Passed", _c("pass"), "Checks passed"),
        ("Warnings", _c("warn"), "Non-blocking"),
        ("Failed", _c("fail"), "Blocking issues"),
    ])


def _validation_checks(report: dict[str, Any]) -> None:
    st.subheader("Validation checks")
    checks = report.get("checks")
    if not (isinstance(checks, list) and checks):
        st.info("No validation checks are recorded.")
        return

    rows: list[dict[str, Any]] = []
    for c in checks:
        c = c or {}
        status = str(c.get("status", "")).lower()
        emoji = _STATUS_EMOJI.get(status, "")
        rows.append({
            "Check": c.get("name", "—"),
            # Emoji + word so the status reads at a glance and stays sortable.
            "Status": f"{emoji} {status}".strip() if status else "—",
            "Detail": c.get("detail", "—"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _fmt_int(value: Any) -> str:
    """Thousands-separated integer, falling back to the raw string if it is not
    an integer (never crash on an unexpected manifest value)."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _dataset_manifest(manifest: dict[str, Any]) -> None:
    st.subheader("Dataset manifest")
    if not manifest:
        st.info("No dataset manifest is recorded.")
        return

    # Key numbers as metric tiles — only those the manifest actually carries.
    tiles: list[tuple[str, Any, str | None]] = []
    if manifest.get("row_count") is not None:
        tiles.append(("Rows", _fmt_int(manifest["row_count"]), None))
    if manifest.get("column_count") is not None:
        tiles.append(("Columns", _fmt_int(manifest["column_count"]), None))
    if manifest.get("target_column") is not None:
        tiles.append(("Target", str(manifest["target_column"]), None))
    if manifest.get("positive_rate") is not None:
        tiles.append(("Positive rate", pct(manifest.get("positive_rate")), "Share of positive class"))
    if tiles:
        stat_cards(tiles)

    dist = manifest.get("target_distribution")
    if isinstance(dist, dict) and dist:
        st.caption("Class balance: " + " · ".join(f"{k} = {_fmt_int(v)}" for k, v in dist.items()))

    # Descriptive scalar fields as a Field/Value table — present keys only.
    field_labels = [
        ("dataset_name", "Dataset name"),
        ("source_kind", "Source kind"),
        ("source_url", "Source URL"),
        ("retrieval_or_generation_date", "Retrieved / generated"),
        ("primary_file", "Primary file"),
        ("license_and_usage", "License & usage"),
        ("target_meaning", "Target meaning"),
    ]
    rows = [
        {"Field": label, "Value": str(manifest[key])}
        for key, label in field_labels
        if manifest.get(key) is not None
    ]
    if rows:
        st.table(pd.DataFrame(rows))


def _known_limitations(manifest: dict[str, Any]) -> None:
    st.subheader("Known limitations")
    # Prefer manifest-supplied limitations/notes verbatim; else the fixed list.
    lims: Any = None
    for key in ("known_limitations", "limitations", "notes"):
        candidate = manifest.get(key)
        if isinstance(candidate, list) and candidate:
            lims = candidate
            break
    if not lims:
        lims = _FALLBACK_LIMITATIONS
    st.markdown("\n".join(f"- {item}" for item in lims))
