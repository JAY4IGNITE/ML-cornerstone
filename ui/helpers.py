"""Shared Streamlit helpers — the contract every page module builds on.

Import surface for pages:
    from ui.helpers import (
        get_service, pct, num, metric_value, page_header, stat_cards,
        risk_band_color, synthetic_warning, require_model, TRAIN_HINT,
    )

Design notes:
  * `get_service()` returns a process-cached ModelService (model loaded once).
  * Formatting mirrors the old frontend/src/lib/format.ts exactly.
  * `require_model()` reproduces the old ErrorState "Model or data not ready"
    gate: it renders the message + train hint and halts the page via st.stop().
  * `synthetic_warning()` reproduces the per-page amber banner; each page passes
    its own title + fallback string (kept verbatim from the React pages).
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import streamlit as st

from loan_risk.serving import ModelService

TRAIN_HINT = (
    "Train the pipeline (`python -m loan_risk.pipeline.run`) so the model "
    "artifacts exist, then reload this page."
)


@st.cache_resource(show_spinner="Loading model…")
def get_service() -> ModelService:
    """One ModelService per process (model + metadata + schema loaded once)."""
    return ModelService()


# ---- formatting (parity with format.ts) ------------------------------------
def pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}%"


def num(x: Optional[float], digits: int = 3) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "—"
    if xf != xf or xf in (float("inf"), float("-inf")):  # NaN / inf
        return "—"
    return f"{xf:.{digits}f}"


def metric_value(x: Any) -> str:
    """ROC/PR-AUC may arrive as a {'error': ...} object on a single-class split."""
    if x is None:
        return "—"
    if isinstance(x, dict):
        return "n/a"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "—"


# ---- risk bands (parity with format.ts colours) ----------------------------
_RISK_BAND_COLOR = {
    "Low": "#059669",       # emerald
    "Moderate": "#d97706",  # amber
    "Elevated": "#ea580c",  # orange
    "High": "#dc2626",      # red
}


def risk_band_color(band: str) -> str:
    return _RISK_BAND_COLOR.get(band, "#475569")  # slate default


def risk_band_badge(band: str) -> str:
    """A small coloured HTML pill for a risk band (use with st.markdown/unsafe)."""
    color = risk_band_color(band)
    return (
        f"<span style='background:{color}1a;color:{color};border:1px solid "
        f"{color}55;border-radius:999px;padding:2px 10px;font-size:0.8rem;"
        f"font-weight:600'>{band} risk</span>"
    )


# ---- layout helpers --------------------------------------------------------
def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def stat_cards(items: Iterable[tuple[str, Any, str | None]]) -> None:
    """Render a KPI row of st.metric tiles: iterable of (label, value, help)."""
    items = list(items)
    cols = st.columns(len(items)) if items else []
    for col, (label, value, sub) in zip(cols, items):
        with col:
            st.metric(label, value, help=sub if sub else None)
            if sub:
                st.caption(sub)


def require_model(service: ModelService) -> None:
    """Gate a page that needs a loaded model. Reproduces the old ErrorState
    'Model or data not ready' path: render the notice + train hint, then stop."""
    if not service.available:
        st.warning("**Model or data not ready** — " + TRAIN_HINT)
        st.stop()


def synthetic_warning(is_synthetic: bool, message: str | None, *,
                      title: str, fallback: str) -> None:
    """Amber banner shown on a synthetic-data model. `message` is the API/metadata
    synthetic_warning when present; otherwise the page-specific `fallback`."""
    if is_synthetic:
        st.warning(f"**{title}** — {message or fallback}")
