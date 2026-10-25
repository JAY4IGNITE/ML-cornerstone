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
import math

import streamlit as st

from loan_risk.config import Config, load_config
from loan_risk.serving import ModelService

TRAIN_HINT = (
    "Train the pipeline (`python -m loan_risk.pipeline.run`) so the model "
    "artifacts exist, then reload this page."
)


@st.cache_resource(show_spinner="Loading model…", max_entries=2)
def _cached_service(signature: tuple, _cfg: Config) -> ModelService:
    return ModelService(_cfg)


def get_service() -> ModelService:
    """Reload when artifacts appear or are replaced, including after training."""
    cfg = load_config()
    directory = cfg.path("artifacts")
    paths = sorted(p for p in directory.rglob("*") if p.is_file()) if directory.exists() else []
    signature = (str(directory), repr(cfg.raw), tuple(
        (str(path.relative_to(directory)), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    ))
    return _cached_service(signature, cfg)


get_service.clear = _cached_service.clear


# ---- formatting (parity with format.ts) ------------------------------------
def pct(x: Optional[float], digits: int = 1) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{value * 100:.{digits}f}%" if math.isfinite(value) else "—"


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
        return num(x, 4)
    except (TypeError, ValueError):
        return "—"


# ---- risk bands (parity with format.ts colours) ----------------------------
_RISK_BAND_COLOR = {
    "Low": "#059669",       # emerald
    "Moderate": "#d97706",  # amber
    "Elevated": "#ea580c",  # orange
    "High": "#dc2626",      # red
    "Critical": "#991b1b",  # deep red
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
        load_error = getattr(service, "load_error", None)
        err_msg = f" ({load_error})" if load_error else ""
        st.warning(f"**Model or data not ready**{err_msg} — " + TRAIN_HINT)
        st.stop()


def synthetic_warning(is_synthetic: bool, message: str | None, *,
                      title: str, fallback: str) -> None:
    """Amber banner shown on a synthetic-data model. `message` is the API/metadata
    synthetic_warning when present; otherwise the page-specific `fallback`."""
    if is_synthetic:
        st.warning(f"**{title}** — {message or fallback}")


# ---- global styling --------------------------------------------------------
# A theme would normally live in .streamlit/config.toml; this deployment cannot
# create that directory, so we inject an equivalent, conservative style pass
# once per rerun from the app chrome. Selectors target stable Streamlit testids
# and degrade to no-ops if a future version renames them.
_BASE_STYLES = """
<style>
  .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1280px; }
  h1 { font-weight: 750; letter-spacing: -0.025em; }
  h2, h3 { font-weight: 650; letter-spacing: -0.015em; }
  hr { margin: 1.25rem 0; opacity: 0.25; }
  [data-testid="stMetric"] {
    background: var(--secondary-background-color); border: 1px solid var(--border-color); border-radius: 0.75rem;
    padding: 0.9rem 1.1rem 0.8rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  [data-testid="stMetricLabel"] p { font-weight: 600; opacity: 0.75; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
  [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-weight: 750; font-size: 1.6rem; }
  [data-testid="stDataFrame"] { border-radius: 0.5rem; border: 1px solid var(--border-color); }
  a { color: var(--primary-color); text-decoration: none; }
  a:hover { text-decoration: underline; }
  [data-testid="baseButton-primary"], .stButton>button[kind="primary"] {
    background: var(--primary-color); border-color: var(--primary-color);
    font-weight: 650; letter-spacing: 0.02em; padding: 0.55rem 1.2rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12); border-radius: 0.5rem;
  }
  .fin-card {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
  }
  .ratio-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
  }
</style>
"""


def inject_base_styles() -> None:
    """Apply the shared, professional style pass. Call once from the app chrome
    before rendering any page. Idempotent per rerun."""
    st.markdown(_BASE_STYLES, unsafe_allow_html=True)
