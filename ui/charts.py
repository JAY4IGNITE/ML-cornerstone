"""Shared Altair chart styling — one professional look across every page.

Altair ships with Streamlit (no new dependency). Pages build their own
encodings (they differ too much to hide behind a generic wrapper) but route the
finished chart through :func:`show`, so fonts, axis colours and gridlines stay
identical everywhere. Palette constants are exported for the rare inline colour.

IMPORTANT: every chart here is fed a *pre-aggregated* frame. Vega serialises the
whole data table into the page, so charts must never receive the raw 300k-row
dataset — aggregate in pandas first, then hand a small summary frame to Altair.
"""
from __future__ import annotations

import altair as alt
import streamlit as st

# Palette — mirrors the CSS accent in ui.helpers and the risk-band colours.
ACCENT = "#2563eb"       # primary blue (selected / emphasis)
ACCENT_SOFT = "#93c5fd"  # light blue (secondary series)
MUTED = "#94a3b8"        # slate (non-highlighted bars)
INK = "#334155"          # axis + label text
GRID = "#e2e8f0"         # gridlines / axis domain
POS = "#dc2626"          # red — "worse"/risk-increasing
NEG = "#059669"          # emerald — "better"/risk-decreasing
AMBER = "#d97706"        # reference lines / thresholds


def finalize(chart: alt.Chart, height: int = 320) -> alt.Chart:
    """Apply the shared styling pass. Call once on the *final* (possibly layered)
    chart — configure_* is only valid at the top level."""
    return (
        chart.properties(height=height)
        .configure_axis(
            labelColor=INK, titleColor=INK, gridColor=GRID, domainColor=GRID,
            labelFontSize=12, titleFontSize=12, titleFontWeight=600,
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelColor=INK, titleColor=INK, labelFontSize=12)
    )


def show(chart: alt.Chart, height: int = 320) -> None:
    """Style and render a chart at full container width."""
    st.altair_chart(finalize(chart, height), use_container_width=True)
