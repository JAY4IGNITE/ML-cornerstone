"""Smoke tests: every page's ``render()`` runs to completion with no exception.

Uses Streamlit's ``AppTest`` to execute each page in a headless script context
against the hermetic, freshly-trained synthetic model. A page that raises (bad
attribute access, unhandled ``None``, a chart built from the wrong shape) shows
up as a non-empty ``at.exception``. Pages that intentionally halt on missing
inputs (``st.stop``) are *not* exceptions and still pass.
"""
from __future__ import annotations

import pytest

from streamlit.testing.v1 import AppTest

# All 13 pages wired into streamlit_app.py's navigation.
PAGES = [
    "overview",
    "analytics",
    "assessment",
    "result",
    "explainability",
    "model_selection",
    "performance",
    "methodology",
    "feature_reference",
    "model_info",
    "fairness",
    "dataset_quality",
    "responsible_use",
]


def _render_script(module: str) -> str:
    return (
        "import importlib\n"
        f"importlib.import_module('ui.pages.{module}').render()\n"
    )


@pytest.mark.parametrize("module", PAGES)
def test_page_renders_without_exception(module, service_with_model, synthetic_raw_df):
    at = AppTest.from_string(_render_script(module), default_timeout=90).run()
    assert not at.exception, f"page '{module}' raised: {at.exception}"


def test_all_pages_are_covered():
    # Guard against a page module being added/removed without updating PAGES.
    from pathlib import Path
    pages_dir = Path(__file__).resolve().parents[1] / "pages"
    on_disk = {
        p.stem for p in pages_dir.glob("*.py")
        if p.stem != "__init__"
    }
    assert set(PAGES) == on_disk, (
        f"PAGES out of sync with ui/pages/: only-in-list={set(PAGES) - on_disk}, "
        f"only-on-disk={on_disk - set(PAGES)}"
    )
