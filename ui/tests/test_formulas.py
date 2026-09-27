"""Tests for the metric-formula registry (``ui/formulas.py``).

Pure data + a light render-tolerance check; no model or dataset needed.
"""
from __future__ import annotations

import pytest

from ui.formulas import (
    CONFUSION_LEGEND,
    METRIC_DEFINITIONS,
    formula,
    render_metric_definitions,
)

# Every key any page passes to render_metric_definitions must resolve, or that
# page silently renders nothing where a formula should be. Keep this in sync
# with the keys referenced across ui/pages/*.py.
REFERENCED_KEYS = {
    "roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "accuracy",
    "fpr", "fnr", "risk_score", "scale_pos_weight", "cv_mean",
    "point_biserial", "disparity_ratio", "group_rate",
}


def test_registry_non_empty():
    assert isinstance(METRIC_DEFINITIONS, dict)
    assert METRIC_DEFINITIONS


def test_every_entry_is_a_triple_of_nonempty_strings():
    for key, entry in METRIC_DEFINITIONS.items():
        assert isinstance(entry, tuple) and len(entry) == 3, key
        title, latex, reading = entry
        for part in (title, latex, reading):
            assert isinstance(part, str) and part.strip(), key


def test_all_referenced_keys_exist():
    missing = REFERENCED_KEYS - set(METRIC_DEFINITIONS)
    assert not missing, f"pages reference undefined metric keys: {sorted(missing)}"


@pytest.mark.parametrize("key", sorted(REFERENCED_KEYS))
def test_referenced_key_present(key):
    assert key in METRIC_DEFINITIONS


def test_render_skips_unknown_keys_without_raising():
    # Mix real + bogus keys; the contract is that unknown ones are skipped.
    render_metric_definitions(["roc_auc", "definitely_not_a_metric", "f1"])


def test_render_empty_iterable_is_noop():
    render_metric_definitions([])


def test_formula_renders_without_caption():
    formula(r"x = y")
    formula(r"x = y", caption="a caption")


def test_confusion_legend_is_descriptive_text():
    assert isinstance(CONFUSION_LEGEND, str)
    for token in ("TN", "FP", "FN", "TP"):
        assert token in CONFUSION_LEGEND
