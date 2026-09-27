"""Unit tests for the pure UI helper functions that carry real logic:
``analytics._age_years`` / ``analytics._resolve`` and ``performance._base_rate``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ui.pages.analytics import _age_years, _resolve
from ui.pages.performance import _base_rate


# ---- _age_years -----------------------------------------------------------
def test_age_years_prefers_canonical_column():
    df = pd.DataFrame({"AGE_YEARS": [25, 40, 63]})
    out = _age_years(df)
    assert list(out) == [25, 40, 63]


def test_age_years_derives_from_days_birth():
    # DAYS_BIRTH is negative days-since-birth; -365.25*40 == exactly 40 years.
    df = pd.DataFrame({"DAYS_BIRTH": [-365.25 * 40, -3650]})
    out = _age_years(df)
    assert out.iloc[0] == pytest.approx(40.0)
    assert out.iloc[1] == pytest.approx(9.9931, abs=1e-3)


def test_age_years_none_when_no_age_column():
    assert _age_years(pd.DataFrame({"SOMETHING_ELSE": [1, 2]})) is None


# ---- _base_rate (confusion-matrix positive rate) --------------------------
def test_base_rate_uses_explicit_n():
    test = {"n": 61503, "confusion_matrix": {"tn": 56481, "fp": 57, "fn": 4876, "tp": 89}}
    assert _base_rate(test) == pytest.approx(4965 / 61503, abs=1e-6)


def test_base_rate_derives_n_from_matrix_when_absent():
    cm = {"tn": 56481, "fp": 57, "fn": 4876, "tp": 89}
    assert _base_rate({"confusion_matrix": cm}) == pytest.approx(4965 / 61503, abs=1e-6)


def test_base_rate_none_on_empty():
    assert _base_rate({}) is None
    assert _base_rate({"confusion_matrix": {}}) is None


# ---- _resolve (dataset path selection from config) ------------------------
def test_resolve_returns_synthetic_when_only_synthetic_exists(synthetic_raw_df):
    # synthetic_raw_df fixture guarantees the synthetic file is on disk; the
    # hermetic temp dir never contains the real raw CSV.
    from ui.helpers import get_service
    cfg = get_service().cfg
    path, is_synthetic = _resolve()
    assert is_synthetic is True
    assert path is not None
    assert path.endswith(cfg["dataset"]["synthetic_file"])


def test_resolve_contract_types():
    path, is_synthetic = _resolve()
    assert isinstance(is_synthetic, bool)
    assert path is None or isinstance(path, str)
    if path is None:
        assert is_synthetic is False
