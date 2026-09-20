"""Data unit/integration tests (07_TESTING_CHECKLIST.md — "Data Tests").

Covers schema validation, missing-target detection, invalid-type detection,
missing-value reporting, duplicate row/identifier detection, leakage scan,
target-distribution check, the DAYS_EMPLOYED sentinel handling, and the
canonical-column contract produced by ``standardize``.

Uses the shared, read-only fixtures from ``tests/conftest.py`` (``cfg``,
``synthetic_raw_df``, ``standardized_df``). ``validate_raw`` and ``standardize``
do not mutate their input, so the session fixtures are passed directly; any test
that needs to mutate the frame works on an explicit ``.copy()``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from loan_risk.data.ingestion import (
    DAYS_EMPLOYED_SENTINEL,
    EMPLOYMENT_MISSING_FLAG,
    RAW_TARGET,
)
from loan_risk.data.ingestion import standardize
from loan_risk.data.validate import ValidationReport, validate_raw
from loan_risk.schema import categorical_names, numeric_names


def _checks(report: ValidationReport) -> dict:
    """Index a report's checks by name for easy assertions."""
    return {c.name: c for c in report.checks}


# ---------------------------------------------------------------------------
# Schema / required-column validation
# ---------------------------------------------------------------------------
def test_required_columns_present(synthetic_raw_df, cfg):
    rep = validate_raw(synthetic_raw_df, cfg, source_file="synthetic")
    checks = _checks(rep)
    assert "required_columns" in checks
    assert checks["required_columns"].status == "pass"


def test_valid_synthetic_has_no_errors(synthetic_raw_df, cfg):
    rep = validate_raw(synthetic_raw_df, cfg)
    assert isinstance(rep, ValidationReport)
    # Clean synthetic data may raise warnings (missingness) but no hard errors.
    assert not rep.has_errors()


# ---------------------------------------------------------------------------
# Missing target detection
# ---------------------------------------------------------------------------
def test_missing_target_detection(synthetic_raw_df, cfg):
    raw = synthetic_raw_df.drop(columns=[RAW_TARGET])  # returns a copy
    rep = validate_raw(raw, cfg)
    assert rep.has_errors()
    checks = _checks(rep)
    assert "target_present" in checks
    assert checks["target_present"].status == "fail"


# ---------------------------------------------------------------------------
# Invalid type detection
# ---------------------------------------------------------------------------
def test_invalid_type_detection(synthetic_raw_df, cfg):
    raw = synthetic_raw_df.head(500).copy()
    raw["AMT_INCOME_TOTAL"] = raw["AMT_INCOME_TOTAL"].astype(object)
    raw.loc[raw.index[0], "AMT_INCOME_TOTAL"] = "not_a_number"
    rep = validate_raw(raw, cfg)
    checks = _checks(rep)
    assert checks["numeric_types"].status == "warn"
    assert "AMT_INCOME_TOTAL" in checks["numeric_types"].data.get("columns", {})


# ---------------------------------------------------------------------------
# Missing-value report
# ---------------------------------------------------------------------------
def test_missing_value_report_present(synthetic_raw_df, cfg):
    rep = validate_raw(synthetic_raw_df, cfg)
    checks = _checks(rep)
    assert "missing_values" in checks
    # Synthetic data injects missingness (EXT_SOURCE / OCCUPATION), so the
    # report must enumerate the affected columns.
    assert checks["missing_values"].data.get("counts")


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
def test_duplicate_row_detection(synthetic_raw_df, cfg):
    raw = synthetic_raw_df.head(200).copy()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)  # exact dup row
    rep = validate_raw(raw, cfg)
    checks = _checks(rep)
    assert checks["duplicate_rows"].status == "warn"
    assert checks["duplicate_rows"].data.get("count", 0) >= 1


def test_duplicate_identifier_detection(synthetic_raw_df, cfg):
    raw = synthetic_raw_df.head(200).copy()
    dup = raw.iloc[[0]].copy()
    dup["AMT_CREDIT"] = dup["AMT_CREDIT"] + 12345  # duplicate id, not a full row
    raw = pd.concat([raw, dup], ignore_index=True)
    rep = validate_raw(raw, cfg)
    checks = _checks(rep)
    assert checks["duplicate_ids"].status == "fail"
    assert checks["duplicate_ids"].data.get("count", 0) >= 1


# ---------------------------------------------------------------------------
# Leakage scan
# ---------------------------------------------------------------------------
def test_leakage_scan_check_exists(synthetic_raw_df, cfg):
    rep = validate_raw(synthetic_raw_df, cfg)
    checks = _checks(rep)
    assert "leakage_scan" in checks
    assert checks["leakage_scan"].status in {"pass", "warn"}


def test_leakage_scan_flags_post_outcome_column(synthetic_raw_df, cfg):
    raw = synthetic_raw_df.head(100).copy()
    raw["LOAN_REPAID_FLAG"] = 1  # obvious post-outcome column
    rep = validate_raw(raw, cfg)
    checks = _checks(rep)
    assert checks["leakage_scan"].status == "warn"
    flagged = checks["leakage_scan"].data.get("columns", [])
    assert any("REPAID" in str(c).upper() for c in flagged)


# ---------------------------------------------------------------------------
# Target distribution
# ---------------------------------------------------------------------------
def test_target_distribution_check(synthetic_raw_df, cfg):
    rep = validate_raw(synthetic_raw_df, cfg)
    checks = _checks(rep)
    assert "target_distribution" in checks
    tc = checks["target_distribution"]
    assert tc.status == "pass"
    assert 0.0 < tc.data.get("positive_rate", 0.0) < 1.0
    assert set(tc.data.get("counts", {}).keys()) <= {"0", "1"}


# ---------------------------------------------------------------------------
# DAYS_EMPLOYED sentinel handling + canonical columns (standardize)
# ---------------------------------------------------------------------------
def test_days_employed_sentinel_handling(synthetic_raw_df, cfg):
    raw = synthetic_raw_df.head(4).copy().reset_index(drop=True)
    raw.loc[0, "DAYS_EMPLOYED"] = DAYS_EMPLOYED_SENTINEL  # pensioner/unemployed
    raw.loc[1, "DAYS_EMPLOYED"] = -3652                   # ~10 years employed
    std = standardize(raw, cfg)

    # sentinel row -> flagged missing, EMPLOYMENT_YEARS becomes NaN
    assert std.loc[0, EMPLOYMENT_MISSING_FLAG] == 1
    assert pd.isna(std.loc[0, "EMPLOYMENT_YEARS"])

    # ordinary row -> flag 0, negative day-count converted to positive years
    assert std.loc[1, EMPLOYMENT_MISSING_FLAG] == 0
    assert std.loc[1, "EMPLOYMENT_YEARS"] == pytest.approx(3652 / 365.25, rel=1e-3)


def test_standardize_produces_canonical_columns(standardized_df):
    cols = set(standardized_df.columns)
    for expected in ("AGE_YEARS", "EMPLOYMENT_YEARS", EMPLOYMENT_MISSING_FLAG):
        assert expected in cols
    # raw day-count columns must be dropped after conversion
    for raw_only in ("DAYS_BIRTH", "DAYS_EMPLOYED"):
        assert raw_only not in cols
    # every schema numeric + categorical column is present
    for c in numeric_names() + categorical_names():
        assert c in cols


def test_standardize_age_conversion(synthetic_raw_df, cfg):
    raw = synthetic_raw_df.head(3).copy().reset_index(drop=True)
    raw.loc[0, "DAYS_BIRTH"] = -15000  # negative day-count from application date
    std = standardize(raw, cfg)
    assert std.loc[0, "AGE_YEARS"] == pytest.approx(15000 / 365.25, rel=1e-6)
