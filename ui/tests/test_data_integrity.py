"""Reject ambiguous records before they can affect model evaluation."""
import numpy as np
import pandas as pd
import pytest

from loan_risk.data.validate import validate_raw
from loan_risk.pipeline.split import make_splits


def test_binary_float_targets_are_valid(synthetic_raw_df, cfg):
    raw = synthetic_raw_df.head(200).copy()
    raw[cfg.target] = raw[cfg.target].astype(float)
    assert not validate_raw(raw, cfg).has_errors()


@pytest.mark.parametrize("target", [0, 1, 0.5, np.nan])
def test_unusable_target_distribution_is_rejected(synthetic_raw_df, cfg, target):
    raw = synthetic_raw_df.head(200).copy()
    raw[cfg.target] = target
    assert validate_raw(raw, cfg).has_errors()


def test_empty_dataset_is_rejected(synthetic_raw_df, cfg):
    assert validate_raw(synthetic_raw_df.iloc[:0], cfg).has_errors()


@pytest.mark.parametrize("identifier", [np.nan, "", "   "])
def test_missing_applicant_id_is_rejected(synthetic_raw_df, cfg, identifier):
    raw = synthetic_raw_df.head(200).copy()
    raw[cfg.id_column] = raw[cfg.id_column].astype(object)
    raw.loc[raw.index[0], cfg.id_column] = identifier
    assert validate_raw(raw, cfg).has_errors()


def test_identical_duplicates_are_reported_and_removed(standardized_df, cfg):
    frame = standardized_df.head(200).copy()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = make_splits(duplicate, cfg)
    assert result.diagnostics["dropped_duplicate_ids"] == 1
    assert sum(len(x) for x in [result.y_train, result.y_val, result.y_test]) == len(frame)


def test_raw_exact_duplicates_do_not_block_training(synthetic_raw_df, cfg):
    raw = synthetic_raw_df.head(200)
    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    report = validate_raw(duplicate, cfg)
    assert not report.has_errors()
    assert any(c.name == "duplicate_ids" and c.status == "warn" for c in report.checks)


@pytest.mark.parametrize("column", ["TARGET", "AMT_CREDIT"])
def test_conflicting_duplicate_applicants_cannot_be_silently_discarded(standardized_df, cfg, column):
    frame = standardized_df.head(200).copy()
    duplicate = frame.iloc[[0]].copy()
    duplicate[column] = 1 - duplicate[column] if column == cfg.target else duplicate[column] + 1
    frame = pd.concat([frame, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="[Cc]onflicting"):
        make_splits(frame, cfg)


@pytest.mark.parametrize("mutation", ["missing_id", "missing_id_column", "fractional_target", "one_class"])
def test_split_enforces_integrity_even_without_raw_validation(standardized_df, cfg, mutation):
    frame = standardized_df.head(200).copy()
    if mutation == "missing_id":
        frame.loc[frame.index[0], cfg.id_column] = np.nan
    elif mutation == "missing_id_column":
        frame = frame.drop(columns=[cfg.id_column])
    elif mutation == "fractional_target":
        frame[cfg.target] = frame[cfg.target].astype(float)
        frame.loc[frame.index[0], cfg.target] = 0.7
    else:
        frame[cfg.target] = 0
    with pytest.raises(ValueError):
        make_splits(frame, cfg)
