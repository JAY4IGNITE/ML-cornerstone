"""Reproducible, leakage-aware train/val/test splitting (03_ML_REQUIREMENTS.md).

- Fixed seed from config for reproducibility.
- Stratified on the target when enabled (handles class imbalance).
- Removes exact duplicate rows before splitting and reports the count.
- Rejects conflicting applicant records instead of silently choosing an outcome.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import Config


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    diagnostics: dict


def make_splits(df: pd.DataFrame, cfg: Config) -> SplitResult:
    target = cfg.target
    id_col = cfg.id_column
    seed = cfg.seed
    test_size = float(cfg["split"]["test_size"])
    val_size = float(cfg["split"]["val_size"])
    stratify_on = cfg["split"]["stratify"]

    diagnostics: dict = {"n_input_rows": int(len(df))}

    if df.empty or target not in df.columns or id_col not in df.columns:
        raise ValueError("Splitting requires nonempty data with target and applicant ID columns.")
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("Test and validation fractions must be strictly between zero and one.")
    missing_ids = df[id_col].isna() | df[id_col].astype(str).str.strip().eq("")
    if missing_ids.any():
        raise ValueError("Applicant IDs must not be missing or blank.")
    if df[target].isna().any() or not df[target].isin([0, 1]).all():
        raise ValueError("Target must contain only nonmissing binary values 0 and 1.")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if df[id_col].duplicated().any():
        raise ValueError("Conflicting records for the same applicant ID require resolution.")
    diagnostics["dropped_duplicate_ids"] = before - len(df)
    diagnostics["dropped_exact_duplicates"] = before - len(df)
    if df[target].nunique() != 2:
        raise ValueError("Target must contain both classes before splitting.")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    strat = y if stratify_on else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    strat_tmp = y_tmp if stratify_on else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size, random_state=seed, stratify=strat_tmp
    )
    if any(y_part.nunique() != 2 for y_part in (y_train, y_val, y_test)):
        raise ValueError("Every split must contain both classes; more labeled records are needed.")

    diagnostics.update({
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "val_positive_rate": round(float(y_val.mean()), 4),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "stratified": bool(stratify_on),
        "seed": seed,
    })
    return SplitResult(X_train, X_val, X_test, y_train, y_val, y_test, diagnostics)
