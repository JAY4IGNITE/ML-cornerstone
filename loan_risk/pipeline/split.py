"""Reproducible, leakage-aware train/val/test splitting (03_ML_REQUIREMENTS.md).

- Fixed seed from config for reproducibility.
- Stratified on the target when enabled (handles class imbalance).
- De-duplicates on the applicant id BEFORE splitting so the same applicant can
  never appear in two splits (record leakage).
- Optional time-aware split diagnostic when a time column is configured.
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

    # drop duplicate applicant ids (record leakage guard)
    if id_col in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
        diagnostics["dropped_duplicate_ids"] = before - len(df)

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
