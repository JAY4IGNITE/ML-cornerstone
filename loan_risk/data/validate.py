"""Data validation — implements the checks in 04_DATASET_PROTOCOL.md and the
Data Tests in 07_TESTING_CHECKLIST.md. Produces a machine-readable report.

Runs against the RAW frame (before standardization) so it verifies the actual
source columns exist and match, and NEVER invents columns (Execution Rule #5).
Checks are non-fatal by default and recorded as pass/warn/fail with details;
`has_errors()` lets callers decide whether to proceed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from .ingestion import (
    DAYS_EMPLOYED_SENTINEL,
    RAW_CATEGORICAL,
    RAW_FEATURE_COLUMNS,
    RAW_ID,
    RAW_TARGET,
)
from ..schema import categories_for


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    source_file: str
    n_rows: int
    n_cols: int
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, **data: Any) -> None:
        self.checks.append(Check(name=name, status=status, detail=detail, data=data))

    def has_errors(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    def summary(self) -> dict[str, int]:
        out = {"pass": 0, "warn": 0, "fail": 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "summary": self.summary(),
            "has_errors": self.has_errors(),
            "checks": [asdict(c) for c in self.checks],
        }


def validate_raw(df: pd.DataFrame, cfg: Config, source_file: str = "") -> ValidationReport:
    rep = ValidationReport(source_file=source_file, n_rows=len(df), n_cols=df.shape[1])
    target = cfg.target
    sentinel = int(cfg["preprocessing"]["days_employed_anomaly"])

    # 1. required columns present
    required = [RAW_ID] + RAW_FEATURE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.add("required_columns", "fail",
                f"Missing required columns: {missing}", missing=missing)
    else:
        rep.add("required_columns", "pass", "All required raw columns present.")

    # 2. target present + distribution
    if target not in df.columns:
        rep.add("target_present", "fail", f"Target column '{target}' missing.")
    else:
        vc = df[target].value_counts(dropna=False).to_dict()
        vc = {str(k): int(v) for k, v in vc.items()}
        n_missing_t = int(df[target].isna().sum())
        pos = int((df[target] == 1).sum())
        rate = pos / len(df) if len(df) else 0.0
        status = "pass"
        detail = f"Target distribution {vc}; positive rate {rate:.4f}."
        if n_missing_t > 0:
            status = "fail"
            detail = f"Target has {n_missing_t} missing values. " + detail
        extra = set(vc) - {"0", "1"}
        if extra:
            status = "fail"
            detail = f"Target has unexpected values {extra}. " + detail
        rep.add("target_distribution", status, detail,
                counts=vc, positive_rate=rate, missing=n_missing_t)

    # 3. dtypes for numeric columns (coercible?)
    numeric_raw = [c for c in RAW_FEATURE_COLUMNS if c not in RAW_CATEGORICAL]
    bad_types: dict[str, int] = {}
    for c in numeric_raw:
        if c in df.columns:
            coerced = pd.to_numeric(df[c], errors="coerce")
            introduced = int(coerced.isna().sum() - df[c].isna().sum())
            if introduced > 0:
                bad_types[c] = introduced
    if bad_types:
        rep.add("numeric_types", "warn",
                f"Non-numeric entries coerced to NaN: {bad_types}", columns=bad_types)
    else:
        rep.add("numeric_types", "pass", "Numeric columns are numeric-coercible.")

    # 4. missing-value report
    miss = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    miss_pct = {c: round(v / len(df), 4) for c, v in miss.items()}
    rep.add("missing_values", "pass" if not miss else "warn",
            f"{len(miss)} columns contain missing values.",
            counts=miss, fraction=miss_pct)

    # 5. infinite values
    inf_cols = {}
    for c in numeric_raw:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
            n_inf = int(np.isinf(col).sum())
            if n_inf:
                inf_cols[c] = n_inf
    rep.add("infinite_values", "fail" if inf_cols else "pass",
            f"Infinite values: {inf_cols}" if inf_cols else "No infinite values.",
            columns=inf_cols)

    # 6. duplicate rows
    dup_rows = int(df.duplicated().sum())
    rep.add("duplicate_rows", "warn" if dup_rows else "pass",
            f"{dup_rows} fully-duplicated rows.", count=dup_rows)

    # 7. duplicate identifiers
    if RAW_ID in df.columns:
        dup_ids = int(df[RAW_ID].duplicated().sum())
        rep.add("duplicate_ids", "fail" if dup_ids else "pass",
                f"{dup_ids} duplicate {RAW_ID} values.", count=dup_ids)

    # 8. invalid numerical ranges (negative amounts etc.)
    range_issues: dict[str, int] = {}
    nonneg = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
              "CNT_FAM_MEMBERS", "CNT_CHILDREN"]
    for c in nonneg:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
            n_bad = int((col < 0).sum())
            if n_bad:
                range_issues[c] = n_bad
    for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
            n_bad = int(((col < 0) | (col > 1)).sum())
            if n_bad:
                range_issues[c] = n_bad
    rep.add("numeric_ranges", "fail" if range_issues else "pass",
            f"Out-of-range numeric values: {range_issues}" if range_issues
            else "Numeric ranges within expected bounds.", columns=range_issues)

    # 9. categorical values vs allowed set
    cat_issues: dict[str, list[str]] = {}
    for c in RAW_CATEGORICAL:
        if c in df.columns:
            allowed = set(categories_for(c))
            present = set(df[c].dropna().astype(str).unique())
            unexpected = sorted(present - allowed)
            if unexpected:
                cat_issues[c] = unexpected
    rep.add("categorical_values", "warn" if cat_issues else "pass",
            f"Unexpected categories: {cat_issues}" if cat_issues
            else "Categorical values within documented sets.", columns=cat_issues)

    # 10. DAYS_EMPLOYED sentinel prevalence (data-quality note, not an error)
    if "DAYS_EMPLOYED" in df.columns:
        col = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce")
        n_sent = int((col == sentinel).sum())
        rep.add("days_employed_sentinel", "pass",
                f"{n_sent} rows use the {sentinel} pensioner/unemployed sentinel "
                f"(handled as missing + indicator).", count=n_sent)

    # 11. leakage heuristic — flag any post-outcome-looking columns
    suspicious = [c for c in df.columns
                  if any(tok in c.upper() for tok in
                         ("PAID", "REPAID", "DEFAULTED", "OUTCOME", "LABEL",
                          "RESULT", "STATUS_FINAL"))
                  and c != target]
    rep.add("leakage_scan", "warn" if suspicious else "pass",
            f"Potential post-outcome columns to review: {suspicious}" if suspicious
            else "No obvious post-outcome columns among used features.",
            columns=suspicious)

    return rep


def write_report(rep: ValidationReport, cfg: Config, filename: str = "validation_report.json") -> Path:
    out_dir = cfg.path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rep.to_dict(), fh, indent=2)
    return path
