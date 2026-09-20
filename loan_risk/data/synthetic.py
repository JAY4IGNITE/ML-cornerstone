"""Synthetic Home Credit-shaped data generator.

Purpose (04_DATASET_PROTOCOL.md explicitly allows synthetic data for "pipeline
stress testing"): produce a fixture with the SAME raw column layout as the real
``application_train.csv`` so the entire pipeline, API and tests run end-to-end
before/without the Kaggle download.

CRITICAL HONESTY RULE: data produced here is SYNTHETIC. Any metric computed on
it is a synthetic-data metric and must be labeled as such everywhere it appears
(manifest, evaluation report, model card). It is NOT a real-world result.

The generator builds a target that is a *noisy* function of realistic risk
signals (external scores, leverage, age, employment, missingness) so models can
learn a genuine—but synthetic—signal instead of noise. The relationship is
intentionally imperfect so metrics stay realistic (AUC well under 1.0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..data.ingestion import DAYS_EMPLOYED_SENTINEL

# Categories reused from the schema so synthetic values are always valid.
from ..schema import categories_for


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    n = int(cfg["synthetic"]["n_rows"])
    pos_rate = float(cfg["synthetic"]["positive_rate"])
    miss_ext = float(cfg["synthetic"]["missing_rate_ext_source"])
    miss_occ = float(cfg["synthetic"]["missing_rate_occupation"])

    df = pd.DataFrame()
    df["SK_ID_CURR"] = np.arange(100001, 100001 + n)

    # --- demographics ---
    age_years = rng.normal(43, 11, n).clip(21, 69)
    df["DAYS_BIRTH"] = (-age_years * 365.25).round().astype(int)

    # employment: a share are pensioners/unemployed -> sentinel value
    is_pensioner = rng.random(n) < 0.18
    emp_years = rng.gamma(shape=2.0, scale=3.0, size=n).clip(0, 45)
    days_emp = (-emp_years * 365.25).round().astype(int)
    days_emp = np.where(is_pensioner, DAYS_EMPLOYED_SENTINEL, days_emp)
    df["DAYS_EMPLOYED"] = days_emp

    df["CNT_CHILDREN"] = rng.poisson(0.5, n).clip(0, 12)
    df["CNT_FAM_MEMBERS"] = (df["CNT_CHILDREN"] + rng.integers(1, 3, n)).clip(1, 15)
    df["REGION_POPULATION_RELATIVE"] = rng.uniform(0.001, 0.07, n).round(6)

    # --- financials ---
    income = rng.lognormal(mean=11.9, sigma=0.45, size=n).clip(25_000, 2_000_000)
    df["AMT_INCOME_TOTAL"] = income.round(-2)
    # credit correlated with income + noise
    credit = (income * rng.uniform(1.0, 6.0, n)).clip(45_000, 3_500_000)
    df["AMT_CREDIT"] = credit.round(-3)
    df["AMT_GOODS_PRICE"] = (credit * rng.uniform(0.8, 1.0, n)).round(-3)
    # annuity ~ credit / term
    term = rng.uniform(12, 60, n)
    df["AMT_ANNUITY"] = (credit / term).clip(2_000, 300_000).round(-2)

    # --- external scores (predictive, lower = riskier) ---
    base = _sigmoid(rng.normal(0.3, 1.0, n))
    ext1 = (base + rng.normal(0, 0.15, n)).clip(0, 1)
    ext2 = (base + rng.normal(0, 0.15, n)).clip(0, 1)
    ext3 = (base + rng.normal(0, 0.15, n)).clip(0, 1)
    df["EXT_SOURCE_1"] = ext1.round(4)
    df["EXT_SOURCE_2"] = ext2.round(4)
    df["EXT_SOURCE_3"] = ext3.round(4)

    # --- categoricals (sampled from canonical schema categories) ---
    def pick(col: str, p=None) -> np.ndarray:
        cats = list(categories_for(col))
        return rng.choice(cats, size=n, p=p)

    df["NAME_CONTRACT_TYPE"] = pick("NAME_CONTRACT_TYPE", p=[0.9, 0.1])
    df["CODE_GENDER"] = pick("CODE_GENDER", p=[0.34, 0.655, 0.005])
    df["FLAG_OWN_CAR"] = pick("FLAG_OWN_CAR", p=[0.34, 0.66])
    df["FLAG_OWN_REALTY"] = pick("FLAG_OWN_REALTY", p=[0.69, 0.31])
    df["NAME_INCOME_TYPE"] = np.where(
        is_pensioner, "Pensioner",
        pick("NAME_INCOME_TYPE",
             p=[0.55, 0.10, 0.23, 0.0, 0.02, 0.03, 0.03, 0.04]),
    )
    df["NAME_EDUCATION_TYPE"] = pick(
        "NAME_EDUCATION_TYPE", p=[0.10, 0.71, 0.03, 0.15, 0.01])
    df["NAME_FAMILY_STATUS"] = pick(
        "NAME_FAMILY_STATUS", p=[0.18, 0.64, 0.10, 0.06, 0.019, 0.001])
    df["NAME_HOUSING_TYPE"] = pick(
        "NAME_HOUSING_TYPE", p=[0.88, 0.04, 0.04, 0.02, 0.01, 0.01])
    df["OCCUPATION_TYPE"] = pick("OCCUPATION_TYPE")

    # --- target: noisy function of real risk signals ---
    ext_mean = np.nanmean(np.vstack([ext1, ext2, ext3]), axis=0)
    credit_income = credit / income
    annuity_income = df["AMT_ANNUITY"].to_numpy() / income
    age_norm = (age_years - 21) / (69 - 21)
    emp_norm = np.where(is_pensioner, 0.0, np.clip(emp_years / 45, 0, 1))

    logit = (
        -1.15                         # base rate control
        - 1.9 * (ext_mean - 0.5)      # lower external score -> higher risk
        + 0.7 * (credit_income - 3.0) / 3.0
        + 0.9 * (annuity_income - 0.15) / 0.15
        - 0.4 * (age_norm - 0.5)      # younger slightly riskier
        - 0.35 * emp_norm             # longer employment -> lower risk
        + rng.normal(0, 1.9, n)       # irreducible noise -> realistic AUC (~0.7x),
                                      # mirroring the real dataset's difficulty
    )
    prob = _sigmoid(logit)
    # calibrate threshold so realized positive rate ~ configured pos_rate
    thresh = np.quantile(prob, 1 - pos_rate)
    df["TARGET"] = (prob >= thresh).astype(int)

    # --- inject realistic missingness AFTER target is set ---
    for col, rate in (("EXT_SOURCE_1", miss_ext), ("EXT_SOURCE_3", miss_ext),
                      ("EXT_SOURCE_2", miss_ext * 0.2)):
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan
    occ_mask = rng.random(n) < miss_occ
    df.loc[occ_mask, "OCCUPATION_TYPE"] = np.nan
    ann_mask = rng.random(n) < 0.005
    df.loc[ann_mask, "AMT_ANNUITY"] = np.nan

    return df


def write_synthetic(cfg: Config) -> Path:
    rng = np.random.default_rng(cfg.seed)
    df = generate(cfg, rng)
    out_dir = cfg.path("data_synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["dataset"]["synthetic_file"]
    df.to_csv(out_path, index=False)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Home Credit-shaped data.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    path = write_synthetic(cfg)
    n = cfg["synthetic"]["n_rows"]
    print(f"[synthetic] wrote {n} rows -> {path}")
    print("[synthetic] REMINDER: this is SYNTHETIC data. Metrics on it are "
          "synthetic-only and must be labeled as such.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
