"""Shared pytest fixtures for the loan-risk test suite.

Fixtures reuse the existing synthetic fixture and the trained artifacts
(read-only) so tests are fast. Everything is session-scoped and, if the
synthetic data or artifacts are missing, builds them exactly once; fixtures that
need a loaded model depend on ``trained_model`` so training always precedes the
first import of backend.main.
"""
from __future__ import annotations

import pandas as pd
import pytest

from loan_risk.config import load_config
from loan_risk.data.ingestion import resolve_source_path, standardize
from loan_risk.pipeline import artifacts


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synthetic_raw_df(cfg) -> pd.DataFrame:
    """Raw synthetic application frame (generates it once if absent)."""
    path = resolve_source_path(cfg)
    if not path.exists():
        from loan_risk.data.synthetic import write_synthetic
        write_synthetic(cfg)
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def standardized_df(cfg, synthetic_raw_df) -> pd.DataFrame:
    return standardize(synthetic_raw_df, cfg)


@pytest.fixture(scope="session")
def trained_model(cfg):
    """The persisted, calibrated end-to-end pipeline (trains once if absent)."""
    if not artifacts.artifacts_exist(cfg):
        from loan_risk.pipeline.run import train
        train(cfg)
    return artifacts.load_model(cfg)


@pytest.fixture(scope="session")
def api_client(trained_model):
    """FastAPI TestClient. Depends on ``trained_model`` so the artifacts exist
    BEFORE backend.main is imported — the app builds its ModelService (which
    loads the model) at import time, so training must happen first regardless of
    the order tests request fixtures."""
    from fastapi.testclient import TestClient

    from backend.main import app
    return TestClient(app)


@pytest.fixture()
def valid_payload() -> dict:
    """A fully-specified, valid applicant payload (all fields present)."""
    return {
        "AMT_INCOME_TOTAL": 180000,
        "AMT_CREDIT": 600000,
        "AMT_ANNUITY": 27000,
        "AMT_GOODS_PRICE": 540000,
        "AGE_YEARS": 35,
        "EMPLOYMENT_YEARS": 5,
        "CNT_FAM_MEMBERS": 2,
        "CNT_CHILDREN": 0,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.5,
        "REGION_POPULATION_RELATIVE": 0.02,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Core staff",
    }
