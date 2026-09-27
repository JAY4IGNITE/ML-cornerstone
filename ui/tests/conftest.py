"""Shared pytest fixtures for the Streamlit UI + service-layer suite.

Hermetic by construction: before anything imports the config we synthesize a
throwaway ``config.hermetic.yaml`` that (a) forces ``dataset.source: synthetic``
and (b) redirects every writable path into an isolated temp dir. So the suite
never reads the git-ignored real Kaggle CSV and never touches (or clobbers) the
developer's real ``./artifacts``. The serving layer also calls ``load_config()``,
so pointing ``LOAN_RISK_CONFIG`` at this file keeps the model the UI loads
consistent with the config the tests assert against.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Repo root is two levels up from ui/tests/. Put it first on sys.path so the
# working-copy ``ui`` and ``loan_risk`` win over any editable install elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="loan_risk_ui_tests_"))
atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)


def _write_hermetic_config() -> Path:
    with (_REPO_ROOT / "config" / "config.yaml").open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["dataset"]["source"] = "synthetic"
    raw["paths"] = {
        "data_raw": (_TMP_ROOT / "data" / "raw").as_posix(),
        "data_synthetic": (_TMP_ROOT / "data" / "synthetic").as_posix(),
        "data_processed": (_TMP_ROOT / "data" / "processed").as_posix(),
        "artifacts": (_TMP_ROOT / "artifacts").as_posix(),
        "reports": (_TMP_ROOT / "reports").as_posix(),
        "manifest": (_TMP_ROOT / "data" / "manifest.json").as_posix(),
    }
    out = _TMP_ROOT / "config.hermetic.yaml"
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    return out


os.environ["LOAN_RISK_CONFIG"] = str(_write_hermetic_config())
# Import config machinery only after LOAN_RISK_CONFIG is set, and clear any cache
# a stray earlier import may have populated with the real config.
from loan_risk.config import load_config  # noqa: E402
from loan_risk.data.ingestion import resolve_source_path, standardize  # noqa: E402
from loan_risk.pipeline import artifacts  # noqa: E402

load_config.cache_clear()


@pytest.fixture(scope="session")
def cfg():
    """The hermetic, synthetic-backed config shared by every test."""
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


@pytest.fixture()
def service_with_model(trained_model):
    """A freshly-built ModelService that has loaded the hermetic trained model.

    ``get_service`` is an ``st.cache_resource`` singleton, so it may have been
    built (model-absent) by an earlier test before training ran. Clear it so the
    rebuilt service picks up the just-trained artifacts.
    """
    from ui.helpers import get_service
    get_service.clear()
    svc = get_service()
    assert svc.available, "hermetic model should be loaded after training"
    return svc


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
