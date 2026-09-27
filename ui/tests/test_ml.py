"""ML unit/integration tests (07_TESTING_CHECKLIST.md — "ML Tests").

Covers: reproducible split, pipeline fit/transform, single-row and batch
inference, probability bounds, unknown-category handling, missing-value
handling, artifact loading, feature-name consistency, engineered-feature
correctness, risk-score/band mapping, and metrics computation.

Model-fitting tests stay fast: the standardized frame is sampled to <=2000 rows
and a single cheap Decision Tree pipeline is fit once per module (module-scoped
fixture), rather than training all four configured models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_risk.pipeline.evaluate import compute_metrics
from loan_risk.pipeline.features import add_engineered_features
from loan_risk.pipeline.models import build_model_specs, build_pipeline
from loan_risk.pipeline.preprocess import get_output_feature_names
from loan_risk.pipeline.risk_score import probability_to_band, probability_to_score
from loan_risk.pipeline.split import SplitResult, make_splits

SAMPLE_N = 2000


def _feature_frame(standardized_df, cfg) -> pd.DataFrame:
    """Model input = standardized frame minus target and id column, matching how
    ``run.train`` fits the pipeline."""
    return standardized_df.drop(columns=[cfg.target, cfg.id_column], errors="ignore")


@pytest.fixture(scope="module")
def small_xy(standardized_df, cfg):
    """A small (<=2000-row) balanced-enough sample: (X features, y target)."""
    sample = standardized_df.sample(
        n=min(SAMPLE_N, len(standardized_df)), random_state=cfg.seed
    ).reset_index(drop=True)
    y = sample[cfg.target].astype(int)
    X = sample.drop(columns=[cfg.target, cfg.id_column], errors="ignore")
    return X, y


@pytest.fixture(scope="module")
def tree_spec(cfg):
    """The Decision Tree spec (cheap to fit) from the model factory."""
    specs, _ = build_model_specs(cfg)
    for s in specs:
        if s.key == "decision_tree":
            return s
    pytest.skip("decision_tree not enabled in config")


@pytest.fixture(scope="module")
def fitted_pipeline(tree_spec, cfg, small_xy):
    X, y = small_xy
    pipe = build_pipeline(tree_spec, cfg)
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------
# Reproducible split
# ---------------------------------------------------------------------------
def test_reproducible_split(standardized_df, cfg):
    s1 = make_splits(standardized_df, cfg)
    s2 = make_splits(standardized_df, cfg)
    assert isinstance(s1, SplitResult)
    # identical indices across identical-seed runs
    assert list(s1.X_train.index) == list(s2.X_train.index)
    assert list(s1.X_val.index) == list(s2.X_val.index)
    assert list(s1.X_test.index) == list(s2.X_test.index)
    # diagnostics report consistent sizes + seed
    assert s1.diagnostics["n_train"] == s2.diagnostics["n_train"]
    assert s1.diagnostics["n_test"] == s2.diagnostics["n_test"]
    assert s1.diagnostics["seed"] == cfg.seed
    # no row lost or duplicated across splits
    total = (s1.diagnostics["n_train"] + s1.diagnostics["n_val"]
             + s1.diagnostics["n_test"])
    assert total == len(standardized_df) - s1.diagnostics.get("dropped_duplicate_ids", 0)


# ---------------------------------------------------------------------------
# Stratification: class balance preserved across every split
# ---------------------------------------------------------------------------
def test_split_stratification_preserves_positive_rate(standardized_df, cfg):
    """With split.stratify enabled, the positive-class rate of train, val and
    test must each stay within a small absolute tolerance of the full-data rate,
    so no split concentrates or starves defaults (03_ML_REQUIREMENTS.md)."""
    assert cfg["split"]["stratify"], "config split.stratify must be enabled"
    s = make_splits(standardized_df, cfg)

    # Reference = rate over exactly the rows that were split. make_splits dedups
    # ids first and the reproducible-split test proves no row is lost across
    # splits, so this union is the full (post-dedup) dataset's positive rate.
    full_y = pd.concat([s.y_train, s.y_val, s.y_test])
    overall_rate = float(full_y.mean())

    tol = 0.02
    for name, y in (("train", s.y_train), ("val", s.y_val), ("test", s.y_test)):
        rate = float(y.mean())
        assert abs(rate - overall_rate) <= tol, (
            f"{name} positive rate {rate:.4f} deviates from overall "
            f"{overall_rate:.4f} by more than {tol}"
        )

    # Diagnostics must report the same per-split rates the object actually holds.
    assert s.diagnostics["stratified"] is True
    assert s.diagnostics["train_positive_rate"] == pytest.approx(
        float(s.y_train.mean()), abs=1e-4
    )
    assert s.diagnostics["val_positive_rate"] == pytest.approx(
        float(s.y_val.mean()), abs=1e-4
    )
    assert s.diagnostics["test_positive_rate"] == pytest.approx(
        float(s.y_test.mean()), abs=1e-4
    )


# ---------------------------------------------------------------------------
# No leakage: applicant ids are partitioned, never shared across splits
# ---------------------------------------------------------------------------
def test_split_no_id_leakage_across_splits(standardized_df, cfg):
    """The SK_ID_CURR sets of train/val/test are pairwise disjoint and together
    cover every unique applicant exactly once. An id in two splits is record
    leakage and would inflate held-out metrics (split.py dedup guard)."""
    id_col = cfg.id_column  # "SK_ID_CURR"
    assert id_col in standardized_df.columns
    s = make_splits(standardized_df, cfg)

    # The id column survives the split (make_splits drops only the target from X).
    for X in (s.X_train, s.X_val, s.X_test):
        assert id_col in X.columns

    train_ids = set(s.X_train[id_col])
    val_ids = set(s.X_val[id_col])
    test_ids = set(s.X_test[id_col])

    # Pairwise disjoint: no applicant appears in more than one split.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Union covers every unique applicant, and its size equals the count of
    # unique ids in the full data (no id dropped, none double-counted).
    union = train_ids | val_ids | test_ids
    n_unique = int(standardized_df[id_col].nunique())
    assert len(union) == n_unique
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(union)


# ---------------------------------------------------------------------------
# Pipeline fit + transform
# ---------------------------------------------------------------------------
def test_pipeline_fit_and_predict_proba(fitted_pipeline, small_xy):
    X, _ = small_xy
    proba = fitted_pipeline.predict_proba(X.head(50))
    assert proba.shape == (50, 2)
    # each row is a valid probability distribution
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_feature_name_consistency(fitted_pipeline, small_xy):
    X, _ = small_xy
    pre = fitted_pipeline.named_steps["preprocess"]
    names = get_output_feature_names(pre)
    transformed = pre.transform(X.head(10))
    transformed = np.asarray(transformed)
    assert transformed.shape[1] == len(names)


# ---------------------------------------------------------------------------
# Inference: one row + a batch, probability bounds
# ---------------------------------------------------------------------------
def test_inference_single_row(fitted_pipeline, small_xy):
    X, _ = small_xy
    one = X.iloc[[0]]
    proba = fitted_pipeline.predict_proba(one)[:, 1]
    assert proba.shape == (1,)
    assert 0.0 <= float(proba[0]) <= 1.0


def test_inference_batch(fitted_pipeline, small_xy):
    X, _ = small_xy
    batch = X.head(32)
    proba = fitted_pipeline.predict_proba(batch)[:, 1]
    assert proba.shape == (32,)


def test_probability_bounds(fitted_pipeline, small_xy):
    X, _ = small_xy
    proba = fitted_pipeline.predict_proba(X.head(200))[:, 1]
    assert float(proba.min()) >= 0.0
    assert float(proba.max()) <= 1.0


# ---------------------------------------------------------------------------
# Robustness: unknown categories + missing values must not crash
# ---------------------------------------------------------------------------
def test_unknown_category_handling(fitted_pipeline, small_xy):
    X, _ = small_xy
    row = X.iloc[[0]].copy()
    row["OCCUPATION_TYPE"] = "Totally Unseen Occupation"  # never seen in fit
    proba = fitted_pipeline.predict_proba(row)[:, 1]
    assert 0.0 <= float(proba[0]) <= 1.0  # handle_unknown='ignore' -> no crash


def test_missing_numeric_handling(fitted_pipeline, small_xy):
    X, _ = small_xy
    row = X.iloc[[0]].copy()
    row["AMT_INCOME_TOTAL"] = np.nan  # median imputation should fill this
    proba = fitted_pipeline.predict_proba(row)[:, 1]
    assert 0.0 <= float(proba[0]) <= 1.0


# ---------------------------------------------------------------------------
# Artifact loading (the persisted, calibrated end-to-end pipeline)
# ---------------------------------------------------------------------------
def test_trained_model_predicts(trained_model, standardized_df, cfg):
    X = _feature_frame(standardized_df, cfg).head(20)
    proba = trained_model.predict_proba(X)
    assert proba.shape == (20, 2)
    p1 = proba[:, 1]
    assert float(p1.min()) >= 0.0 and float(p1.max()) <= 1.0


# ---------------------------------------------------------------------------
# Engineered feature correctness
# ---------------------------------------------------------------------------
def test_engineered_features_correctness():
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [100000.0, 200000.0],
        "AMT_CREDIT": [300000.0, 400000.0],
        "AMT_ANNUITY": [15000.0, 20000.0],
        "AMT_GOODS_PRICE": [270000.0, 380000.0],
        "EMPLOYMENT_YEARS": [5.0, 10.0],
        "AGE_YEARS": [40.0, 50.0],
        "CNT_FAM_MEMBERS": [2.0, 4.0],
        "EXT_SOURCE_1": [0.4, 0.6],
        "EXT_SOURCE_2": [0.5, 0.7],
        "EXT_SOURCE_3": [0.6, 0.8],
    })
    out = add_engineered_features(df)
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(300000.0 / 100000.0)
    assert out.loc[1, "CREDIT_INCOME_RATIO"] == pytest.approx(400000.0 / 200000.0)
    assert out.loc[0, "ANNUITY_INCOME_RATIO"] == pytest.approx(15000.0 / 100000.0)
    assert out.loc[0, "CREDIT_ANNUITY_RATIO"] == pytest.approx(300000.0 / 15000.0)
    assert out.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx((0.4 + 0.5 + 0.6) / 3)
    assert out.loc[0, "INCOME_PER_FAM_MEMBER"] == pytest.approx(100000.0 / 2.0)


def test_engineered_divide_by_zero_is_nan():
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [0.0],
        "AMT_CREDIT": [300000.0],
        "AMT_ANNUITY": [15000.0],
        "AMT_GOODS_PRICE": [270000.0],
        "EMPLOYMENT_YEARS": [5.0],
        "AGE_YEARS": [40.0],
        "CNT_FAM_MEMBERS": [2.0],
        "EXT_SOURCE_1": [0.4],
        "EXT_SOURCE_2": [0.5],
        "EXT_SOURCE_3": [0.6],
    })
    out = add_engineered_features(df)
    # zero denominator -> NaN (never inf), per the no-infinite-values rule
    assert pd.isna(out.loc[0, "CREDIT_INCOME_RATIO"])
    assert not np.isinf(out["CREDIT_INCOME_RATIO"]).any()


# ---------------------------------------------------------------------------
# Risk score + band mapping
# ---------------------------------------------------------------------------
def test_probability_to_score_mapping():
    assert probability_to_score(0.5) == 50
    assert probability_to_score(0.0) == 0
    assert probability_to_score(1.0) == 100
    assert probability_to_score(0.234) == 23
    # clamps out-of-range inputs
    assert probability_to_score(-0.5) == 0
    assert probability_to_score(1.5) == 100


def test_risk_bands_monotonic(cfg):
    band_names = [b["name"] for b in cfg["risk_score"]["bands"]]

    def band_index(p: float) -> int:
        return band_names.index(probability_to_band(p, cfg))

    probs = [0.01, 0.05, 0.10, 0.15, 0.25, 0.35, 0.5, 0.75, 0.99]
    indices = [band_index(p) for p in probs]
    # band index never decreases as probability increases
    assert indices == sorted(indices)
    # lowest prob -> first band, highest prob -> last band
    assert band_index(0.001) == 0
    assert band_index(0.999) == len(band_names) - 1


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def test_compute_metrics_ranges():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    # correlated-but-noisy scores so both classes are represented
    y_prob = np.clip(0.25 + 0.5 * y_true + rng.normal(0, 0.2, size=500), 0, 1)
    m = compute_metrics(y_true, y_prob)
    for key in ("roc_auc", "pr_auc", "brier_score"):
        assert key in m
        assert isinstance(m[key], float)  # not an error dict
        assert 0.0 <= m[key] <= 1.0
    assert 0.0 <= m["accuracy"] <= 1.0


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.15, 0.85, 0.95])
    m = compute_metrics(y_true, y_prob)
    assert m["roc_auc"] == pytest.approx(1.0)
