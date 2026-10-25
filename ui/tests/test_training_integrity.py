"""Training must preserve the validation holdout and tie serving to evidence."""
import copy
import json

import pytest

from loan_risk.config import Config
from loan_risk.pipeline import artifacts, run
from loan_risk.pipeline.split import make_splits


def test_training_never_fits_validation_and_persists_auditable_bundle(cfg, standardized_df, tmp_path, monkeypatch):
    raw = copy.deepcopy(cfg.raw)
    raw["paths"]["artifacts"] = str(tmp_path / "artifacts")
    raw["paths"]["reports"] = str(tmp_path / "reports")
    raw["paths"]["manifest"] = str(tmp_path / "manifest.json")
    for name, settings in raw["models"].items():
        settings["enabled"] = name == "decision_tree"
    isolated = Config(raw, cfg.root)
    split = make_splits(standardized_df, isolated)
    train_rows = set(split.X_train.index)
    val_rows = set(split.X_val.index)
    assert train_rows.isdisjoint(val_rows)
    original_calibrate = run.calibrate_pipeline
    original_select = run.cross_val_select
    original_tune = run.tune_model

    def calibrate(pipe, X, y, config):
        assert set(X.index) <= train_rows
        return original_calibrate(pipe, X, y, config)

    def select(specs, X, y, config):
        assert set(X.index) <= train_rows
        return original_select(specs, X, y, config)

    def tune(spec, X, y, config):
        assert set(X.index) <= train_rows
        return original_tune(spec, X, y, config)

    monkeypatch.setattr(run, "calibrate_pipeline", calibrate)
    monkeypatch.setattr(run, "cross_val_select", select)
    monkeypatch.setattr(run, "tune_model", tune)
    metrics = run.train(isolated)
    metadata = artifacts.load_metadata(isolated)
    assert metadata["decision_threshold"] == metrics["selected_threshold"]["value"]
    assert metrics["selected_threshold"]["selection_basis"] == "held_out_validation"
    assert metadata["model_key"] == "decision_tree"
    assert metrics["baseline_test_metrics"]["roc_auc"] == 0.5
    assert metrics["final_test_metrics"]["n"] == len(split.y_test)
    artifacts.verify_bundle(isolated, metadata)
    assert set(artifacts.load_all_models(isolated)) == {"decision_tree"}
    # A report from another run must not be silently displayed for this model.
    path = isolated.path("artifacts") / "metrics.json"
    path.write_text(json.dumps({"fake": True}))
    with pytest.raises(ValueError, match="checksum"):
        artifacts.verify_bundle(isolated, metadata)
