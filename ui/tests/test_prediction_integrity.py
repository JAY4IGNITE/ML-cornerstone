"""Regressions for truthful, validated risk estimates."""
import numpy as np
import pytest

from loan_risk.pipeline.explain import _rank_contributions, get_feature_reason
from loan_risk.pipeline.risk_score import probability_to_decision, probability_to_score
from loan_risk.serving.service import ModelService
from loan_risk.serving.validation import validate_payload


class FixedModel:
    classes_ = np.array([0, 1])

    def __init__(self, probability=0.2):
        self.probability = probability

    def predict_proba(self, X):
        return np.array([[1 - self.probability, self.probability]] * len(X))


@pytest.fixture
def service(cfg, monkeypatch):
    monkeypatch.setattr(ModelService, "_try_load", lambda self: None)
    svc = ModelService(cfg)
    svc.model = FixedModel()
    svc.models = {"decision_tree": svc.model}
    svc.metadata = {"model_key": "decision_tree", "model_name": "Decision Tree",
                    "decision_threshold": 0.3}
    return svc


def test_selected_identity_and_threshold(service, valid_payload):
    result = service.predict(valid_payload, with_explanation=False)
    assert result["model_key"] == "decision_tree"
    assert result["decision_threshold"] == 0.3
    assert result["predicted_target"] == 0


def test_unavailable_model_does_not_silently_fallback(service, valid_payload):
    with pytest.raises(ValueError, match="unavailable"):
        service.predict(valid_payload, model_key="xgboost", with_explanation=False)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_model_probability_is_rejected(service, valid_payload, value):
    service.model.probability = value
    with pytest.raises(ValueError):
        service.predict(valid_payload, with_explanation=False)


def test_service_validates_before_inference(service, valid_payload):
    valid_payload["EXT_SOURCE_1"] = 99
    with pytest.raises(ValueError):
        service.predict(valid_payload, with_explanation=False)


@pytest.mark.parametrize("changes", [{"CNT_CHILDREN": 0.5}, {"CNT_CHILDREN": True},
    {"EMPLOYMENT_YEARS": 80}, {"CNT_FAM_MEMBERS": 1, "CNT_CHILDREN": 2},
    {"unknown_field": 1}])
def test_invalid_applicant_relationships_rejected(valid_payload, changes):
    normalized, errors = validate_payload(valid_payload | changes)
    assert normalized is None and errors


def test_optional_missing_inputs_use_training_nan(service, valid_payload):
    valid_payload["OCCUPATION_TYPE"] = None
    valid_payload["AMT_ANNUITY"] = None
    frame = service.to_model_frame(valid_payload)
    assert np.isnan(frame["AMT_ANNUITY"].iloc[0])
    assert frame["OCCUPATION_TYPE"].isna().all()


@pytest.mark.parametrize("p", [0.01, 0.1, 0.2, 0.9])
def test_risk_output_does_not_invent_lending_policy(p):
    decision = probability_to_decision(p, threshold=0.15)
    text = str(decision).lower()
    for claim in ["approved", "declined", "25%+ collateral", "prime financing", "repayment expected"]:
        assert claim not in text
    assert decision["predicted_target"] == int(p >= 0.15)


def test_nan_not_converted_to_certain_risk():
    with pytest.raises(ValueError):
        probability_to_score(np.nan)


def test_explanation_does_not_invent_history_or_hide_demographics():
    reason = get_feature_reason("EXT_SOURCE_1", "External score 1", -0.2, "decreases_risk")
    assert "reliable repayment history" not in reason.lower()
    ranked = _rank_contributions(["CODE_GENDER_F", "AMT_CREDIT"], [0.8, 0.1], 2)
    assert ranked[0]["feature"] == "CODE_GENDER_F"


def test_single_class_metrics_are_json_safe():
    import json
    from loan_risk.pipeline.evaluate import compute_metrics
    result = compute_metrics([0, 0], [0.1, 0.2])
    assert result["roc_auc"] is None
    json.dumps(result, allow_nan=False)


def test_bootstrap_intervals_are_reproducible():
    from loan_risk.pipeline.evaluate import bootstrap_intervals
    y = [0, 1] * 20
    p = [0.1, 0.9] * 20
    first = bootstrap_intervals(y, p, repeats=10)
    assert first == bootstrap_intervals(y, p, repeats=10)
    assert first["metrics"]["roc_auc"] == {"lower": 1.0, "upper": 1.0}


def test_outside_training_range_is_disclosed(service, valid_payload):
    service.metadata["input_profile"] = {"AMT_CREDIT": {"min": 1, "max": 100}}
    result = service.predict(valid_payload, with_explanation=False)
    assert result["outside_training_range"] == ["AMT_CREDIT"]
    assert result["assessed_at"]
