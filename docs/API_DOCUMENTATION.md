# Python service contract

The application is Streamlit and calls Python in-process. There is no separate FastAPI server or supported `/api/*` HTTP endpoint. Historical HTTP examples have been removed because they described an absent backend.

## Load and inspect

```python
from loan_risk.serving import ModelService, validate_payload
service = ModelService()
if not service.available:
    raise RuntimeError(service.load_error)
models = service.get_available_models()
info = service.model_info()
schema = service.get_feature_schema()
metrics = service.get_metrics()
```

`ModelService` loads the selected `model.joblib` and only candidates registered in the same release. It verifies persisted SHA-256 hashes before deserialization. Metadata and feature schema are required. The metrics snapshot stays paired with the loaded model. In Streamlit, `ui.helpers.get_service()` caches by artifact fingerprint and refreshes when a release changes.

Checksums detect accidental corruption and mismatched files. They are not signatures: joblib artifacts must be from a trusted source.

## Applicant inputs

The canonical numeric and categorical fields are defined in `loan_risk/schema.py` and exported in `artifacts/feature_schema.json`. Supply AGE_YEARS and EMPLOYMENT_YEARS, not the raw negative day counts. Submit the Home Credit normalized external scores and original currency scale; no commercial credit-score conversion exists.

`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EMPLOYMENT_YEARS`, `OCCUPATION_TYPE`, `AMT_ANNUITY`, and `AMT_GOODS_PRICE` may be omitted or null. Missing values become training-compatible NaN values with the appropriate missingness flags. Other fields are required.

Validation rejects unknown fields, invalid categories, nonfinite numbers, out-of-bounds measurements, booleans masquerading as numbers, fractional household counts, children exceeding household size minus the applicant, and employment duration exceeding age. Numeric strings may be normalized to numbers.

```python
normalized, errors = validate_payload(payload)
# errors: [{"field": "...", "message": "..."}]
if errors:
    raise ValueError(errors)
result = service.predict(normalized, model_key=None, with_explanation=True)
```

The service validates again at its own boundary. `model_key=None` uses the metadata-selected model. An unavailable explicit key raises ValueError rather than silently substituting another model. Invalid probability outputs also raise ValueError. A missing model raises RuntimeError.

## Prediction result

| Field | Meaning |
|---|---|
| model_key, model_name, model_version | Actual estimator and release used |
| trained_at, assessed_at, dataset_source, synthetic | Provenance snapshot |
| default_probability | Calibrated estimate for the dataset target, in [0,1] |
| risk_score | Rounded probability multiplied by 100 |
| risk_band | Descriptive interval from the release's stored score configuration |
| predicted_target, target_label | Below/at-or-above threshold classification; not observed outcome |
| decision, decision_action | Analyst review indicator; never loan terms or approval |
| decision_threshold, threshold_basis | Per-model held-out validation threshold, or explicitly unvalidated legacy fallback |
| explanation_available, explanation | Explanation backend, scope, output units, contributions and limitations |
| all_model_predictions | Available registered candidates and their own thresholds |
| comparison_unavailable | Candidate comparisons that failed |
| missing_inputs, outside_training_range | Input-quality disclosures |
| limitations, disclaimer | Scope and uncertainty statements |

No submitted record is persisted by `persist_prediction`; Streamlit stores only the latest payload and result in browser-session state. Failed submissions clear the previous result to avoid confusion.

## Explanation contract

SHAP contributions explain one uncalibrated base estimator. XGBoost contributions use raw-margin units; supported sklearn tree contributions use base probability units. Linear contributions use log-odds. None is a decomposition of the whole calibrated ensemble probability. Tree fallback is unsigned global importance. Sensitive inputs used by the model remain visible. Text describes the measured contribution rather than inventing applicant history, causation, or adverse-action reasons.

## Reports

`get_metrics`, `get_manifest`, `get_validation_report`, and `model_info` expose the installed release's evidence. New releases include model/candidate files, metrics, feature schema, dataset manifest, and validation report, with metadata written last. Legacy artifacts lack integrity evidence and are disclosed as such.

This service has no production authentication or legal/fairness certification. See the model card and responsible-use documentation.
