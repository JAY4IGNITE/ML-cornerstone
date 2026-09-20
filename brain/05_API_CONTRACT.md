# API Contract

## Endpoints

### GET /api/health
Returns service health and model availability.

### GET /api/model/info
Returns:
- Model name
- Model version
- Training dataset manifest reference
- Feature schema version
- Calibration status
- Training timestamp if available

### POST /api/validate-input
Validates an applicant payload without producing a prediction.

### POST /api/predict
Returns:
- prediction_id when persistence is enabled
- default_probability
- risk_score
- risk_band
- model_version
- explanation_available
- limitations

## API Rules
- Use Pydantic schemas.
- Validate numeric bounds and categorical values.
- Handle missing and unknown values intentionally.
- Do not expose stack traces or internal paths.
- Do not log sensitive applicant information.
- Do not retrain during prediction.
- Load the complete preprocessing + model pipeline.
- Return consistent error schemas.
- Ensure probability is between 0 and 1.
- Add tests for valid, invalid, missing, and unknown inputs.

## Responsible Output
The response must clearly state that the output is an analytical estimate and not an autonomous lending decision.
