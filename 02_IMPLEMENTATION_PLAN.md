# Implementation Plan

Execute in phases and validate each phase before proceeding.

## Phase 1 — Repository Audit
- Inspect all folders and files.
- Identify current frontend/backend/ML implementation.
- Identify existing datasets and notebooks.
- Produce an audit report.
- Do not delete or replace working code blindly.

## Phase 2 — Data Ingestion and Validation
- Add dataset manifest.
- Validate schema, target, types, missingness, duplicates, invalid values, and leakage.
- Confirm table grain and relationships.
- Save a machine-readable validation report.

## Phase 3 — Baseline Pipeline
- Build train/validation/test split.
- Use sklearn Pipeline and ColumnTransformer.
- Implement numeric and categorical preprocessing.
- Fit transformations only on training data.
- Train Logistic Regression and tree baselines.

## Phase 4 — Feature Engineering
- Add only features supported by source columns.
- Document formula, inputs, meaning, and leakage risk.
- Build historical aggregations only with correct applicant identifiers and temporal logic.

## Phase 5 — Evaluation
- Compare models using accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, and calibration.
- Keep final test set untouched until final evaluation.
- Perform threshold analysis.
- Report cross-validation variation.

## Phase 6 — Explainability
- Implement global feature importance.
- Implement SHAP where technically appropriate.
- Provide local explanations tied to actual predictions.
- Preserve feature names after transformations.

## Phase 7 — Backend
- Implement /api/health
- Implement /api/model/info
- Implement /api/predict
- Implement /api/validate-input
- Add robust Pydantic validation and safe error handling.

## Phase 8 — Frontend
- Build overview, assessment, result, explainability, model metrics, and limitations pages.
- Clearly distinguish probability, score, and risk band.
- Include loading, error, empty, and validation states.

## Phase 9 — Testing and Documentation
- Add unit and integration tests.
- Run tests and report actual output.
- Write README, data dictionary, model card, evaluation report, API docs, and responsible-use documentation.

## Phase 10 — Deployment Readiness
- Verify environment variables and secret handling.
- Add Docker configuration if appropriate.
- Confirm reproducible setup and inference.
- Do not claim deployment readiness until checks pass.
