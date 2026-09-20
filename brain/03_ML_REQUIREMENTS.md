# Machine Learning Requirements

## Target
Primary target: TARGET from Home Credit Default Risk.

Before training:
- Inspect target values.
- Confirm what 0 and 1 mean from the dataset documentation.
- Report class balance.
- Do not assume the target means a generic default without documenting its exact definition.

## Data Splitting
- Use a reproducible split.
- Use stratification where suitable.
- Prevent duplicate applicant/record leakage.
- If temporal information is available and relevant, evaluate a time-aware split.

## Preprocessing
- Numeric and categorical pipelines.
- Median imputation for numeric fields when justified.
- Explicit unknown handling for categorical fields.
- One-hot encoding for nominal categories.
- Scaling for linear models where needed.
- Fit all transformations only on training data.

## Feature Engineering
Potential features, only when source data supports them:
- Credit-to-income ratio
- Annuity-to-income ratio
- Loan-to-income ratio
- Employment duration
- Aggregated bureau credit count
- Previous application count
- Historical repayment aggregates
- Missingness indicators

For each feature, document:
- Formula
- Source columns
- Unit
- Interpretation
- Leakage assessment

## Models
Minimum:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost if dependency and data suitability permit

Use identical evaluation splits and comparable preprocessing.

## Evaluation
Required:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix
- Brier score
- Calibration curve
- Threshold analysis

Do not choose a model from accuracy alone.

## Calibration
Evaluate whether predicted probabilities match observed frequencies.
Use calibration procedures without leakage.
Document whether probabilities are calibrated and how.

## Explainability
- Global importance
- Local SHAP explanations where supported
- Explain direction and contribution carefully
- Do not describe feature importance as causation
- Do not invent adverse-action reasons

## Risk Score
Store:
- Raw default probability
- Configured risk score
- Risk band
- Model version
- Score definition

The score must be explicitly defined and must not be represented as a guaranteed outcome.
