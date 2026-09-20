# Intelligent Loan Risk Assessment System — Project Context

## Purpose
Build an academic but engineering-quality loan default risk assessment platform.

## Primary ML Task
Predict the probability of repayment difficulty/default using the Kaggle Home Credit Default Risk dataset.

## Important Distinction
Do not confuse:
- Loan approval prediction: predicts an approval label.
- Loan default prediction: predicts repayment difficulty/default.

The primary project task is default-risk prediction. Approval workflows, if added, must remain separate.

## Primary Dataset
Home Credit Default Risk (Kaggle):
- Main file: application_train.csv
- Target: TARGET
- Supporting files may include bureau, previous applications, installments, POS cash balance, and credit-card balance data.
- Verify exact schemas, licenses, target meaning, row counts, and relationships from downloaded files.

## Required Product Capabilities
1. Dataset ingestion and validation
2. Leakage-aware preprocessing
3. Feature engineering
4. EDA
5. Multiple ML models
6. Probability calibration
7. SHAP/global and local explanations
8. Transparent risk score
9. FastAPI prediction service
10. React dashboard
11. Testing, documentation, and reproducibility

## Technology Preference
- Python
- Pandas/Polars/DuckDB where appropriate
- Scikit-learn
- XGBoost
- SHAP
- FastAPI + Pydantic
- React + Vite + TypeScript
- Tailwind CSS + shadcn/ui
- PostgreSQL/Supabase if persistence is required
- Docker where useful

## Non-Negotiable Principles
- Never fabricate metrics, data, explanations, or test results.
- Never use test data to fit preprocessing or tune thresholds.
- Never silently invent missing columns.
- Never claim production readiness, fairness, or legal compliance without evidence.
- Do not build an autonomous lending decision-maker.
