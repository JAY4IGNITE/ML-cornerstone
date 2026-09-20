# Dataset Protocol

## Selected Source
Home Credit Default Risk, downloaded from the official Kaggle competition source.

## Required Files
Start with:
- application_train.csv

Add supporting files incrementally only after validating their keys and grain:
- bureau.csv
- previous_application.csv
- installments_payments.csv
- POS_CASH_balance.csv
- credit_card_balance.csv

## Dataset Manifest
Create a manifest containing:
- Source URL
- Retrieval date
- File names
- License and usage notes
- Row and column counts
- Target column and meaning
- Feature descriptions
- Missingness
- Known limitations

## Validation Rules
Check:
- Required columns
- Data types
- Missing values
- Infinite values
- Duplicate rows
- Duplicate identifiers
- Invalid numerical ranges
- Target distribution
- Join cardinality
- Train/test overlap
- Post-outcome fields
- Potential leakage

## Joining Rules
- Understand the grain of every table.
- Aggregate one-to-many tables before joining to application-level data.
- Prevent row multiplication.
- Verify row counts before and after joins.
- Check whether historical data was available at the prediction time.
- Save join diagnostics.

## Memory-Constrained Processing
Do not load all large tables into memory at once.
Use chunking, Polars, DuckDB, or Parquet when appropriate.
Save intermediate aggregated features.

## Dataset Separation
Do not merge unrelated Kaggle and Hugging Face datasets into the primary training data.
Use other datasets for separate experiments, external validation, or pipeline stress testing.
