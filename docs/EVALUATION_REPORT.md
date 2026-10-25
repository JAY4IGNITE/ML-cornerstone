# Evaluation report — audited release

Generated from `artifacts/metrics.json` and `artifacts/metadata.json` after the 2026-10-26 audit. These replace the historical September metrics previously quoted in this document.

- Model: **XGBoost**, version `20261025T184404Z-52e96b89`.
- Recorded training timestamp: `2026-10-25T18:44:05Z` (UTC).
- Dataset: real Home Credit application data, SHA-256 `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f`.
- Rows: 307,511; train 196,806, validation 49,202, test 61,503.
- Test positive rate: 8.0728%; target is the dataset's payment-difficulty label, not loan approval or a verified universal definition of default.

## Leakage controls and selection

All hyperparameter tuning, model-selection cross-validation, preprocessing fitting, and probability calibration use TRAIN only. Validation rows select the F1 review threshold. The calibrated model is not subsequently refit on validation. The test partition is used only for the final report and diagnostics. All partitions are stratified with seed 42 and disjoint applicant IDs. Conflicting duplicate applicants stop training.

Search uses four randomized configurations on at most 30,000 training rows for the forest and XGBoost. Selection uses three-fold CV on TRAIN. Because parameters have already been searched on TRAIN, these CV values are selection diagnostics, not unbiased estimates. The test results provide the separate evaluation. This test partition has been used in historical project reports; a fresh external or prospective holdout remains necessary before real deployment.

| Candidate | Mean CV ROC-AUC | Standard deviation |
|---|---:|---:|
| Logistic Regression | 0.7435 | 0.0008 |
| Decision Tree | 0.7194 | 0.0021 |
| Random Forest | 0.7495 | 0.0003 |
| XGBoost | 0.7564 | 0.0010 |

All four candidates are calibrated with sigmoid calibration, three folds, on TRAIN. Each stores its own threshold and calibrated validation metrics in `candidate_details`. Per-model `val_metrics` in the comparison table describe uncalibrated models and must not be substituted for calibrated performance.

## Held-out performance

The selected model uses threshold **0.168923**, chosen by validation F1 and saved with the model. This is a research operating point, not an institution's lending policy.

| Metric | Measured value |
|---|---:|
| ROC-AUC | 0.7612 |
| Average precision | 0.2507 |
| Brier score | 0.0677 |
| Accuracy at review threshold | 0.8521 |
| Precision at review threshold | 0.2513 |
| Recall at review threshold | 0.4203 |
| F1 at review threshold | 0.3146 |

Confusion matrix: TN 50,321, FP 6,217, FN 2,878, TP 2,087.

The model detects about 42.0% of positive outcomes at this threshold, and 25.1% of flagged records are positive. It therefore misses many payment-difficulty cases and produces many false alarms. These are material limitations, not production-grade guarantees.

The constant training-prevalence baseline has ROC-AUC 0.500, average precision 0.0807, Brier 0.0742, and accuracy 0.9193. Average precision improves by 3.11 times over that baseline. High accuracy alone is uninformative here: at threshold 0.5, the current selected model predicts no positives and has zero recall.

Calibration Brier before: 0.195242; after: 0.067714. Brier measures overall probability error, combining calibration and discrimination; use the reliability curve and bin counts as well.

## Sampling uncertainty

200 test-row bootstrap resamples, seed 42, percentile 95% intervals. These hold the fitted model fixed, so they do not capture training variability, temporal change, dataset shift, or uncertainty for an individual applicant.

| Metric | 95% lower | 95% upper |
|---|---:|---:|
| roc_auc | 0.7532 | 0.7682 |
| pr_auc | 0.2400 | 0.2622 |
| brier_score | 0.0661 | 0.0695 |
| recall | 0.4058 | 0.4327 |
| precision | 0.2406 | 0.2587 |

## Explanations and fairness

Global permutation importance and age/gender subgroup diagnostics are stored in the report. Sensitive features are shown honestly in attribution instead of hidden. No fairness certification or mitigation has been established.

Local SHAP explains the first uncalibrated fold, with output units explicitly identified. Logistic contributions are in log-odds. These do not decompose the final calibrated ensemble probability. Without supported SHAP, tree importance is shown as unsigned global importance, not an applicant-specific explanation. Narratives describe the calculated contribution only and do not invent repayment history or causal reasons.

## Reproduction and remaining evidence

Run `python -m loan_risk.pipeline.run` using `config/config.yaml`. The release records training configuration, dependency versions, source hash, unique version, fitted candidates, thresholds, schema, data report, manifest, and SHA-256 hashes. Numerical reproducibility requires compatible dependencies and the same input CSV. Checksums detect accidental corruption or mixed runs; joblib files must come from a trusted source.

This release is a research decision-support system. Missing evidence includes external-population and temporal validation, cost-based threshold selection, approved lending policy, independent fairness review, authentication and secure deployment, and ongoing outcome/drift monitoring. Performance has not been shown to be outstanding or universally accurate.

The holdout design follows [scikit-learn threshold guidance](https://scikit-learn.org/1.8/modules/classification_threshold.html) and [calibration guidance](https://scikit-learn.org/stable/modules/calibration).
