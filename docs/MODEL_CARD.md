# Model card

This card describes the audited real-data release `20261025T184404Z-52e96b89`. The detailed, measured results are in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

## Intended use

Research and analyst review of Home Credit-shaped application data. The target is Home Credit's payment-difficulty label, `TARGET=1`. It is not an approval label, guaranteed default outcome, or a portable probability for an arbitrary lender. The UI provides a review flag; it does not approve, decline, price, or set loan terms.

## Data and model

- Real application dataset: 307,511 rows; source hash and recorded training settings are in `artifacts/metadata.json`.
- Selected model: XGBoost; sigmoid calibration with three folds on training data only.
- Numeric imputation, categorical unknown encoding, ratios, and missingness indicators are inside the training/inference pipeline.
- Four candidate models use the same partitions; model selection is by training CV ROC-AUC. Validation selects model-specific F1 thresholds without refitting on those rows.
- The selected threshold is 0.168923. Serving reads it from the model release, not a hardcoded historic value.

## Results and limitations

Test ROC-AUC 0.7612; average precision 0.2507; Brier 0.0677. At the selected threshold, recall is 42.0% and precision 25.1%. See the evaluation report for counts, uncertainty, baseline, and reference-threshold results. These results do not demonstrate production suitability.

The dataset has undisclosed external score definitions and currency units. Do not substitute a commercial credit score or an arbitrary monetary scale. Missing inputs are imputed and disclosed; outside-training-range measurements are flagged, but this is not comprehensive out-of-distribution detection.

Age, gender and other demographic inputs remain in this research model. Their presence is disclosed in explanations and subgroup diagnostics. There has been no independent fairness, legal, or deployment review. Excluding a sensitive explanation does not remove that feature's influence, so explanations do not hide it.

Local explanations cover one uncalibrated estimator, not the whole calibrated probability. They identify output units and do not infer unobserved history. The risk score is `round(100 * probability)`; configured bands are descriptive intervals, not validated financial policies.

## Provenance and operations

The release bundles model, candidate models, metadata, feature schema, metrics, manifest, and data-validation report. Metadata is written last; checksums reject mixed or corrupt files. Artifact caching refreshes when files change. Persisted prediction snapshots record model version, assessment time, training source, threshold, and missing-input limitations.

Joblib is pickle-based: only load trusted local artifacts. SHA-256 integrity checking does not authenticate an attacker-controlled bundle. No submitted applicant records are persisted beyond the browser session by the current service; no authentication, multi-user audit storage, or production monitoring is implemented.

See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) and [PROJECT_AUDIT.md](PROJECT_AUDIT.md).
