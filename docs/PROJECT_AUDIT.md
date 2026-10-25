# Project audit and remediation — 2026-10-26

This audit inspected the data ingestion and validation, split/preprocessing/model-selection/calibration pipeline, artifact persistence, prediction service, explanatory text, Streamlit pages, session state, tests, and documentation. Findings below are concrete defects reproduced or established from code. This is not a claim that every possible defect or deployment risk has been eliminated.

## Verified issues and fixes

| Severity | Finding | Remediation and evidence |
|---|---|---|
| High | Threshold selected on validation rows already included in calibrated-model fitting | Tuning, CV and calibration now use TRAIN only. Validation selects thresholds without subsequent refitting. Integration test asserts every fitting input excludes validation indices. |
| High | Training did not persist the candidate models exposed by the UI; old files could be mixed with a new selected model | Save all calibrated candidates, per-model thresholds, and a registered candidate manifest. Ignore orphan candidate files. Verify bundle hashes before loading. |
| High | Default/fallback model identities were hardcoded to XGBoost | Use metadata-selected identity. Unknown requested models raise an error; no silent substitution. |
| High | Configured threshold could differ from the evaluated threshold | Persist each calibrated candidate's validation threshold and use it at serving. Legacy fallback is explicitly unvalidated. |
| High | Model output produced approval, decline, pricing and collateral suggestions without supporting evidence | Replace with analytical review flags; preserve the distinction between model class and observed outcome. |
| High | Explanation templates invented repayment history, job stability and causal conclusions from feature names | Text now describes only calculated attribution. No factual borrower-history claims are generated. |
| High | First-fold SHAP was described as an exact explanation of the calibrated ensemble | Explicit first-estimator scope and raw-margin/probability/log-odds units. Unsigned tree fallback is disclosed as global importance. |
| High | Gender contributions were suppressed even when the model used gender | Show actual model inputs in attribution; retain explicit fairness limitations and subgroup diagnostics. Hiding an attribution is not bias mitigation. |
| High | Service accepted unvalidated input and invalid model probabilities could be clamped into plausible scores | Validate at service boundary; reject nonfinite or invalid probability distributions and unsupported classes. |
| Medium | None handling could differ from training NaN imputation | Convert canonical inputs to numeric NaN/object NaN consistently. Exercise fully missing optional inputs. |
| Medium | Unknown fields, booleans and impossible household relationships accepted | Forbid extras and booleans; require integral household counts and consistent family/employment relationships. |
| Medium | Valid float binary targets rejected; empty/single-class data and missing IDs were not reliably blocked | Validate numeric target values and both classes, reject missing IDs, require viable splits. |
| High | Conflicting duplicate applicant records silently kept only the first record | Remove only exact duplicates with counts; reject conflicting applicant records. |
| Medium | Artifact/report provenance could disagree | Unique release ID, source hash, runtime versions, training config, file hashes, manifest/report snapshots and metadata written last. |
| Medium | Missing-model cache did not refresh after training | Fingerprint artifacts for the Streamlit service cache. |
| Medium | Demo preset buttons changed auxiliary data but not existing keyed widgets | Apply preset values through callbacks before widgets render; tested repeated edits/preset changes. |
| Medium | Assess Another Applicant retained the old result and remained on the result page | Clear form and prediction state and navigate to Assessment; verified via interactive navigation test. |
| Medium | Failed submissions left stale prior results | Clear previous prediction when a new assessment is submitted, including validation failures. |
| Medium | Analytics silently preferred real data even for synthetic configuration, with stale file caching | Honor configured source, fingerprint dataset files, and label analytics as configured-dataset statistics. |
| Medium | Dashboard used hardcoded winning model, calibration method, performance values and statistical-significance claims | Read recorded evidence and remove unsupported claims; widen comparison charts to actual score ranges. |
| Medium | Unsigned explanations displayed risk directions; empty importance could crash | Handle empty/unavailable importance, show global fallback without local signed claims, guard no-model actions. |
| Medium | NaN metric values could enter JSON for single-class fairness groups | Report undefined ranking metrics as null with notes; enforce finite JSON serialization. |
| Medium | Headline accuracy concealed class imbalance and no-positive predictions at 0.5 | Add baseline metrics, operating-threshold confusion matrix, bootstrap intervals, and explicit recall/precision interpretation. |
| Medium | Documentation described an absent HTTP backend and obsolete results | Replace with actual in-process service contract and regenerated current evaluation/model card. |

## Installed release and measured performance

The real 307,511-row dataset was validated: 12 checks passed, one missing-data warning, zero failures. The release selects XGBoost by training-only three-fold CV. Model artifacts were regenerated locally, verified, and installed. Prior artifacts remain in `reports/pre-audit-artifacts/` for recovery.

On 61,503 held-out rows: ROC-AUC 0.7612 (95% bootstrap interval 0.7532–0.7682), average precision 0.2507, Brier 0.0677. At the validation-selected threshold 0.168923: recall 42.0%, precision 25.1%, F1 0.3146. The constant-prevalence average-precision baseline is 0.0807. The current model has zero recall at the reference threshold 0.5 despite 91.9% accuracy.

This does not establish an improvement over every prior fitted artifact or outstanding real-world accuracy. The main improvement is the validity, traceability and honesty of the pipeline and its outputs. The existing historical test split has been reused for this audit; fresh independent data is still required to establish external performance.

## Validation

- Full hermetic pytest suite: 127 passing tests, three upstream SHAP/matplotlib deprecation warnings.
- Regression groups reproduced failures before fixes for model identity, invalid probabilities, validation, misleading recommendations, explanation claims, training holdout use, data integrity and UI state.
- Real release: all four models evaluated all three illustrative presets, with finite probabilities and model-specific thresholds. Fully missing optional inputs also produced a valid estimate with missingness disclosure.
- Installed model/report/source hashes and versions verified. Serialization compression was checked to preserve predictions exactly.
- All 13 pages and the complete Streamlit entrypoint passed against the installed real-data release. Streamlit AppTest was used for page execution and interactive controls. This is headless runtime verification, not visual browser QA.
- Python compile checks and Git whitespace checks pass. Ruff is unavailable in the local environment; no lint-pass claim is made.

## Remaining limitations and production work

- The model still misses about 58% of positive cases at its current research operating point; about 75% of flags are false positives. Do not automate credit decisions.
- Calibration and predictive performance must be evaluated on the intended lender/population, currency conventions, external-score definitions, time period and licensed data.
- Demographic inputs remain in the research model. Subgroup diagnostics are not an independent fairness audit or a mitigation strategy.
- CV after hyperparameter search is a model-selection diagnostic; it is not nested-CV evidence. Bootstrap intervals quantify only fixed-model test sampling uncertainty.
- Local SHAP explains one base fold, not the full calibrated ensemble. No individual confidence interval or causal explanation is established.
- No authentication, roles, durable case/audit storage, secure hosted deployment, outcome feedback or drift monitor is implemented. These require concrete operational requirements and separate validation before real use.
- Artifact hashes detect corruption, not malicious replacement of trusted files. Only load trusted joblib artifacts.
- Deprecated Streamlit sizing parameters and SHAP plotting warnings remain compatible in the tested runtime; dependency migration should be managed before future upgrades.

## Methodology sources

The threshold correction follows [scikit-learn threshold guidance](https://scikit-learn.org/1.8/modules/classification_threshold.html). Calibration and disjoint fitting requirements follow [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration). Baselines and average precision follow [the model-evaluation reference](https://scikit-learn.org/stable/modules/model_evaluation.html).
