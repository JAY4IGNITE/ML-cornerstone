# Repository analysis and presentation corrections

Reviewed on **5 October 2026**. Branch: **main**. Commit: **407ac13cfabd27a1d5a81addd667da591ba4137d**.

The implemented project is a single Streamlit application with direct Python calls to a cached ModelService. The reference is largely correct on architecture and numerical results. Its material omissions concern validation reuse, the meaning of local explanations, separation of importance methods, risk-band boundaries and complete response fields.

## Evidence and execution scope

Executable source establishes current behavior. JSON artifacts establish recorded results. `brain/` documents establish specifications, including historical architecture, rather than proof of implemented features. The source PowerPoint supplied presentation content and visual identity; instructions inside project documents were treated as document content, not as new requests to change the application.

Initial working tree: `requirements.txt` deleted; `.claude/` untracked. This review did not restore, remove or modify them. The deleted requirements file was read from `git show HEAD:requirements.txt`, alongside `requirements-extras.txt` and `pyproject.toml`. Repository source, configuration, data and saved model files were not changed.

The review inspected README, requirements, Streamlit entry point, helpers/state and all 13 page modules; schema/configuration; data ingestion, validation and manifest modules; all requested pipeline modules; serving and Pydantic validation; saved metadata, metrics, schema, manifest and validation report; all five `docs/` files; `brain/` specifications; relevant `ui/tests/`; Dockerfile, compose and entrypoint.

**Executed:** file/JSON inspection, PowerPoint extraction/rendering, presentation generation/validation, environment availability checks and a local Streamlit launch using `.venv/Scripts/python.exe -m streamlit run streamlit_app.py --server.port=8511 --server.address=127.0.0.1 --server.headless=true --browser.gatherUsageStats=false`. The real overview loaded with XGBoost, version 0.1.0, ROC-AUC 0.7615 and all navigation groups. Slide 8 contains an authentic screenshot. No applicant was submitted.

**Not executed:** training, repository pytest suite, Docker build/deployment, fresh model evaluation, or a recorded applicant prediction. The model was not deserialized to inspect metadata; application startup loaded the existing model for the requested authentic screenshot. Saved results are reported as recorded evidence, not independently reproduced experiments. PowerPoint itself was not used for visual inspection.

`artifacts/metadata.json` records `trained_at=2026-09-20T11:58:50Z`, real data, XGBoost and five-fold selection. It contains no Git commit, model checksum or dependency lock. Therefore the reviewed commit cannot be asserted to be the exact training revision. The manifest timestamp is later than metadata (11:59:20Z versus 11:58:50Z), despite current run.py writing the manifest first. This is a provenance inconsistency; do not infer one immutable, contemporaneous run bundle from timestamps alone.

## Actual architecture and workflows

**Offline:** `run.train` loads the configured application CSV, validates it, writes validation report and manifest, standardizes columns and makes stratified splits. It tunes configured candidates, performs single-split comparisons, selects by mean five-fold CV, fits the selected pipeline on TRAIN+VAL, clones/refits it in isotonic calibration folds, selects an analytical threshold, evaluates the held-out test set, computes explanation/subgroup diagnostics and saves artifacts.

**Online:** `assessment._handle_submit` validates form input through `validate_payload`, then calls `ModelService.predict` directly. `to_model_frame` builds canonical columns plus missingness indicators. The saved estimator performs feature engineering/preprocessing and returns the positive probability. Score and band derive from that probability, with a best-effort optional explanation. `ui.state.save_prediction` stores the last normalized payload/result in browser-session state for the Result and Explainability pages.

`ui.helpers.get_service` uses `st.cache_resource`. `ModelService._try_load` loads `model.joblib`, metadata and schema once per cached instance. Analytical methods read metrics/manifest/report files. `persist_prediction` returns `None`; no database or durable prediction record exists. The eight response keys are `default_probability`, `risk_score`, `risk_band`, `model_version`, `explanation_available`, `explanation`, `limitations`, `disclaimer`. `prediction_id` is conditional on a future nonempty persistence return.

The form uses saved `feature_schema.json`, while Pydantic is generated from current `schema.py`. Both share the same source contract but can drift if code and artifacts are changed separately; documentation that says they *cannot* drift is too strong. There is no active React/FastAPI/REST network boundary. Old `api` configuration labels and endpoint comments are historical remnants.

## Data, model and methodological findings

- Manifest dimensions: 307,511 rows, 122 raw columns, 24,825 positive and 282,686 negative labels. TARGET=1 denotes client payment difficulties: late payment beyond the dataset-defined threshold on at least one of the first installments. TARGET=0 denotes all other cases. This is the repository's documented meaning; the review did not independently verify the original competition data dictionary.
- Input subset: 12 numeric and 9 categorical fields, four missingness indicators, seven engineered features. Thus preprocessing receives 23 numeric columns plus nine categorical fields before one-hot expansion. ID and target are excluded. No auxiliary table joins are implemented.
- Standardization converts negative DAYS_BIRTH/DAYS_EMPLOYED to years using 365.25. Employment sentinel 365243 and actual missing values map to NaN with a flag. Each missing EXT_SOURCE gets a flag. Ratios safely map zero denominators to NaN; the external score mean skips missing values. Numeric median imputation, categorical constant fill and one-hot encoding are in the estimator pipeline. Scaling applies to Logistic Regression only. Pydantic rejects invalid categories before the encoder, although the encoder itself handles unseen categories.
- Saved validation report: 10 pass, 1 missingness warning, 0 fail. A name-based leakage heuristic is not proof of complete leakage prevention. Duplicate ID validation blocks the run; split.py also defensively deduplicates.
- Seed 42, train 196,806 / validation 49,202 / test 61,503. Validation is .20 of the non-test .80, so approximate fractions are 64/16/20, not 60/20/20.
- RF/XGBoost search: RandomizedSearchCV, 40,000 stratified TRAIN+VAL rows, eight trials, three folds, ROC-AUC. The single-split VAL comparison happens after this tuning and is not independent. Five-fold selection on TRAIN+VAL reuses chosen hyperparameters. This is not nested CV and CV results should not be interpreted as an independent estimate of the entire search process.
- XGBoost wins mean CV ROC-AUC: LR .7432634 (SD .0050310), DT .7250001 (.0048727), RF .7498013 (.0048708), XGBoost .7570623 (.0047375). RF tuned parameters are 300 trees/depth 12/min_leaf 20. XGBoost tuned parameters are 300 trees/depth 4/rate 0.03/subsample 0.7/colsample 0.8. XGBoost defaults are 400/depth 5/.05/.8/.8. LR/DT use fixed configuration. Class weighting handles imbalance; no SMOTE/oversampling claim is warranted.
- `calibrate_pipeline` clones the selected pipeline and fits CalibratedClassifierCV(method=isotonic,cv=3) on TRAIN+VAL. Fold pipelines are refitted; final probabilities average calibrated folds. The earlier full TRAIN+VAL fit supplies the uncalibrated comparison. Preprocessing is fit within the relevant folds, not on test data.
- Threshold .16649552683035532 comes from calibrated predictions on VAL rows that already entered TRAIN+VAL calibration. Selection is not fully independent and can be optimistic. The test set remains outside fitting/selection. A separate calibration/threshold holdout or properly out-of-fold selection is proposed future methodology, not an implemented feature.
- `risk_score=round(p*100)` is a risk percentage-like integer, not conventional creditworthiness. Score and bands clamp p to [0,1]. Bands use the unrounded p: Low [0,.08), Moderate [.08,.20), Elevated [.20,.40), High [.40,1]. The configuration's 1.01 exclusive upper limit includes p=1. Python round uses ties-to-even. The analytical F 1 threshold is distinct from bands.

## Recorded evaluation and diagnostic evidence

| Test metric | Saved value |
|---|---:|
| ROC-AUC | 0.7615026541 |
| PR-AUC key (average precision) | 0.2504281999 |
| Brier after calibration | 0.0675000087 |
| Brier before calibration | 0.1960558891 |
| Accuracy at 0.5 | 0.9197925304 |
| Precision / recall / F 1 at 0.5 | 0.609589 / 0.017925 / 0.034827 |
| TN / FP / FN / TP at 0.5 | 56,481 / 57 / 4,876 / 89 |
| Test precision / recall / F 1 at selected threshold | 0.261461 / 0.395166 / 0.314700 |

Source: `artifacts/metrics.json.final_test_metrics`, `.selected_threshold`, `.calibration`. PR-AUC uses average_precision_score. All-negative accuracy on this test split is 0.9192722, explaining why accuracy alone is weak evidence. After-calibration bins [.6,.7) and [.7,.8) have only 12 and 16 observations. Brier improvement compares a full-data base fit with a calibrated fold ensemble, not solely the effect of a fixed calibration map on an identical estimator.

Native `tree_feature_importances` averages across three calibration folds, matching post-transform names and averaging each feature over folds where it exists. Top features are EXT_SOURCE_MEAN .1667383, EXT_SOURCE_3 .0409456, Higher education dummy .0351922. This is native importance, not SHAP and not necessarily split frequency. Permutation importance uses original canonical input columns, the calibrated estimator and ROC-AUC drops on 5,000 sampled test rows with five repeats. Top features are EXT_SOURCE_3 .0662455, EXT_SOURCE_2 .0561675, AMT_CREDIT .0251545. Different feature spaces/methods produce different rankings; neither establishes causation.

Local explanations use the first calibration fold's underlying model. Successful SHAP explains that model's output, not an exact decomposition of the averaged isotonic-calibrated probability. SHAP installed in saved metadata does not prove success on any particular prediction. Tree fallback is unsigned global importance, with direction `n/a`, not individual signed attribution. `explanation_available` can be true for fallback contributions.

Fairness code and saved results contain gender and age subgroup diagnostics at the selected threshold. Selection rate means fraction flagged at p>=threshold, not approval rate. Gender F/M: n=40,561/20,940, selection=9.36%/17.71%, FNR=67.03%/51.76%, FPR=7.58%/14.26%, AUC=.7528/.7637. Age <30/60+: n=8,905/7,191, selection=24.26%/2.63%, FNR=46.42%/87.11%. Gender XNA n=2 is low-support; AUC is undefined. Groups below 100 are flagged and excluded from disparity ratios. This is limited one-split subgroup evidence, not a formal fairness audit, mitigation, certification or legal clearance.

## Material conflicts and scope distinctions

| Source statement | Current evidence and correction |
|---|---|
| README architecture says eight pages; brain frontend spec lists eight | streamlit_app.py has 13 pages in four groups (2+3+5+3). Treat eight as outdated. |
| README responsible-use section says no fairness analysis performed | fairness.py and metrics.fairness implement/report diagnostics. Current RESPONSIBLE_USE.md and MODEL_CARD.md already distinguish diagnostics from no formal audit; their no-audit statements are accurate. |
| brain context/plan/API contract describe FastAPI/REST and older frontend | Streamlit imports ModelService in-process. docs/API_DOCUMENTATION.md explicitly marks its endpoint details historical. No active endpoints should be diagrammed. |
| README/model-card/artifact wording calls the selected threshold validation-based | Correct but incomplete: VAL participated in calibration. Current run.py acknowledges reuse but minimizes its risk in a comment; the presentation avoids the unsupported claim of negligible optimism. |
| Preprocess/model comments describe fitting on TRAIN only | In final fitting/CV/calibration the relevant fitting data include TRAIN+VAL partitions. The valid guarantee is exclusion of test statistics. |
| Some explanatory text calls native importance Gini or split count | The implementation reads each estimator's feature_importances_. Label it native tree importance and distinguish permutation ROC-AUC drops. |
| Result/explainability UI says per-applicant and shows a signed caption for any contributions | Fallback outputs global unsigned importances. Present method/scope explicitly. The fallback note also incorrectly says SHAP not installed when failure could instead trigger fallback. |
| MODEL_CARD sparse-bin discussion mentions 2-5 cases | Current saved after-calibration high bins have 12 and 16. Use artifact counts. |
| Reference says synthetic fixture for offline tests only | Docker entrypoint can train a synthetic demo when no model/real CSV exists. Say tests and demonstrations; saved shipped metadata remains real. |
| Requirements comments call XGBoost/SHAP optional | HEAD requirements.txt also includes them; pyproject optional extras and requirements-extras provide them. Distinguish training's optional candidate skip from loading a saved XGBoost model, which needs its dependency. |
| Reproducible pipeline and artifact provenance wording | Seed/config support repeatability, but dependency ranges, absent training commit/hash, timestamp inconsistency and deleted local requirements limit stronger claims. |
| Persistence seam and session state | Session state is implemented; durable prediction persistence/authentication/monitoring are future work. Docker existence is not proof of production deployment. |

## Slide-by-slide correction table

Verdicts describe the original reference claim. Accurate content is retained; incomplete claims receive qualifications. Evidence abbreviations: **M**=`artifacts/metrics.json`, **MD**=`artifacts/metadata.json`, **MF**=`data/manifest.json`, **CFG**=`config/config.yaml`. Paths below are repository-relative.

| Slide number | Existing claim | Evidence | Verdict | Corrected wording |
|---:|---|---|---|---|
| 1 | Estimates repayment difficulty, not approval | README; MF.target_meaning | accurate | Estimates repayment difficulty and supports human review. |
| 1 | Home Credit, calibrated XGBoost, Streamlit | MD; streamlit_app.py | accurate | Retain project identity, four names/institution/guide placeholders and model context in notes. |
| 2 | Imbalanced target and probability objectives | MF.target_distribution; M.final_test_metrics | incomplete | About 8.07% positives; accuracy can mask missed positives. |
| 2 | Score=round(probability×100) | pipeline/risk_score.py | accurate | Retain formula; higher is greater risk, not conventional creditworthiness. |
| 2 | Four named bands and sequential score-to-band arrow | CFG.risk_score; probability_to_band | misleading | Show exact probability intervals; both score and band derive from unrounded probability. |
| 2 | Performance, quality, explanation/fairness views | streamlit_app.py; ui/pages/ | accurate | Retain as implemented analytical/diagnostic views. |
| 3 | Streamlit calls ModelService directly; no HTTP API | streamlit_app.py; helpers.get_service; service.py | accurate | Retain direct in-process boundary and cached service. |
| 3 | Data→ML→serving→UI single chain | pipeline/run.py; artifacts.py; service.py | incomplete | Separate offline training and online inference with saved-artifact reuse arrow. |
| 3 | Candidate models, then selection & tuning | run.train; tuning.py | misleading | Tune configured candidates before comparisons and five-fold model selection. |
| 3 | Form→validation→features→calibrated model | validation.py; to_model_frame; models.py | incomplete | Add canonical frame/missingness, saved transforms, best-effort explanation and no retraining. |
| 4 | 307511 rows,122 columns,8.07%, class counts | MF | accurate | Retain raw dimensions and counts. |
| 4 | Dataset/preprocessing without feature-subset count | schema.py; feature_schema.json; preprocess.py | incomplete | 12 numeric+9 categorical application inputs,4 flags,7 engineered features. |
| 4 |20,000 synthetic fixture for offline tests only | CFG.synthetic; docker/entrypoint.sh; MD.synthetic | misleading | Fixture supports tests/demos; saved model records real data. |
| 4 | Sentinel, median, linear scaling, one-hot handling | ingestion.py; preprocess.py | accurate | Retain, clarify missingness flags and serving category validation. |
| 5 | Seven feature descriptions use “vs” | features.add_engineered_features | incomplete | Replace informal comparisons with exact numerator/denominator formulas. |
| 5 | Age/employment derived from DAYS_* | ingestion.standardize | incomplete | Specify negative day-count/365.25, sentinel exception and years supplied by serving. |
| 5 | Shared feature contract | FeatureEngineer; service.to_model_frame | accurate | Shared saved engineering, with zero-safe division/NaN handling explained. |
| 6 | Four models and CV chart | M.per_model_cross_validation | accurate | Retain means and truncated-axis label, add fold SD. |
| 6 | Split counts and 40,000-row randomized search | M.split_diagnostics; M.hyperparameter_tuning; run.py | incomplete |64/16/20 split; search samples TRAIN+VAL, RF/XGBoost only,8 trials/3 folds. |
| 6 | Five-fold selection and refit/calibrate | tuning.cross_val_select; calibrate_pipeline | incomplete | Three-fold calibration clones/refits base pipelines; validation reused, not nested CV. |
| 6 | XGBoost 300/depth 4/.03/.7/.8 | M.hyperparameter_tuning.xgboost | accurate | Retain as tuned values; distinguish config defaults 400/depth 5/.05/.8/.8 in notes. |
| 7 | ROC-AUC 0.7615, PR-AUC 0.2504, Brier 0.0675, accuracy 0.9198 | M.final_test_metrics; M.calibration | accurate | Retain rounded metrics; label PR-AUC as AP and accuracy threshold. |
| 7 | Confusion matrix TN 56481 FP 57 FN 4876 TP 89 | M.final_test_metrics.confusion_matrix | accurate | Retain explicit Actual rows and Predicted columns at 0.5. |
| 7 | Precision 0.6096 recall 0.0179 F 1.0348 | M.final_test_metrics | accurate | Retain and explain all-negative accuracy≈.9193. |
| 7 | Threshold≈.1665 validation-selected, test P/R/F 1 | M.selected_threshold; run.train | incomplete | Selection reused calibration rows and can be optimistic; test remained outside fitting/selection. |
| 8 |13 pages/four groups; cached direct service | streamlit_app.py; helpers.py | accurate | Retain, name groups and include authentic overview screenshot. |
| 8 | Probability 0.18 example explicitly illustrative | risk_score.py; CFG | accurate | Retain illustration only: score 18, Moderate; no prediction claimed. |
| 8 | Response lists six fields | ModelService.predict | incomplete | Add model_version and explanation_available; prediction_id absent with no-op persistence. |
| 8 | explanation = feature contributions | local_explanation; service.predict | misleading | Optional method-labeled output; it may be unsigned global fallback, not applicant attribution. |
| 9 | SHAP for trees when available | explain.local_explanation; MD backends | incomplete | SHAP attempted on first calibration fold; availability does not guarantee success. |
| 9 | Native fallback and combined global-driver list | M.global_importance; M.permutation_importance | misleading | Separate fold-averaged native and test permutation rankings; unsigned fallback is global. |
| 9 | Gender/age diagnostics, selection/FNR/FPR/AUC | fairness.py; M.fairness | accurate | Retain with compact subgroup values and low-support XNA n=2. |
| 9 | Selection rate without definition | fairness._slice_metrics | incomplete | Fraction flagged at p≥diagnostic threshold, not approved. |
| 9 | Diagnostic, not certification | M.fairness.disclaimer | accurate | Retain, add no formal audit/mitigation/legal clearance and README conflict in notes. |
|10| Reproducible delivered pipeline/artifacts | implementation; pyproject; metadata | incomplete | Shared pipeline and saved estimator/reports delivered; avoid exact reproducibility guarantee. |
|10| Application-only, shift, sparse bins, low recall | MF; M; code | accurate | Retain, with saved high-bin counts in notes. |
|10| Limitations omit validation reuse/local approximation | run.py; explain.py | incomplete | Add validation reuse, approximate local explanations and limited subgroup diagnostics. |
|10| Multi-table, monitoring, auth, stored predictions future | serving.persist_prediction; current app/Docker | accurate | Retain future status, add independent calibration/threshold validation. |

## Ten project-specific viva questions and answers

1. **Why is this repayment-risk estimation rather than loan approval prediction?** TARGET is documented as payment difficulties, not an approval decision. The service returns probability, score, band and limitations; it does not approve/decline applicants. Evidence: MF.target_meaning; service.predict; README.
2. **Why is the split 64/16/20 when both configuration fractions are 0.20?** split.make_splits removes 20% test first, then assigns 20% of the remaining 80% to validation: .2×.8=.16. Saved counts are 196806/49202/61503. Evidence: split.py and M.split_diagnostics.
3. **Which parts of model selection reuse validation data?** Tuning receives TRAIN+VAL, then the tuned candidates are compared on VAL and scored by five-fold CV on TRAIN+VAL. Calibration also uses TRAIN+VAL, followed by threshold selection on those VAL rows. This is not nested CV. Test stays excluded. Evidence: run.train and tuning.py.
4. **What does three-fold isotonic calibration actually save?** calibrate_pipeline clones the base pipeline and CalibratedClassifierCV refits fold pipelines/calibrators. The saved estimator averages calibrated fold probabilities and includes engineering/preprocessing. It does not simply attach calibration to the already-fitted full-data model. Evidence: calibrate.py; models.py; artifacts.py.
5. **Why is 91.98% accuracy insufficient?** The test positive rate is 8.0728%; all-negative accuracy is 91.9272%. At 0.5 the model catches 89 of 4965 positives (recall 1.79%). ROC-AUC, average precision, calibration and threshold trade-offs add necessary context. Evidence: M.final_test_metrics.
6. **How do training and serving keep features consistent?** Ingestion converts raw day counts to years and creates missingness flags; serving accepts years and mirrors those flags. FeatureEngineer and fitted preprocessing live inside saved pipelines. UI schema JSON and Pydantic originate in schema.py, although artifact/code version mismatch could still cause drift. Evidence: ingestion.standardize; service.to_model_frame; schema.py; preprocess.py.
7. **Is 18/100 a good credit score, and what happens at band boundaries?** No: it means round(.18×100), increasing with estimated payment difficulty. .18 is Moderate. Exactly 0.08 is Moderate, .20 Elevated, .40 High and 1.00 High. Bands use unrounded probability, independently of the analytical threshold. Evidence: risk_score.py; CFG.risk_score.bands.
8. **Why can native and permutation rankings differ?** Native importance averages transformed-feature importances over calibration folds. Permutation shuffles original canonical input columns and measures ROC-AUC drops on 5000 test rows, repeated five times, through the full calibrated pipeline. Neither gives causal effects. Evidence: explain.py; M.global_importance/permutation_importance.
9. **Does a successful SHAP installation guarantee a faithful explanation of the served probability?** No. Per-request SHAP can fail. It uses the first fold's underlying model, not the final averaged calibrated probability. Tree fallback is unsigned global importance, never individualized signed risk reasons. Evidence: explain._unwrap/local_explanation; service.predict.
10. **What fairness and production functionality is actually delivered?** Gender/age subgroup diagnostics exist, including flagged rates and error rates, with low-support groups marked. No formal audit/mitigation/certification is established. Streamlit serves directly and session state retains the last result; persistence remains a no-op, and authentication/monitoring are future work. Evidence: fairness.py; M.fairness; ui/state.py; service.persist_prediction; Docker configuration.
