# Evaluation Report — Intelligent Loan Risk Assessment System

_All figures below are read directly from `artifacts/metrics.json` (generated_at: 2026-09-20T11:58:50Z)._

> ## Real-data results, with important scope limits
> The model was trained and evaluated on the **real Kaggle Home Credit Default Risk** `application_train.csv` (`dataset_source: "real"`, `synthetic: false`). The metrics below are genuine held-out performance on that population. **However**, this system deliberately uses only the **application-level feature subset** — it does **not** join the auxiliary bureau / previous-application / installment tables that the strongest Kaggle solutions rely on. So these numbers represent an honest application-only baseline, not the ceiling achievable on this dataset, and they are **not** evidence of fairness, calibration in deployment, or fitness for real lending decisions. See `docs/RESPONSIBLE_USE.md`.

This report documents the **default-risk** prediction task (predicting repayment difficulty / `TARGET=1`), **not** loan approval. No number here has been estimated, rounded up, or invented; values are quoted as stored in `artifacts/metrics.json`. Where the artifact does not contain a value, that is stated explicitly.

---

## 1. Dataset Version

Source of truth: `data/manifest.json` and the `dataset_source` field of `artifacts/metrics.json`.

| Property | Value |
|---|---|
| Dataset name | Home Credit Default Risk |
| Source kind | **real** (`source_kind: "real"`) |
| Source URL | https://www.kaggle.com/competitions/home-credit-default-risk |
| Primary file | `application_train.csv` |
| Retrieval date | 2026-09-20T07:41:16Z |
| License / usage | Kaggle Home Credit Default Risk competition rules apply; must be accepted on Kaggle |
| Row count | 307,511 |
| Column count | 122 (raw file); the pipeline consumes an application-level subset |
| Target column | `TARGET` |
| Positive rate | 0.0807 |
| Target distribution | `TARGET=0`: 282,686 · `TARGET=1`: 24,825 |

**Target meaning (from manifest):** `TARGET=1` = client with payment difficulties (late payment beyond the dataset-defined threshold on at least one of the first installments); `TARGET=0` = all other cases. The manifest notes this should be confirmed against `HomeCredit_columns_description.csv`.

**Manifest-declared known limitations:**
- Serving subset uses application-level features only; supporting tables (bureau, previous_application, ...) are an offline/advanced extension and not required for a single real-time prediction.
- `DAYS_EMPLOYED` uses the documented `365243` sentinel for pensioners/unemployed; treated as missing with an indicator. The validation report counts **55,374** rows using this sentinel.

**Real-data missingness (from `data/manifest.json`, relevant to the model's features):** `EXT_SOURCE_1` is missing in **56.4%** of rows, `EXT_SOURCE_3` in **19.8%**, `EXT_SOURCE_2` in **0.2%**, and `OCCUPATION_TYPE` in **31.4%**. This heavy, structured missingness is why the pipeline adds explicit missingness indicators (`EXT_SOURCE_1_MISSING`, `EXT_SOURCE_2_MISSING`, `EXT_SOURCE_3_MISSING`, `DAYS_EMPLOYED_MISSING`) rather than silently imputing — and §10 shows `EXT_SOURCE_1_MISSING` is among the most important features, confirming the missingness itself carries signal on real data.

---

## 2. Split Strategy

Source: `split_diagnostics` in `artifacts/metrics.json`, with configuration from `config/config.yaml` (`split` section).

| Property | Value |
|---|---|
| Input rows | 307,511 |
| Dropped duplicate IDs (leakage guard) | 0 |
| Train rows (`n_train`) | 196,806 |
| Validation rows (`n_val`) | 49,202 |
| Test rows (`n_test`) | 61,503 |
| Stratified | true |
| Random seed | 42 |
| Train positive rate | 0.0807 |
| Validation positive rate | 0.0807 |
| Test positive rate | 0.0807 |

Configured split ratios (`config/config.yaml`): `test_size: 0.20`, `val_size: 0.20` (fraction of the train-remainder), `stratify: true`. The `split` block no longer defines a `time_column` key, so no time-aware split was performed. Deduplication ran and found 0 duplicate applicant IDs, so no rows were dropped for leakage. Stratification held the positive rate at the 0.0807 population rate across all three splits.

---

## 3. Models Trained

Source: `per_model_validation` and `skipped_models` in `artifacts/metrics.json`.

Four models were trained and evaluated on identical splits:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost — **selected model**

**Skipped models:** none. `skipped_models` is an empty list (`[]`), so all four candidate models trained successfully; `metadata.json` confirms `xgboost_installed: true` and `shap_installed: true`.

**Selection rule:** the model was selected by **`roc_auc`** (`selection_metric: "roc_auc"`), not by accuracy alone, per `03_ML_REQUIREMENTS.md`. Selection uses `selection_basis: "cross_validation_5fold_mean"` — a 5-fold stratified cross-validation on TRAIN+VAL, choosing the highest **mean** ROC-AUC (not a single validation-split argmax), with hyperparameters tuned via RandomizedSearchCV (§4.1). **XGBoost** had the best mean 5-fold CV ROC-AUC (0.7571 ± 0.0047), ahead of Random Forest (0.7498 ± 0.0049), Logistic Regression (0.7433 ± 0.0050), and Decision Tree (0.7250 ± 0.0049), and was selected. Its single validation-split ROC-AUC (0.7566) is reported head-to-head in §5.1 but is not the selection basis.

**Validation gate summary (`validation_summary`):** 10 pass, 1 warn, 0 fail. The single warning is the expected missing-value report on real data (67 columns contain missing values); no check failed.

---

## 4. Hyperparameters

Fixed config values from the `models` section of `config/config.yaml`. For models covered by the `tuning` block (`random_forest`, `xgboost`), the **selected** estimator uses RandomizedSearchCV-tuned parameters (§4.1), not these fixed values. Calibration parameters from the `calibration` section; CV folds (5) from the `selection` block.

| Model | Hyperparameters |
|---|---|
| Logistic Regression | `C: 1.0`, `max_iter: 1000`, `class_weight: balanced` |
| Decision Tree | `max_depth: 6`, `min_samples_leaf: 50`, `class_weight: balanced` |
| Random Forest | `n_estimators: 300`, `max_depth: 12`, `min_samples_leaf: 20`, `n_jobs: -1`, `class_weight: balanced` |
| XGBoost **(selected)** | Config defaults (`n_estimators: 400`, `max_depth: 5`, `learning_rate: 0.05`, `subsample: 0.8`) are **superseded by the tuned params actually used**: `n_estimators: 300`, `max_depth: 4`, `learning_rate: 0.03`, `subsample: 0.7`, `colsample_bytree: 0.8` (colsample_bytree unchanged). `scale_pos_weight` still set from the train-split class balance. |

**Calibration configuration:** `method: isotonic`, `cv: 3`. **Selection metric:** `roc_auc`. **CV folds (selection):** 5 (from the `selection.cv_folds` key; drives 5-fold stratified cross-validation model selection on TRAIN+VAL by best mean ROC-AUC, not an evaluation setting). **Global random seed:** 42.

### 4.1 Hyperparameter tuning

Source: `hyperparameter_tuning` in `artifacts/metrics.json` (`metadata.json` records `hyperparameters_tuned: true`). Models in the `tuning` block are tuned with **RandomizedSearchCV** (`n_iter: 8`, scored by `roc_auc`) on a 40,000-row stratified subsample, then refit on the full training data. The selected estimator uses the tuned parameters below rather than the fixed config defaults in §4.

| Model | Tuned CV ROC-AUC | Selected params |
|---|---|---|
| Random Forest | 0.7371 | `n_estimators: 300`, `max_depth: 12`, `min_samples_leaf: 20` |
| XGBoost **(selected)** | 0.7401 | `n_estimators: 300`, `max_depth: 4`, `learning_rate: 0.03`, `subsample: 0.7`, `colsample_bytree: 0.8` |

The tuned CV score here (a subsample-based search score) is not directly comparable to the full 5-fold selection CV in §3; it is the search's own internal criterion used to pick each model's hyperparameters.

---

## 5. Metrics

### 5.1 Per-model validation comparison

All rows evaluated on the validation split (`n = 49202`, decision threshold `0.5`, positive rate `0.0807284256737531`). Values quoted exactly from `per_model_validation[*].val_metrics`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.6819438234218121 | 0.15686159271231268 | 0.6719536757301108 | 0.25434792967074854 | 0.7419027098600217 | 0.21834771884170817 | 0.2059024041663295 |
| Decision Tree | 0.6648306979391082 | 0.1485993375624544 | 0.6664149043303121 | 0.24301124627036952 | 0.7226657712766729 | 0.18894690548714888 | 0.20949097885629564 |
| Random Forest | 0.7241575545709524 | 0.17199671996719967 | 0.6336858006042296 | 0.2705578845533699 | 0.7481138030329041 | 0.22504923913300393 | 0.1881380649890652 |
| **XGBoost (selected)** | 0.7003983577903338 | 0.16694501144306303 | 0.6795065458207452 | 0.26803714186404487 | **0.7565655893487444** | 0.23798123242869207 | 0.19749189502694128 |

XGBoost's validation ROC-AUC is 0.7566 and PR-AUC 0.2380. Selection is by 5-fold stratified CV mean ROC-AUC on TRAIN+VAL (XGBoost 0.7571 ± 0.0047, the best mean — see §3), not by the highest single-split validation ROC-AUC. Note the models with `class_weight: balanced` (all but XGBoost, which uses `scale_pos_weight`) trade accuracy for recall at the 0.5 threshold — e.g. Logistic Regression reaches recall 0.672 but accuracy only 0.682. This is expected under an ~8% positive rate and is exactly why ROC-AUC, not accuracy, is the selection criterion.

### 5.2 Final held-out TEST metrics (selected model: XGBoost)

Evaluated once on the held-out test split (`n = 61503`, decision threshold `0.5`, positive rate `0.08072776937710356`). Values quoted exactly from `final_test_metrics`.

| Metric | Value |
|---|---|
| Accuracy | 0.9197925304456693 |
| Precision | 0.6095890410958904 |
| Recall | 0.017925478348439074 |
| F1 | 0.03482684406182743 |
| ROC-AUC | 0.7615026541337846 |
| PR-AUC | 0.25042819987443476 |
| Brier score | 0.06750000869660358 |

The high test accuracy (0.9198) is essentially the trivial "predict every applicant repays" baseline (1 − 0.0807 = 0.9193), which shows why accuracy is not a meaningful headline for this imbalanced task. ROC-AUC (0.7615) and PR-AUC (0.2504) are the informative discrimination measures. At the default 0.5 threshold recall is very low (0.0179) because the calibrated probabilities cluster well below 0.5 — see the confusion matrix and threshold analysis below, which is why the pipeline reports a validation-selected operating point (§8) rather than defaulting to 0.5.

For context: published Home Credit solutions exceed ~0.79–0.80 ROC-AUC, but only by engineering features across the auxiliary bureau/previous-application/installment tables. An application-only ROC-AUC of ~0.762 is a credible, honest baseline for the feature subset this system deliberately restricts itself to.

---

## 6. Confusion Matrix (Final TEST, threshold 0.5)

Rendered from `final_test_metrics.confusion_matrix` (`n = 61503`).

|  | Predicted: No default (0) | Predicted: Default (1) |
|---|---|---|
| **Actual: No default (0)** | TN = 56,481 | FP = 57 |
| **Actual: Default (1)** | FN = 4,876 | TP = 89 |

Totals: 56,481 + 57 + 4,876 + 89 = 61,503. Of 4,965 actual defaulters (FN + TP), the model flags only 89 at the 0.5 threshold; of 56,538 non-defaulters, only 57 are false-flagged. At threshold 0.5 the model is extremely conservative about predicting "default" — the operating point must move well below 0.5 to be useful (§8).

---

## 7. Calibration

Source: `calibration` in `artifacts/metrics.json`. Method: **isotonic**, `cv: 3` (from config and `metadata.json`).

| | Brier score |
|---|---|
| Before calibration | 0.19605588912963867 |
| After calibration | 0.06750000869660358 |

Lower Brier is better. Isotonic calibration (cv=3) reduced the Brier score from 0.1961 to 0.0675 on the test set, and `metadata.json` records `improved: true`. The calibrated Brier (0.06750000869660358) equals the `final_test_metrics.brier_score`, as expected since final metrics are computed on the calibrated model.

### 7.1 Reliability curve — after calibration

From `calibration.curve_after`. Each row is a probability bin: `mean_predicted` is the average predicted probability in the bin; `observed_frequency` is the actual default rate among those cases; `count` is the number of cases.

| Bin | Mean predicted | Observed frequency | Count |
|---|---|---|---|
| 0.0–0.1 | 0.043718847757490346 | 0.04314343222193052 | 45,708 |
| 0.1–0.2 | 0.14032422114831558 | 0.13825757575757575 | 10,560 |
| 0.2–0.3 | 0.23357738428041108 | 0.24379481522338664 | 3,626 |
| 0.3–0.4 | 0.34707466080679117 | 0.3668085106382979 | 1,175 |
| 0.4–0.5 | 0.44942149619951294 | 0.4479166666666667 | 288 |
| 0.5–0.6 | 0.5457374736414118 | 0.6016949152542372 | 118 |
| 0.6–0.7 | 0.651282681359185 | 0.6666666666666666 | 12 |
| 0.7–0.8 | 0.7409787997603416 | 0.625 | 16 |

In the well-populated bins (0.0–0.5, together 61,357 of the 61,503 test cases), mean predicted probability tracks observed frequency closely — calibration is good where the vast majority of predictions live. The higher bins are unreliable indicators only because their counts are tiny (down to 12 and 16 cases); those rows should not be read as evidence of mis- or well-calibration. `calibration.curve_before` (10 bins) is also present in the artifact and shows the pre-calibration model systematically **over-predicting** risk (e.g. mean predicted 0.079 vs observed 0.006 in the lowest bin) — which isotonic calibration corrects.

---

## 8. Threshold Analysis

Source: `threshold_analysis` in `artifacts/metrics.json` (evaluated on the test split, `n = 61503`, 4,965 actual positives). Shows how precision, recall, F1, and the flagged rate move as the decision threshold changes. Values quoted exactly.

| Threshold | Precision | Recall | F1 | Flagged rate | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.1289103312642124 | 0.8449144008056395 | 0.22369157757218652 | 0.5291124010210884 | 4195 | 28347 | 770 | 28191 |
| 0.10 | 0.1894903450459006 | 0.6028197381671702 | 0.2883429672447014 | 0.2568167406467977 | 2993 | 12802 | 1972 | 43736 |
| 0.15 | 0.24619608740418716 | 0.43343403826787513 | 0.3140230555960893 | 0.14212314846430255 | 2152 | 6589 | 2813 | 49949 |
| 0.20 | 0.29283667621776505 | 0.30876132930513595 | 0.30058823529411766 | 0.08511779913174967 | 1533 | 3702 | 3432 | 52836 |
| 0.25 | 0.36181307661452067 | 0.18167170191339377 | 0.24188790560471976 | 0.040534608067899124 | 902 | 1591 | 4063 | 54947 |
| 0.30 | 0.4033561218147918 | 0.13071500503524672 | 0.19744447824764222 | 0.026161325463798513 | 649 | 960 | 4316 | 55578 |
| 0.35 | 0.4387646432374867 | 0.08298086606243706 | 0.13956639566395665 | 0.015267547924491488 | 412 | 527 | 4553 | 56011 |
| 0.40 | 0.5023041474654378 | 0.04390735146022155 | 0.08075569549916652 | 0.007056566346357088 | 218 | 216 | 4747 | 56322 |
| 0.45 | 0.5514705882352942 | 0.030211480362537766 | 0.057284704983769336 | 0.004422548493569419 | 150 | 122 | 4815 | 56416 |
| 0.50 | 0.6095890410958904 | 0.017925478348439074 | 0.03482684406182743 | 0.0023738679414012326 | 89 | 57 | 4876 | 56481 |
| 0.55 | 0.6438356164383562 | 0.0094662638469285 | 0.018658197697499008 | 0.0011869339707006163 | 47 | 26 | 4918 | 56512 |
| 0.60 | 0.6428571428571429 | 0.0036253776435045317 | 0.0072100941317844985 | 0.00045526234492626377 | 18 | 10 | 4947 | 56528 |
| 0.65 | 0.5909090909090909 | 0.0026183282980866062 | 0.005213555243633447 | 0.0003577061281563501 | 13 | 9 | 4952 | 56529 |
| 0.70 | 0.625 | 0.002014098690835851 | 0.004015257980325236 | 0.0002601499113864364 | 10 | 6 | 4955 | 56532 |
| 0.75 | 1.0 | 0.0010070493454179255 | 0.002012072434607646 | 0.00008129684730826138 | 5 | 0 | 4960 | 56538 |
| 0.80 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 4965 | 56538 |
| 0.85 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 4965 | 56538 |
| 0.90 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 4965 | 56538 |
| 0.95 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 4965 | 56538 |

**Selected operating point** (`selected_threshold` in the artifact): threshold **0.16649552683035532**, chosen on the **validation** set (never the test set, to avoid leakage). At selection it achieves validation F1 **0.3128786398280242** (precision 0.2556691152986266, recall 0.4030715005035247). Its **test-set generalization** at the same threshold is F1 **0.31470045713369155** (precision 0.26146055437100213, recall 0.39516616314199393). The threshold is reported for analysis and is **not auto-applied** at serving; the API's risk bands use the configured probability cut points.

The default 0.5 threshold is a poor operating point on this data: it yields test F1 = 0.035 and catches only ~2% of defaulters (recall 0.0179). The validation-selected ~0.167 threshold lifts test F1 to ~0.315 by trading precision down (~0.262) for materially higher recall (~0.395). The table makes the trade-off explicit: lowering the threshold catches more defaulters (higher recall, higher TP) at the cost of flagging far more non-defaulters (higher FP, higher flagged rate). At threshold 0.05, recall reaches 0.845 but 52.9% of all applicants are flagged and precision collapses to 0.129.

---

## 9. Error Analysis (interpretation of the real numbers above)

This section interprets the confusion matrix (§6) and threshold table (§8). The numbers are real held-out results; the interpretation is analytical and still bounded by the application-subset scope limit.

**Class imbalance sets the terms.** The positive (default) rate is ~8.07%, so 4,965 of 61,503 test applicants actually default. Any model can reach ~92% accuracy by predicting "no default" for everyone (baseline 0.919), so accuracy carries almost no signal here. FP vs FN is the real trade-off.

**At threshold 0.5 the model is badly mis-tuned toward false negatives.** The test confusion matrix shows FN = 4,876 and TP = 89: the model misses 4,876 of 4,965 real defaulters while raising only 57 false positives. In lending, a **false negative** (a would-be defaulter scored as low-risk) is typically the costlier error — it maps to an approved loan that later defaults. A **false positive** (a repaying applicant flagged as risky) mostly costs a lost-good-customer / extra-review burden. The ~86:1 FN:FP ratio at 0.5 (4,876/57 ≈ 85.5) means the default threshold optimizes almost entirely against the cheaper error, which is the wrong direction for risk screening — hence the validation-selected lower threshold.

**The threshold curve shows the lever to fix it.** Because the calibrated probabilities cluster low (74.3% of the test set lands in the 0.0–0.1 bin, §7.1), the decision threshold must move well below 0.5. Moving to the validation-selected ~0.166 raises test recall from ~0.018 to ~0.395 and test F1 from ~0.035 to ~0.315. If the business cost of a missed defaulter dominates, an even lower threshold (e.g. 0.10, recall 0.603; or 0.05, recall 0.845) buys more recall — but §8 quantifies the price: threshold 0.10 flags 25.7% of applicants (FP = 12,802) and threshold 0.05 flags 52.9% (FP = 28,347). The right operating point is a business decision about the relative cost of FN vs FP, and this report deliberately does not pick one; it surfaces the full curve so a human can choose.

**Discrimination is modest — an application-only baseline.** Test ROC-AUC is 0.7615 and PR-AUC is 0.2504. PR-AUC is the more honest ceiling under imbalance, and ~0.250 (against a 0.081 no-skill baseline) says the ranking is meaningfully better than random but far from decisive. No threshold in §8 achieves both high precision and high recall simultaneously — the best F1 anywhere in the table is ~0.314 (threshold 0.15). The model is a moderate ranker, not a decisive classifier, which reinforces that its output should inform a human reviewer rather than gate decisions automatically. Adding the auxiliary Home Credit tables is the known path to stronger discrimination and is out of scope for this application-level system.

---

## 10. Global Feature Importance (context)

Source: `global_importance` in `artifacts/metrics.json`. Method: **`tree_feature_importances`** (XGBoost native importances), **averaged across the 3 calibration folds** (`folds_averaged: 3`). The artifact explicitly labels this as **"Association with predicted risk, not causation. Averaged across 3 calibration folds."**

Top features by importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `EXT_SOURCE_MEAN` | 0.16673832635084787 |
| 2 | `EXT_SOURCE_3` | 0.040945593267679214 |
| 3 | `NAME_EDUCATION_TYPE_Higher education` | 0.03519216055671374 |
| 4 | `EXT_SOURCE_2` | 0.03492886448899905 |
| 5 | `CODE_GENDER_M` | 0.028533594061930973 |
| 6 | `CODE_GENDER_F` | 0.02694863888124625 |
| 7 | `EXT_SOURCE_3_MISSING` | 0.026703275740146637 |
| 8 | `CREDIT_GOODS_RATIO` | 0.024684120590488117 |
| 9 | `EXT_SOURCE_1_MISSING` | 0.020947180067499478 |
| 10 | `NAME_EDUCATION_TYPE_Secondary / secondary special` | 0.020633169760306675 |
| 11 | `NAME_INCOME_TYPE_Working` | 0.020576393231749535 |
| 12 | `FLAG_OWN_CAR_Y` | 0.01990690641105175 |

Two observations matter for responsible use:

1. **The external-source score family dominates.** The engineered `EXT_SOURCE_MEAN` is the single most important feature (0.167), and `EXT_SOURCE_3` (rank 2) / `EXT_SOURCE_2` (rank 4) also rank highly. Critically, the top-7 missingness indicator is **`EXT_SOURCE_3_MISSING` (rank 7, 0.027)**, confirming that whether an external score is *present* is itself predictive on real data — vindicating the decision to model missingness explicitly rather than impute it away. On real Home Credit data the external scores carry heavy structured missingness (§1), so these indicators are populated and meaningful.

2. **Protected-attribute proxies appear.** `CODE_GENDER_M` (rank 5, 0.029) and `CODE_GENDER_F` (rank 6, 0.027) are direct protected attributes, and features like `NAME_EDUCATION_TYPE_*` and `NAME_INCOME_TYPE_*` also appear. Their presence is a statistical **association**, not a causal or endorsed decision factor. `metrics.json` now includes a fairness diagnostic (§12) that slices selection rate, FNR, FPR, and ROC-AUC by `CODE_GENDER` and `AGE_BAND` at the operating threshold, with simple disparity ratios. This is a screen for human review, **not** a fairness certification — the disparities it surfaces (e.g., male selection rate ~0.177 vs female ~0.094) are exactly why the review flagged in `docs/RESPONSIBLE_USE.md` remains required before any real-world use.

---

## 11. Permutation Feature Importance

Source: `permutation_importance` in `artifacts/metrics.json`. Method: **`permutation_importance_roc_auc`** — model-agnostic, computed on the held-out **TEST** split (`n_repeats: 5`, 5,000-sample cap), with the per-column drops aggregated back to the **original input features**. Each value is the mean drop in ROC-AUC when that feature is randomly shuffled. The artifact labels this as **"Association with predictive value, not causation."**

| Rank | Feature | Mean ROC-AUC drop | Std |
|---|---|---|---|
| 1 | `EXT_SOURCE_3` | 0.06624550442623234 | 0.008401998924190067 |
| 2 | `EXT_SOURCE_2` | 0.05616746493284595 | 0.006004440358783353 |
| 3 | `AMT_CREDIT` | 0.02515450104877357 | 0.0025956792698282553 |
| 4 | `AMT_ANNUITY` | 0.025007192401518406 | 0.002576962882807134 |
| 5 | `EXT_SOURCE_1` | 0.018022996057479347 | 0.005258962597280224 |
| 6 | `AMT_GOODS_PRICE` | 0.017287124284181797 | 0.002651035396045877 |

Permutation importance is a useful cross-check on the native tree importances in §10: it measures actual predictive contribution on unseen data rather than split-frequency inside the trees. The external-source scores again dominate (`EXT_SOURCE_3` and `EXT_SOURCE_2` account for the two largest drops), and the loan-amount family (`AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`) follows. Because this method aggregates to original features, the protected attribute `CODE_GENDER` (0.0036) ranks well below the financial features here — a different lens on the raw one-hot ranking in §10, and one reason the fairness diagnostic (§12) is reported alongside importances rather than inferred from them.

---

## 12. Fairness Diagnostic

Source: `fairness` in `artifacts/metrics.json`. **This is a diagnostic only.** The artifact disclaimer states it plainly: *"Diagnostic only. Raw group metrics + simple disparity ratios on one held-out split. NOT a fairness certification, legal-compliance assessment, or mitigation. Disparities are flags for human review."* It is **not** a fairness audit or certification, and it does not clear the model for real-world use.

Metrics are sliced by `CODE_GENDER` (F / M / XNA) and `AGE_BAND`, computed at the selected operating threshold (**0.16649552683035532**), reporting per-group selection rate, false-negative rate (FNR), false-positive rate (FPR), ROC-AUC, and precision, plus `low_support` flags for tiny groups.

**By `CODE_GENDER`:**

| Group | n | Selection rate | FNR | FPR | ROC-AUC |
|---|---|---|---|---|---|
| F | 40,561 | 0.09356278198269273 | 0.6703102961918195 | 0.07581179589131876 | 0.7528124856876606 |
| M | 20,940 | 0.1771251193887297 | 0.5176139032409582 | 0.1425761522513423 | 0.763697170182073 |
| XNA | 2 | 0.0 | — | 0.0 | — (`low_support: true`) |

**By `AGE_BAND`:**

| Group | n | Selection rate | FNR | FPR | ROC-AUC |
|---|---|---|---|---|---|
| 0–30 | 8,905 | 0.2425603593486805 | 0.46421471172962225 | 0.20521585010760857 | 0.7394826508728362 |
| 30–40 | 16,487 | 0.16400800630800025 | 0.5424588086185045 | 0.1329398349989939 | 0.7665323197316786 |
| 40–50 | 15,303 | 0.10658040907011697 | 0.6258620689655172 | 0.08463550873223503 | 0.7631350467027674 |
| 50–60 | 13,617 | 0.06021884409194389 | 0.7442129629629629 | 0.0469693405473222 | 0.7428476796309365 |
| 60+ | 7,191 | 0.026282853566958697 | 0.8711484593837535 | 0.02092478782557799 | 0.7201502784315366 |

**Disparity ratios (min/max across groups):** for `CODE_GENDER`, selection rate ratio 0.53 (male selection rate ~0.177 vs female ~0.094 — males are flagged materially more often), FNR ratio 0.77, FPR ratio 0.53, ROC-AUC ratio 0.99. For `AGE_BAND`, selection rate ratio 0.11 (0–30 at ~0.243 vs 60+ at ~0.026), FNR ratio 0.53, FPR ratio 0.10, ROC-AUC ratio 0.94. These gaps are **flags for human review, not verdicts**: they show the operating point behaves quite differently across groups (younger applicants and males are flagged far more often), which is exactly the kind of disparate behaviour the fairness review flagged in `docs/RESPONSIBLE_USE.md` must examine before any real-world use.

---

## Reproducibility

| Artifact | Value |
|---|---|
| Metrics generated at | 2026-09-20T11:58:50Z |
| Model | XGBoost, version 0.1.0 (`artifacts/metadata.json`) |
| Trained at | 2026-09-20T11:58:50Z |
| Dataset source | real (`application_train.csv`, 307,511 rows) |
| Feature schema version | 1.0 |
| Random seed | 42 |
| Selection metric | roc_auc |
| Calibration | isotonic, cv=3 |

To reproduce: ensure `data/raw/application_train.csv` is present (see `README.md` / `python -m loan_risk.data.download`), confirm `dataset.source: real` in `config/config.yaml`, and run `python -m loan_risk.pipeline.run`. All figures above are regenerated into `artifacts/metrics.json`.

---

> **Scope reminder:** every figure above is real held-out performance on the application-level feature subset. It is an honest baseline, not the achievable ceiling (which requires the auxiliary tables), and it is **not** evidence of fairness or fitness for any real lending decision. See `docs/RESPONSIBLE_USE.md`.
