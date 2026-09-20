# Evaluation Report — Intelligent Loan Risk Assessment System

_All figures below are read directly from `artifacts/metrics.json` (generated_at: 2026-09-20T07:43:10Z)._

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

Configured split ratios (`config/config.yaml`): `test_size: 0.20`, `val_size: 0.20` (fraction of the train-remainder), `stratify: true`, `time_column: null` (no time-aware split was performed; the configuration disables it). Deduplication ran and found 0 duplicate applicant IDs, so no rows were dropped for leakage. Stratification held the positive rate at the 0.0807 population rate across all three splits.

---

## 3. Models Trained

Source: `per_model_validation` and `skipped_models` in `artifacts/metrics.json`.

Four models were trained and evaluated on identical splits:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost — **selected model**

**Skipped models:** none. `skipped_models` is an empty list (`[]`), so all four candidate models trained successfully; `metadata.json` confirms `xgboost_installed: true` and `shap_installed: true`.

**Selection rule:** the model was selected by **`roc_auc`** (`selection_metric: "roc_auc"`), not by accuracy alone, per `03_ML_REQUIREMENTS.md`. **XGBoost** had the highest validation ROC-AUC (0.7599377908236276) and was selected.

**Validation gate summary (`validation_summary`):** 10 pass, 1 warn, 0 fail. The single warning is the expected missing-value report on real data (67 columns contain missing values); no check failed.

---

## 4. Hyperparameters

Source: `models` section of `config/config.yaml`. Calibration parameters from the `calibration` section.

| Model | Hyperparameters |
|---|---|
| Logistic Regression | `C: 1.0`, `max_iter: 1000`, `class_weight: balanced` |
| Decision Tree | `max_depth: 6`, `min_samples_leaf: 50`, `class_weight: balanced` |
| Random Forest | `n_estimators: 300`, `max_depth: 12`, `min_samples_leaf: 20`, `n_jobs: -1`, `class_weight: balanced` |
| XGBoost **(selected)** | `n_estimators: 400`, `max_depth: 5`, `learning_rate: 0.05`, `subsample: 0.8`, `colsample_bytree: 0.8`, `scale_pos_weight` set from the train-split class balance |

**Calibration configuration:** `method: isotonic`, `cv: 3`. **Selection metric:** `roc_auc`. **CV folds (evaluation):** 5. **Global random seed:** 42.

---

## 5. Metrics

### 5.1 Per-model validation comparison

All rows evaluated on the validation split (`n = 49202`, decision threshold `0.5`, positive rate `0.0807284256737531`). Values quoted exactly from `per_model_validation[*].val_metrics`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.6819438234218121 | 0.15686159271231268 | 0.6719536757301108 | 0.25434792967074854 | 0.7419027098600217 | 0.21834771884170817 | 0.2059024041663295 |
| Decision Tree | 0.6648306979391082 | 0.1485993375624544 | 0.6664149043303121 | 0.24301124627036952 | 0.7226657712766729 | 0.18894690548714888 | 0.20949097885629564 |
| Random Forest | 0.7241575545709524 | 0.17199671996719967 | 0.6336858006042296 | 0.2705578845533699 | 0.7481138030329041 | 0.22504923913300393 | 0.1881380649890652 |
| **XGBoost (selected)** | 0.7188528921588553 | 0.17293532338308457 | 0.6563444108761329 | 0.273743896676642 | **0.7599377908236276** | 0.24332328706758183 | 0.18709228329976882 |

XGBoost has the highest validation ROC-AUC (0.7599) and the highest PR-AUC (0.2433), so it is selected. Note the models with `class_weight: balanced` (all but XGBoost, which uses `scale_pos_weight`) trade accuracy for recall at the 0.5 threshold — e.g. Logistic Regression reaches recall 0.672 but accuracy only 0.682. This is expected under an ~8% positive rate and is exactly why ROC-AUC, not accuracy, is the selection criterion.

### 5.2 Final held-out TEST metrics (selected model: XGBoost)

Evaluated once on the held-out test split (`n = 61503`, decision threshold `0.5`, positive rate `0.08072776937710356`). Values quoted exactly from `final_test_metrics`.

| Metric | Value |
|---|---|
| Accuracy | 0.9199713835097475 |
| Precision | 0.6405228758169934 |
| Recall | 0.01973816717019134 |
| F1 | 0.03829620945681907 |
| ROC-AUC | 0.7666792436510453 |
| PR-AUC | 0.255592473882044 |
| Brier score | 0.06722983139426303 |

The high test accuracy (0.9200) is essentially the trivial "predict every applicant repays" baseline (1 − 0.0807 = 0.9193), which shows why accuracy is not a meaningful headline for this imbalanced task. ROC-AUC (0.7667) and PR-AUC (0.2556) are the informative discrimination measures. At the default 0.5 threshold recall is very low (0.0197) because the calibrated probabilities cluster well below 0.5 — see the confusion matrix and threshold analysis below, which is why the pipeline reports a validation-selected operating point (§8) rather than defaulting to 0.5.

For context: published Home Credit solutions exceed ~0.79–0.80 ROC-AUC, but only by engineering features across the auxiliary bureau/previous-application/installment tables. An application-only ROC-AUC of ~0.767 is a credible, honest baseline for the feature subset this system deliberately restricts itself to.

---

## 6. Confusion Matrix (Final TEST, threshold 0.5)

Rendered from `final_test_metrics.confusion_matrix` (`n = 61503`).

|  | Predicted: No default (0) | Predicted: Default (1) |
|---|---|---|
| **Actual: No default (0)** | TN = 56,483 | FP = 55 |
| **Actual: Default (1)** | FN = 4,867 | TP = 98 |

Totals: 56,483 + 55 + 4,867 + 98 = 61,503. Of 4,965 actual defaulters (FN + TP), the model flags only 98 at the 0.5 threshold; of 56,538 non-defaulters, only 55 are false-flagged. At threshold 0.5 the model is extremely conservative about predicting "default" — the operating point must move well below 0.5 to be useful (§8).

---

## 7. Calibration

Source: `calibration` in `artifacts/metrics.json`. Method: **isotonic**, `cv: 3` (from config and `metadata.json`).

| | Brier score |
|---|---|
| Before calibration | 0.18692637979984283 |
| After calibration | 0.06722983139426303 |

Lower Brier is better. Isotonic calibration reduced the Brier score from 0.1869 to 0.0672 on the test set, and `metadata.json` records `improved: true`. The calibrated Brier (0.06722983139426303) equals the `final_test_metrics.brier_score`, as expected since final metrics are computed on the calibrated model.

### 7.1 Reliability curve — after calibration

From `calibration.curve_after`. Each row is a probability bin: `mean_predicted` is the average predicted probability in the bin; `observed_frequency` is the actual default rate among those cases; `count` is the number of cases.

| Bin | Mean predicted | Observed frequency | Count |
|---|---|---|---|
| 0.0–0.1 | 0.04341510795088114 | 0.04166300237446135 | 45,484 |
| 0.1–0.2 | 0.13922159457088787 | 0.13996689350744895 | 10,874 |
| 0.2–0.3 | 0.23904944684456195 | 0.2524489224741114 | 3,573 |
| 0.3–0.4 | 0.3440523048902948 | 0.3669201520912547 | 1,052 |
| 0.4–0.5 | 0.44001556287344534 | 0.44141689373297005 | 367 |
| 0.5–0.6 | 0.549325700547244 | 0.6216216216216216 | 111 |
| 0.6–0.7 | 0.6271820550873166 | 0.6857142857142857 | 35 |
| 0.7–0.8 | 0.7093557874361675 | 0.6 | 5 |
| 0.8–0.9 | 0.8546523054440816 | 1.0 | 2 |

In the well-populated bins (0.0–0.5, together 61,350 of the 61,503 test cases), mean predicted probability tracks observed frequency closely — calibration is good where the vast majority of predictions live. The higher bins are unreliable indicators only because their counts are tiny (down to 5 and 2 cases); those rows should not be read as evidence of mis- or well-calibration. `calibration.curve_before` (10 bins) is also present in the artifact and shows the pre-calibration model systematically **over-predicting** risk (e.g. mean predicted 0.070 vs observed 0.009 in the lowest bin) — which isotonic calibration corrects.

---

## 8. Threshold Analysis

Source: `threshold_analysis` in `artifacts/metrics.json` (evaluated on the test split, `n = 61503`, 4,965 actual positives). Shows how precision, recall, F1, and the flagged rate move as the decision threshold changes. Values quoted exactly.

| Threshold | Precision | Recall | F1 | Flagged rate | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.12939018798716184 | 0.8525679758308157 | 0.2246815286624204 | 0.5319252719379542 | 4233 | 28482 | 732 | 28056 |
| 0.10 | 0.19164741869030527 | 0.6183282980866063 | 0.2926038886770873 | 0.26045883940620784 | 3070 | 12949 | 1895 | 43589 |
| 0.15 | 0.24565364642131407 | 0.43826787512588117 | 0.3148375895247052 | 0.14402549469131587 | 2176 | 6682 | 2789 | 49856 |
| 0.20 | 0.3008746355685131 | 0.31178247734138975 | 0.3062314540059347 | 0.08365445588020097 | 1548 | 3597 | 3417 | 52941 |
| 0.25 | 0.35560423512230743 | 0.19617321248741187 | 0.2528556593977155 | 0.044534412955465584 | 974 | 1765 | 3991 | 54773 |
| 0.30 | 0.410941475826972 | 0.13011077542799598 | 0.19764417928713476 | 0.02555972879371738 | 646 | 926 | 4319 | 55612 |
| 0.35 | 0.4345991561181435 | 0.08298086606243706 | 0.1393539658379841 | 0.01541388224964636 | 412 | 536 | 4553 | 56002 |
| 0.40 | 0.5 | 0.05236656596173213 | 0.09480401093892434 | 0.008454872120059184 | 260 | 260 | 4705 | 56278 |
| 0.45 | 0.5360824742268041 | 0.03141993957703928 | 0.0593607305936073 | 0.004731476513340813 | 156 | 135 | 4809 | 56403 |
| 0.50 | 0.6405228758169934 | 0.01973816717019134 | 0.03829620945681907 | 0.002487683527632798 | 98 | 55 | 4867 | 56483 |
| 0.55 | 0.6442307692307693 | 0.013494461228600202 | 0.026435194318405998 | 0.0016909744240118367 | 67 | 37 | 4898 | 56501 |
| 0.60 | 0.6904761904761905 | 0.005840886203423968 | 0.011583782704214101 | 0.0006828935173893956 | 29 | 13 | 4936 | 56525 |
| 0.65 | 0.6 | 0.0012084592145015106 | 0.002412060301507538 | 0.00016259369461652277 | 6 | 4 | 4959 | 56534 |
| 0.70 | 0.7142857142857143 | 0.0010070493454179255 | 0.002011263073209976 | 0.00011381558623156594 | 5 | 2 | 4960 | 56536 |
| 0.75 | 1.0 | 0.0004028197381671702 | 0.0008053150795248641 | 0.00003251873892330455 | 2 | 0 | 4963 | 56538 |
| 0.80 | 1.0 | 0.0004028197381671702 | 0.0008053150795248641 | 0.00003251873892330455 | 2 | 0 | 4963 | 56538 |
| 0.85 | 1.0 | 0.0004028197381671702 | 0.0008053150795248641 | 0.00003251873892330455 | 2 | 0 | 4963 | 56538 |
| 0.90 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 4965 | 56538 |
| 0.95 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 4965 | 56538 |

**Selected operating point** (`selected_threshold` in the artifact): threshold **0.16136377056439719**, chosen on the **validation** set (never the test set, to avoid leakage). At selection it achieves validation F1 **0.34466342949633433** (precision 0.27713979482468226, recall 0.4556898288016113). Its **test-set generalization** at the same threshold is F1 **0.3161587029386546** (precision 0.2578840284842319, recall 0.40845921450151057). The threshold is reported for analysis and is **not auto-applied** at serving; the API's risk bands use the configured probability cut points.

The default 0.5 threshold is a poor operating point on this data: it yields test F1 = 0.038 and catches only ~2% of defaulters (recall 0.0197). The validation-selected ~0.161 threshold lifts test F1 to ~0.316 by trading precision down (~0.258) for materially higher recall (~0.408). The table makes the trade-off explicit: lowering the threshold catches more defaulters (higher recall, higher TP) at the cost of flagging far more non-defaulters (higher FP, higher flagged rate). At threshold 0.05, recall reaches 0.853 but 53.2% of all applicants are flagged and precision collapses to 0.129.

---

## 9. Error Analysis (interpretation of the real numbers above)

This section interprets the confusion matrix (§6) and threshold table (§8). The numbers are real held-out results; the interpretation is analytical and still bounded by the application-subset scope limit.

**Class imbalance sets the terms.** The positive (default) rate is ~8.07%, so 4,965 of 61,503 test applicants actually default. Any model can reach ~92% accuracy by predicting "no default" for everyone (baseline 0.919), so accuracy carries almost no signal here. FP vs FN is the real trade-off.

**At threshold 0.5 the model is badly mis-tuned toward false negatives.** The test confusion matrix shows FN = 4,867 and TP = 98: the model misses 4,867 of 4,965 real defaulters while raising only 55 false positives. In lending, a **false negative** (a would-be defaulter scored as low-risk) is typically the costlier error — it maps to an approved loan that later defaults. A **false positive** (a repaying applicant flagged as risky) mostly costs a lost-good-customer / extra-review burden. The ~89:1 FN:FP ratio at 0.5 means the default threshold optimizes almost entirely against the cheaper error, which is the wrong direction for risk screening — hence the validation-selected lower threshold.

**The threshold curve shows the lever to fix it.** Because the calibrated probabilities cluster low (73.9% of the test set lands in the 0.0–0.1 bin, §7.1), the decision threshold must move well below 0.5. Moving to the validation-selected ~0.161 raises test recall from ~0.020 to ~0.408 and test F1 from ~0.038 to ~0.316. If the business cost of a missed defaulter dominates, an even lower threshold (e.g. 0.10, recall 0.618; or 0.05, recall 0.853) buys more recall — but §8 quantifies the price: threshold 0.10 flags 26.0% of applicants (FP = 12,949) and threshold 0.05 flags 53.2% (FP = 28,482). The right operating point is a business decision about the relative cost of FN vs FP, and this report deliberately does not pick one; it surfaces the full curve so a human can choose.

**Discrimination is modest — an application-only baseline.** Test ROC-AUC is 0.7667 and PR-AUC is 0.2556. PR-AUC is the more honest ceiling under imbalance, and ~0.256 (against a 0.081 no-skill baseline) says the ranking is meaningfully better than random but far from decisive. No threshold in §8 achieves both high precision and high recall simultaneously — the best F1 anywhere in the table is ~0.315 (threshold 0.15). The model is a moderate ranker, not a decisive classifier, which reinforces that its output should inform a human reviewer rather than gate decisions automatically. Adding the auxiliary Home Credit tables is the known path to stronger discrimination and is out of scope for this application-level system.

---

## 10. Global Feature Importance (context)

Source: `global_importance` in `artifacts/metrics.json`. Method: **`tree_feature_importances`** (XGBoost native importances). The artifact explicitly labels this as **"Association with predicted risk, not causation."**

Top features by importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `EXT_SOURCE_MEAN` | 0.10468602925539017 |
| 2 | `NAME_EDUCATION_TYPE_Higher education` | 0.03681366890668869 |
| 3 | `CODE_GENDER_M` | 0.033770330250263214 |
| 4 | `EXT_SOURCE_3` | 0.026819396764039993 |
| 5 | `CODE_GENDER_F` | 0.02523353509604931 |
| 6 | `EXT_SOURCE_1_MISSING` | 0.024964166805148125 |
| 7 | `NAME_EDUCATION_TYPE_Secondary / secondary special` | 0.02415873110294342 |
| 8 | `EXT_SOURCE_2` | 0.021648896858096123 |
| 9 | `CREDIT_GOODS_RATIO` | 0.021514438092708588 |
| 10 | `NAME_CONTRACT_TYPE_Cash loans` | 0.02052740380167961 |
| 11 | `NAME_INCOME_TYPE_Pensioner` | 0.0200092401355505 |
| 12 | `FLAG_OWN_CAR_N` | 0.019938109442591667 |

Two observations matter for responsible use:

1. **The external-source score family dominates.** The engineered `EXT_SOURCE_MEAN` is the single most important feature (0.105), and `EXT_SOURCE_3` / `EXT_SOURCE_2` also rank highly. Critically, **`EXT_SOURCE_1_MISSING` (rank 6, 0.025)** confirms that whether an external score is *present* is itself predictive on real data — vindicating the decision to model missingness explicitly rather than impute it away. On real Home Credit data `EXT_SOURCE_1` is absent for 56.4% of applicants (§1), so this indicator is populated and meaningful.

2. **Protected-attribute proxies appear.** `CODE_GENDER_M` (rank 3) and `CODE_GENDER_F` (rank 5) are direct protected attributes, and features like `NAME_EDUCATION_TYPE_*` and `NAME_INCOME_TYPE_Pensioner` (an age proxy) also appear. Their presence is a statistical **association**, not a causal or endorsed decision factor. This is precisely the kind of signal that requires the fairness review flagged in `docs/RESPONSIBLE_USE.md` before any real-world use — the model has not been audited for disparate impact, and gender's high importance is a concrete reason that audit is necessary.

---

## Reproducibility

| Artifact | Value |
|---|---|
| Metrics generated at | 2026-09-20T07:43:10Z |
| Model | XGBoost, version 0.1.0 (`artifacts/metadata.json`) |
| Trained at | 2026-09-20T07:43:10Z |
| Dataset source | real (`application_train.csv`, 307,511 rows) |
| Feature schema version | 1.0 |
| Random seed | 42 |
| Selection metric | roc_auc |
| Calibration | isotonic, cv=3 |

To reproduce: ensure `data/raw/application_train.csv` is present (see `README.md` / `python -m loan_risk.data.download`), confirm `dataset.source: real` in `config/config.yaml`, and run `python -m loan_risk.pipeline.run`. All figures above are regenerated into `artifacts/metrics.json`.

---

> **Scope reminder:** every figure above is real held-out performance on the application-level feature subset. It is an honest baseline, not the achievable ceiling (which requires the auxiliary tables), and it is **not** evidence of fairness or fitness for any real lending decision. See `docs/RESPONSIBLE_USE.md`.
