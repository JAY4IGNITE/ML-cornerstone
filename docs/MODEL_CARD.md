# Model Card — Intelligent Loan Risk Assessment System

> **Real-data results, application-subset scope.** This model is trained and
> evaluated on the **real Kaggle Home Credit Default Risk** `application_train.csv`
> (`artifacts/metadata.json.synthetic = false`, `artifacts/metrics.json.synthetic
> = false`). Metrics are genuine held-out performance, but the system uses only
> the **application-level feature subset** (no auxiliary bureau/previous-application
> tables), so these are an honest baseline — **not** the achievable ceiling, and
> **not** evidence of fairness or fitness for real lending decisions.

| | |
|---|---|
| **Model** | XGBoost (`model_key: xgboost`) |
| **Version** | 0.1.0 |
| **Task** | Default-risk probability estimation (repayment difficulty), NOT loan approval |
| **Trained at** | 2026-09-20T11:58:50Z |
| **Dataset source** | real (`application_train.csv`) |
| **Feature schema version** | 1.0 |
| **Selection metric** | ROC-AUC |
| **Selected among** | Logistic Regression, Decision Tree, Random Forest, XGBoost |

Source: `artifacts/metadata.json`, `artifacts/metrics.json`.

---

## Intended use

- Produce an **analytical estimate** of the probability that a loan applicant
  will experience repayment difficulty (default risk), as an aid to human
  analysts and underwriters.
- Support review workflows: surface a probability, a transparent 0–100 risk
  score (`risk_score = round(default_probability * 100)`), and a risk band
  (Low / Moderate / Elevated / High per `config/config.yaml: risk_score.bands`).
- **Human oversight is required.** The output is an estimate to inform a person,
  not a decision. (`metadata.json.responsible_use_note`: "Analytical estimate of
  default probability. NOT an autonomous lending decision. Requires human
  oversight.")

## Out-of-scope use

- **Autonomous lending decisions.** Do not wire this model to auto-approve,
  auto-decline, price, or otherwise decide on credit without a human in the loop.
- **Loan approval prediction.** This model predicts repayment difficulty, not an
  approval label. Approval workflows, if added, must remain separate
  (`01_PROJECT_CONTEXT.md`).
- **Fairness / regulatory compliance claims.** A fairness **diagnostic** exists
  (`metrics.json.fairness`: per-group selection_rate / FNR / FPR / ROC-AUC sliced
  by `CODE_GENDER` and `AGE_BAND` at the operating threshold, plus simple disparity
  ratios), but it is a screen for human review — **not** a fairness certification,
  legal-compliance assessment, or mitigation. Do not represent this model as fair,
  unbiased, or legally compliant without evidence. This matters concretely here:
  `CODE_GENDER` is among the top features (see below).
- **Adverse-action reasons.** Do not use feature importances to generate
  adverse-action explanations or reasons for denial — importance is association,
  not causation (see Known risks).
- **Ground-truth or guaranteed outcomes.** The risk score is an estimate, not a
  guaranteed outcome.

## Training data

- **REAL** Kaggle Home Credit Default Risk `application_train.csv`
  (`data/raw/application_train.csv`; competition rules apply, accepted on Kaggle).
- **Rows:** 307,511 (`manifest.json.row_count`). **Columns:** 122 in the raw file;
  the pipeline consumes an application-level subset.
- **Target prevalence:** positive rate **0.0807** (~8.1%); class distribution
  `TARGET=0`: 282,686, `TARGET=1`: 24,825 (`manifest.json.target_distribution`).
- **Target meaning:** `TARGET=1` = client with payment difficulties (late payment
  beyond the dataset-defined threshold on at least one of the first
  installments); `TARGET=0` = all other cases (`manifest.json.target_meaning`).
- **Feature scope:** application-level subset only (12 numeric + 9 categorical,
  plus engineered features and missingness indicators for `DAYS_EMPLOYED` and each
  `EXT_SOURCE`). See `docs/DATA_DICTIONARY.md`.
- **Real-data missingness handled explicitly:** on the real data `EXT_SOURCE_1` is
  missing in 56.4% of rows, `EXT_SOURCE_3` in 19.8%, `OCCUPATION_TYPE` in 31.4%,
  and 55,374 rows carry the `DAYS_EMPLOYED` pensioner/unemployed sentinel. The
  pipeline adds `*_MISSING` indicator columns so the model can distinguish
  "missing" from an imputed value — and `EXT_SOURCE_3_MISSING` turns out to be a
  top-ranked feature (rank 7 in the current fold-averaged global importance, above
  `EXT_SOURCE_1_MISSING`; see below).

## Evaluation data

Held-out test split, from `metrics.json.split_diagnostics` (reproducible,
stratified, seed 42):

| Split | Rows | Positive rate |
|-------|------|---------------|
| Train | 196,806 | 0.0807 |
| Validation | 49,202 | 0.0807 |
| Test | 61,503 | 0.0807 |

- Input rows: 307,511; duplicate IDs dropped: 0; stratified: true.
- Split fractions from `config/config.yaml`: `test_size = 0.20`,
  `val_size = 0.20` (of the train remainder).
- Preprocessing is fit on training data only (no test/val leakage). The operating
  threshold is selected on the **validation** split, never on test.

## Metrics (real held-out test)

**Final test metrics** — XGBoost, threshold 0.5, n = 61,503 (verbatim from
`metrics.json.final_test_metrics`).

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7615 |
| PR-AUC | 0.2504 |
| Brier score | 0.0675 |
| Accuracy | 0.9198 |
| Precision | 0.6096 |
| Recall | 0.0179 |
| F1 | 0.0348 |

**Confusion matrix (test, threshold 0.5):**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN = 56,481 | FP = 57 |
| **Actual 1** | FN = 4,876 | TP = 89 |

**Reading these numbers honestly.** Accuracy (0.92) is essentially the base rate
of always predicting "no default" (~91.9%), so it is not informative on its own —
this is why the model is selected on ROC-AUC, not accuracy
(`03_ML_REQUIREMENTS.md`: "Do not choose a model from accuracy alone"). At the
default 0.5 threshold, recall is very low (0.0179: only 89 of 4,965 defaulters
flagged); the calibrated probabilities concentrate below 0.5 for this imbalanced
target. ROC-AUC (0.7615) shows the model ranks risk meaningfully better than
chance. `metrics.json` reports a `selected_threshold` of ≈ 0.167 (0.1665) chosen on the
**validation** set (val F1 ≈ 0.313 at that operating threshold), whose test generalization is F1 ≈ 0.315
(precision ≈ 0.262, recall ≈ 0.395), plus a full `threshold_analysis` table — the
operating threshold should be chosen deliberately for the use case, not left at 0.5.

**Model selection (5-fold stratified cross-validation ROC-AUC on TRAIN+VAL)** —
selection_basis `cross_validation_5fold_mean`, choosing the best mean ROC-AUC (std
reported), from `metrics.json.per_model_cross_validation`:

| Model | 5-fold CV ROC-AUC (mean ± std) |
|-------|--------------------|
| Logistic Regression | 0.7433 ± 0.0050 |
| Decision Tree | 0.7250 ± 0.0049 |
| Random Forest | 0.7498 ± 0.0049 |
| **XGBoost (selected)** | **0.7571 ± 0.0047** |

The single-split validation ROC-AUC is still reported as a head-to-head
(`metrics.json.per_model_validation`; XGBoost 0.7566) but is no longer the
selection basis.

XGBoost was selected by the highest mean ROC-AUC across a 5-fold stratified
cross-validation on TRAIN+VAL (`metadata.json.selection_basis =
cross_validation_5fold_mean`, `selection_metric = roc_auc`), not by the highest
ROC-AUC on a single validation split; on that validation split it also had the
highest PR-AUC (0.2380). No models were skipped
(`metrics.json.skipped_models = []`).

The selected XGBoost is **hyperparameter-tuned** (`metadata.json.hyperparameters_tuned
= true`) via `RandomizedSearchCV` (n_iter 8 over a 40,000-row stratified subsample;
XGBoost tuned CV ROC-AUC 0.7401 vs Random Forest 0.7371, from
`metrics.json.hyperparameter_tuning`). The tuned parameters actually in use —
`n_estimators: 300`, `max_depth: 4`, `learning_rate: 0.03`, `subsample: 0.7`,
`colsample_bytree: 0.8` — override the `config/config.yaml` defaults
(400 / 5 / 0.05 / 0.8 / 0.8); `scale_pos_weight` is still set from the train-split
class balance.

**Global feature importance** (XGBoost, `method: tree_feature_importances`,
fold-averaged across the 3 calibration folds) — top drivers of *predicted* risk
(association, not causation), from `metrics.json.global_importance`:
`EXT_SOURCE_MEAN` (0.167), `EXT_SOURCE_3` (0.041),
`NAME_EDUCATION_TYPE_Higher education` (0.035), `EXT_SOURCE_2` (0.035),
`CODE_GENDER_M` (0.029), `CODE_GENDER_F` (0.027), **`EXT_SOURCE_3_MISSING` (0.027)**,
`CREDIT_GOODS_RATIO` (0.025), `EXT_SOURCE_1_MISSING` (0.021). The external-score
family dominates, the *missingness* of `EXT_SOURCE_3` is itself a top-ranked
feature, and `CODE_GENDER` appears prominently — the last point is a direct
trigger for the fairness review noted under Known risks.

## Calibration

From `metadata.json.calibration` / `metrics.json.calibration`:

- **Method:** isotonic regression, 3-fold CV (`config/config.yaml: calibration`).
- **Brier score before calibration:** 0.1961
- **Brier score after calibration:** 0.0675
- **Improved:** yes.
- Probabilities are adjusted toward observed frequencies; `metrics.json` stores
  the reliability curve before and after (`calibration.curve_before` /
  `curve_after`).

Calibration was performed without using the test split to fit the calibrator. On
the real data the after-calibration reliability curve tracks observed frequency
closely in the well-populated bins (0.0–0.5 hold 61,350 of 61,503 test cases); the
high-probability bins have very few cases (down to 2–5), so calibration quality
there is weakly evidenced.

## Limitations

- **Application-level subset only.** No full bureau history, previous
  applications, installment, POS, or credit-card balance data. Those supporting
  tables are a documented offline/advanced extension and are not used here
  (`manifest.json.known_limitations`, `schema.py` scope note). This caps
  discrimination well below published multi-table Home Credit solutions (~0.79–0.80
  ROC-AUC); ~0.762 (test ROC-AUC 0.7615) is the honest application-only baseline.
- **Low recall at default threshold.** At threshold 0.5 the model flags very few
  positives (recall 0.0179); a use-case-appropriate threshold must be chosen (the
  reported validation-selected point is ≈ 0.167, i.e. 0.1665).
- **Sparse high-probability calibration bins** (see Calibration).
- **Fairness diagnostic only, no audit.** A per-group fairness **diagnostic** is
  now shipped (`metrics.json.fairness`, sliced by `CODE_GENDER` and `AGE_BAND` at
  the operating threshold ≈ 0.1665); it already surfaces disparities (e.g. male
  selection rate ~0.177 vs female ~0.094). It is a screen for human review, **not**
  a fairness certification or audit — no formal disparate-impact / equalized-odds
  assessment or mitigation has been performed, despite `CODE_GENDER` ranking among
  the most important features (see Known risks).

## Known risks

- **Do not treat scores as ground truth.** Outputs are probability estimates, not
  verified outcomes or guarantees.
- **Association, not causation.** Feature importances and SHAP values describe
  association with the model's predictions, not causal effects
  (`metrics.json.global_importance.interpretation`:
  "Association with predicted risk, not causation.").
- **No invented adverse-action reasons.** The model does not, and must not be used
  to, generate reasons for denial or adverse-action notices.
- **Concrete bias risk.** `CODE_GENDER_M` and `CODE_GENDER_F` are among the top
  features by importance. A fairness **diagnostic** (`metrics.json.fairness`) now
  slices selection_rate, FNR, FPR, and ROC-AUC by `CODE_GENDER` and `AGE_BAND` at
  the operating threshold and already flags disparities for review (e.g. male
  selection rate ~0.177 vs female ~0.094). This is a diagnostic screen, **not** a
  disparate-impact analysis or certification; do **not** assume the model is fair
  across groups. A full fairness audit is still required before any real use.
- **Distribution shift.** Trained on a static snapshot of the Home Credit
  population; performance on other populations or future time periods is unknown
  until re-evaluated.

## Responsible use

- **Non-autonomous.** This system is an analytical aid and must not make
  autonomous lending decisions. A human is responsible for any decision informed
  by its output (`metadata.json.responsible_use_note`).
- See the companion **`docs/RESPONSIBLE_USE.md`** for human-oversight, data /
  geographic / temporal limitations, potential bias, fairness-analysis
  limitations, privacy considerations, and the non-autonomous use statement.
- Before any real-world or production consideration: choose an operating threshold
  deliberately, perform a fairness / bias assessment (especially given the
  `CODE_GENDER` importance above), and — for stronger discrimination — engineer the
  auxiliary Home Credit tables. Production readiness, fairness, and legal
  compliance must not be claimed without evidence (`01_PROJECT_CONTEXT.md`
  Non-Negotiable Principles).

---

## Sources

| Section | Sourced from |
|---------|--------------|
| Model identity, version, selection metric, calibration summary | `artifacts/metadata.json` |
| Final test metrics, confusion matrix, per-model validation, split sizes, importance, thresholds | `artifacts/metrics.json` |
| Training data provenance, row count, positive rate, target meaning, missingness, limitations | `data/manifest.json` |
| Risk-score definition, calibration method, split config, bands | `config/config.yaml`, `artifacts/metadata.json` |
| Feature scope and definitions | `loan_risk/schema.py`, `docs/DATA_DICTIONARY.md` |
| Task framing and non-negotiable principles | `01_PROJECT_CONTEXT.md`, `03_ML_REQUIREMENTS.md` |
