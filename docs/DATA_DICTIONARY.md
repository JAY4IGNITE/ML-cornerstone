# Data Dictionary — Intelligent Loan Risk Assessment System

Task: **default-risk prediction** (probability that a client has repayment difficulty),
NOT loan approval. See `01_PROJECT_CONTEXT.md`.

Every feature below is documented directly from `loan_risk/schema.py`
(`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, `ENGINEERED_FEATURES`), which is the
single source of truth for the whole system (synthetic generation, validation,
preprocessing, API request models, and the frontend form). The canonical
conversion of the raw `DAYS_*` columns is documented from
`loan_risk/data/ingestion.py`. No feature, column, unit, or bound in this
document was invented; each is copied from those modules.

When real Kaggle data is used, column names match the public **Home Credit
Default Risk** `application_train.csv` schema — `validate.py` confirms these
columns exist and match, and never invents columns (per the schema module's
Non-Negotiable Principles). The synthetic fixture is generated with the same raw
column layout on purpose.

> **A note on the raw file vs. the model columns.** The raw
> `application_train.csv` stores age and employment as *negative day-counts*
> relative to the application date: `DAYS_BIRTH` and `DAYS_EMPLOYED`. Ingestion
> converts these once into the human-friendly canonical columns `AGE_YEARS` and
> `EMPLOYMENT_YEARS` that the model trains and serves on (Section 2). Sections 1
> and 3 therefore list the **canonical model columns**; Section 2 documents the
> raw→canonical conversion.

---

## 1. Application-level input features

These are the columns an API caller / applicant form supplies (from
`schema.py: NUMERIC_FEATURES` and `CATEGORICAL_FEATURES`). Scope is an
*application-level subset* of the real Home Credit schema — the fields a single
applicant form can realistically supply at prediction time. Supporting tables
(bureau, previous_application, …) are the documented advanced/offline path and
are not required for a single real-time prediction.

### 1a. Numeric input features

| Name | Type | Meaning | Unit | Source | Valid range | Missing-value behavior |
|------|------|---------|------|--------|-------------|------------------------|
| `AMT_INCOME_TOTAL` | numeric | Total declared income of the client | currency | `application_train.csv` | 0 – 5,000,000 | median imputation (should rarely be missing) |
| `AMT_CREDIT` | numeric | Credit amount of the loan | currency | `application_train.csv` | 0 – 5,000,000 | median imputation |
| `AMT_ANNUITY` | numeric | Loan annuity (periodic payment) | currency | `application_train.csv` | 0 – 500,000 | median imputation |
| `AMT_GOODS_PRICE` | numeric | Price of the goods for which the loan is granted | currency | `application_train.csv` | 0 – 5,000,000 | median imputation |
| `AGE_YEARS` | numeric | Client age in years. This IS the canonical model column; the raw file's `DAYS_BIRTH` (negative day-count) is converted to it once during ingestion, so training and serving share it. | years | `application_train.csv` (derived from `DAYS_BIRTH`, see Section 2) | 18 – 100 | required; validated on input |
| `EMPLOYMENT_YEARS` | numeric | Years employed at current job. Canonical model column derived once at ingestion from `DAYS_EMPLOYED`; the documented 365243 sentinel (pensioners/unemployed) becomes missing with a `DAYS_EMPLOYED_MISSING` indicator. Omit at serving for pensioners/unemployed → flag set, value imputed. | years | `application_train.csv` (derived from `DAYS_EMPLOYED`, see Section 2) | 0 – 50 | median imputation + `DAYS_EMPLOYED_MISSING` flag |
| `CNT_FAM_MEMBERS` | numeric | Number of family members | count | `application_train.csv` | 1 – 20 | median imputation |
| `CNT_CHILDREN` | numeric | Number of children the client has | count | `application_train.csv` | 0 – 20 | median imputation |
| `EXT_SOURCE_1` | numeric | Normalized score from external data source 1 | score 0-1 | `application_train.csv` | 0 – 1 | frequently missing; median imputation + indicator |
| `EXT_SOURCE_2` | numeric | Normalized score from external data source 2 | score 0-1 | `application_train.csv` | 0 – 1 | median imputation + indicator |
| `EXT_SOURCE_3` | numeric | Normalized score from external data source 3 | score 0-1 | `application_train.csv` | 0 – 1 | frequently missing; median imputation + indicator |
| `REGION_POPULATION_RELATIVE` | numeric | Normalized population of the region where the client lives | ratio | `application_train.csv` | 0 – 0.15 | median imputation |

Numeric imputation strategy (`median`), the categorical unknown token, and
scaling-for-linear-models are configured in `config/config.yaml`
(`preprocessing:`). All transformations are fit on training data only.

### 1b. Categorical input features

Allowed values are the canonical Home Credit categories declared in
`schema.py`. Any value outside the allowed set, or a missing value, is mapped to
the unknown token `__UNKNOWN__` (`config/config.yaml: preprocessing.categorical_unknown_token`).

| Name | Type | Meaning | Unit | Source | Allowed values | Missing-value behavior |
|------|------|---------|------|--------|----------------|------------------------|
| `NAME_CONTRACT_TYPE` | categorical | Loan type: cash or revolving | category | `application_train.csv` | `Cash loans`, `Revolving loans` | unknown token |
| `CODE_GENDER` | categorical | Gender of the client | category | `application_train.csv` | `M`, `F`, `XNA` | unknown token |
| `FLAG_OWN_CAR` | categorical | Whether the client owns a car | category | `application_train.csv` | `Y`, `N` | unknown token |
| `FLAG_OWN_REALTY` | categorical | Whether the client owns real estate | category | `application_train.csv` | `Y`, `N` | unknown token |
| `NAME_INCOME_TYPE` | categorical | Client's income source | category | `application_train.csv` | `Working`, `State servant`, `Commercial associate`, `Pensioner`, `Unemployed`, `Student`, `Businessman`, `Maternity leave` | unknown token |
| `NAME_EDUCATION_TYPE` | categorical | Highest level of education achieved | category | `application_train.csv` | `Lower secondary`, `Secondary / secondary special`, `Incomplete higher`, `Higher education`, `Academic degree` | unknown token |
| `NAME_FAMILY_STATUS` | categorical | Family status of the client | category | `application_train.csv` | `Single / not married`, `Married`, `Civil marriage`, `Widow`, `Separated`, `Unknown` | unknown token |
| `NAME_HOUSING_TYPE` | categorical | Housing situation of the client | category | `application_train.csv` | `House / apartment`, `Rented apartment`, `With parents`, `Municipal apartment`, `Office apartment`, `Co-op apartment` | unknown token |
| `OCCUPATION_TYPE` | categorical | Occupation of the client (often missing in real data) | category | `application_train.csv` | `Laborers`, `Core staff`, `Accountants`, `Managers`, `Drivers`, `Sales staff`, `Cleaning staff`, `Cooking staff`, `Private service staff`, `Medicine staff`, `Security staff`, `High skill tech staff`, `Waiters/barmen staff`, `Low-skill Laborers`, `Realty agents`, `Secretaries`, `IT staff`, `HR staff` | unknown token (missingness is itself informative) |

---

## 2. Derived / canonical columns (ingestion conversion)

Documented from `loan_risk/data/ingestion.py`. The raw application file stores
age/employment as negative day-counts relative to the application date. Ingestion
(`standardize()`) converts them **exactly once** so the training data and the API
inputs share identical columns (consistent train/inference pipeline). Raw
`DAYS_*` columns are dropped after conversion. This same standardization runs for
both the synthetic fixture and real Kaggle data.

| Canonical column | Type | Derived from (raw) | Conversion | Unit | Missing-value behavior |
|------------------|------|--------------------|------------|------|------------------------|
| `AGE_YEARS` | numeric | `DAYS_BIRTH` (negative day-count) | `AGE_YEARS = (-DAYS_BIRTH) / 365.25` | years | required; validated on input |
| `EMPLOYMENT_YEARS` | numeric | `DAYS_EMPLOYED` (negative day-count) | `EMPLOYMENT_YEARS = (-DAYS_EMPLOYED) / 365.25`; the `365243` sentinel → `NaN` | years | median imputation + `DAYS_EMPLOYED_MISSING` flag |
| `DAYS_EMPLOYED_MISSING` | numeric (0/1 indicator) | `DAYS_EMPLOYED` | `1` if `DAYS_EMPLOYED == 365243` (pensioner/unemployed sentinel) else `0` | flag (0/1) | always produced by ingestion; not user-supplied |

**The `365243` sentinel.** In the raw Home Credit data, `DAYS_EMPLOYED` uses the
documented value `365243` for pensioners/unemployed clients. Ingestion detects
this sentinel (`DAYS_EMPLOYED_SENTINEL = 365243`, overridable via
`config/config.yaml: preprocessing.days_employed_anomaly`), sets
`EMPLOYMENT_YEARS` to missing for those rows, and raises the
`DAYS_EMPLOYED_MISSING` indicator so the model can distinguish "genuinely 0 years
employed" from "not employed / pensioner." The missing `EMPLOYMENT_YEARS` is then
median-imputed at the preprocessing stage.

The canonical model frame produced by ingestion is: `SK_ID_CURR` (id),
`TARGET` (if present), the 12 numeric columns, `DAYS_EMPLOYED_MISSING`, and the 9
categorical columns.

---

## 3. Engineered features

Documented from `schema.py: ENGINEERED_FEATURES` and implemented once in
`loan_risk/pipeline/features.py` (a stateless transformer applied identically in
training and serving). Every engineered feature is point-in-time: it uses only
application-level fields available at prediction time, so it introduces no
leakage. Ratios guard against division-by-zero by returning `NaN` (later imputed),
never infinity.

| Name | Formula | Inputs | Unit | Interpretation | Leakage assessment |
|------|---------|--------|------|----------------|--------------------|
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` | `AMT_CREDIT`, `AMT_INCOME_TOTAL` | ratio | Loan size relative to income; higher = more leverage. | Safe: both known at application time. |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | `AMT_ANNUITY`, `AMT_INCOME_TOTAL` | ratio | Payment burden relative to income (affordability). | Safe: both known at application time. |
| `CREDIT_ANNUITY_RATIO` | `AMT_CREDIT / AMT_ANNUITY` | `AMT_CREDIT`, `AMT_ANNUITY` | ratio (~term) | Approximate loan term in payment periods. | Safe: both known at application time. |
| `CREDIT_GOODS_RATIO` | `AMT_CREDIT / AMT_GOODS_PRICE` | `AMT_CREDIT`, `AMT_GOODS_PRICE` | ratio | Loan vs goods value; >1 implies borrowing beyond price. | Safe: both known at application time. |
| `EMPLOYMENT_AGE_RATIO` | `EMPLOYMENT_YEARS / AGE_YEARS` | `EMPLOYMENT_YEARS`, `AGE_YEARS` | ratio | Share of life in current employment; stability proxy. | Safe: derived from applicant attributes. |
| `EXT_SOURCE_MEAN` | `mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)` ignoring NaN | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` | score 0-1 | Aggregate external creditworthiness signal. | Safe: external scores are inputs, not outcomes. |
| `INCOME_PER_FAM_MEMBER` | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` | `AMT_INCOME_TOTAL`, `CNT_FAM_MEMBERS` | currency | Per-capita household income. | Safe: both known at application time. |

---

## 4. Target column

Documented from `data/manifest.json`.

| Name | Type | Meaning | Values |
|------|------|---------|--------|
| `TARGET` | binary label | `TARGET=1`: client with payment difficulties (late payment beyond the dataset-defined threshold on at least one of the first installments); `TARGET=0`: all other cases. Confirm against `HomeCredit_columns_description.csv` when using real data. | `0` / `1` |

---

## 5. Data source: real data and the synthetic fixture

The shipped model trains on the **real** Kaggle Home Credit Default Risk
`application_train.csv` (`data/manifest.json`: `source_kind: real`, 307,511 rows,
positive rate 0.0807). On the real data, missingness is substantial and is modeled
explicitly with `*_MISSING` indicators rather than silently imputed:
`EXT_SOURCE_1` ≈ 56.4%, `OCCUPATION_TYPE` ≈ 31.4%, `EXT_SOURCE_3` ≈ 19.8%,
`EXT_SOURCE_2` ≈ 0.2%, `AMT_GOODS_PRICE` ≈ 0.1% (from `manifest.json.missingness`);
55,374 rows carry the `DAYS_EMPLOYED` pensioner/unemployed sentinel.

A labeled **synthetic** Home Credit-shaped fixture generated by
`loan_risk/data/synthetic.py` remains available (`dataset.source: synthetic`) for
offline pipeline stress-testing without the Kaggle download. It produces the same
raw column layout as `application_train.csv`; any metric computed on it is
synthetic and is NOT a real-world result. The system flips between the two sources
by configuration alone, with no code change.

When the synthetic fixture is used, its values are generated as follows (from
`synthetic.py` and `config/config.yaml: synthetic`):

- **Rows / prevalence:** 20,000 rows; configured `positive_rate = 0.081` (~8.1%,
  matching the documented Home Credit default rate). The target is a *noisy*
  function of realistic risk signals (external scores, leverage, age, employment,
  missingness), intentionally imperfect so AUC stays realistically under 1.0.
- **Age / employment:** `DAYS_BIRTH` from ages ~N(43, 11) clipped to [21, 69];
  ~18% of rows are pensioners and receive the `365243` `DAYS_EMPLOYED` sentinel.
- **Financials:** income is log-normal, clipped to [25,000, 2,000,000]; credit,
  goods price, and annuity are derived from income with noise.
- **External scores:** `EXT_SOURCE_1/2/3` share a latent base signal (lower =
  riskier) plus noise, clipped to [0, 1].
- **Categoricals:** sampled from the canonical schema categories (so all values
  are valid).
- **Injected missingness** — configured target rates (`config/config.yaml:
  synthetic`): `missing_rate_ext_source = 0.35`, `missing_rate_occupation = 0.31`.
  A representative synthetic run measured `EXT_SOURCE_1` ≈ 35.0%, `EXT_SOURCE_3` ≈
  34.5%, `EXT_SOURCE_2` ≈ 6.8%, `OCCUPATION_TYPE` ≈ 31.4%, `AMT_ANNUITY` ≈ 0.5%.
  (These are the *synthetic* rates; the real-data rates are higher — see the top
  of this section. `manifest.json.missingness` now reflects whichever source is
  active.)

Known limitations (from `manifest.json.known_limitations`):

- Serving subset uses application-level features only; supporting tables
  (bureau, previous_application, …) are an offline/advanced extension and not
  required for a single real-time prediction.
- `DAYS_EMPLOYED` uses the documented `365243` sentinel for pensioners/unemployed;
  treated as missing with an indicator.

(When the synthetic fixture is the active source, distributions approximate but do
not equal the real Home Credit population; the fixture carries its own
`synthetic_warning` in the manifest.)

---

## Sources

| Section | Sourced from |
|---------|--------------|
| Numeric & categorical input features (Sec. 1) | `loan_risk/schema.py` (`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`) |
| Preprocessing behavior (imputation, unknown token, scaling) | `config/config.yaml` (`preprocessing:`) |
| Derived/canonical columns & sentinel (Sec. 2) | `loan_risk/data/ingestion.py` |
| Engineered features (Sec. 3) | `loan_risk/schema.py` (`ENGINEERED_FEATURES`), implemented in `loan_risk/pipeline/features.py` |
| Target meaning (Sec. 4) | `data/manifest.json` (`target_meaning`) |
| Data source, real missingness & known limitations (Sec. 5) | `data/manifest.json` (`source_kind`, `missingness`, `known_limitations`) |
| Synthetic fixture generation (Sec. 5) | `loan_risk/data/synthetic.py`, `config/config.yaml` (`synthetic:`) |
