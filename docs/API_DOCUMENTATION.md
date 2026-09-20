# API Documentation — Loan Default Risk API

FastAPI service that estimates the **probability of loan default** for a single
applicant. Every output is an **analytical estimate, not an autonomous lending
decision**.

- App: `backend/main.py` (`title: "Loan Default Risk API"`, `version: 0.1.0`)
- Base URL (dev): `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs` (OpenAPI/Swagger UI)
- Start: `uvicorn backend.main:app --reload --port 8000`

> The shipped model is trained on **real** Kaggle Home Credit Default Risk data;
> `GET /api/model/info` reports `synthetic: false`. Metrics are genuine held-out
> results from an **application-level feature subset** only — an honest baseline,
> not the achievable ceiling and not evidence of fairness. If you retrain on the
> synthetic fixture instead, the API advertises that honestly (`synthetic: true`,
> plus a synthetic-data warning prefixed to each prediction's `limitations`).

## Table of contents

- [Conventions](#conventions)
- [Endpoints](#endpoints)
  - [GET /api/health](#get-apihealth)
  - [GET /api/model/info](#get-apimodelinfo)
  - [POST /api/validate-input](#post-apivalidate-input)
  - [POST /api/predict](#post-apipredict)
  - [GET /api/metrics](#get-apimetrics)
  - [GET /api/feature-schema](#get-apifeature-schema)
  - [GET /api/dataset/quality](#get-apidatasetquality)
- [Request schema: ApplicantInput](#request-schema-applicantinput)
- [Response schemas](#response-schemas)
- [Error responses](#error-responses)
- [Model version behavior](#model-version-behavior)

## Conventions

- All request and response bodies are JSON (`Content-Type: application/json`).
- CORS allows the frontend dev origins (`http://localhost:5173`,
  `http://127.0.0.1:5173`) for `GET` and `POST`.
- Validation is performed by Pydantic models built dynamically from the shared
  feature contract in `loan_risk/schema.py`, so API bounds/enums match the ML
  pipeline and the frontend form.
- Errors always use a single [`ErrorResponse`](#error-responses) shape and never
  expose stack traces or internal filesystem paths.

## Endpoints

| Method | Path                   | Purpose                                        | Non-200 statuses          |
| ------ | ---------------------- | ---------------------------------------------- | ------------------------- |
| GET    | `/api/health`          | Service + model availability                   | —                         |
| GET    | `/api/model/info`      | Model name/version/manifest/calibration        | `503`                     |
| POST   | `/api/validate-input`  | Validate a payload without predicting          | `422`                     |
| POST   | `/api/predict`         | Probability + score + band + explanation       | `422`, `503`              |
| GET    | `/api/metrics`         | Full evaluation metrics (last training run)    | `404`                     |
| GET    | `/api/feature-schema`  | Input contract driving the form                | —                         |
| GET    | `/api/dataset/quality` | Validation report + manifest summary           | `404`                     |

A `GET /` root route (not in the OpenAPI schema) returns
`{"service", "docs", "health"}` pointers.

### GET /api/health

Returns service health and model availability. Always `200`; `status` is `ok`
when a model is loaded, otherwise `degraded`.

**Response — [`HealthResponse`](#healthresponse):**

```json
{
  "status": "ok",
  "model_available": true,
  "api_version": "0.1.0",
  "detail": "Model loaded and ready."
}
```

```bash
curl http://127.0.0.1:8000/api/health
```

### GET /api/model/info

Returns model identity, dataset provenance, feature-schema version, and
calibration status. Returns `503 model_unavailable` if no trained model is
loaded.

**Response — [`ModelInfoResponse`](#modelinforesponse)** (values below are from
the shipped **real-data** run, read from `artifacts/metadata.json`; note the
served `manifest_reference` is the basename only, never an absolute path):

```json
{
  "model_name": "XGBoost",
  "model_version": "0.1.0",
  "model_key": "xgboost",
  "trained_at": "2026-09-20T07:43:10Z",
  "dataset_source": "real",
  "synthetic": false,
  "synthetic_warning": null,
  "manifest_reference": "manifest.json",
  "feature_schema_version": "1.0",
  "calibration": {
    "method": "isotonic",
    "cv": 3,
    "brier_before": 0.18692637979984283,
    "brier_after": 0.06722983139426303,
    "improved": true,
    "status": "calibrated (isotonic/sigmoid via CV); probabilities adjusted toward observed frequencies"
  },
  "selection_metric": "roc_auc",
  "responsible_use_note": "Analytical estimate of default probability. NOT an autonomous lending decision. Requires human oversight."
}
```

```bash
curl http://127.0.0.1:8000/api/model/info
```

### POST /api/validate-input

Validates an applicant payload **without** producing a prediction. If the body
passes Pydantic validation, returns `valid: true` and the normalized fields;
otherwise returns `422` with the [error schema](#error-responses).

**Request:** [`ApplicantInput`](#request-schema-applicantinput) (same body as
`/api/predict`).

**Response — [`ValidationResponse`](#validationresponse):**

```json
{
  "valid": true,
  "normalized_fields": { "AMT_INCOME_TOTAL": 180000.0, "...": "..." },
  "errors": []
}
```

```bash
curl -X POST http://127.0.0.1:8000/api/validate-input \
  -H "Content-Type: application/json" \
  -d @applicant.json
```

### POST /api/predict

Returns a default-probability estimate, transparent risk score and band, model
version, an optional explanation, limitations, and a disclaimer. Returns `503
model_unavailable` if no model is loaded, or `422` for an invalid payload.

**Request:** [`ApplicantInput`](#request-schema-applicantinput).

Full valid request body (the example applicant — income 180,000, credit 600,000,
annuity 27,000, goods 540,000, age 35, 5 years employed, 2 family members, 0
children, external scores ~0.5/0.6/0.5, region population 0.02, Cash loans, F, no
car, owns realty, Working, Higher education, Married, House/apartment, Core
staff):

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 600000,
    "AMT_ANNUITY": 27000,
    "AMT_GOODS_PRICE": 540000,
    "AGE_YEARS": 35,
    "EMPLOYMENT_YEARS": 5,
    "CNT_FAM_MEMBERS": 2,
    "CNT_CHILDREN": 0,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.5,
    "REGION_POPULATION_RELATIVE": 0.02,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "F",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "OCCUPATION_TYPE": "Core staff"
  }'
```

**Response — [`PredictResponse`](#predictresponse).** The structure is shown
below; the numeric `default_probability` is produced by the model at request time
(the illustrative value here is internally consistent with the score/band rules,
not from a recorded run). `limitations` and `disclaimer` are the exact strings
the service emits for the shipped real-data model (when trained on the synthetic
fixture, a "MODEL TRAINED ON SYNTHETIC DATA ..." line is prepended to
`limitations`):

```json
{
  "prediction_id": null,
  "default_probability": 0.12,
  "risk_score": 12,
  "risk_band": "Moderate",
  "model_version": "0.1.0",
  "explanation_available": true,
  "explanation": {
    "method": "shap_tree",
    "interpretation": "SHAP values: signed contribution to the model's risk output for this applicant.",
    "note": null,
    "contributions": [
      { "feature": "ANNUITY_INCOME_RATIO", "contribution": 0.031, "direction": "increases_risk" },
      { "feature": "EXT_SOURCE_MEAN", "contribution": -0.024, "direction": "decreases_risk" }
    ]
  },
  "limitations": [
    "Estimate of default probability, not a guaranteed outcome.",
    "Trained on an application-level feature subset; does not use full credit bureau history.",
    "Requires human oversight; not an autonomous lending decision."
  ],
  "disclaimer": "The risk score is an analytical estimate of default probability, NOT a guaranteed outcome and NOT an autonomous lending decision."
}
```

Notes:
- `prediction_id` is `null` — the service is stateless today (persistence is a
  documented no-op seam).
- `risk_score = round(default_probability * 100)`.
- `risk_band` comes from the configured probability intervals: `Low` `[0.00,
  0.08)`, `Moderate` `[0.08, 0.20)`, `Elevated` `[0.20, 0.40)`, `High` `[0.40,
  1.00]`.
- The explanation `method` varies by backend: `shap_tree` when SHAP is installed
  and the model supports it, otherwise a labeled fallback
  (`linear_coefficient_times_value`, `tree_importance_fallback`, or
  `unavailable`). `explanation_available` is `false` if no contributions are
  produced.

### GET /api/metrics

Returns the full evaluation metrics from the last training run (the parsed
`artifacts/metrics.json`). Returns `404 not_found` if the file is absent.

Top-level keys include: `generated_at`, `dataset_source`, `synthetic`,
`synthetic_warning`, `selection_metric`, `selected_model`, `split_diagnostics`,
`validation_summary`, `per_model_validation`, `final_test_metrics`,
`threshold_analysis`, `optimal_f1_threshold`, `calibration`, `global_importance`,
`skipped_models`.

```bash
curl http://127.0.0.1:8000/api/metrics
```

### GET /api/feature-schema

Returns the machine-readable input contract (the parsed
`artifacts/feature_schema.json`), with `numeric_features`,
`categorical_features`, and `engineered_features`. Returns `{}` if the schema is
not loaded. Each numeric entry carries `name`, `label`, `unit`, `meaning`, `min`,
`max`, `example`, `missing_behavior`; each categorical entry carries `name`,
`label`, `meaning`, `categories`, `example`, `missing_behavior`.

```bash
curl http://127.0.0.1:8000/api/feature-schema
```

### GET /api/dataset/quality

Returns the dataset validation report and manifest summary:
`{ "validation_report": {...}, "manifest": {...} }`. Returns `404 not_found` if
neither is available.

```bash
curl http://127.0.0.1:8000/api/dataset/quality
```

## Request schema: ApplicantInput

Shared by `POST /api/validate-input` and `POST /api/predict`. Fields, types, and
bounds are derived from `loan_risk/schema.py`. Numeric fields are validated
against inclusive `[min, max]` bounds; categorical fields must be one of the
allowed values. **Optional** fields may be omitted (they are frequently missing
in real data and are imputed downstream); all other fields are **required**.

### Numeric fields

| Field                        | Type    | Required | Min | Max        | Unit      |
| ---------------------------- | ------- | -------- | --- | ---------- | --------- |
| `AMT_INCOME_TOTAL`           | number  | yes      | 0   | 5,000,000  | currency  |
| `AMT_CREDIT`                 | number  | yes      | 0   | 5,000,000  | currency  |
| `AMT_ANNUITY`                | number  | no       | 0   | 500,000    | currency  |
| `AMT_GOODS_PRICE`            | number  | no       | 0   | 5,000,000  | currency  |
| `AGE_YEARS`                  | number  | yes      | 18  | 100        | years     |
| `EMPLOYMENT_YEARS`           | number  | yes      | 0   | 50         | years     |
| `CNT_FAM_MEMBERS`            | number  | yes      | 1   | 20         | count     |
| `CNT_CHILDREN`               | number  | yes      | 0   | 20         | count     |
| `EXT_SOURCE_1`               | number  | no       | 0   | 1          | score 0-1 |
| `EXT_SOURCE_2`               | number  | no       | 0   | 1          | score 0-1 |
| `EXT_SOURCE_3`               | number  | no       | 0   | 1          | score 0-1 |
| `REGION_POPULATION_RELATIVE` | number  | yes      | 0   | 0.15       | ratio     |

### Categorical fields

| Field                 | Required | Allowed values                                                                                                                                                                                 |
| --------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NAME_CONTRACT_TYPE`  | yes      | `Cash loans`, `Revolving loans`                                                                                                                                                                |
| `CODE_GENDER`         | yes      | `M`, `F`, `XNA`                                                                                                                                                                                 |
| `FLAG_OWN_CAR`        | yes      | `Y`, `N`                                                                                                                                                                                        |
| `FLAG_OWN_REALTY`     | yes      | `Y`, `N`                                                                                                                                                                                        |
| `NAME_INCOME_TYPE`    | yes      | `Working`, `State servant`, `Commercial associate`, `Pensioner`, `Unemployed`, `Student`, `Businessman`, `Maternity leave`                                                                     |
| `NAME_EDUCATION_TYPE` | yes      | `Lower secondary`, `Secondary / secondary special`, `Incomplete higher`, `Higher education`, `Academic degree`                                                                                  |
| `NAME_FAMILY_STATUS`  | yes      | `Single / not married`, `Married`, `Civil marriage`, `Widow`, `Separated`, `Unknown`                                                                                                            |
| `NAME_HOUSING_TYPE`   | yes      | `House / apartment`, `Rented apartment`, `With parents`, `Municipal apartment`, `Office apartment`, `Co-op apartment`                                                                            |
| `OCCUPATION_TYPE`     | no       | `Laborers`, `Core staff`, `Accountants`, `Managers`, `Drivers`, `Sales staff`, `Cleaning staff`, `Cooking staff`, `Private service staff`, `Medicine staff`, `Security staff`, `High skill tech staff`, `Waiters/barmen staff`, `Low-skill Laborers`, `Realty agents`, `Secretaries`, `IT staff`, `HR staff` |

> **Note on `EMPLOYMENT_YEARS`:** the API requires this field. Internally, an
> omitted employment value would set a `DAYS_EMPLOYED_MISSING` indicator
> (pensioners/unemployed), but because the field is required at the API layer,
> callers must supply a value in `[0, 50]`. `DAYS_BIRTH` / `DAYS_EMPLOYED` from
> the raw dataset are converted to `AGE_YEARS` / `EMPLOYMENT_YEARS` during
> ingestion, so the API accepts these human-friendly columns directly.

## Response schemas

### HealthResponse

| Field             | Type    | Notes                          |
| ----------------- | ------- | ------------------------------ |
| `status`          | string  | `ok` or `degraded`             |
| `model_available` | boolean | true if a model is loaded      |
| `api_version`     | string  | e.g. `0.1.0`                   |
| `detail`          | string  | human-readable status message  |

### ModelInfoResponse

| Field                    | Type            | Notes                                                     |
| ------------------------ | --------------- | --------------------------------------------------------- |
| `model_name`             | string          | e.g. `Random Forest`                                      |
| `model_version`          | string          | e.g. `0.1.0`                                              |
| `model_key`              | string \| null  | internal key, e.g. `random_forest`                        |
| `trained_at`             | string \| null  | ISO-8601 training timestamp                               |
| `dataset_source`         | string          | `synthetic` or `real`                                     |
| `synthetic`              | boolean         | true when trained on synthetic data                       |
| `synthetic_warning`      | string \| null  | warning text when synthetic                               |
| `manifest_reference`     | string \| null  | path/reference to the dataset manifest                    |
| `feature_schema_version` | string          | e.g. `1.0`                                                |
| `calibration`            | object          | [`CalibrationInfo`](#calibrationinfo)                     |
| `selection_metric`       | string \| null  | metric used to select the model, e.g. `roc_auc`          |
| `responsible_use_note`   | string          | responsible-use statement                                 |

#### CalibrationInfo

| Field          | Type            | Notes                                    |
| -------------- | --------------- | ---------------------------------------- |
| `method`       | string \| null  | e.g. `isotonic`                          |
| `brier_before` | number \| null  | Brier score before calibration           |
| `brier_after`  | number \| null  | Brier score after calibration            |
| `improved`     | boolean \| null | whether calibration improved the Brier   |
| `status`       | string \| null  | human-readable calibration status        |

### ValidationResponse

| Field               | Type            | Notes                                 |
| ------------------- | --------------- | ------------------------------------- |
| `valid`             | boolean         | true when the payload is valid        |
| `normalized_fields` | object \| null  | the parsed/normalized applicant fields |
| `errors`            | array           | empty on success                      |

### PredictResponse

| Field                   | Type            | Notes                                                    |
| ----------------------- | --------------- | -------------------------------------------------------- |
| `prediction_id`         | string \| null  | `null` (stateless service today)                         |
| `default_probability`   | number          | `[0.0, 1.0]`                                             |
| `risk_score`            | integer         | `[0, 100]`, `round(default_probability * 100)`          |
| `risk_band`             | string          | `Low` / `Moderate` / `Elevated` / `High`                 |
| `model_version`         | string          | from model metadata                                      |
| `explanation_available` | boolean         | true if contributions were produced                     |
| `explanation`           | object \| null  | [`ExplanationBlock`](#explanationblock)                  |
| `limitations`           | array<string>   | includes a synthetic-data warning when applicable        |
| `disclaimer`            | string          | analytical-estimate / not-a-lending-decision statement   |

#### ExplanationBlock

| Field            | Type            | Notes                                                        |
| ---------------- | --------------- | ------------------------------------------------------------ |
| `method`         | string          | `shap_tree`, `linear_coefficient_times_value`, `tree_importance_fallback`, or `unavailable` |
| `interpretation` | string \| null  | how to read the contributions (association, not causation)   |
| `note`           | string \| null  | present for fallback/approximate explanations                |
| `contributions`  | array           | list of [`Contribution`](#contribution)                      |

#### Contribution

| Field          | Type    | Notes                                                       |
| -------------- | ------- | ----------------------------------------------------------- |
| `feature`      | string  | post-transform feature name                                 |
| `contribution` | number  | signed contribution (or unsigned in fallback)               |
| `direction`    | string  | `increases_risk`, `decreases_risk`, or `n/a` (unsigned)     |

## Error responses

All errors share one consistent shape. Stack traces and internal filesystem
paths are never exposed to the client.

**`ErrorResponse`:**

| Field    | Type                   | Notes                                            |
| -------- | ---------------------- | ------------------------------------------------ |
| `error`  | string                 | machine-readable code (see below)                |
| `detail` | string                 | human-readable message                           |
| `fields` | array \| null          | per-field errors (validation only)               |

| Status | `error`             | When                                                        |
| ------ | ------------------- | ----------------------------------------------------------- |
| `422`  | `validation_error`  | payload fails validation (bad type, out-of-bounds, unknown category, missing required field) |
| `503`  | `model_unavailable` | no trained model loaded (`/api/model/info`, `/api/predict`) |
| `404`  | `not_found`         | requested report not available (`/api/metrics`, `/api/dataset/quality`) |
| `500`  | `internal_error`    | unexpected server error (generic message; no internals leaked) |

**`422` validation error** — each entry in `fields` has `field`, `message`, and
`type`:

```json
{
  "error": "validation_error",
  "detail": "One or more input fields are invalid.",
  "fields": [
    {
      "field": "CODE_GENDER",
      "message": "Value error, CODE_GENDER must be one of ['F', 'M', 'XNA']; got 'Z'",
      "type": "value_error"
    }
  ]
}
```

**`503` model unavailable:**

```json
{
  "error": "model_unavailable",
  "detail": "No trained model is loaded. Train the pipeline first.",
  "fields": null
}
```

**`404` not found:**

```json
{
  "error": "not_found",
  "detail": "Metrics not available. Train the pipeline first.",
  "fields": null
}
```

## Model version behavior

- The model version originates in `artifacts/metadata.json` (`model_version`,
  set from `api.version` in `config/config.yaml`; currently `0.1.0`).
- `GET /api/model/info` exposes it as `model_version`, alongside `model_name`,
  `model_key`, `trained_at`, `feature_schema_version`, `selection_metric`, and
  the calibration block.
- Every `POST /api/predict` response echoes `model_version`, so a prediction can
  be traced to the exact model that produced it.
- **Synthetic flag:** when the model was trained on synthetic data,
  `GET /api/model/info` returns `synthetic: true` with a `synthetic_warning`, and
  each prediction's `limitations` array is prefixed with a synthetic-data
  warning. Treat all such outputs as demonstration-only, not real risk estimates.
- The model is loaded once at startup and is never retrained during a request. If
  the artifact is missing, `/api/model/info` and `/api/predict` return `503
  model_unavailable` and `/api/health` reports `degraded`.
