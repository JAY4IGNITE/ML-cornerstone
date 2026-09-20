# Intelligent Loan Risk Assessment System

A config-driven machine-learning system that estimates the **probability that a
loan applicant will experience repayment difficulty (default)**. It pairs a
scikit-learn training pipeline with a FastAPI prediction service and a React
dashboard.

> **This is a default-risk estimator, not a loan-approval engine.** It predicts
> repayment difficulty; it does **not** decide whether a loan should be granted,
> and it is **not** an autonomous lending decision-maker. Every output is an
> analytical estimate that requires human oversight. See
> [Responsible use](#responsible-use).

> **Trained on real data, with a scope limit.** The shipped model is trained on
> the real Kaggle Home Credit Default Risk `application_train.csv` (307,511 rows).
> Metrics are genuine held-out results — but the system uses only the
> **application-level feature subset** (no auxiliary bureau / previous-application
> tables), so they are an honest baseline, **not** the achievable ceiling and
> **not** evidence of fairness or fitness for real lending. The labeled synthetic
> fixture remains available (`dataset.source: synthetic`) for offline testing.

---

## Table of contents

- [Project overview](#project-overview)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Dataset acquisition](#dataset-acquisition)
- [Training](#training)
- [Evaluation](#evaluation)
- [API startup](#api-startup)
- [Frontend startup](#frontend-startup)
- [Deployment (Docker)](#deployment-docker)
- [Testing](#testing)
- [Responsible use](#responsible-use)

---

## Project overview

The system predicts the **probability of default** (Home Credit's `TARGET`,
where `1` = client with payment difficulty and `0` = all other cases) for a
single applicant described by application-level features.

It is deliberately **not** a loan-approval predictor. Approval predicts an
approval label; default prediction estimates repayment difficulty. The two are
kept separate by design (see [`brain/01_PROJECT_CONTEXT.md`](brain/01_PROJECT_CONTEXT.md)).

The output of a prediction is a set of three transparent, separately reported
signals plus context:

- **`default_probability`** — a calibrated probability in `[0, 1]`.
- **`risk_score`** — an explicitly defined mapping `round(default_probability * 100)`, i.e. an integer `0–100`.
- **`risk_band`** — a configured probability interval label (`Low` / `Moderate` / `Elevated` / `High`).

Every response also carries `limitations` and a `disclaimer` stating that the
result is an estimate and not an autonomous lending decision.

Key principles enforced throughout the codebase
([`brain/09_EXECUTION_RULES.md`](brain/09_EXECUTION_RULES.md)): never fabricate metrics or
data, never fit preprocessing on test data, use configuration instead of
hardcoded values, and keep the training and serving pipelines consistent.

## Architecture

The system is built in four layers, all deriving from a single feature contract.

### Feature-schema contract (`loan_risk/schema.py`)

One module defines every input column — its type, bounds, allowed categories,
missing-value behavior, and engineered-feature formulas. Synthetic-data
generation, data validation, preprocessing, the backend Pydantic models, and the
frontend form all derive from it, so they cannot drift apart. Training persists
this contract as a machine-readable `artifacts/feature_schema.json`, which the
API serves at `GET /api/feature-schema`.

### Data layer (`loan_risk/data/`)

- `synthetic.py` — generates a Home Credit-shaped labeled fixture for stress-testing.
- `download.py` — one-command acquisition of the real Kaggle dataset (no data committed to the repo).
- `ingestion.py` — loads the raw application file and standardizes it into the canonical model frame (e.g. converting the raw negative day-count `DAYS_BIRTH` / `DAYS_EMPLOYED` columns to `AGE_YEARS` / `EMPLOYMENT_YEARS` once, so train and serve share identical columns).
- `validate.py` — schema, type, missingness, duplicate, range, and leakage checks, written to a machine-readable report.
- `manifest.py` — builds the dataset manifest (source, counts, target meaning, missingness, limitations).

### ML pipeline (`loan_risk/pipeline/`)

A scikit-learn `Pipeline` + `ColumnTransformer` design. `run.py` orchestrates the
full flow: ensure a dataset exists → validate → reproducible stratified split
(test held out) → tune hyperparameters (RandomizedSearchCV) and select the best
model via 5-fold stratified cross-validation on TRAIN+VAL (selection_basis
`cross_validation_5fold_mean`), choosing the highest mean ROC-AUC — not a single
validation-split argmax → refit on TRAIN+VAL and calibrate probabilities (CV, no
test leakage) → final evaluation on the untouched TEST set → global explainability
→ persist artifacts.
Supporting modules cover preprocessing, feature engineering, splitting, model
specs, calibration, evaluation, explainability (SHAP with graceful fallback), and
the risk-score mapping.

### FastAPI backend (`backend/`)

Loads the fitted pipeline, metadata, and feature schema **once at startup** and
never retrains during a request. `schemas.py` builds the applicant request model
dynamically from `loan_risk/schema.py` (bounds + categorical enums). Errors use a
consistent schema and never leak stack traces or internal paths. Full details:
[`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md).

### React frontend (`frontend/`)

A React + Vite + TypeScript + Tailwind dashboard. During development it runs on
port `5173` and proxies `/api` to the backend at `http://127.0.0.1:8000`, so the
two run side by side with no CORS friction.

### Config-driven design (`config/config.yaml`)

Nothing that varies — paths, seeds, thresholds, hyperparameters, risk bands, CORS
origins, the dataset source — is hardcoded in logic. `loan_risk/config.py` loads
`config/config.yaml` by default, or a file pointed to by the `LOAN_RISK_CONFIG`
environment variable.

## Repository structure

Verified against the working tree. Directories marked *(git-ignored)* are
produced by running the pipeline and are not committed.

```
ml-cornerstone/
├── config/
│   └── config.yaml                 # central configuration (paths, seeds, models, bands)
├── loan_risk/                      # Python package: ML pipeline + shared schema
│   ├── config.py                   # YAML config loader (single source of truth)
│   ├── schema.py                   # feature contract shared by data / ML / API / frontend
│   ├── data/
│   │   ├── synthetic.py            # synthetic Home Credit-shaped generator
│   │   ├── download.py             # real Kaggle dataset acquisition
│   │   ├── ingestion.py            # raw file -> canonical model frame
│   │   ├── validate.py             # data-validation checks + report
│   │   └── manifest.py             # dataset manifest builder
│   └── pipeline/
│       ├── preprocess.py           # numeric/categorical ColumnTransformer
│       ├── features.py             # engineered feature definitions
│       ├── split.py                # reproducible stratified split
│       ├── models.py               # model specs (LR / DecisionTree / RF / XGBoost)
│       ├── calibrate.py            # probability calibration
│       ├── evaluate.py             # metrics, threshold + calibration analysis
│       ├── explain.py              # global + local explainability (SHAP w/ fallback)
│       ├── risk_score.py           # probability -> score / band mapping
│       ├── artifacts.py            # model / metadata / schema / metrics persistence
│       └── run.py                  # training orchestrator (entrypoint)
├── backend/                        # FastAPI prediction service
│   ├── main.py                     # app, routes, consistent error handlers
│   ├── schemas.py                  # Pydantic request/response models (built from schema.py)
│   └── service.py                  # model loading + prediction service layer
├── frontend/                       # React + Vite + TypeScript dashboard
│   ├── package.json
│   ├── vite.config.ts              # dev server :5173, proxies /api -> :8000
│   ├── Dockerfile                  # build the Vite bundle, serve via nginx
│   ├── nginx.conf                  # SPA routing + /api proxy to backend:8000
│   ├── index.html
│   └── src/
│       ├── main.tsx / App.tsx / index.css
│       ├── components/             # Layout, ui, states
│       ├── lib/                    # api client, types, formatters
│       ├── pages/                  # 8 pages: Overview, Assessment, Result, Explainability, Performance, DatasetQuality, ModelInfo, ResponsibleUse
│       └── store/                  # prediction context
├── artifacts/                      # trained pipeline + metadata (git-ignored; regenerated)
│   ├── model.joblib
│   ├── metadata.json
│   ├── feature_schema.json
│   └── metrics.json
├── data/                           # datasets (git-ignored) + manifest
│   ├── raw/application_train.csv           # real Kaggle data (current source)
│   ├── synthetic/application_train_synthetic.csv
│   └── manifest.json
├── reports/                        # validation report + plots (git-ignored)
│   └── validation_report.json
├── docs/
│   ├── API_DOCUMENTATION.md
│   ├── DATA_DICTIONARY.md
│   ├── MODEL_CARD.md
│   ├── EVALUATION_REPORT.md
│   └── RESPONSIBLE_USE.md
├── tests/
│   ├── conftest.py
│   ├── test_api.py                 # API contract + error-leak tests
│   ├── test_data.py                # ingestion / validation / manifest tests
│   └── test_ml.py                  # split, preprocessing, pipeline tests
├── docker/
│   └── entrypoint.sh               # first-run train (synthetic fallback) then serve
├── Dockerfile                      # backend + ML pipeline image
├── docker-compose.yml              # local stack: backend :8000 + nginx frontend :5173
├── dataset.py                      # standalone kagglehub download helper
├── pyproject.toml
├── requirements.txt
├── requirements-extras.txt
└── README.md
```

## Setup

Requires **Python ≥ 3.11**. Commands below use the Windows layout (this repo
lives at `C:\Users\ramuv\ml-cornerstone`); on macOS/Linux, activate with
`source .venv/bin/activate` instead.

```bat
:: from the repo root
python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` covers the core runtime (pandas, numpy, pyarrow,
scikit-learn, joblib, PyYAML, pydantic, FastAPI, uvicorn, matplotlib) plus the
test tooling (pytest, httpx). `pip install -e .` installs the `loan_risk` package
in editable mode and registers three console scripts:

| Console script         | Equivalent module command                 |
| ---------------------- | ------------------------------------------ |
| `loan-risk-synth`      | `python -m loan_risk.data.synthetic`       |
| `loan-risk-download`   | `python -m loan_risk.data.download`        |
| `loan-risk-train`      | `python -m loan_risk.pipeline.run`         |

Optional ML accelerators (`xgboost`, `shap`) live in a separate file so the
pipeline still runs and tests still pass without them — XGBoost is skipped if
absent, and SHAP explanations fall back to the model's native
importances/coefficients:

```bat
pip install -r requirements-extras.txt
```

## Dataset acquisition

The system is configured for the **real** dataset. Choose a source with
`dataset.source` in [`config/config.yaml`](config/config.yaml):

```yaml
dataset:
  source: "real"   # "real" (current) or "synthetic"
```

### Synthetic fixture (offline testing)

Generates a labeled Home Credit-shaped fixture (20,000 rows, ~8.1% positive rate)
into `data/synthetic/`:

```bat
python -m loan_risk.data.synthetic
```

Training also auto-generates this fixture if it is missing and
`dataset.source: synthetic`. The generator prints a reminder that the data — and
any metric computed on it — is synthetic-only.

### Real Kaggle data

The real Home Credit Default Risk dataset is a Kaggle competition dataset and is
**not** committed to the repo (`data/raw/` is git-ignored). To fetch it:

```bat
python -m loan_risk.data.download
```

This requires the Kaggle CLI plus an API token (`kaggle.json` under
`%USERPROFILE%\.kaggle\`, or the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment
variables) **and having accepted the competition rules** at
`https://www.kaggle.com/competitions/home-credit-default-risk/rules`. If
credentials or rule acceptance are missing, the script prints exact manual
instructions and exits non-zero — it never fabricates data. `application_train.csv`
may also be placed in `data/raw/` manually.

`dataset.py` at the repo root is a minimal standalone `kagglehub` alternative
that logs in interactively and downloads the competition bundle into the
`kagglehub` cache; copy `application_train.csv` from there into `data/raw/` if you
use it.

Once `application_train.csv` is present, set `dataset.source: "real"` in the
config (already the default) and run training — no code changes are needed.

## Training

```bat
python -m loan_risk.pipeline.run
```

The run validates the data, writes a manifest, splits reproducibly (stratified,
test held out), trains every enabled model, selects the best by
`evaluation.selection_metric` (`roc_auc`), refits and calibrates, evaluates on the
untouched test set, and computes global feature importance. Every number written
is computed from real predictions; synthetic runs are flagged synthetic
throughout.

It writes these artifacts into `artifacts/`:

| File                          | Contents                                                              |
| ----------------------------- | --------------------------------------------------------------------- |
| `model.joblib`                | the fitted, calibrated end-to-end pipeline (the serving contract)     |
| `metadata.json`               | model name/version, training timestamp, dataset source, calibration   |
| `feature_schema.json`         | the exact input contract the API and frontend derive from             |
| `metrics.json`                | full evaluation results (per-model validation + 5-fold CV, hyperparameter tuning, test, threshold, calibration, global + permutation importance, fairness diagnostic) |

It also refreshes `data/manifest.json` and `reports/validation_report.json`.

## Evaluation

Full evaluation output is written to `artifacts/metrics.json` and the dataset
validation report to `reports/validation_report.json`. A written narrative lives
in [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md); the model card is at
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

The values below are read from `artifacts/metrics.json` / `artifacts/metadata.json`
(regenerated by training, git-ignored) and are **real** held-out results from the
shipped run (trained `2026-09-20T11:58:50Z`, `dataset_source: real`).

**Selected model: XGBoost**, selected by ROC-AUC. Split:
196,806 train / 49,202 validation / 61,503 test (seed 42, stratified, ~8.07%
positive rate in every split).

Final held-out test metrics at the default 0.5 threshold:

| Metric      | Value |
| ----------- | ----- |
| ROC-AUC     | 0.762 |
| PR-AUC      | 0.250 |
| Brier score | 0.067 |
| Accuracy    | 0.920 |
| Precision   | 0.610 |
| Recall      | 0.018 |
| F1          | 0.035 |

Because the classes are imbalanced (~8.1% positive), the 0.5-threshold recall is
low; `metrics.json` includes a full threshold analysis and a `selected_threshold`
of ≈0.1665 **chosen on the validation set** (its test generalization is F1 ≈0.315,
recall ≈0.395) so a decision threshold can be chosen against the operating
trade-off rather than defaulting to 0.5. Probability calibration (isotonic, cv=3)
improved the Brier score from **~0.196** to **~0.067** on this run. For context,
published multi-table Home Credit solutions reach ~0.79–0.80 ROC-AUC by joining
the auxiliary bureau/previous-application tables; ~0.762 here is the honest
application-only baseline. Full narrative:
[`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md).

## API startup

Start the FastAPI service (loads the trained artifacts at startup):

```bat
uvicorn backend.main:app --reload --port 8000
```

- Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/api/health`

If no trained model is present, `GET /api/health` reports `degraded` and
`GET /api/model/info` and `POST /api/predict` return `503 model_unavailable` —
train the pipeline first. Full endpoint, schema, and error reference:
[`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md).

## Frontend startup

```bat
cd frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and proxies `/api` requests to the
backend at `http://127.0.0.1:8000`, so start the backend first. Other scripts:
`npm run build` (`tsc --noEmit && vite build`) and `npm run typecheck`
(`tsc --noEmit`).

The dashboard has eight pages, all present in `frontend/src/pages/`: Overview,
Assessment, Result, Explainability, Performance, Dataset Quality, Model Info, and
Responsible Use (routed in `frontend/src/App.tsx`). The API client, shared
components, types, and prediction store are in place.

## Deployment (Docker)

A `docker-compose.yml` brings up the full local stack — the FastAPI backend and
the nginx-served React frontend — with one command:

```bat
docker compose up --build
```

This is a **local demo stack, not a production deployment**: no TLS, no auth, no
persistence guarantees. Two services come up:

- **`backend`** — the `Dockerfile` image (Python + ML pipeline + FastAPI),
  published on `http://localhost:8000`. Its `docker/entrypoint.sh` runs on start:
  if a trained `artifacts/model.joblib` already exists it serves it directly;
  otherwise it trains first. When no artifact **and** no real-data CSV
  (`data/raw/application_train.csv`) are present, it forces training on the
  **synthetic** fixture via a generated `LOAN_RISK_CONFIG` override (so a fresh
  clone starts cleanly without the real Kaggle data). If a real-data CSV is
  mounted, it trains on the configured source (`real`) instead. `artifacts/`,
  `data/`, and `reports/` are mounted from the host so trained models and
  generated data persist between runs.
- **`frontend`** — the `frontend/Dockerfile` image builds the Vite bundle and
  serves it with nginx (container port `80`, published on `http://localhost:5173`).
  `frontend/nginx.conf` handles SPA routing and proxies `/api` to the backend
  service, so no CORS configuration is needed.

Startup is **health-gated**: the backend exposes a `/api/health` healthcheck with
a long `start_period` to allow first-run training to finish, and the frontend
`depends_on` the backend being healthy before it starts. To serve a real-data
model, either mount pre-trained artifacts into `artifacts/` or place
`application_train.csv` under `data/raw/` (see [Dataset acquisition](#dataset-acquisition))
before bringing the stack up.

## Testing

The suite uses pytest. Session-scoped fixtures reuse the existing artifacts and
dataset read-only, building them once if absent (so a first run may train on the
configured source). Run it with the virtual environment's interpreter:

```bat
.venv\Scripts\python -m pytest
```

- `tests/test_api.py` — the API testing checklist: health, model info, a valid
  prediction, invalid category, missing required field, out-of-bounds numeric,
  optional-field omission, the consistent error-response schema for both the 422
  (validation) and 500 (internal-error) paths (asserting no tracebacks or
  filesystem paths leak), `validate-input`, and the metrics, feature-schema, and
  dataset-quality support endpoints.
- `tests/test_data.py` — ingestion, standardization, validation, and manifest.
- `tests/test_ml.py` — the reproducible stratified split (including
  stratification-preserves-positive-rate and no-ID-leakage-across-splits
  assertions), preprocessing, and pipeline behavior.

`pyproject.toml` sets `testpaths = ["tests"]`.

## Responsible use

- **Not an autonomous decision-maker.** Outputs are analytical estimates of
  default probability and must not be treated as automated approve/deny
  decisions. Human oversight is required for any real use.
- **Estimate, not a guarantee.** The probability, score, and band describe
  modeled risk, not a certain outcome. Feature importance describes association,
  **not** causation, and the system does not invent adverse-action reasons.
- **Real data, honest labeling.** The shipped model is trained on the real Kaggle
  data (`synthetic: false` in `GET /api/model/info`). If you switch back to the
  synthetic fixture, the API advertises it honestly (`synthetic: true`, plus a
  synthetic warning prepended to prediction `limitations`).
- **Scope limits.** The served model uses an application-level feature subset and
  does not use full credit-bureau history, which caps discrimination below
  published multi-table solutions. It carries no fairness, bias, or legal
  compliance guarantees; **no fairness analysis has been performed**, and
  `CODE_GENDER` is among the most important features — see
  [`docs/RESPONSIBLE_USE.md`](docs/RESPONSIBLE_USE.md).
- **Privacy.** The service does not log raw applicant PII, and error responses
  never expose stack traces or internal paths.

See [`docs/RESPONSIBLE_USE.md`](docs/RESPONSIBLE_USE.md),
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and the spec files
[`brain/01_PROJECT_CONTEXT.md`](brain/01_PROJECT_CONTEXT.md) and
[`brain/05_API_CONTRACT.md`](brain/05_API_CONTRACT.md) for the full intended-use and
out-of-scope statements.
