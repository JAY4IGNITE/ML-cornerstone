# Responsible Use — Intelligent Loan Risk Assessment System

This document states plainly what the system is, what it is **not**, and what has and has **not** been validated. It is a policy, not a results report: it deliberately contains **no metric numbers, no fairness numbers, and no certifications**. For the current data source and measured results, read it alongside `docs/MODEL_CARD.md` and `docs/EVALUATION_REPORT.md`, which are the records of what the model was trained and evaluated on.

> ## ⚠️ This system is a decision-support tool, not a decision-maker.
> It produces an **estimate of the probability that an applicant defaults** (experiences repayment difficulty, `TARGET=1`). It does **not** approve, decline, price, or set terms on a loan, and it is **not** an autonomous lending system. Per `01_PROJECT_CONTEXT.md`: _"Do not build an autonomous lending decision-maker."_

---

## 1. Purpose

- The system estimates a **default probability** and derives a transparent 0–100 risk score and a risk band (Low / Moderate / High / Critical, defined on probability in `config/config.yaml: risk_score.bands`). The score mapping is explicit: `risk_score = round(default_probability * 100)`.
- Its purpose is to **support human review** — to help a qualified reviewer rank, flag, and reason about applications. It is an analytical input to a person's judgment, not a verdict.
- It is **not a loan-approval system.** It predicts repayment difficulty, not an approval label. Any approval workflow, if one is ever built, must remain **separate** from this default-risk model (`01_PROJECT_CONTEXT.md`).
- It is **not an autonomous decision-maker.** The in-process service returns analyst review indicators and limitations; the model metadata carries the corresponding responsible-use note.

---

## 2. Human Oversight

**Human oversight is required for any use of the output.**

- Every output — probability, risk score, and band — is an estimate meant to **inform a human reviewer**, never to act on its own.
- A qualified person makes any decision that affects an applicant. The model ranks and flags; it does not approve, decline, price, or set terms.
- The system must **not** be wired to auto-approve or auto-decline applicants, or to trigger any applicant-facing action without a human in the loop.
- The risk score is an estimate, not a guaranteed outcome, and must never be presented to a reviewer or an applicant as one.

---

## 3. Scope and Data Limitations

- **Application-level feature subset only.** Serving uses an application-level subset of features. The supporting Home Credit tables (bureau / credit-history, previous applications, installments, POS-cash, credit-card balance) are **not** used in a prediction; they are a documented offline/advanced extension (`data/manifest.json` known_limitations, `loan_risk/schema.py` scope note). **There is no full credit-bureau history behind a prediction.**
- **Home Credit dataset framing.** The system is built on the Kaggle Home Credit Default Risk dataset. Its population reflects a specific lending context and **may not generalize** to other geographies, lenders, loan products, or applicant populations. No external-population validation has been performed.
- **Associations, not causation.** Feature importances and any explanations describe **association with the model's predictions, not causal effects** (`artifacts/metrics.json` labels importance as _"Association with predicted risk, not causation."_). An importance value does not mean an attribute causes default and must not be used to justify treating any group differently, or to generate adverse-action reasons.
- **No temporal or drift validation.** The split is not time-aware (`config/config.yaml: split` defines only `test_size` / `val_size` / `stratify` — no time column), and there is no data-drift or concept-drift monitoring. Estimates can silently degrade if input distributions or default behavior shift.

---

## 4. Fairness and Bias — Not Audited

**No fairness or bias audit has been performed. This must be stated plainly and not glossed over.**

- **A per-group diagnostic exists, but it is not an audit.** `artifacts/metrics.json` now includes a fairness *diagnostic* that slices performance by gender (`CODE_GENDER`) and age band at the model's operating threshold, reporting per-group selection rate, false-negative / false-positive rate, and ROC-AUC alongside simple disparity ratios. This is a screen to surface disparities for human review — it is **not** a disparate-impact analysis, an equalized-odds / demographic-parity / calibration-by-group study, a bias mitigation, or a fairness certification. It carries an explicit diagnostic-only disclaimer, and the disparities it surfaces are reasons the review below remains **required**, not evidence the system is fair.
- The feature set includes demographic fields and attributes that can correlate with protected groups. Using such features in a risk model can produce **disparate impact** even without explicit intent. Whether they should be used at all in a real system is a legal/ethical decision that has **not** been made here.
- Therefore **the system makes no fairness claim, and none can be inferred.** It has **not** been shown to be fair, unbiased, or non-discriminatory across protected groups. Per `01_PROJECT_CONTEXT.md`, the project must _"never claim production readiness, fairness, or legal compliance without evidence"_ — and there is no such evidence today.

---

## 5. Legal and Regulatory Review — Not Performed

**No legal or regulatory review has been done.**

- There has been **no** review against credit and lending regulations or data-protection law — including, without limitation, **ECOA** (Equal Credit Opportunity Act), **FCRA** (Fair Credit Reporting Act), and **GDPR** (General Data Protection Regulation).
- The system carries **no certification** of any kind and must **not** be represented as compliant, approved, or production-ready.
- **It must not be used for real credit decisions** without a fairness/bias assessment on real, licensed data and a documented legal and regulatory review appropriate to the jurisdiction and use case.

---

## 6. Privacy and Data Handling

Grounded in `loan_risk/serving/service.py`, `ui/state.py`, and `.gitignore`:

- The service does not deliberately log applicant payloads or persist submissions. Inputs and results remain in Streamlit session state until cleared or the session ends.
- Validation failures return field errors; model-loading failures expose generic messages. Exception diagnostics are logged server-side for maintenance; deployment logging needs an independent privacy review.
- There is no separate HTTP prediction API in this build. Streamlit calls the Python service directly.
- Secrets and raw datasets are excluded from source control. Trained model artifacts are tracked for the local demo; do not treat them as anonymized raw datasets or load untrusted joblib files.
- There is no authentication, role-based access, durable audit history, retention policy, or TLS termination in this application. Local functional testing is not a security or privacy certification.

---

## 7. Non-Autonomous Use Statement

**This system must not be used as an autonomous lending decision-maker.**

- It **estimates** default probability; it does **not** decide, approve, decline, price, or set terms.
- Every prediction requires **human review** before any action affecting an applicant.
- It must **not** be represented as production-ready, fair, or legally compliant. Its performance, fairness, and compliance on real applicants are **unverified** — see `docs/MODEL_CARD.md` and `docs/EVALUATION_REPORT.md` for what has actually been measured.

---

## What has and has NOT been validated

| Item | Status |
|---|---|
| Purpose framing (default-risk estimate, human-oversight tool) | Stated and enforced in code/config |
| Applicant PII kept out of logs | Service payload logging is absent; independent deployment review remains needed |
| No stack traces / internal paths returned to clients | Generic model-load and UI inference errors; no standalone HTTP API |
| Secrets excluded from source control | Verified in `.gitignore` |
| Fairness / subgroup performance | **NOT validated** — no fairness or bias audit exists |
| Legal / regulatory review (ECOA, FCRA, GDPR) | **NOT performed** |
| API authentication / access control | **Absent** — not implemented |
| Generalization beyond the Home Credit population | **NOT validated** |
| Temporal stability / drift | **NOT validated** — split is not time-aware, no monitoring |
| Production readiness / compliance | **NOT established** — no certification of any kind |

_Be honest about the boundary: this build demonstrates an engineering pipeline. It is not evidence that the model is accurate, fair, or safe for real lending decisions. Any real-world use requires retraining and re-evaluation on licensed data, a fairness assessment, and a documented legal and regulatory review._
