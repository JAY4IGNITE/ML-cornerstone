"""API tests for the loan-risk FastAPI backend (07_TESTING_CHECKLIST.md -> API Tests).

Every test exercises the real FastAPI app through the shared `api_client`
TestClient fixture (tests/conftest.py), which loads the already-trained
artifacts. Nothing is retrained here and no source files are modified. The
`valid_payload` fixture supplies a fully-specified, valid applicant.
"""
from __future__ import annotations


# 1. Health endpoint -------------------------------------------------------
def test_health_ok(api_client):
    resp = api_client.get("/api/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in ("status", "model_available", "api_version", "detail"):
        assert key in body, f"missing key {key!r} in health response"
    assert body["model_available"] is True
    assert body["status"] == "ok"
    assert isinstance(body["detail"], str) and body["detail"]


# 2. Model info endpoint ---------------------------------------------------
def test_model_info(api_client, cfg):
    resp = api_client.get("/api/model/info")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in ("model_name", "model_version", "synthetic", "calibration"):
        assert key in body, f"missing key {key!r} in model info response"
    # The API must advertise its training source honestly: the `synthetic`
    # flag must match the configured dataset source, whichever it is (real or
    # synthetic), never a hardcoded assumption.
    expected_synthetic = cfg["dataset"]["source"] == "synthetic"
    assert body["synthetic"] is expected_synthetic
    assert isinstance(body["model_name"], str) and body["model_name"]
    assert isinstance(body["model_version"], str) and body["model_version"]
    assert isinstance(body["calibration"], dict)


# 3. Valid prediction ------------------------------------------------------
def test_predict_valid(api_client, valid_payload):
    resp = api_client.post("/api/predict", json=valid_payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert 0 <= body["risk_score"] <= 100
    assert isinstance(body["risk_band"], str) and body["risk_band"]
    assert isinstance(body["limitations"], list) and len(body["limitations"]) > 0
    assert isinstance(body["disclaimer"], str) and body["disclaimer"]
    assert body["model_version"]


# 4. Invalid payload — bad categorical value -------------------------------
def test_predict_invalid_category(api_client, valid_payload):
    bad = dict(valid_payload)
    bad["CODE_GENDER"] = "Z"  # not in the allowed enum {M, F, XNA}
    resp = api_client.post("/api/predict", json=bad)
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"] == "validation_error"
    assert isinstance(body.get("fields"), list) and body["fields"]


# 5. Missing required field ------------------------------------------------
def test_predict_missing_required_field(api_client, valid_payload):
    bad = dict(valid_payload)
    del bad["AMT_INCOME_TOTAL"]  # required (NOT in OPTIONAL_FIELDS)
    resp = api_client.post("/api/predict", json=bad)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"] == "validation_error"


# 6. Invalid numerical value — out of documented bounds --------------------
def test_predict_numeric_out_of_bounds(api_client, valid_payload):
    bad = dict(valid_payload)
    bad["AGE_YEARS"] = 5  # below the documented minimum of 18
    resp = api_client.post("/api/predict", json=bad)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"] == "validation_error"


# 7. Optional field omitted still yields a prediction ----------------------
def test_predict_optional_field_omitted(api_client, valid_payload):
    payload = dict(valid_payload)
    del payload["EXT_SOURCE_1"]  # optional; imputed downstream
    resp = api_client.post("/api/predict", json=payload)
    assert resp.status_code == 200, resp.text
    assert 0.0 <= resp.json()["default_probability"] <= 1.0


# 8. Error response schema — consistent shape, no internal leaks -----------
def test_error_response_schema_no_leaks(api_client, valid_payload):
    bad = dict(valid_payload)
    bad["CODE_GENDER"] = "Z"
    resp = api_client.post("/api/predict", json=bad)
    assert resp.status_code == 422, resp.text
    body = resp.json()
    # Every error body carries string `error` and `detail`.
    assert isinstance(body["error"], str) and body["error"]
    assert isinstance(body["detail"], str) and body["detail"]
    # Never leak stack traces or internal filesystem paths to the client.
    assert "Traceback" not in resp.text
    assert "C:\\" not in body["detail"]


# 9. validate-input endpoint ----------------------------------------------
def test_validate_input_valid(api_client, valid_payload):
    resp = api_client.post("/api/validate-input", json=valid_payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["valid"] is True


def test_validate_input_invalid(api_client, valid_payload):
    bad = dict(valid_payload)
    bad["CODE_GENDER"] = "Z"
    resp = api_client.post("/api/validate-input", json=bad)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"] == "validation_error"


# 10. Read-only support endpoints ------------------------------------------
def test_metrics_endpoint(api_client):
    resp = api_client.get("/api/metrics")
    assert resp.status_code == 200, resp.text
    assert "final_test_metrics" in resp.json()


def test_feature_schema_endpoint(api_client):
    resp = api_client.get("/api/feature-schema")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["numeric_features"], list) and body["numeric_features"]
    assert isinstance(body["categorical_features"], list) and body["categorical_features"]


def test_dataset_quality_endpoint(api_client):
    resp = api_client.get("/api/dataset/quality")
    assert resp.status_code == 200, resp.text


# 11. Unhandled server error (500) leaks nothing sensitive -----------------
def test_predict_internal_error_leaks_nothing(monkeypatch, valid_payload):
    """Force the prediction path to raise an *unexpected* exception and prove the
    client sees ONLY the generic ErrorResponse — never a stack trace, the raw
    exception text/type, or an internal filesystem path (05_API_CONTRACT.md;
    backend/main.py `unhandled_exception_handler`).

    This test builds its own TestClient with ``raise_server_exceptions=False``
    instead of reusing the shared ``api_client``: Starlette's ServerErrorMiddleware
    *always* re-raises after producing the 500 response
    (starlette/middleware/errors.py), so the default client
    (``raise_server_exceptions=True``) would propagate the exception into the
    test rather than return the response we need to inspect.

    Patching the service keeps this hermetic — the availability gate is satisfied
    with a sentinel and predict is replaced, so no real artifacts are loaded and
    the training layer is never touched. ``monkeypatch`` reverts both attributes
    afterwards, leaving the shared service untouched for other tests."""
    from fastapi.testclient import TestClient

    import backend.main as main
    from backend.schemas import ErrorResponse

    # An adversarial failure whose message is deliberately packed with exactly
    # the kinds of internals that must NOT reach the client.
    secret_marker = "SENSITIVE-INTERNAL-9f83c2"

    def _boom(*_args, **_kwargs):
        raise RuntimeError(
            "model matrix failure at "
            "C:\\Users\\ramuv\\ml-cornerstone\\backend\\service.py:112 "
            "(fallback /Users/ramuv/loan-risk/secrets.env) "
            f"{secret_marker} "
            "Traceback (most recent call last): ..."
        )

    # Satisfy `service.available` (model is not None) without loading a real
    # artifact, then make the prediction itself blow up.
    monkeypatch.setattr(main.service, "model", object())
    monkeypatch.setattr(main.service, "predict", _boom)

    client = TestClient(main.app, raise_server_exceptions=False)
    resp = client.post("/api/predict", json=valid_payload)

    # (1) A clean 500 — the generic internal-error path fired.
    assert resp.status_code == 500, resp.text
    assert resp.headers["content-type"].startswith("application/json")

    # (2) Body matches the ErrorResponse schema: string `error` + `detail`, and
    #     no keys outside the documented contract.
    body = resp.json()
    assert {"error", "detail"} <= set(body), body
    assert set(body) <= {"error", "detail", "fields"}, body
    assert isinstance(body["error"], str) and body["error"]
    assert isinstance(body["detail"], str) and body["detail"]
    assert body["error"] == "internal_error"
    ErrorResponse(**body)  # constructs cleanly => shape conforms to the schema

    # (3) The detail returned to the client is the generic message, NOT the raw
    #     exception text.
    assert secret_marker not in body["detail"]

    # (4) Nothing sensitive anywhere in the serialized body: no stack/traceback
    #     marker, no raw exception repr/type, no internal filesystem path, no
    #     internal file/dir names, no injected secret.
    raw = resp.text
    forbidden = [
        "Traceback",       # stack-trace marker
        "RuntimeError",    # raw exception type / repr
        secret_marker,     # injected secret payload
        "service.py",      # internal source filename
        "secrets.env",     # internal filename
        "C:\\",            # Windows absolute path
        "/Users/",         # POSIX absolute path
        "ml-cornerstone",  # internal project directory name
    ]
    for token in forbidden:
        assert token not in raw, f"sensitive token {token!r} leaked in 500 body: {raw!r}"
