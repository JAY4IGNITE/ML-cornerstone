"""FastAPI application (05_API_CONTRACT.md).

Endpoints:
  GET  /api/health         service + model availability
  GET  /api/model/info     model name/version/manifest/calibration
  POST /api/validate-input validate a payload WITHOUT predicting
  POST /api/predict         probability + score + band + explanation

Safety:
  * Pydantic validation with numeric bounds + categorical enums.
  * Consistent ErrorResponse schema; no stack traces / internal paths leaked.
  * No raw applicant PII logged.
  * Model loaded once at startup; never retrained during a request.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from loan_risk.config import load_config

from .schemas import (
    ApplicantInput,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
    ValidationResponse,
)
from .service import ModelService

logger = logging.getLogger("loan_risk.api")
logging.basicConfig(level=logging.INFO)

cfg = load_config()
service = ModelService(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if service.available:
        logger.info("Model loaded: %s v%s (synthetic=%s)",
                    service.metadata.get("model_name"),
                    service.metadata.get("model_version"),
                    service.metadata.get("synthetic"))
    else:
        logger.warning("Model NOT loaded: %s", service.load_error)
    yield


app = FastAPI(
    title=cfg["api"]["title"],
    version=cfg["api"]["version"],
    description=("Loan DEFAULT-risk estimation API. Outputs are analytical "
                 "estimates, NOT autonomous lending decisions."),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["api"]["cors_origins"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---- consistent error handling ---------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    fields = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body"),
         "message": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            detail="One or more input fields are invalid.",
            fields=fields,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log server-side, but never leak internals to the client.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail="An internal error occurred. Please try again later.",
        ).model_dump(),
    )


# ---- endpoints -------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if service.available else "degraded",
        model_available=service.available,
        api_version=cfg["api"]["version"],
        detail=("Model loaded and ready." if service.available
                else "Model artifact not found — train the pipeline first."),
    )


@app.get("/api/model/info", response_model=ModelInfoResponse,
         responses={503: {"model": ErrorResponse}}, tags=["model"])
def model_info():
    if not service.available:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(error="model_unavailable",
                                  detail="No trained model is loaded.").model_dump(),
        )
    return service.model_info()


@app.post("/api/validate-input", response_model=ValidationResponse, tags=["prediction"])
def validate_input(applicant: ApplicantInput) -> ValidationResponse:
    # Reaching here means Pydantic already validated bounds + categories.
    return ValidationResponse(
        valid=True,
        normalized_fields=applicant.model_dump(),
        errors=[],
    )


@app.post("/api/predict", response_model=PredictResponse,
          responses={503: {"model": ErrorResponse}}, tags=["prediction"])
def predict(applicant: ApplicantInput):
    if not service.available:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="model_unavailable",
                detail="No trained model is loaded. Train the pipeline first.",
            ).model_dump(),
        )
    result = service.predict(applicant.model_dump())
    return result


# ---- read-only support endpoints (power the dashboard; not core contract) ----
def _or_404(payload, what: str):
    if payload is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(error="not_found",
                                  detail=f"{what} not available. Train the pipeline first.").model_dump(),
        )
    return payload


@app.get("/api/metrics", tags=["reports"],
         responses={404: {"model": ErrorResponse}})
def metrics():
    """Full evaluation metrics (real numbers from the last training run)."""
    return _or_404(service.get_metrics(), "Metrics")


@app.get("/api/feature-schema", tags=["reports"])
def feature_schema():
    """Input contract driving the assessment form."""
    return service.get_feature_schema() or {}


@app.get("/api/dataset/quality", tags=["reports"],
         responses={404: {"model": ErrorResponse}})
def dataset_quality():
    """Validation report + manifest summary for the Dataset Quality page."""
    report = service.get_validation_report()
    manifest = service.get_manifest()
    if report is None and manifest is None:
        return _or_404(None, "Dataset quality report")
    return {"validation_report": report, "manifest": manifest}


@app.get("/", include_in_schema=False)
def root():
    return {"service": cfg["api"]["title"], "docs": "/docs",
            "health": "/api/health"}
