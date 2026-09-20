"""End-to-end training orchestrator (02_IMPLEMENTATION_PLAN.md phases 2-6).

Flow:
  1. Ensure a dataset exists (auto-generate synthetic fixture if configured).
  2. Load raw -> validate -> write validation report + manifest.
  3. Standardize -> reproducible stratified split (test held out).
  4. Train each enabled model on TRAIN, evaluate on VAL.
  5. Select the best model by configured metric (NOT accuracy alone).
  6. Refit best on TRAIN+VAL, calibrate probabilities (CV, no test leakage).
  7. Select the F1 operating point on VALIDATION, then evaluate on the untouched
     TEST set (metrics + threshold generalization + calibration curve).
  8. Global explainability.
  9. Persist model + metadata + feature_schema + metrics.

Every number written is computed from real predictions. Synthetic runs are
flagged synthetic throughout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config, load_config
from ..data.ingestion import resolve_source_path, standardize
from ..data.manifest import build_manifest, write_manifest
from ..data.synthetic import write_synthetic
from ..data.validate import validate_raw, write_report
from .artifacts import (
    METADATA_FILE,
    METRICS_FILE,
    save_feature_schema,
    save_json,
    save_model,
)
from .calibrate import calibrate_pipeline
from .evaluate import (
    best_f1_threshold,
    calibration_curve_points,
    compute_metrics,
    threshold_analysis,
)
from .explain import global_importance, permutation_global_importance, shap_available
from .fairness import fairness_report
from .models import build_model_specs, build_pipeline, xgboost_available
from .split import make_splits
from .tuning import apply_params, cross_val_select, tune_model

import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prob(estimator, X) -> np.ndarray:
    return estimator.predict_proba(X)[:, 1]


def _portable_reference(path: Path, root: Path) -> str:
    """A portable, non-leaking reference to ``path`` for persisted metadata.

    Prefer the path relative to the repo root; if ``path`` lives outside the
    root (a configured absolute path), fall back to the bare filename so we
    never persist — or later expose via the API — an absolute host path.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def ensure_dataset(cfg: Config) -> None:
    path = resolve_source_path(cfg)
    if path.exists():
        return
    if cfg["dataset"]["source"] == "synthetic":
        print("[run] synthetic fixture missing — generating it now ...")
        write_synthetic(cfg)
    else:
        raise FileNotFoundError(
            f"Real data not found at {path}. Run `python -m loan_risk.data.download` "
            f"or set dataset.source: synthetic in config.yaml."
        )


def train(cfg: Config) -> dict[str, Any]:
    cfg.ensure_dirs()
    ensure_dataset(cfg)
    source = cfg["dataset"]["source"]
    is_synth = source == "synthetic"

    # --- load + validate raw ---
    raw_path = resolve_source_path(cfg)
    df_raw = pd.read_csv(raw_path)
    report = validate_raw(df_raw, cfg, source_file=str(raw_path.name))
    report_path = write_report(report, cfg)
    print(f"[run] validation: {report.summary()} -> {report_path}")
    if report.has_errors():
        errs = [c.name for c in report.checks if c.status == "fail"]
        raise SystemExit(f"[run] BLOCKER: validation failed ({errs}). See {report_path}.")

    # --- manifest ---
    manifest = build_manifest(df_raw, cfg, generated_at=_now(),
                              validation_summary=report.summary())
    manifest_path = write_manifest(manifest, cfg)
    print(f"[run] manifest -> {manifest_path}")

    # --- standardize + split ---
    df = standardize(df_raw, cfg)
    split = make_splits(df, cfg)
    print(f"[run] split: {split.diagnostics}")

    id_col = cfg.id_column

    def _X(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns=[id_col], errors="ignore")

    # class imbalance -> scale_pos_weight for xgboost
    pos = float(split.y_train.mean())
    spw = (1 - pos) / pos if pos > 0 else 1.0

    specs, skipped = build_model_specs(cfg, scale_pos_weight=spw)
    for note in skipped:
        print(f"[run] {note}")

    X_train, X_val, X_test = _X(split.X_train), _X(split.X_val), _X(split.X_test)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([split.y_train, split.y_val])

    # --- optional hyperparameter tuning (RandomizedSearchCV on a subsample) ---
    # Tune BEFORE selection so the model comparison is between tuned models.
    tuning_results: dict[str, Any] = {}
    tuned_specs: list = []
    for spec in specs:
        res = tune_model(spec, X_trainval, y_trainval, cfg)
        if res is not None:
            tuning_results[spec.key] = res
            tuned_specs.append(apply_params(spec, res["best_params"]))
            print(f"[run] tuned {spec.display_name}: {res['best_params']} "
                  f"(cv {res['scoring']}={res['best_cv_score']:.4f})")
        else:
            tuned_specs.append(spec)
    specs = tuned_specs

    # --- per-model held-out VALIDATION metrics (train on TRAIN, eval VAL) ---
    # Kept as an interpretable head-to-head comparison on a single held-out split.
    selection_metric = cfg["evaluation"]["selection_metric"]
    per_model: dict[str, Any] = {}
    spec_by_key: dict[str, Any] = {}
    for spec in specs:
        print(f"[run] training {spec.display_name} (val comparison) ...")
        pipe = build_pipeline(spec, cfg)
        pipe.fit(X_train, split.y_train)
        val_prob = _prob(pipe, X_val)
        per_model[spec.key] = {
            "display_name": spec.display_name,
            "val_metrics": compute_metrics(split.y_val, val_prob),
        }
        spec_by_key[spec.key] = spec

    if not spec_by_key:
        raise SystemExit("[run] BLOCKER: no models were trained.")

    # --- select best model ---
    cv_folds = int(cfg.get("selection", {}).get("cv_folds", 1))
    per_model_cv: dict[str, Any] = {}
    if cv_folds > 1:
        # Stratified k-fold CV on TRAIN+VAL, select by MEAN score (std reported).
        print(f"[run] cross-validated selection ({cv_folds}-fold) ...")
        best_key, per_model_cv = cross_val_select(specs, X_trainval, y_trainval, cfg)
        for spec in specs:
            m = per_model_cv[spec.key]
            per_model[spec.key]["cv_metrics"] = m
            print(f"[run]   {spec.display_name}: cv {m['scoring']}="
                  f"{m['mean']:.4f} +/- {m['std']:.4f}")
        selection_basis = f"cross_validation_{cv_folds}fold_mean"
        print(f"[run] selected: {spec_by_key[best_key].display_name} "
              f"(cv mean {per_model_cv[best_key]['scoring']}="
              f"{per_model_cv[best_key]['mean']:.4f})")
    else:
        def _score(key: str) -> float:
            v = per_model[key]["val_metrics"].get(selection_metric)
            return v if isinstance(v, (int, float)) else -1.0
        best_key = max(spec_by_key, key=_score)
        selection_basis = "single_validation_split"
        print(f"[run] selected: {spec_by_key[best_key].display_name} "
              f"(val {selection_metric}={_score(best_key):.4f})")

    best_spec = spec_by_key[best_key]

    # --- refit best on TRAIN+VAL, calibrate (X_trainval/y_trainval built above) ---
    refit = build_pipeline(best_spec, cfg)
    refit.fit(X_trainval, y_trainval)
    uncal_test_prob = _prob(refit, X_test)

    calibrated = calibrate_pipeline(refit, X_trainval, y_trainval, cfg)
    cal_test_prob = _prob(calibrated, X_test)

    # --- select the F1-optimal operating point on VALIDATION (never TEST) ---
    # A decision threshold is a tunable operating point, so it must NOT be chosen
    # by looking at the held-out TEST set. We select it on the calibrated model's
    # VALIDATION predictions (the same probability scale the API serves) and then
    # only REPORT how that fixed threshold generalizes to TEST. Caveat: VAL is
    # in-sample for the calibrated model (it was refit on TRAIN+VAL), so the VAL
    # figures are mildly optimistic — but a single scalar threshold has
    # negligible capacity to overfit and TEST stays completely untouched by
    # selection. The threshold is reported for analysis, not auto-applied at
    # serving (risk bands use the configured probability cut points).
    cal_val_prob = _prob(calibrated, X_val)
    sel = best_f1_threshold(split.y_val, cal_val_prob)
    sel_thr = sel["threshold"]
    test_at_sel = compute_metrics(split.y_test, cal_test_prob, threshold=sel_thr)
    selected_threshold = {
        "value": sel_thr,
        "selection_basis": "validation",
        "val_f1": sel["f1"],
        "val_precision": sel["precision"],
        "val_recall": sel["recall"],
        "test_f1": test_at_sel["f1"],
        "test_precision": test_at_sel["precision"],
        "test_recall": test_at_sel["recall"],
        "note": (
            "F1-optimal operating point chosen on the VALIDATION set (never the "
            "test set, to avoid leakage), shown here with its test-set "
            "generalization. Not auto-applied at serving — risk bands use the "
            "configured probability cut points."
        ),
    }

    # --- final evaluation on untouched TEST ---
    test_metrics = compute_metrics(split.y_test, cal_test_prob)
    thr = threshold_analysis(split.y_test, cal_test_prob)
    calib_before = calibration_curve_points(split.y_test, uncal_test_prob)
    calib_after = calibration_curve_points(split.y_test, cal_test_prob)
    from sklearn.metrics import brier_score_loss
    brier_before = float(brier_score_loss(split.y_test.astype(int), uncal_test_prob))
    brier_after = float(brier_score_loss(split.y_test.astype(int), cal_test_prob))

    print(f"[run] TEST roc_auc={test_metrics['roc_auc']} "
          f"pr_auc={test_metrics['pr_auc']} brier={test_metrics['brier_score']:.4f}")

    # --- global explainability (on calibrated model) ---
    # Gini/coef importance (fold-averaged), plus permutation importance on the
    # held-out TEST set — the latter is model-agnostic, computed out-of-sample,
    # and aggregated per ORIGINAL input feature (not fragmented across one-hot
    # dummies), so it is the more defensible headline importance.
    importance = global_importance(calibrated, cfg)
    perm_importance = permutation_global_importance(
        calibrated, X_test, split.y_test, cfg)

    # --- fairness slicing diagnostic on TEST (CODE_GENDER, age bands) ---
    fairness = fairness_report(X_test, split.y_test, cal_test_prob, cfg,
                               threshold=sel_thr)

    # --- persist ---
    save_model(calibrated, cfg)
    save_feature_schema(cfg)

    metadata = {
        "model_name": best_spec.display_name,
        "model_key": best_key,
        "model_version": cfg["api"]["version"],
        "trained_at": _now(),
        "dataset_source": source,
        "synthetic": is_synth,
        "synthetic_warning": manifest.get("synthetic_warning"),
        # relative path so it is portable and does not leak an absolute
        # filesystem path through /api/model/info. When the configured manifest
        # lives outside the repo root (e.g. an absolute path set via
        # LOAN_RISK_CONFIG, which config.py explicitly supports), relative_to
        # would raise — fall back to the bare filename, which is equally
        # portable and leak-safe.
        "manifest_reference": _portable_reference(cfg.path("manifest"), cfg.root),
        "feature_schema_version": "1.0",
        "selection_metric": selection_metric,
        "selection_basis": selection_basis,
        "hyperparameters_tuned": bool(tuning_results),
        "calibration": {
            "method": cfg["calibration"]["method"],
            "cv": cfg["calibration"]["cv"],
            "brier_before": brier_before,
            "brier_after": brier_after,
            "improved": brier_after < brier_before,
            "status": ("calibrated (isotonic/sigmoid via CV); probabilities "
                       "adjusted toward observed frequencies"),
        },
        "explainability_backends": {
            "shap_installed": shap_available(),
            "xgboost_installed": xgboost_available(),
        },
        "risk_score_definition": "risk_score = round(default_probability * 100)",
        "responsible_use_note": (
            "Analytical estimate of default probability. NOT an autonomous "
            "lending decision. Requires human oversight."
        ),
    }
    save_json(metadata, METADATA_FILE, cfg)

    metrics_blob = {
        "generated_at": _now(),
        "dataset_source": source,
        "synthetic": is_synth,
        "synthetic_warning": manifest.get("synthetic_warning"),
        "selection_metric": selection_metric,
        "selection_basis": selection_basis,
        "selected_model": best_spec.display_name,
        "split_diagnostics": split.diagnostics,
        "validation_summary": report.summary(),
        "per_model_validation": per_model,
        "per_model_cross_validation": per_model_cv,
        "hyperparameter_tuning": tuning_results,
        "final_test_metrics": test_metrics,
        "threshold_analysis": thr,
        "selected_threshold": selected_threshold,
        "calibration": {
            "brier_before": brier_before, "brier_after": brier_after,
            "curve_before": calib_before, "curve_after": calib_after,
        },
        "global_importance": importance,
        "permutation_importance": perm_importance,
        "fairness": fairness,
        "skipped_models": skipped,
    }
    metrics_path = save_json(metrics_blob, METRICS_FILE, cfg)
    print(f"[run] metrics -> {metrics_path}")
    print(f"[run] artifacts -> {cfg.path('artifacts')}")
    if is_synth:
        print("[run] NOTE: SYNTHETIC data — all metrics above are synthetic-only.")
    return metrics_blob


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the loan default-risk pipeline.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
