"""End-to-end training orchestrator (02_IMPLEMENTATION_PLAN.md phases 2-6).

Flow:
  1. Ensure a dataset exists (auto-generate synthetic fixture if configured).
  2. Load raw -> validate -> write validation report + manifest.
  3. Standardize -> reproducible stratified split (test held out).
  4. Train each enabled model on TRAIN, evaluate on VAL.
  5. Select the best model by configured metric (NOT accuracy alone).
  6. Fit calibrated candidates on TRAIN; reserve VAL for operating thresholds.
  7. Select the F1 operating point on VALIDATION, then evaluate on the untouched
     TEST set (metrics + threshold generalization + calibration curve).
  8. Global explainability.
  9. Persist model + metadata + feature_schema + metrics.

Every number written is computed from real predictions. Synthetic runs are
flagged synthetic throughout.
"""
from __future__ import annotations

import argparse
import platform
from importlib.metadata import version
import joblib
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
    file_sha256,
)
from .calibrate import calibrate_pipeline
from .evaluate import (
    best_f1_threshold,
    bootstrap_intervals,
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

    # --- optional hyperparameter tuning (RandomizedSearchCV on a subsample) ---
    # Tune BEFORE selection so the model comparison is between tuned models.
    tuning_results: dict[str, Any] = {}
    tuned_specs: list = []
    for spec in specs:
        res = tune_model(spec, X_train, split.y_train, cfg)
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
        # Stratified k-fold CV on TRAIN, select by MEAN score (std reported).
        print(f"[run] cross-validated selection ({cv_folds}-fold) ...")
        best_key, per_model_cv = cross_val_select(specs, X_train, split.y_train, cfg)
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

    # Fit and calibrate on TRAIN only. Validation stays independent of all
    # model fitting, tuning, and calibration; do not refit after threshold selection.
    candidates = {}
    candidate_details = {}
    for spec in specs:
        print(f"[run] calibrating {spec.display_name} on training folds ...")
        pipe = build_pipeline(spec, cfg)
        calibrated_candidate = calibrate_pipeline(pipe, X_train, split.y_train, cfg)
        val_prob = _prob(calibrated_candidate, X_val)
        point = best_f1_threshold(split.y_val, val_prob)
        candidates[spec.key] = calibrated_candidate
        candidate_details[spec.key] = {
            "display_name": spec.display_name,
            "decision_threshold": point["threshold"],
            "threshold_basis": "held_out_validation_f1",
            "calibration_method": cfg["calibration"]["method"],
            "validation_metrics": compute_metrics(split.y_val, val_prob),
        }
    calibrated = candidates[best_key]
    refit = build_pipeline(best_spec, cfg)
    refit.fit(X_train, split.y_train)
    uncal_test_prob = _prob(refit, X_test)
    cal_test_prob = _prob(calibrated, X_test)
    sel_thr = candidate_details[best_key]["decision_threshold"]
    val_at_sel = compute_metrics(split.y_val, _prob(calibrated, X_val), threshold=sel_thr)
    test_at_sel = compute_metrics(split.y_test, cal_test_prob, threshold=sel_thr)
    selected_threshold = {
        "value": sel_thr,
        "selection_basis": "held_out_validation",
        "val_f1": val_at_sel["f1"],
        "val_precision": val_at_sel["precision"],
        "val_recall": val_at_sel["recall"],
        "test_f1": test_at_sel["f1"],
        "test_precision": test_at_sel["precision"],
        "test_recall": test_at_sel["recall"],
        "note": ("F1 threshold selected on validation rows excluded from fitting, "
                 "tuning and calibration. Applied at serving for this model. "
                 "A research operating point, not a lending policy."),
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
    # Metadata is written last as the bundle completion marker. Its checksums
    # let serving fail closed while a run is incomplete or mixed.
    model_dir = cfg.path("artifacts") / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for key, estimator in candidates.items():
        joblib.dump(estimator, model_dir / f"{key}.joblib", compress=3)
    save_model(calibrated, cfg)
    save_feature_schema(cfg)
    save_json(manifest, "dataset_manifest.json", cfg)
    save_json(report.to_dict(), "validation_report.json", cfg)

    metadata = {
        "model_name": best_spec.display_name,
        "model_key": best_key,
        "model_version": _now().replace(":", "").replace("-", "") + "-" + file_sha256(raw_path)[:8],
        "api_version": cfg["api"]["version"],
        "candidate_details": candidate_details,
        "decision_threshold": sel_thr,
        "risk_score_config": cfg["risk_score"],
        "dataset_sha256": file_sha256(raw_path),
        "training_config": cfg.raw,
        "runtime_versions": {name: version(name) for name in
                             ("scikit-learn", "numpy", "pandas", "joblib")},
        "python_version": platform.python_version(),
        "training_rows": len(X_train),
        "input_profile": {col: {"min": float(X_train[col].min()),
                                "max": float(X_train[col].max())}
                          for col in X_train.select_dtypes(include="number")
                          if X_train[col].notna().any()},
        "evaluation_protocol": "train_only_fit_validation_threshold_test_report",
        "selection_note": "CV scores after hyperparameter search are selection diagnostics, not unbiased generalization estimates.",
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
        "test_metrics_at_selected_threshold": test_at_sel,
        "baseline_test_metrics": compute_metrics(split.y_test, np.full(len(split.y_test), pos)),
        "test_confidence_intervals": bootstrap_intervals(
            split.y_test, cal_test_prob, threshold=sel_thr, seed=cfg.seed,
            repeats=int(cfg.get("evaluation", {}).get("bootstrap_repeats", 200))),
        "evaluation_protocol": metadata["evaluation_protocol"],
        "candidate_details": candidate_details,
        "model_version": metadata["model_version"],
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
    bundle_files = ["model.joblib", "metrics.json", "feature_schema.json",
                    "dataset_manifest.json", "validation_report.json"] + [
        f"models/{key}.joblib" for key in candidates]
    metadata["file_hashes"] = {name: file_sha256(cfg.path("artifacts") / name)
                               for name in bundle_files}
    save_json(metadata, METADATA_FILE, cfg)
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
