// Types mirroring the backend Pydantic schemas (backend/schemas.py). Keeping
// these in sync is what prevents API/UI drift; the /api/feature-schema endpoint
// is the runtime source of truth for form fields.

export interface HealthResponse {
  status: string;
  model_available: boolean;
  api_version: string;
  detail: string;
}

export interface CalibrationInfo {
  method?: string | null;
  brier_before?: number | null;
  brier_after?: number | null;
  improved?: boolean | null;
  status?: string | null;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  model_key?: string | null;
  trained_at?: string | null;
  dataset_source: string;
  synthetic: boolean;
  synthetic_warning?: string | null;
  manifest_reference?: string | null;
  feature_schema_version: string;
  calibration: CalibrationInfo;
  selection_metric?: string | null;
  responsible_use_note: string;
}

export interface NumericFeature {
  name: string;
  label: string;
  unit: string;
  meaning: string;
  min: number | null;
  max: number | null;
  example: number | string | null;
  missing_behavior: string;
}

export interface CategoricalFeature {
  name: string;
  label: string;
  meaning: string;
  categories: string[];
  example: number | string | null;
  missing_behavior: string;
}

export interface EngineeredFeature {
  name: string;
  formula: string;
  inputs: string[];
  unit: string;
  interpretation: string;
  leakage_assessment: string;
}

export interface FeatureSchema {
  numeric_features: NumericFeature[];
  categorical_features: CategoricalFeature[];
  engineered_features: EngineeredFeature[];
}

export interface Contribution {
  feature: string;
  contribution: number;
  direction: string;
}

export interface ExplanationBlock {
  method: string;
  interpretation?: string | null;
  note?: string | null;
  contributions: Contribution[];
}

export interface PredictResponse {
  prediction_id?: string | null;
  default_probability: number;
  risk_score: number;
  risk_band: string;
  model_version: string;
  explanation_available: boolean;
  explanation?: ExplanationBlock | null;
  limitations: string[];
  disclaimer: string;
}

export interface ApiError {
  error: string;
  detail: string;
  fields?: { field: string; message: string; type: string }[] | null;
}

// ---- metrics.json shape (subset we render) ----
export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface MetricSet {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | { error: string };
  pr_auc: number | { error: string };
  brier_score: number;
  confusion_matrix: ConfusionMatrix;
  n: number;
  positive_rate: number;
}

export interface ThresholdRow {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  flagged_rate: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface CalibrationPoint {
  bin_lower: number;
  bin_upper: number;
  mean_predicted: number;
  observed_frequency: number;
  count: number;
}

export interface GlobalImportance {
  method: string;
  interpretation?: string;
  note?: string;
  features: { feature: string; importance: number }[];
}

export interface Metrics {
  generated_at: string;
  dataset_source: string;
  synthetic: boolean;
  synthetic_warning?: string | null;
  selection_metric: string;
  selected_model: string;
  split_diagnostics: Record<string, unknown>;
  validation_summary: Record<string, number>;
  per_model_validation: Record<
    string,
    { display_name: string; val_metrics: MetricSet }
  >;
  final_test_metrics: MetricSet;
  threshold_analysis: ThresholdRow[];
  selected_threshold: {
    value: number;
    selection_basis: string;
    val_f1: number;
    val_precision: number;
    val_recall: number;
    test_f1: number;
    test_precision: number;
    test_recall: number;
    note: string;
  };
  calibration: {
    brier_before: number;
    brier_after: number;
    curve_before: CalibrationPoint[];
    curve_after: CalibrationPoint[];
  };
  global_importance: GlobalImportance;
  skipped_models: string[];
}

export interface DatasetQuality {
  validation_report: {
    source_file: string;
    n_rows: number;
    n_cols: number;
    summary: Record<string, number>;
    has_errors: boolean;
    checks: {
      name: string;
      status: string;
      detail: string;
      data: Record<string, unknown>;
    }[];
  } | null;
  manifest: Record<string, unknown> | null;
}
