// Dashboard landing page. Gives an at-a-glance summary of the model and dataset
// and routes to every section. Framed as an analytical tool, not a lending
// decision (06_FRONTEND_REQUIREMENTS.md). health and model info load via useAsync
// with its standard loading/error/empty handling — there is no special pre-training
// path, so when those artifacts are absent this page shows the same error state as
// the others. Only metrics is caught (falls back to null) so its stat cards degrade
// to placeholders instead of failing the whole view.
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { Alert, Badge, Card, StatCard } from "../components/ui";
import { ErrorState, LoadingState, useAsync } from "../components/states";
import { metricValue, pct } from "../lib/format";

const SECTIONS: { to: string; title: string; desc: string }[] = [
  {
    to: "/assessment",
    title: "Risk Assessment",
    desc: "Enter applicant details to estimate default probability.",
  },
  {
    to: "/performance",
    title: "Model Performance",
    desc: "Validation comparison, test metrics, calibration and thresholds.",
  },
  {
    to: "/explainability",
    title: "Explainability",
    desc: "Global drivers and per-applicant contribution breakdowns.",
  },
  {
    to: "/dataset",
    title: "Dataset Quality",
    desc: "Validation checks and the dataset manifest with known limits.",
  },
  {
    to: "/model",
    title: "Model Information",
    desc: "Selected model, version, training details and calibration.",
  },
  {
    to: "/responsible-use",
    title: "Responsible Use",
    desc: "Scope, limitations and appropriate use of these estimates.",
  },
];

export default function Overview() {
  const { data, loading, error, reload } = useAsync(
    () => Promise.all([api.health(), api.modelInfo(), api.metrics().catch(() => null)]),
    [],
  );

  if (loading) return <LoadingState label="Loading overview" />;
  if (error || !data) return <ErrorState error={error} onRetry={reload} />;

  const [health, model, metrics] = data;
  const test = metrics?.final_test_metrics;

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">Loan Default-Risk Dashboard</h1>
          <p className="mt-1 max-w-2xl text-sm text-[color:var(--color-muted)]">
            An analytical tool for estimating and understanding loan default risk. Outputs are
            probability estimates for review, not automated lending decisions.
          </p>
        </div>
        <Badge
          className={
            health.model_available
              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
              : "border-amber-200 bg-amber-50 text-amber-700"
          }
        >
          {health.model_available ? "Model ready" : "Model not ready"} · API v{health.api_version}
        </Badge>
      </div>

      {model.synthetic && (
        <Alert tone="warning" title="Synthetic data">
          {model.synthetic_warning ??
            "This model was trained on synthetic data. Results are for demonstration only and must not inform real lending."}
        </Alert>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Selected model" value={model.model_name} sub="Best model by selection metric" />
        <StatCard label="Model version" value={model.model_version} sub={`Schema ${model.feature_schema_version}`} />
        <StatCard
          label="Test ROC-AUC"
          value={test ? metricValue(test.roc_auc) : "—"}
          sub={test ? "Held-out test set" : "Train the pipeline to populate"}
        />
        <StatCard
          label="Positive rate"
          value={test ? pct(test.positive_rate) : "—"}
          sub={test ? "Default rate in test set" : "Not available yet"}
        />
      </div>

      <Card title="Explore" subtitle="Jump into any part of the analysis.">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {SECTIONS.map((s) => (
            <Link
              key={s.to}
              to={s.to}
              className="group rounded-[var(--radius-card)] border border-[color:var(--color-line)] bg-[color:var(--color-surface)] p-4 transition-colors hover:border-[color:var(--color-accent)] focus:outline-none focus-visible:border-[color:var(--color-accent)]"
            >
              <h3 className="text-sm font-semibold text-[color:var(--color-ink)] group-hover:text-[color:var(--color-accent)]">
                {s.title}
              </h3>
              <p className="mt-1 text-sm text-[color:var(--color-muted)]">{s.desc}</p>
            </Link>
          ))}
        </div>
      </Card>
    </div>
  );
}
