// Dataset Quality page. Renders the machine-readable validation report + the
// dataset manifest from /api/dataset/quality. Never fabricates — shows exactly
// the checks and provenance the backend recorded. Status is shown as text AND
// color (never color-only).
import { api } from "../lib/api";
import { Alert, Badge, Card, StatCard } from "../components/ui";
import { EmptyState, ErrorState, LoadingState, useAsync } from "../components/states";

function statusClass(status: string): string {
  switch (status) {
    case "pass":
      return "border-emerald-300 bg-emerald-50 text-emerald-800";
    case "warn":
      return "border-amber-300 bg-amber-50 text-amber-800";
    case "fail":
      return "border-red-300 bg-red-50 text-red-800";
    default:
      return "border-slate-300 bg-slate-50 text-slate-700";
  }
}

function asString(v: unknown, fallback = "—"): string {
  if (v == null) return fallback;
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return fallback;
}

export default function DatasetQuality() {
  const { data, loading, error, reload } = useAsync(() => api.datasetQuality(), []);

  if (loading) return <LoadingState label="Loading dataset quality" />;
  if (error || !data) return <ErrorState error={error} onRetry={reload} />;

  const report = data.validation_report;
  const manifest = data.manifest;

  if (!report && !manifest) {
    return (
      <EmptyState
        title="No dataset report available"
        message="Train the pipeline (python -m loan_risk.pipeline.run) to generate the validation report and manifest."
      />
    );
  }

  const isSynthetic =
    manifest?.source_kind === "synthetic" || manifest?.synthetic_warning != null;
  const knownLimitations = Array.isArray(manifest?.known_limitations)
    ? (manifest!.known_limitations as unknown[])
    : [];

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Dataset Quality</h1>
        <p className="mt-1 text-sm text-[color:var(--color-muted)]">
          Validation checks and provenance for the dataset the current model was trained on.
        </p>
      </div>

      {isSynthetic && (
        <Alert tone="warning" title="Synthetic dataset">
          {asString(
            manifest?.synthetic_warning,
            "This is a synthetic, Home Credit-shaped fixture for pipeline stress testing. Distributions approximate but do not equal the real population.",
          )}
        </Alert>
      )}

      {report && (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard label="Rows" value={report.n_rows.toLocaleString()} />
            <StatCard label="Columns" value={report.n_cols} />
            <StatCard label="Checks passed" value={report.summary.pass ?? 0} sub="pass" />
            <StatCard
              label="Warnings / failures"
              value={`${report.summary.warn ?? 0} / ${report.summary.fail ?? 0}`}
              sub={report.has_errors ? "has failures" : "no failures"}
            />
          </div>

          <Card title="Validation checks" subtitle={`Source file: ${report.source_file}`}>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-[color:var(--color-line)] text-xs uppercase tracking-wide text-[color:var(--color-muted)]">
                    <th scope="col" className="py-2 pr-4">Check</th>
                    <th scope="col" className="py-2 pr-4">Status</th>
                    <th scope="col" className="py-2">Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {report.checks.map((c) => (
                    <tr key={c.name} className="border-b border-[color:var(--color-line)] last:border-0">
                      <td className="py-2 pr-4 font-medium">{c.name}</td>
                      <td className="py-2 pr-4">
                        <Badge className={statusClass(c.status)}>{c.status}</Badge>
                      </td>
                      <td className="py-2 text-[color:var(--color-muted)]">{c.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {manifest && (
        <Card title="Dataset manifest" subtitle="Provenance recorded at training time">
          <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
            <div>
              <dt className="text-xs uppercase tracking-wide text-[color:var(--color-muted)]">Dataset</dt>
              <dd className="text-sm font-medium">{asString(manifest.dataset_name)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-[color:var(--color-muted)]">Source kind</dt>
              <dd className="text-sm font-medium">{asString(manifest.source_kind)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-[color:var(--color-muted)]">Target column</dt>
              <dd className="text-sm font-medium">{asString(manifest.target_column)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-[color:var(--color-muted)]">Positive rate</dt>
              <dd className="text-sm font-medium">{asString(manifest.positive_rate)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-[color:var(--color-muted)]">Row count</dt>
              <dd className="text-sm font-medium">
                {typeof manifest.row_count === "number"
                  ? manifest.row_count.toLocaleString()
                  : asString(manifest.row_count)}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-[color:var(--color-muted)]">Column count</dt>
              <dd className="text-sm font-medium">{asString(manifest.column_count)}</dd>
            </div>
          </dl>

          <div className="mt-4">
            <div className="text-xs uppercase tracking-wide text-[color:var(--color-muted)]">Target meaning</div>
            <p className="mt-1 text-sm">{asString(manifest.target_meaning)}</p>
          </div>

          {knownLimitations.length > 0 && (
            <div className="mt-4">
              <div className="text-xs uppercase tracking-wide text-[color:var(--color-muted)]">Known limitations</div>
              <ul className="mt-1 list-inside list-disc space-y-1 text-sm text-[color:var(--color-ink)]">
                {knownLimitations.map((l, i) => (
                  <li key={i}>{asString(l)}</li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
