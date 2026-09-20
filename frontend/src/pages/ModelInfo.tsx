// Model Information. Surfaces the full model identity, provenance, calibration
// and responsible-use note from /api/model/info (06_FRONTEND_REQUIREMENTS.md).
// Renders only what the API returns; a 503 (model not trained) becomes an
// ErrorState via useAsync.
import { api } from "../lib/api";
import type { ModelInfo as ModelInfoType } from "../lib/types";
import { Alert, Badge, Card, StatCard } from "../components/ui";
import { ErrorState, LoadingState, useAsync } from "../components/states";
import { num } from "../lib/format";

// null/undefined fields render as an em dash rather than blank or "null".
function DefRow({ label, value }: { label: string; value: React.ReactNode }) {
  const display = value === null || value === undefined || value === "" ? "—" : value;
  return (
    <div className="grid grid-cols-1 gap-1 py-2 sm:grid-cols-[14rem_1fr]">
      <dt className="text-sm font-medium text-[color:var(--color-muted)]">{label}</dt>
      <dd className="text-sm text-[color:var(--color-ink)]">{display}</dd>
    </div>
  );
}

function brier(x: number | null | undefined): string {
  return x === null || x === undefined ? "—" : num(x, 4);
}

function Content({ model }: { model: ModelInfoType }) {
  const cal = model.calibration;
  const calStatus = cal.status ?? (cal.improved == null ? "—" : cal.improved ? "Improved" : "Not improved");

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Model" value={model.model_name} sub={model.model_key ?? undefined} />
        <StatCard label="Version" value={model.model_version} sub={`Schema ${model.feature_schema_version}`} />
        <StatCard label="Calibration" value={calStatus} sub={cal.method ?? "Method not reported"} />
      </div>

      {model.synthetic && (
        <Alert tone="warning" title="Trained on synthetic data">
          {model.synthetic_warning ??
            "This model was trained on synthetic data and is not validated for real lending populations."}
        </Alert>
      )}

      <Card title="Identity & provenance">
        <dl className="divide-y divide-[color:var(--color-line)]">
          <DefRow label="Model name" value={model.model_name} />
          <DefRow label="Model version" value={model.model_version} />
          <DefRow label="Model key" value={model.model_key} />
          <DefRow label="Trained at" value={model.trained_at} />
          <DefRow label="Dataset source" value={model.dataset_source} />
          <DefRow
            label="Synthetic data"
            value={<Badge className={model.synthetic ? "border-amber-300 bg-amber-50 text-amber-800" : "border-emerald-300 bg-emerald-50 text-emerald-800"}>{model.synthetic ? "Yes" : "No"}</Badge>}
          />
          <DefRow label="Feature schema version" value={model.feature_schema_version} />
          <DefRow label="Selection metric" value={model.selection_metric} />
          <DefRow
            label="Manifest reference"
            value={model.manifest_reference ? <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs">{model.manifest_reference}</code> : null}
          />
        </dl>
      </Card>

      <Card title="Calibration" subtitle="Whether predicted probabilities match observed frequencies">
        <dl className="divide-y divide-[color:var(--color-line)]">
          <DefRow label="Method" value={cal.method} />
          <DefRow label="Brier score (before)" value={brier(cal.brier_before)} />
          <DefRow label="Brier score (after)" value={brier(cal.brier_after)} />
          <DefRow
            label="Improved"
            value={
              cal.improved == null ? (
                "—"
              ) : (
                <Badge className={cal.improved ? "border-emerald-300 bg-emerald-50 text-emerald-800" : "border-slate-200 bg-slate-50 text-slate-600"}>
                  {cal.improved ? "Yes" : "No"}
                </Badge>
              )
            }
          />
          <DefRow label="Status" value={cal.status} />
        </dl>
        <p className="mt-3 text-xs text-[color:var(--color-muted)]">
          A lower Brier score is better. Calibration adjusts raw scores so a stated probability is closer to the
          real-world frequency of default.
        </p>
      </Card>

      <Card title="Responsible use">
        <p className="text-sm text-[color:var(--color-ink)]">{model.responsible_use_note}</p>
      </Card>
    </div>
  );
}

export default function ModelInfo() {
  const { data, loading, error, reload } = useAsync(() => api.modelInfo(), []);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Model Information</h1>
        <p className="mt-1 text-sm text-[color:var(--color-muted)]">
          Identity, provenance and calibration for the deployed model.
        </p>
      </div>

      {loading ? (
        <LoadingState label="Loading model information" />
      ) : error || !data ? (
        <ErrorState error={error} onRetry={reload} />
      ) : (
        <Content model={data} />
      )}
    </div>
  );
}
