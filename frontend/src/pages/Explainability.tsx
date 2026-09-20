// Explainability. TWO parts, both grounded strictly in what the API/store
// returns (06_FRONTEND_REQUIREMENTS.md):
//   1. GLOBAL feature importance from /api/metrics (model-level, ranked bars).
//   2. LOCAL per-applicant explanation from the last prediction in the store.
// Explanations describe ASSOCIATION with predicted risk, never causation, and
// must not be used as adverse-action reasons.
import { Link } from "react-router-dom";
import { usePrediction } from "../store/prediction";
import { api } from "../lib/api";
import type { Metrics, PredictResponse } from "../lib/types";
import { Alert, Badge, Card } from "../components/ui";
import { ErrorState, LoadingState, useAsync } from "../components/states";
import { num, pct, riskBandClass } from "../lib/format";

// "+0.1234" for positive, "-0.1234" for negative (toFixed already signs it),
// "0.0000" for zero.
function signed(x: number, digits = 4): string {
  const s = num(x, digits);
  return x > 0 ? `+${s}` : s;
}

function directionLabel(direction: string): string {
  if (direction === "increases_risk") return "Increases risk";
  if (direction === "decreases_risk") return "Decreases risk";
  return direction;
}

function GlobalImportanceSection({ metrics }: { metrics: Metrics }) {
  const gi = metrics.global_importance;
  const features = gi?.features ?? [];
  const maxImportance = features.reduce((m, f) => Math.max(m, f.importance), 0);

  return (
    <Card title="Global feature importance" subtitle="Model-level drivers across the evaluation data">
      {metrics.synthetic && (
        <div className="mb-4">
          <Alert tone="warning" title="Synthetic data">
            {metrics.synthetic_warning ??
              "These importances were computed on synthetic data and do not reflect a real lending population."}
          </Alert>
        </div>
      )}

      <dl className="mb-4 grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-[color:var(--color-muted)]">Method</dt>
          <dd className="mt-0.5 font-medium text-[color:var(--color-ink)]">
            <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs">{gi.method}</code>
          </dd>
        </div>
      </dl>

      {(gi.interpretation || gi.note) && (
        <p className="mb-4 text-sm text-[color:var(--color-muted)]">
          {gi.interpretation}
          {gi.interpretation && gi.note ? " " : ""}
          {gi.note}
        </p>
      )}

      {features.length === 0 ? (
        <p className="text-sm text-[color:var(--color-muted)]">No importance values are available.</p>
      ) : (
        <ul className="space-y-2">
          {features.map((f) => {
            const width = maxImportance > 0 ? (f.importance / maxImportance) * 100 : 0;
            return (
              <li key={f.feature} className="grid grid-cols-[10rem_1fr_5rem] items-center gap-3">
                <span className="truncate text-sm font-medium text-[color:var(--color-ink)]" title={f.feature}>
                  {f.feature}
                </span>
                <div
                  className="h-3 overflow-hidden rounded-full bg-slate-100"
                  role="img"
                  aria-label={`${f.feature} importance ${num(f.importance, 4)}`}
                >
                  <div
                    className="h-full rounded-full bg-[color:var(--color-accent)]"
                    style={{ width: `${Math.min(100, Math.max(2, width))}%` }}
                  />
                </div>
                <span className="text-right text-sm tabular-nums text-[color:var(--color-ink)]">
                  {num(f.importance, 4)}
                </span>
              </li>
            );
          })}
        </ul>
      )}

      <p className="mt-4 text-xs text-[color:var(--color-muted)]">
        These values show <span className="font-semibold">association</span> with predicted risk, NOT causation.
        A high-importance feature is one the model relies on, not a proven cause of default.
      </p>
    </Card>
  );
}

function LocalExplanationSection({ result }: { result: PredictResponse | null }) {
  if (!result) {
    return (
      <Card title="Applicant-level explanation">
        <div className="py-8 text-center">
          <h3 className="text-sm font-semibold text-[color:var(--color-ink)]">No applicant assessed yet</h3>
          <p className="mx-auto mt-2 max-w-md text-sm text-[color:var(--color-muted)]">
            Submit an applicant on the Risk Assessment page to see which features drove that specific estimate.
          </p>
          <p className="mt-4">
            <Link
              className="font-medium text-[color:var(--color-accent)] underline"
              to="/assessment"
            >
              Go to Risk Assessment
            </Link>
          </p>
        </div>
      </Card>
    );
  }

  const explanation = result.explanation;
  const hasContributions = !!explanation && explanation.contributions.length > 0;
  const maxAbs = hasContributions
    ? explanation!.contributions.reduce((m, c) => Math.max(m, Math.abs(c.contribution)), 0)
    : 0;

  return (
    <Card
      title="Applicant-level explanation"
      subtitle="Drivers of the most recent estimate on this device"
      action={<Badge className={riskBandClass(result.risk_band)}>{result.risk_band} risk</Badge>}
    >
      <p className="mb-4 text-sm text-[color:var(--color-muted)]">
        Estimated default probability{" "}
        <span className="font-semibold text-[color:var(--color-ink)]">{pct(result.default_probability, 1)}</span>{" "}
        · model version {result.model_version}.
      </p>

      {hasContributions ? (
        <>
          <dl className="mb-4 grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-[color:var(--color-muted)]">Method</dt>
              <dd className="mt-0.5 font-medium text-[color:var(--color-ink)]">
                <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs">{explanation!.method}</code>
              </dd>
            </div>
          </dl>

          {(explanation!.interpretation || explanation!.note) && (
            <p className="mb-4 text-sm text-[color:var(--color-muted)]">
              {explanation!.interpretation}
              {explanation!.interpretation && explanation!.note ? " " : ""}
              {explanation!.note}
            </p>
          )}

          <div className="overflow-hidden rounded-lg border border-[color:var(--color-line)]">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-[color:var(--color-muted)]">
                <tr>
                  <th scope="col" className="px-3 py-2 font-medium">Feature</th>
                  <th scope="col" className="px-3 py-2 font-medium">Contribution</th>
                  <th scope="col" className="px-3 py-2 font-medium">Direction</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[color:var(--color-line)]">
                {explanation!.contributions.map((c) => {
                  const increases = c.direction === "increases_risk";
                  const textClass = increases ? "text-red-600" : "text-emerald-700";
                  const barClass = increases ? "bg-red-500" : "bg-emerald-500";
                  const width = maxAbs > 0 ? (Math.abs(c.contribution) / maxAbs) * 100 : 0;
                  return (
                    <tr key={c.feature}>
                      <td className="px-3 py-2 font-medium text-[color:var(--color-ink)]">{c.feature}</td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <span className={`w-16 tabular-nums font-semibold ${textClass}`}>
                            {signed(c.contribution)}
                          </span>
                          <div className="h-2 w-24 overflow-hidden rounded-full bg-slate-100" aria-hidden="true">
                            <div
                              className={`h-full rounded-full ${barClass}`}
                              style={{ width: `${Math.min(100, Math.max(2, width))}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className={`px-3 py-2 font-medium ${textClass}`}>{directionLabel(c.direction)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      ) : !result.explanation_available ? (
        <Alert tone="info" title="No per-applicant explanation">
          This model does not produce per-applicant feature contributions. See the global feature importance above
          for model-level drivers.
        </Alert>
      ) : (
        <Alert tone="info" title="No contributions returned">
          An explanation was expected for this applicant but no feature contributions were returned.
        </Alert>
      )}

      <div className="mt-4">
        <Alert tone="warning" title="Approximate — not adverse-action reasons">
          Feature contributions are approximate attributions for a single estimate. They must not be used as
          adverse-action reasons or as a substitute for a documented, human lending decision.
        </Alert>
      </div>
    </Card>
  );
}

export default function Explainability() {
  const global = useAsync(() => api.metrics(), []);
  const { result } = usePrediction();

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Explainability</h1>
        <p className="mt-1 text-sm text-[color:var(--color-muted)]">
          How the model relates features to predicted default risk — at the model level and for your most recent
          applicant.
        </p>
      </div>

      {global.loading ? (
        <LoadingState label="Loading feature importance" />
      ) : global.error || !global.data ? (
        <ErrorState error={global.error} onRetry={global.reload} />
      ) : (
        <GlobalImportanceSection metrics={global.data} />
      )}

      <LocalExplanationSection result={result} />
    </div>
  );
}
