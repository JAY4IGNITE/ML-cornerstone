// Prediction Result. Clearly SEPARATES default probability, risk score, risk
// band, model version, explanation availability, limitations and the
// not-a-lending-decision disclaimer (06_FRONTEND_REQUIREMENTS.md).
import { Link } from "react-router-dom";
import { usePrediction } from "../store/prediction";
import { Alert, Badge, Button, Card, StatCard } from "../components/ui";
import { EmptyState } from "../components/states";
import { pct, riskBandClass } from "../lib/format";

export default function Result() {
  const { result } = usePrediction();

  if (!result) {
    return (
      <EmptyState
        title="No assessment yet"
        message="Submit an applicant on the Risk Assessment page to see a result here."
      />
    );
  }

  const prob = result.default_probability;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Prediction Result</h1>
        <Badge className={riskBandClass(result.risk_band)}>{result.risk_band} risk</Badge>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Default probability" value={pct(prob, 1)} sub="Estimated P(default)" />
        <StatCard label="Risk score" value={`${result.risk_score}/100`} sub="round(probability × 100)" />
        <StatCard label="Risk band" value={result.risk_band} sub="Configured probability band" />
      </div>

      {/* Probability bar — not color-only; value + band label shown in text too. */}
      <Card title="Default probability">
        <div className="flex items-center gap-3">
          <div
            className="h-3 flex-1 overflow-hidden rounded-full bg-slate-100"
            role="img"
            aria-label={`Default probability ${pct(prob, 1)}`}
          >
            <div
              className="h-full rounded-full bg-[color:var(--color-accent)]"
              style={{ width: `${Math.min(100, Math.max(2, prob * 100))}%` }}
            />
          </div>
          <span className="w-16 text-right text-sm font-semibold tabular-nums">{pct(prob, 1)}</span>
        </div>
        <p className="mt-2 text-xs text-[color:var(--color-muted)]">
          Model version {result.model_version}.{" "}
          {result.explanation_available ? (
            <>
              A per-applicant explanation is available —{" "}
              <Link className="font-medium text-[color:var(--color-accent)] underline" to="/explainability">
                view explainability
              </Link>
              .
            </>
          ) : (
            "No per-applicant explanation is available for this model."
          )}
        </p>
      </Card>

      <Card title="Limitations">
        <ul className="list-inside list-disc space-y-1 text-sm text-[color:var(--color-ink)]">
          {result.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </Card>

      <Alert tone="warning" title="Not a lending decision">
        {result.disclaimer}
      </Alert>

      <div>
        <Link to="/assessment">
          <Button variant="secondary">← New assessment</Button>
        </Link>
      </div>
    </div>
  );
}
