// Responsible Use. Honest, precise statement of what this system is and is not.
// Content is grounded in the project's principles (01_PROJECT_CONTEXT.md /
// 09_EXECUTION_RULES.md); the live synthetic flag and responsible-use note are
// pulled from /api/model/info when available (best-effort, never blocks the page).
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { Alert, Card } from "../components/ui";
import { useAsync } from "../components/states";

export default function ResponsibleUse() {
  // Best-effort: the page is fully useful even if the model is not trained yet.
  const { data: model } = useAsync(() => api.modelInfo().catch(() => null), []);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Responsible Use</h1>
        <p className="mt-1 text-sm text-[color:var(--color-muted)]">
          How this system may and may not be used, and the limitations you must keep in mind.
        </p>
      </div>

      <Alert tone="danger" title="Not an autonomous lending decision-maker">
        This system produces analytical estimates and is NOT an autonomous lending decision-maker. A qualified human
        must make any lending decision.
      </Alert>

      {model?.synthetic && (
        <Alert tone="warning" title="Synthetic data in use">
          {model.synthetic_warning ??
            "The active model is trained on synthetic data. Outputs are for demonstration only and do not reflect a real lending population."}
        </Alert>
      )}

      {model?.responsible_use_note && (
        <Card title="Model's responsible-use note">
          <p className="text-sm text-[color:var(--color-ink)]">{model.responsible_use_note}</p>
        </Card>
      )}

      <Card title="Human oversight">
        <p className="text-sm text-[color:var(--color-ink)]">
          Every output is a decision-support estimate for a qualified human reviewer. The system does not approve,
          deny, price, or otherwise act on a loan. A human is accountable for the final decision and for weighing
          context the model never sees — the applicant's full circumstances, policy, and applicable regulation.
        </p>
      </Card>

      <Card title="Data limitations">
        <ul className="list-inside list-disc space-y-1 text-sm text-[color:var(--color-ink)]">
          <li>
            The model is currently trained on {model?.synthetic === false ? "the configured dataset" : "synthetic data"},
            which limits how far its outputs generalise to a real lending population.
          </li>
          <li>Inputs are an application-level subset of features, not a complete financial profile.</li>
          <li>There is no full credit-bureau history, no repayment timeline, and no account-level behaviour.</li>
          <li>Missing optional inputs are imputed, which adds uncertainty that is not shown per applicant.</li>
        </ul>
        <p className="mt-3 text-sm text-[color:var(--color-muted)]">
          See <Link className="font-medium text-[color:var(--color-accent)] underline" to="/model">Model Information</Link>{" "}
          for provenance and calibration, and{" "}
          <Link className="font-medium text-[color:var(--color-accent)] underline" to="/performance">Model Performance</Link>{" "}
          for evaluation metrics.
        </p>
      </Card>

      <Card title="Intended vs out-of-scope use">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <h3 className="text-sm font-semibold text-[color:var(--color-ink)]">Intended</h3>
            <ul className="mt-2 list-inside list-disc space-y-1 text-sm text-[color:var(--color-ink)]">
              <li>Decision support and triage for a human reviewer.</li>
              <li>Exploring how application features relate to estimated risk.</li>
              <li>Education and demonstration of a risk-scoring workflow.</li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-[color:var(--color-ink)]">Out of scope</h3>
            <ul className="mt-2 list-inside list-disc space-y-1 text-sm text-[color:var(--color-ink)]">
              <li>Automated approve/deny or pricing decisions.</li>
              <li>Generating adverse-action reasons or legal justifications.</li>
              <li>Any use presented as compliant, audited, or production-ready.</li>
            </ul>
          </div>
        </div>
      </Card>

      <Card title="Potential bias & fairness">
        <ul className="list-inside list-disc space-y-1 text-sm text-[color:var(--color-ink)]">
          <li className="font-medium">No fairness audit has been performed on this model.</li>
          <li>
            Protected-attribute proxies (for example age- and gender-related fields) are present in the inputs and can
            correlate with model outputs.
          </li>
          <li>
            Feature importance and contributions show association with predicted risk, not causation, and are not
            evidence of a fair or unbiased outcome.
          </li>
        </ul>
      </Card>

      <Card title="Privacy">
        <ul className="list-inside list-disc space-y-1 text-sm text-[color:var(--color-ink)]">
          <li>No raw personally identifiable information is logged by the application.</li>
          <li>Secrets and credentials are kept out of source control and configuration in code.</li>
          <li>Applicant inputs are used to produce an estimate and are not persisted as an identity record.</li>
        </ul>
      </Card>

      <Card title="Non-autonomous use statement">
        <p className="text-sm text-[color:var(--color-ink)]">
          This tool is explicitly non-autonomous. It cannot and must not be configured to make or execute lending
          decisions on its own. Estimates are inputs to human judgement, and responsibility for any decision remains
          with the qualified people and institutions using it.
        </p>
      </Card>
    </div>
  );
}
