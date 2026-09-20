// Model Performance. Presents validation model comparison, held-out test
// metrics, confusion matrix, threshold analysis and calibration. Values are
// rendered only from metrics.json — nothing is fabricated. ROC/PR-AUC may arrive
// as {error} (single-class split), so metricValue handles them.
import { api } from "../lib/api";
import { Alert, Badge, Card, StatCard } from "../components/ui";
import { ErrorState, LoadingState, useAsync } from "../components/states";
import { metricValue, num, pct } from "../lib/format";

function thClass(extra = "") {
  return `px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-[color:var(--color-muted)] ${extra}`;
}
function tdClass(extra = "") {
  return `px-3 py-2 text-sm text-[color:var(--color-ink)] tabular-nums ${extra}`;
}

export default function Performance() {
  const { data: metrics, loading, error, reload } = useAsync(() => api.metrics(), []);

  if (loading) return <LoadingState label="Loading performance" />;
  if (error || !metrics) return <ErrorState error={error} onRetry={reload} />;

  const test = metrics.final_test_metrics;
  const cm = test.confusion_matrix;
  const sel = metrics.selected_threshold;
  const validation = Object.entries(metrics.per_model_validation);
  const reliability = metrics.calibration.curve_after;

  // The selected threshold comes off the precision–recall curve, so it almost
  // never equals a 0.05-spaced sweep row exactly. Highlight the CLOSEST sweep
  // row instead of testing for equality (which would essentially never match).
  const sweep = metrics.threshold_analysis;
  const nearestThreshold = sweep.reduce(
    (best, row) =>
      Math.abs(row.threshold - sel.value) < Math.abs(best - sel.value) ? row.threshold : best,
    sweep.length ? sweep[0].threshold : sel.value,
  );

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Model Performance</h1>
        <p className="mt-1 text-sm text-[color:var(--color-muted)]">
          Selected model <span className="font-medium text-[color:var(--color-ink)]">{metrics.selected_model}</span>{" "}
          by <span className="font-medium text-[color:var(--color-ink)]">{metrics.selection_metric}</span>.
        </p>
      </div>

      {metrics.synthetic && (
        <Alert tone="warning" title="Synthetic data">
          {metrics.synthetic_warning ??
            "These metrics are computed on synthetic data and are illustrative only."}
        </Alert>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Test ROC-AUC" value={metricValue(test.roc_auc)} sub="Held-out test set" />
        <StatCard label="Test PR-AUC" value={metricValue(test.pr_auc)} sub="Precision–recall AUC" />
        <StatCard label="Brier score" value={num(test.brier_score, 4)} sub="Lower is better" />
        <StatCard label="F1" value={num(test.f1, 4)} sub={`At threshold ${num(test.threshold, 2)}`} />
      </div>

      <Card title="Validation comparison" subtitle="Per-model metrics on the validation split.">
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="border-b border-[color:var(--color-line)]">
                <th scope="col" className={thClass()}>Model</th>
                <th scope="col" className={thClass()}>Accuracy</th>
                <th scope="col" className={thClass()}>Precision</th>
                <th scope="col" className={thClass()}>Recall</th>
                <th scope="col" className={thClass()}>F1</th>
                <th scope="col" className={thClass()}>ROC-AUC</th>
                <th scope="col" className={thClass()}>PR-AUC</th>
                <th scope="col" className={thClass()}>Brier</th>
              </tr>
            </thead>
            <tbody>
              {validation.map(([key, entry]) => {
                const m = entry.val_metrics;
                const selected = key === metrics.selected_model || entry.display_name === metrics.selected_model;
                return (
                  <tr
                    key={key}
                    className={`border-b border-[color:var(--color-line)] ${selected ? "bg-blue-50" : ""}`}
                  >
                    <th scope="row" className="px-3 py-2 text-left text-sm font-medium text-[color:var(--color-ink)]">
                      {entry.display_name}
                      {selected && (
                        <Badge className="ml-2 border-blue-200 bg-blue-100 text-blue-700">Selected</Badge>
                      )}
                    </th>
                    <td className={tdClass()}>{num(m.accuracy, 4)}</td>
                    <td className={tdClass()}>{num(m.precision, 4)}</td>
                    <td className={tdClass()}>{num(m.recall, 4)}</td>
                    <td className={tdClass()}>{num(m.f1, 4)}</td>
                    <td className={tdClass()}>{metricValue(m.roc_auc)}</td>
                    <td className={tdClass()}>{metricValue(m.pr_auc)}</td>
                    <td className={tdClass()}>{num(m.brier_score, 4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Final test metrics" subtitle="Held-out test set for the selected model.">
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="border-b border-[color:var(--color-line)]">
                <th scope="col" className={thClass()}>Accuracy</th>
                <th scope="col" className={thClass()}>Precision</th>
                <th scope="col" className={thClass()}>Recall</th>
                <th scope="col" className={thClass()}>F1</th>
                <th scope="col" className={thClass()}>ROC-AUC</th>
                <th scope="col" className={thClass()}>PR-AUC</th>
                <th scope="col" className={thClass()}>Brier</th>
                <th scope="col" className={thClass()}>n</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className={tdClass()}>{num(test.accuracy, 4)}</td>
                <td className={tdClass()}>{num(test.precision, 4)}</td>
                <td className={tdClass()}>{num(test.recall, 4)}</td>
                <td className={tdClass()}>{num(test.f1, 4)}</td>
                <td className={tdClass()}>{metricValue(test.roc_auc)}</td>
                <td className={tdClass()}>{metricValue(test.pr_auc)}</td>
                <td className={tdClass()}>{num(test.brier_score, 4)}</td>
                <td className={tdClass()}>{test.n}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Confusion matrix" subtitle={`Test set at threshold ${num(test.threshold, 2)}.`}>
        <div className="overflow-x-auto">
          <table className="border-collapse">
            <thead>
              <tr>
                <td className="px-3 py-2" />
                <th scope="col" colSpan={2} className={thClass("text-center")}>
                  Predicted
                </th>
              </tr>
              <tr className="border-b border-[color:var(--color-line)]">
                <td className="px-3 py-2" />
                <th scope="col" className={thClass("text-center")}>0 (No default)</th>
                <th scope="col" className={thClass("text-center")}>1 (Default)</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-[color:var(--color-line)]">
                <th scope="row" className={thClass()}>Actual 0</th>
                <td className={tdClass("text-center")}>
                  {cm.tn} <span className="text-xs text-[color:var(--color-muted)]">TN</span>
                </td>
                <td className={tdClass("text-center")}>
                  {cm.fp} <span className="text-xs text-[color:var(--color-muted)]">FP</span>
                </td>
              </tr>
              <tr>
                <th scope="row" className={thClass()}>Actual 1</th>
                <td className={tdClass("text-center")}>
                  {cm.fn} <span className="text-xs text-[color:var(--color-muted)]">FN</span>
                </td>
                <td className={tdClass("text-center")}>
                  {cm.tp} <span className="text-xs text-[color:var(--color-muted)]">TP</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      <Card
        title="Threshold analysis"
        subtitle={`F1-optimal threshold ${num(sel.value, 2)}, selected on validation (val F1 ${num(sel.val_f1, 3)}) and reported on test (test F1 ${num(sel.test_f1, 3)}, precision ${num(sel.test_precision, 3)}, recall ${num(sel.test_recall, 3)}). Not auto-applied at serving.`}
      >
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="border-b border-[color:var(--color-line)]">
                <th scope="col" className={thClass()}>Threshold</th>
                <th scope="col" className={thClass()}>Precision</th>
                <th scope="col" className={thClass()}>Recall</th>
                <th scope="col" className={thClass()}>F1</th>
                <th scope="col" className={thClass()}>Flagged rate</th>
              </tr>
            </thead>
            <tbody>
              {sweep.map((row) => {
                const isSelected = Math.abs(row.threshold - nearestThreshold) < 1e-9;
                return (
                  <tr
                    key={row.threshold}
                    className={`border-b border-[color:var(--color-line)] ${isSelected ? "bg-blue-50" : ""}`}
                  >
                    <th scope="row" className="px-3 py-2 text-left text-sm font-medium text-[color:var(--color-ink)] tabular-nums">
                      {num(row.threshold, 2)}
                      {isSelected && (
                        <Badge className="ml-2 border-blue-200 bg-blue-100 text-blue-700">Nearest to selected</Badge>
                      )}
                    </th>
                    <td className={tdClass()}>{num(row.precision, 3)}</td>
                    <td className={tdClass()}>{num(row.recall, 3)}</td>
                    <td className={tdClass()}>{num(row.f1, 3)}</td>
                    <td className={tdClass()}>{pct(row.flagged_rate)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card
        title="Calibration"
        subtitle="Reliability of predicted probabilities after calibration."
      >
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <StatCard label="Brier before" value={num(metrics.calibration.brier_before, 4)} sub="Pre-calibration" />
          <StatCard label="Brier after" value={num(metrics.calibration.brier_after, 4)} sub="Post-calibration" />
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="border-b border-[color:var(--color-line)]">
                <th scope="col" className={thClass()}>Bin</th>
                <th scope="col" className={thClass()}>Mean predicted</th>
                <th scope="col" className={thClass()}>Observed frequency</th>
                <th scope="col" className={thClass()}>Count</th>
              </tr>
            </thead>
            <tbody>
              {reliability.map((p, i) => (
                <tr key={i} className="border-b border-[color:var(--color-line)]">
                  <th scope="row" className="px-3 py-2 text-left text-sm font-medium text-[color:var(--color-ink)] tabular-nums">
                    {pct(p.bin_lower, 0)}–{pct(p.bin_upper, 0)}
                  </th>
                  <td className={tdClass()}>{pct(p.mean_predicted)}</td>
                  <td className={tdClass()}>{pct(p.observed_frequency)}</td>
                  <td className={tdClass()}>{p.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Alert tone="info" title="Interpretation">
        These are association metrics evaluated on{" "}
        {metrics.synthetic ? "synthetic data" : "the held-out test set"}. They describe statistical
        performance on this dataset and do not establish causation or fitness for real lending
        decisions.
      </Alert>
    </div>
  );
}
