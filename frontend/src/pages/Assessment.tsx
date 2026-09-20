// Loan Risk Assessment form. Fields are generated from /api/feature-schema so
// the form never drifts from the model contract. Client-side validation mirrors
// the server bounds/enums; the server remains the source of truth (errors from
// a 422 are surfaced per-field).
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiRequestError } from "../lib/api";
import type { FeatureSchema } from "../lib/types";
import { Alert, Button, Card, Field, SelectField } from "../components/ui";
import { ErrorState, LoadingState, useAsync } from "../components/states";
import { usePrediction } from "../store/prediction";

type FormValues = Record<string, string>;

function buildInitial(schema: FeatureSchema): FormValues {
  const v: FormValues = {};
  for (const f of schema.numeric_features) {
    v[f.name] = f.example != null ? String(f.example) : "";
  }
  for (const f of schema.categorical_features) {
    v[f.name] = f.example != null ? String(f.example) : f.categories[0] ?? "";
  }
  return v;
}

const OPTIONAL = new Set([
  "EXT_SOURCE_1",
  "EXT_SOURCE_2",
  "EXT_SOURCE_3",
  "OCCUPATION_TYPE",
  "AMT_GOODS_PRICE",
  "AMT_ANNUITY",
]);

export default function Assessment() {
  const { data: schema, loading, error, reload } = useAsync(() => api.featureSchema(), []);
  const navigate = useNavigate();
  const { setPrediction } = usePrediction();

  const [values, setValues] = useState<FormValues | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<unknown>(null);

  const current = useMemo(() => {
    if (values) return values;
    return schema ? buildInitial(schema) : {};
  }, [values, schema]);

  if (loading) return <LoadingState label="Loading form" />;
  if (error || !schema) return <ErrorState error={error} onRetry={reload} />;

  const set = (name: string, value: string) =>
    setValues((prev) => ({ ...(prev ?? buildInitial(schema)), [name]: value }));

  function validate(): { ok: boolean; payload: Record<string, unknown> } {
    const errs: Record<string, string> = {};
    const payload: Record<string, unknown> = {};

    for (const f of schema!.numeric_features) {
      const raw = current[f.name]?.trim() ?? "";
      if (raw === "") {
        if (!OPTIONAL.has(f.name)) errs[f.name] = "Required.";
        continue;
      }
      const n = Number(raw);
      if (!Number.isFinite(n)) {
        errs[f.name] = "Must be a number.";
        continue;
      }
      if (f.min != null && n < f.min) errs[f.name] = `Must be ≥ ${f.min}.`;
      else if (f.max != null && n > f.max) errs[f.name] = `Must be ≤ ${f.max}.`;
      else payload[f.name] = n;
    }
    for (const f of schema!.categorical_features) {
      const raw = current[f.name]?.trim() ?? "";
      if (raw === "") {
        if (!OPTIONAL.has(f.name)) errs[f.name] = "Required.";
        continue;
      }
      if (!f.categories.includes(raw)) errs[f.name] = "Not an allowed value.";
      else payload[f.name] = raw;
    }
    setFieldErrors(errs);
    return { ok: Object.keys(errs).length === 0, payload };
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    const { ok, payload } = validate();
    if (!ok) return;
    setSubmitting(true);
    try {
      const result = await api.predict(payload);
      setPrediction(result, payload);
      navigate("/result");
    } catch (err) {
      // Map server field errors back onto the form when present.
      if (err instanceof ApiRequestError && err.fields?.length) {
        const mapped: Record<string, string> = {};
        for (const fe of err.fields) mapped[fe.field] = fe.message;
        setFieldErrors((prev) => ({ ...prev, ...mapped }));
      }
      setSubmitError(err);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Loan Risk Assessment</h1>
        <p className="mt-1 text-sm text-[color:var(--color-muted)]">
          Enter applicant details to estimate default probability. Optional fields (external
          scores, occupation) may be left blank and will be imputed.
        </p>
      </div>

      <form onSubmit={onSubmit} noValidate className="space-y-4">
        <Card title="Financials">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {schema.numeric_features
              .filter((f) => f.name.startsWith("AMT_"))
              .map((f) => (
                <Field
                  key={f.name}
                  label={f.label + (OPTIONAL.has(f.name) ? " (optional)" : "")}
                  unit={f.unit}
                  type="number"
                  step="any"
                  value={current[f.name] ?? ""}
                  onChange={(e) => set(f.name, e.target.value)}
                  error={fieldErrors[f.name]}
                  hint={f.min != null ? `Range ${f.min}–${f.max}` : undefined}
                />
              ))}
          </div>
        </Card>

        <Card title="Applicant profile">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {schema.numeric_features
              .filter((f) => !f.name.startsWith("AMT_") && !f.name.startsWith("EXT_"))
              .map((f) => (
                <Field
                  key={f.name}
                  label={f.label + (OPTIONAL.has(f.name) ? " (optional)" : "")}
                  unit={f.unit}
                  type="number"
                  step="any"
                  value={current[f.name] ?? ""}
                  onChange={(e) => set(f.name, e.target.value)}
                  error={fieldErrors[f.name]}
                  hint={f.min != null ? `Range ${f.min}–${f.max}` : undefined}
                />
              ))}
            {schema.categorical_features.map((f) => (
              <SelectField
                key={f.name}
                label={f.label + (OPTIONAL.has(f.name) ? " (optional)" : "")}
                options={OPTIONAL.has(f.name) ? ["", ...f.categories] : f.categories}
                value={current[f.name] ?? ""}
                onChange={(e) => set(f.name, e.target.value)}
                error={fieldErrors[f.name]}
              />
            ))}
          </div>
        </Card>

        <Card title="External credit scores" subtitle="Optional — often missing in real data; imputed if blank.">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {schema.numeric_features
              .filter((f) => f.name.startsWith("EXT_"))
              .map((f) => (
                <Field
                  key={f.name}
                  label={f.label + " (optional)"}
                  unit={f.unit}
                  type="number"
                  step="any"
                  value={current[f.name] ?? ""}
                  onChange={(e) => set(f.name, e.target.value)}
                  error={fieldErrors[f.name]}
                  hint="0–1"
                />
              ))}
          </div>
        </Card>

        {submitError != null && (
          <ErrorState error={submitError} />
        )}

        <div className="flex items-center gap-3">
          <Button type="submit" disabled={submitting}>
            {submitting ? "Assessing…" : "Assess risk"}
          </Button>
          <Alert tone="info">
            This produces an analytical estimate, not a lending decision.
          </Alert>
        </div>
      </form>
    </div>
  );
}
