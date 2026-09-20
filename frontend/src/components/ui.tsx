// Shared, accessible UI primitives. Deliberately lightweight (no heavy UI
// dependency): clean financial-analytics styling with semantic HTML, labels,
// and visible focus. Color is never the only signal.
import type { ReactNode, SelectHTMLAttributes, InputHTMLAttributes } from "react";
import { useId } from "react";

export function Card({
  title,
  subtitle,
  children,
  className = "",
  action,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
}) {
  return (
    <section
      className={`rounded-[var(--radius-card)] border border-[color:var(--color-line)] bg-[color:var(--color-surface)] p-5 shadow-sm ${className}`}
    >
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-base font-semibold text-[color:var(--color-ink)]">{title}</h2>}
            {subtitle && <p className="mt-1 text-sm text-[color:var(--color-muted)]">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Button({
  children,
  variant = "primary",
  ...props
}: {
  children: ReactNode;
  variant?: "primary" | "secondary";
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50";
  const styles =
    variant === "primary"
      ? "bg-[color:var(--color-accent)] text-[color:var(--color-accent-fg)] hover:bg-blue-800"
      : "border border-[color:var(--color-line)] bg-white text-[color:var(--color-ink)] hover:bg-slate-50";
  return (
    <button className={`${base} ${styles}`} {...props}>
      {children}
    </button>
  );
}

export function Badge({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${className}`}
    >
      {children}
    </span>
  );
}

export function Field({
  label,
  hint,
  error,
  unit,
  ...props
}: {
  label: string;
  hint?: string;
  error?: string;
  unit?: string;
} & InputHTMLAttributes<HTMLInputElement>) {
  const id = useId();
  const describedBy = [hint ? `${id}-hint` : "", error ? `${id}-err` : ""].filter(Boolean).join(" ");
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-medium text-[color:var(--color-ink)]">
        {label}
        {unit && <span className="ml-1 font-normal text-[color:var(--color-muted)]">({unit})</span>}
      </label>
      <input
        id={id}
        aria-describedby={describedBy || undefined}
        aria-invalid={!!error}
        className={`rounded-lg border bg-white px-3 py-2 text-sm text-[color:var(--color-ink)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--color-accent)] ${
          error ? "border-red-400" : "border-[color:var(--color-line)]"
        }`}
        {...props}
      />
      {hint && !error && (
        <span id={`${id}-hint`} className="text-xs text-[color:var(--color-muted)]">
          {hint}
        </span>
      )}
      {error && (
        <span id={`${id}-err`} className="text-xs font-medium text-red-600">
          {error}
        </span>
      )}
    </div>
  );
}

export function SelectField({
  label,
  options,
  hint,
  error,
  ...props
}: {
  label: string;
  options: string[];
  hint?: string;
  error?: string;
} & SelectHTMLAttributes<HTMLSelectElement>) {
  const id = useId();
  const describedBy = [hint ? `${id}-hint` : "", error ? `${id}-err` : ""].filter(Boolean).join(" ");
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-sm font-medium text-[color:var(--color-ink)]">
        {label}
      </label>
      <select
        id={id}
        aria-describedby={describedBy || undefined}
        aria-invalid={!!error}
        className={`rounded-lg border bg-white px-3 py-2 text-sm text-[color:var(--color-ink)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--color-accent)] ${
          error ? "border-red-400" : "border-[color:var(--color-line)]"
        }`}
        {...props}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
      {hint && !error && (
        <span id={`${id}-hint`} className="text-xs text-[color:var(--color-muted)]">
          {hint}
        </span>
      )}
      {error && (
        <span id={`${id}-err`} className="text-xs font-medium text-red-600">
          {error}
        </span>
      )}
    </div>
  );
}

export function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: ReactNode;
  sub?: string;
}) {
  return (
    <div className="rounded-[var(--radius-card)] border border-[color:var(--color-line)] bg-[color:var(--color-surface)] p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-[color:var(--color-muted)]">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold text-[color:var(--color-ink)]">{value}</div>
      {sub && <div className="mt-1 text-xs text-[color:var(--color-muted)]">{sub}</div>}
    </div>
  );
}

export function Alert({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warning" | "danger";
  title?: string;
  children: ReactNode;
}) {
  const tones = {
    info: "border-blue-200 bg-blue-50 text-blue-900",
    warning: "border-amber-200 bg-amber-50 text-amber-900",
    danger: "border-red-200 bg-red-50 text-red-900",
  } as const;
  return (
    <div role="note" className={`rounded-lg border p-3 text-sm ${tones[tone]}`}>
      {title && <div className="mb-1 font-semibold">{title}</div>}
      {children}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span role="status" aria-live="polite" className="inline-flex items-center gap-2 text-sm text-[color:var(--color-muted)]">
      <span
        aria-hidden="true"
        className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-[color:var(--color-accent)]"
      />
      {label}…
    </span>
  );
}
