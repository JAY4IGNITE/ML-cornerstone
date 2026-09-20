// Small formatting helpers used across pages.

export function pct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`;
}

export function num(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

export function metricValue(x: number | { error: string } | null | undefined): string {
  if (x == null) return "—";
  if (typeof x === "object") return "n/a";
  return x.toFixed(4);
}

export const RISK_BAND_CLASS: Record<string, string> = {
  Low: "text-[color:var(--color-risk-low)] bg-emerald-50 border-emerald-200",
  Moderate: "text-[color:var(--color-risk-moderate)] bg-amber-50 border-amber-200",
  Elevated: "text-[color:var(--color-risk-elevated)] bg-orange-50 border-orange-200",
  High: "text-[color:var(--color-risk-high)] bg-red-50 border-red-200",
};

export function riskBandClass(band: string): string {
  return RISK_BAND_CLASS[band] ?? "text-slate-700 bg-slate-50 border-slate-200";
}
