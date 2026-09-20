// App shell: skip-link, header with model/health status, sidebar nav, content.
import { NavLink, useLocation } from "react-router-dom";
import { useEffect } from "react";
import { api } from "../lib/api";
import { Badge } from "./ui";
import { useAsync } from "./states";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/assessment", label: "Risk Assessment" },
  { to: "/result", label: "Prediction Result" },
  { to: "/explainability", label: "Explainability" },
  { to: "/performance", label: "Model Performance" },
  { to: "/dataset", label: "Dataset Quality" },
  { to: "/model", label: "Model Information" },
  { to: "/responsible-use", label: "Responsible Use" },
];

function StatusPill() {
  const { data } = useAsync(() => api.health(), []);
  const info = useAsync(() => api.modelInfo().catch(() => null), []);
  if (!data) return <Badge className="border-slate-200 bg-slate-50 text-slate-500">checking…</Badge>;
  const ok = data.model_available;
  const synthetic = info.data?.synthetic;
  return (
    <div className="flex items-center gap-2">
      {synthetic && (
        <Badge className="border-amber-300 bg-amber-50 text-amber-800">Synthetic data</Badge>
      )}
      <Badge
        className={
          ok
            ? "border-emerald-300 bg-emerald-50 text-emerald-800"
            : "border-red-300 bg-red-50 text-red-800"
        }
      >
        <span
          aria-hidden="true"
          className={`mr-1.5 inline-block h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-red-500"}`}
        />
        API {ok ? "ready" : "degraded"}
      </Badge>
    </div>
  );
}

export default function Layout({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation();
  useEffect(() => {
    // Move focus to main on route change for screen-reader users.
    document.getElementById("main-content")?.focus();
  }, [pathname]);

  return (
    <div className="min-h-full">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2 focus:shadow"
      >
        Skip to content
      </a>

      <header className="sticky top-0 z-40 border-b border-[color:var(--color-line)] bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[color:var(--color-accent)] text-sm font-bold text-white">
              LR
            </div>
            <div>
              <div className="text-sm font-semibold leading-tight">Loan Default Risk</div>
              <div className="text-xs text-[color:var(--color-muted)]">Analytical assessment · not a lending decision</div>
            </div>
          </div>
          <StatusPill />
        </div>
      </header>

      <div className="mx-auto flex max-w-7xl gap-6 px-4 py-6">
        <nav aria-label="Primary" className="hidden w-56 shrink-0 md:block">
          <ul className="space-y-1">
            {NAV.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    `block rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-[color:var(--color-accent)] text-white"
                        : "text-[color:var(--color-ink)] hover:bg-slate-100"
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <main id="main-content" tabIndex={-1} className="min-w-0 flex-1 focus:outline-none">
          {/* Mobile nav */}
          <nav aria-label="Primary mobile" className="mb-4 flex gap-2 overflow-x-auto md:hidden">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `whitespace-nowrap rounded-lg px-3 py-1.5 text-xs font-medium ${
                    isActive ? "bg-[color:var(--color-accent)] text-white" : "bg-slate-100 text-slate-700"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          {children}
        </main>
      </div>
    </div>
  );
}
