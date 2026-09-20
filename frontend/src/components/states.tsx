// Reusable UX states (06_FRONTEND_REQUIREMENTS.md: loading / empty / error /
// API-failure / unsupported states) + a tiny data-fetching hook.
import { useCallback, useEffect, useState } from "react";
import { ApiRequestError } from "../lib/api";
import { Alert, Button, Card, Spinner } from "./ui";

export function LoadingState({ label }: { label?: string }) {
  return (
    <div className="flex items-center justify-center py-16">
      <Spinner label={label ?? "Loading"} />
    </div>
  );
}

export function EmptyState({ title, message }: { title: string; message: string }) {
  return (
    <Card>
      <div className="py-10 text-center">
        <h3 className="text-sm font-semibold text-[color:var(--color-ink)]">{title}</h3>
        <p className="mx-auto mt-2 max-w-md text-sm text-[color:var(--color-muted)]">{message}</p>
      </div>
    </Card>
  );
}

export function ErrorState({
  error,
  onRetry,
}: {
  error: unknown;
  onRetry?: () => void;
}) {
  const isApi = error instanceof ApiRequestError;
  const network = isApi && error.status === 0;
  const unavailable = isApi && error.status === 503;
  const notReady = isApi && (error.status === 404 || unavailable);

  const title = network
    ? "Cannot reach the API"
    : notReady
      ? "Model or data not ready"
      : "Something went wrong";
  const message = isApi
    ? error.message
    : "An unexpected error occurred while loading this view.";

  return (
    <Alert tone={network ? "danger" : "warning"} title={title}>
      <p>{message}</p>
      {notReady && (
        <p className="mt-2 text-xs">
          Train the pipeline (<code>python -m loan_risk.pipeline.run</code>) and ensure the
          backend is running, then retry.
        </p>
      )}
      {onRetry && (
        <div className="mt-3">
          <Button variant="secondary" onClick={onRetry}>
            Retry
          </Button>
        </div>
      )}
    </Alert>
  );
}

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: unknown;
  reload: () => void;
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  const run = useCallback(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => {
        if (active) setData(d);
      })
      .catch((e) => {
        if (active) setError(e);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => run(), [run]);
  return { data, loading, error, reload: run };
}
