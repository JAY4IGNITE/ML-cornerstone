// Lightweight shared state for the latest prediction + the applicant payload
// that produced it, so Result and Explainability pages can render without a
// re-submit. No external state library needed.
import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { PredictResponse } from "../lib/types";

interface PredictionState {
  result: PredictResponse | null;
  payload: Record<string, unknown> | null;
  setPrediction: (result: PredictResponse, payload: Record<string, unknown>) => void;
  clear: () => void;
}

const Ctx = createContext<PredictionState | undefined>(undefined);

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [payload, setPayload] = useState<Record<string, unknown> | null>(null);

  const value = useMemo<PredictionState>(
    () => ({
      result,
      payload,
      setPrediction: (r, p) => {
        setResult(r);
        setPayload(p);
      },
      clear: () => {
        setResult(null);
        setPayload(null);
      },
    }),
    [result, payload],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function usePrediction(): PredictionState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("usePrediction must be used within PredictionProvider");
  return ctx;
}
