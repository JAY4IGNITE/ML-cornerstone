// Typed API client. Single place that talks to the backend so error handling
// and the base URL live in one spot (no hardcoded hosts scattered around).

import type {
  DatasetQuality,
  FeatureSchema,
  HealthResponse,
  Metrics,
  ModelInfo,
  PredictResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiRequestError extends Error {
  status: number;
  code: string;
  fields?: { field: string; message: string; type: string }[] | null;
  constructor(
    status: number,
    code: string,
    message: string,
    fields?: { field: string; message: string; type: string }[] | null,
  ) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
    this.fields = fields;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // Network / server unreachable => API failure state.
    throw new ApiRequestError(0, "network_error", "Cannot reach the API server.");
  }

  const text = await res.text();
  const body = text ? JSON.parse(text) : null;

  if (!res.ok) {
    const code = body?.error ?? "error";
    const detail = body?.detail ?? `Request failed (${res.status}).`;
    throw new ApiRequestError(res.status, code, detail, body?.fields ?? null);
  }
  return body as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  modelInfo: () => request<ModelInfo>("/api/model/info"),
  featureSchema: () => request<FeatureSchema>("/api/feature-schema"),
  metrics: () => request<Metrics>("/api/metrics"),
  datasetQuality: () => request<DatasetQuality>("/api/dataset/quality"),
  validateInput: (payload: Record<string, unknown>) =>
    request<{ valid: boolean; errors: unknown[] }>("/api/validate-input", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  predict: (payload: Record<string, unknown>) =>
    request<PredictResponse>("/api/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
