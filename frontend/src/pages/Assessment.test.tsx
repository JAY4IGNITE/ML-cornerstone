// Behavioural tests for the Assessment form: it must block submission and show
// a per-field "Required." error when a required field is cleared, and it must
// call the predict API with the parsed payload when the form is valid.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import Assessment from "./Assessment";
import { PredictionProvider } from "../store/prediction";
import { api } from "../lib/api";
import type { FeatureSchema } from "../lib/types";

// Replace the network client; keep ApiRequestError (used via instanceof) real.
vi.mock("../lib/api", async (importActual) => {
  const actual = await importActual<typeof import("../lib/api")>();
  return {
    ...actual,
    api: {
      featureSchema: vi.fn(),
      modelInfo: vi.fn(),
      predict: vi.fn(),
    },
  };
});

const schema: FeatureSchema = {
  numeric_features: [
    {
      name: "AGE_YEARS",
      label: "Age",
      unit: "years",
      meaning: "Applicant age",
      min: 18,
      max: 100,
      example: 30,
      missing_behavior: "required",
    },
  ],
  categorical_features: [],
  engineered_features: [],
};

function renderForm() {
  return render(
    <MemoryRouter>
      <PredictionProvider>
        <Assessment />
      </PredictionProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.mocked(api.featureSchema).mockResolvedValue(schema);
  vi.mocked(api.modelInfo).mockResolvedValue({
    model_name: "xgboost",
    model_version: "v1",
    dataset_source: "test",
    synthetic: false,
    feature_schema_version: "v1",
    calibration: {},
    responsible_use_note: "",
  });
  vi.mocked(api.predict).mockResolvedValue({
    default_probability: 0.12,
    risk_score: 12,
    risk_band: "Low",
    model_version: "v1",
    explanation_available: false,
    limitations: [],
    disclaimer: "Not a lending decision.",
  });
});

describe("Assessment form", () => {
  it("shows a Required error and does not submit when a required field is empty", async () => {
    const user = userEvent.setup();
    renderForm();

    // Wait for the schema-driven form to render its field.
    // The unit span renders inside the <label>, so the accessible name is
    // "Age (years)" — match by prefix rather than exact text.
    const age = await screen.findByLabelText(/^Age/);
    await user.clear(age);
    await user.click(screen.getByRole("button", { name: /assess risk/i }));

    expect(await screen.findByText("Required.")).toBeInTheDocument();
    expect(api.predict).not.toHaveBeenCalled();
  });

  it("submits the parsed payload when the form is valid", async () => {
    const user = userEvent.setup();
    renderForm();

    await screen.findByLabelText(/^Age/); // pre-filled from example (30)
    await user.click(screen.getByRole("button", { name: /assess risk/i }));

    await waitFor(() => expect(api.predict).toHaveBeenCalledTimes(1));
    expect(api.predict).toHaveBeenCalledWith({ AGE_YEARS: 30 });
  });
});
