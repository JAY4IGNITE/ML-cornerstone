// Exercises the useAsync loading -> error path through a real page. When
// api.metrics() rejects with a network error, Performance must render the
// ErrorState "Cannot reach the API" alert instead of crashing or hanging on
// the spinner.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import Performance from "./Performance";
import { api, ApiRequestError } from "../lib/api";

vi.mock("../lib/api", async (importActual) => {
  const actual = await importActual<typeof import("../lib/api")>();
  return {
    ...actual,
    api: { metrics: vi.fn() },
  };
});

beforeEach(() => {
  vi.mocked(api.metrics).mockReset();
});

describe("Performance page async states", () => {
  it("shows the loading state while metrics are in flight", () => {
    // A promise that never settles keeps the hook in its loading state.
    vi.mocked(api.metrics).mockReturnValue(new Promise(() => {}));
    render(<Performance />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("renders the API-failure error state when metrics cannot be reached", async () => {
    vi.mocked(api.metrics).mockRejectedValue(
      new ApiRequestError(0, "network_error", "Cannot reach the API server."),
    );
    render(<Performance />);

    expect(await screen.findByText("Cannot reach the API")).toBeInTheDocument();
    expect(screen.getByText("Cannot reach the API server.")).toBeInTheDocument();
  });
});
