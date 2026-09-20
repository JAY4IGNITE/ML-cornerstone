// Vitest setup: jsdom environment + jest-dom matchers, and a per-test cleanup
// so component trees don't leak between tests.
import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
});
