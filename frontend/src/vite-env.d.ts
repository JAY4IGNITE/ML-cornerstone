/// <reference types="vite/client" />

// Typed project-specific env vars (see vite.config.ts / .env). The API base URL
// is configurable so there is no hardcoded backend host in the bundle.
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
