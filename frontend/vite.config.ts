import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The API base URL is configurable (no hardcoded host). Defaults to the local
// FastAPI dev server; override with VITE_API_BASE_URL at build/run time.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
