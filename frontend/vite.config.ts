import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

const apiProxy = {
  "/api": {
    target: "http://localhost:8000",
    changeOrigin: true,
  },
};

export default defineConfig({
  plugins: [react()],
  // `server.proxy` only applies to `vite dev`; `preview.proxy` is the
  // equivalent for `vite preview` (used by Playwright's webServer in
  // playwright.config.ts) -- both must be set for /api to resolve in either mode.
  server: { proxy: apiProxy },
  preview: { proxy: apiProxy },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/setupTests.ts"],
    exclude: ["**/node_modules/**", "**/e2e/**"],
  },
});
