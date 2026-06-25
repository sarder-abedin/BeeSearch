import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

const PORT = 4173;
const API_PORT = 8000;
const REPO_ROOT = path.resolve(import.meta.dirname, "..");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          executablePath: "/opt/pw-browsers/chromium",
        },
      },
    },
  ],
  webServer: [
    {
      // Dev-only mock LLM + mock search (backend/app/mock_llm.py, mock_search.py)
      // so the E2E run is deterministic and needs neither a reachable Ollama
      // server nor network access to the real search backends.
      command: `python -m uvicorn backend.app.main:app --port ${API_PORT}`,
      url: `http://localhost:${API_PORT}/api/health`,
      cwd: REPO_ROOT,
      env: { ...process.env, BEESEARCH_MOCK_LLM: "1" },
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
    {
      command: `npm run preview -- --port ${PORT} --strictPort`,
      url: `http://localhost:${PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
  ],
});
