import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/**
 * Runs against the REAL FastAPI backend (with BEESEARCH_MOCK_LLM=1, see
 * playwright.config.ts's webServer) and the production Vite build (`vite
 * preview`), exercising the full Mode 2 Phase B pipeline: React -> HTTP ->
 * FastAPI -> agents.notebook_pipeline_graph -> all 7 agent nodes.
 *
 * Reuses the Phase A fixture (fixtures/battery-storage-briefing.txt) -- the
 * pipeline's summarization/citation/KG/study-guide/podcast stages all run on
 * the full document text directly (no BM25 ranking sensitivity the way Phase
 * A's chat retrieval has), so the fixture's exact paragraph mix doesn't matter
 * here; it's reused only to avoid a second near-duplicate fixture file.
 *
 * Each pipeline stage's distinct prompt is matched by its own branch in
 * backend/app/mock_llm.py (added alongside this test), so Summary / Citations
 * / Knowledge Graph / Study Guide / Podcast each render different, assertable
 * mock content instead of all falling through to one generic canned answer.
 *
 * Knowledge graph PNG rendering depends on the system graphviz `dot` binary
 * plus the Python `graphviz` package being installed (agents/notebook_advanced
 * .py::render_dot_bytes) -- neither is guaranteed in every sandbox/CI image,
 * and the feature already degrades gracefully (503 -> "preview unavailable"
 * caption) when missing. The assertion below accepts either outcome so this
 * test doesn't couple to one environment's optional dependencies.
 */

const FIXTURE_PATH = path.join(import.meta.dirname, "fixtures", "battery-storage-briefing.txt");

function uniqueName(label: string): string {
  return `${label} ${Math.random().toString(36).slice(2, 8)}`;
}

async function createNotebookViaUi(page: Page, name: string): Promise<void> {
  await page.getByLabel("Notebook name").fill(name);
  await page.getByRole("button", { name: "Create notebook" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();
}

test("shows the no-sources guard instead of run controls for an empty notebook", async ({ page }) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("Empty Pipeline Notebook"));

  await page.getByRole("tab", { name: "Analysis Pipeline" }).click();
  await expect(
    page.getByText("Add at least one source in the Sources panel before running the analysis pipeline."),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Run Full Pipeline" })).toHaveCount(0);
});

test("runs the full 7-agent pipeline end-to-end with distinct content per sub-tab", async ({ page }) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("Pipeline Notebook"));

  await page.getByLabel("Upload sources").setInputFiles(FIXTURE_PATH);
  await expect(page.getByText("battery-storage-briefing.txt")).toBeVisible();

  await page.getByRole("tab", { name: "Analysis Pipeline" }).click();
  await page.getByRole("button", { name: "Run Full Pipeline" }).click();

  // Mid-run: at least one of the 7 agents' progress labels should be observable
  // over real network polling (700ms interval) before the run completes.
  await expect(
    page.getByText(/Agent \d — (Document Ingestion|Summarization|Retrieval|Citation Verification|Knowledge Graph|Study Guide|Podcast Script)/),
  ).toBeVisible({ timeout: 15_000 });

  // 7 sequential agents, several with their own mock-LLM call(s) at ~0.4s each,
  // plus real HybridStore/BM25 indexing -- a cold backend can push this well
  // past the chat flow's latency, hence the larger timeout than notebook.spec.ts.
  await expect(page.getByText("Done.")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/Pipeline failed/)).toHaveCount(0);

  // Ingestion (default landing sub-tab)
  await expect(page.getByText(/Loaded 1 source\(s\) with \d+ chunk\(s\)/)).toBeVisible();
  await expect(page.getByText("1 document(s) ingested.")).toBeVisible();

  // Summary -- single-source branch in mock_llm.py
  await page.getByRole("tab", { name: "Summary" }).click();
  await expect(page.getByText(/mock single-document summary of 'battery-storage-briefing\.txt'/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Download .md" })).toBeVisible();

  // Retrieval
  await page.getByRole("tab", { name: "Retrieval" }).click();
  await expect(page.getByText(/Retrieval mode:/)).toBeVisible();
  await expect(page.getByText(/chunk\(s\) retrieved/)).toBeVisible();

  // Citations -- exercises the mock_llm.py branch added specifically to avoid
  // colliding with the generic Mode 3 "JSON array" branch (see mock_llm.py
  // comment); a real regression here would previously have crashed this stage.
  await page.getByRole("tab", { name: "Citations" }).click();
  await expect(page.getByRole("heading", { name: "Citation Verification Report" })).toBeVisible();
  await expect(page.getByText(/Verified 3 claims/)).toBeVisible();
  // Scoped to table cells (not getByText) since the summary paragraph above
  // also contains these substrings ("1 ✅ HIGH · 1 🟡 MEDIUM · 1 ❌ LOW"),
  // which would otherwise trip Playwright's strict-mode multi-match check.
  await expect(page.getByRole("cell", { name: /HIGH/ })).toBeVisible();
  await expect(page.getByRole("cell", { name: /MEDIUM/ })).toBeVisible();
  await expect(page.getByRole("cell", { name: /LOW/ })).toBeVisible();

  // Knowledge Graph -- accept either a rendered preview or the graceful
  // "unavailable" degradation, depending on whether this environment has
  // graphviz installed (see file header comment).
  await page.getByRole("tab", { name: "Knowledge Graph" }).click();
  await expect(
    page.getByAltText("Knowledge graph").or(page.getByText(/Knowledge graph preview unavailable/)),
  ).toBeVisible({ timeout: 10_000 });
  await page.getByText("View DOT source").click();
  await expect(page.getByText(/digraph knowledge_graph/)).toBeVisible();

  // Study Guide
  await page.getByRole("tab", { name: "Study Guide" }).click();
  await expect(page.getByRole("heading", { name: "Key Concepts" })).toBeVisible();
  await expect(page.getByText("Mock Concept A")).toBeVisible();

  // Real browser download of the client-side-generated Study Guide markdown --
  // verifies the actual anchor/Blob download mechanic end-to-end (the frontend
  // component tests mock this module out entirely, since jsdom doesn't support it).
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download .md" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("study_guide.md");

  // Podcast
  await page.getByRole("tab", { name: "Podcast" }).click();
  await expect(page.getByText(/HOST: Welcome back to the show/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Download .txt" })).toBeVisible();
});
