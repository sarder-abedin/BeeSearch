import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/**
 * Runs against the REAL FastAPI backend (with BEESEARCH_MOCK_LLM=1, see
 * playwright.config.ts's webServer) and the production Vite build (`vite
 * preview`), exercising the full Mode 2 Phase E pipeline: React -> HTTP ->
 * FastAPI -> agents.graph -> all 8 Research Report workflow steps, including
 * the academic/web search steps (mocked via backend/app/mock_search.py,
 * installed alongside mock_llm.py under the same BEESEARCH_MOCK_LLM=1 flag --
 * see backend/app/main.py).
 *
 * Reuses the Phase A/B/C/D fixture (battery-storage-briefing.txt) -- the
 * report-generation mock response is keyed off prompt *shape*
 * (backend/app/mock_llm.py), not the document's actual text, so the
 * fixture's exact paragraph mix doesn't matter here; it's reused only to
 * avoid a second near-duplicate fixture file (same rationale as the other
 * notebook-*.spec.ts files).
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

test("generates a search-mode report for a notebook with no sources, with a Markdown download", async ({
  page,
}) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("Report Search Notebook"));

  await page.getByRole("tab", { name: "Research Report" }).click();
  await expect(
    page.getByText("No sources in this notebook — will search academic literature only."),
  ).toBeVisible();

  await page.getByLabel("Research goal or question").fill("What is grid-scale battery storage?");
  await page.getByRole("button", { name: "Generate Research Report" }).click();

  // Mid-run: at least one of the 8 workflow steps' progress labels should be
  // observable over real network polling (700ms interval) before completion
  // -- query generation + academic search alone are backed by a mock LLM
  // call (0.4s) plus a mock search call (0.5s), see mock_llm.py/mock_search.py.
  await expect(
    page.getByText(
      /Generating search queries|Searching arXiv \+ Semantic Scholar|Analysing sources|Compiling references|Generating report|Evaluating quality/,
    ),
  ).toBeVisible({ timeout: 15_000 });

  await expect(page.getByText("Done.")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/^Research workflow failed:/)).toHaveCount(0);

  await expect(
    page.getByText("Mock Research Report: What is grid-scale battery storage?"),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "Executive Summary" })).toBeVisible();

  await expect(page.getByRole("heading", { name: "Key Findings", level: 3 })).toBeVisible();
  await expect(page.getByText(/Mock key finding one/).first()).toBeVisible();
  await expect(page.getByText("Mock key finding two.").first()).toBeVisible();

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download Report (Markdown)" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("research_report.md");

  await page.getByRole("tab", { name: "References" }).click();
  await expect(page.getByText(/References \(\d+\)/)).toBeVisible();
  await expect(page.getByRole("button", { name: /^\[1\] A Survey of/ })).toBeVisible();
});

test("generates a hybrid report with web search enabled and exports BibTeX/RIS citations", async ({ page }) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("Report Hybrid Notebook"));

  await page.getByLabel("Upload sources").setInputFiles(FIXTURE_PATH);
  await expect(page.getByText("battery-storage-briefing.txt")).toBeVisible();

  await page.getByRole("tab", { name: "Research Report" }).click();
  await expect(
    page.getByText("No sources in this notebook — will search academic literature only."),
  ).toHaveCount(0);

  await page.getByLabel("Research goal or question").fill("What is grid-scale battery storage?");
  await page.getByRole("checkbox", { name: "Include web search (DuckDuckGo)" }).click();
  await page.getByRole("button", { name: "Generate Research Report" }).click();

  await expect(page.getByText("Done.")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/^Research workflow failed:/)).toHaveCount(0);
  await expect(page.getByText(/Web search was enabled but/)).toHaveCount(0);

  await expect(page.getByText("Mock Research Report: What is grid-scale battery storage?")).toBeVisible();
  await expect(page.getByText(/\[Source 1\]/).first()).toBeVisible();

  await page.getByRole("tab", { name: "References" }).click();
  const paperCard = page.getByRole("button", { name: /^\[1\] A Survey of/ });
  await expect(paperCard).toBeVisible();
  await paperCard.click();
  await expect(page.getByText("Authors: A. Researcher; B. Scholar")).toBeVisible();
  await expect(page.getByText("Journal/Venue: Mock Journal of Research")).toBeVisible();
  await expect(page.getByRole("link", { name: "https://example.org/mock-paper-1" })).toHaveAttribute(
    "href",
    "https://example.org/mock-paper-1",
  );
  await expect(page.getByText(/Source: arXiv preprint/)).toBeVisible();
  await expect(page.getByText(/Citations: 12/)).toBeVisible();

  await expect(page.getByRole("button", { name: /Mock web result for/ }).first()).toBeVisible();

  const bibtexDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export BibTeX (.bib)" }).click();
  expect((await bibtexDownload).suggestedFilename()).toBe("references.bib");

  const risDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export RIS (.ris)" }).click();
  expect((await risDownload).suggestedFilename()).toBe("references.ris");
});
