import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/**
 * Runs against the REAL FastAPI backend (with BEESEARCH_MOCK_LLM=1, see
 * playwright.config.ts's webServer) and the production Vite build (`vite
 * preview`), exercising the full Mode 2 Phase C surface: React -> HTTP ->
 * FastAPI -> agents.notebook_advanced's 9 standalone tools.
 *
 * Two fixtures are uploaded into one notebook:
 *  - storage-review-with-references.txt (NEW, added alongside this spec):
 *    plain prose followed by a genuine "References" heading and four
 *    bracketed bibliography entries. Citation Timeline is the only one of
 *    the 9 tools that needs this -- agents/notebook_advanced.py::
 *    extract_citation_timeline() skips any source where
 *    tools.text_parsing.extract_references_section() returns empty, and
 *    battery-storage-briefing.txt (Phase A/B's fixture, reused here for the
 *    other 8 tools) has no references section at all, so without this new
 *    fixture Citation Timeline would only ever exercise its graceful
 *    "no references found" empty-result path, never its real rendering path.
 *  - battery-storage-briefing.txt (existing, reused as-is): the other 8
 *    tools run on whatever prose they're given, with no special section
 *    requirements, so the existing fixture is reused rather than adding a
 *    second near-duplicate file (same rationale as notebook-pipeline.spec.ts).
 *
 * Uploaded in that order, so the references fixture is always source 1 --
 * pinning down which per-source mock title suffix
 * (backend/app/mock_llm.py's "Mock Cited Work {source_num}-1/-2") the
 * Citation Timeline assertions below expect.
 *
 * Knowledge graph / mind map PNG rendering depends on the system graphviz
 * `dot` binary plus the Python `graphviz` package (agents/notebook_advanced
 * .py::render_dot_bytes) -- neither is guaranteed in every sandbox/CI image,
 * and the feature already degrades gracefully (503 -> "preview unavailable"
 * caption) when missing, so those assertions accept either outcome.
 */

const FIXTURE_WITH_REFS = path.join(import.meta.dirname, "fixtures", "storage-review-with-references.txt");
const FIXTURE_PLAIN = path.join(import.meta.dirname, "fixtures", "battery-storage-briefing.txt");

function uniqueName(label: string): string {
  return `${label} ${Math.random().toString(36).slice(2, 8)}`;
}

async function createNotebookViaUi(page: Page, name: string): Promise<void> {
  await page.getByLabel("Notebook name").fill(name);
  await page.getByRole("button", { name: "Create notebook" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();
}

test("shows the source-comparison guard for a notebook with fewer than two sources", async ({ page }) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("Single Source Advanced Notebook"));

  await page.getByLabel("Upload sources").setInputFiles(FIXTURE_PLAIN);
  await expect(page.getByText("battery-storage-briefing.txt")).toBeVisible();

  await page.getByRole("tab", { name: "Advanced Tools" }).click();
  await page.getByRole("radio", { name: "Compare" }).click();
  await expect(page.getByText("Add at least two sources to use source comparison.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Compare Sources" })).toHaveCount(0);
});

test("runs all 9 advanced tools end-to-end with distinct content per tool", async ({ page }) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("Advanced Tools Notebook"));

  await page.getByLabel("Upload sources").setInputFiles([FIXTURE_WITH_REFS, FIXTURE_PLAIN]);
  await expect(page.getByText("storage-review-with-references.txt")).toBeVisible();
  await expect(page.getByText("battery-storage-briefing.txt")).toBeVisible();

  await page.getByRole("tab", { name: "Advanced Tools" }).click();

  // Cross-Document Summary is the default tool on first render -- no radio click needed.
  await expect(page.getByRole("heading", { name: "Cross-Document Summary" })).toBeVisible();
  await page.getByRole("button", { name: "Generate Summary" }).click();
  await expect(page.getByText(/mock cross-document summary/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("button", { name: "Download .md" })).toBeVisible();

  // FAQ
  await page.getByRole("radio", { name: "FAQ" }).click();
  await page.getByRole("button", { name: "Generate FAQ" }).click();
  const faqQuestion = "What is the main topic of these sources (mock)?";
  await expect(page.getByRole("button", { name: faqQuestion })).toBeVisible({ timeout: 15_000 });
  await page.getByRole("button", { name: faqQuestion }).click();
  await expect(page.getByText("This is a mock grounded FAQ answer (BEESEARCH_MOCK_LLM=1).")).toBeVisible();

  // Literature Review
  await page.getByRole("radio", { name: "Lit Review" }).click();
  await page.getByRole("button", { name: "Generate Literature Review" }).click();
  await expect(page.getByRole("heading", { name: "1. Introduction" })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/mock literature review introduction/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Download .pdf" })).toBeVisible();

  // Mind Map
  await page.getByRole("radio", { name: "Mind Map" }).click();
  await page.getByRole("button", { name: "Generate Mind Map" }).click();
  await expect(
    page.getByAltText("Mind map").or(page.getByText(/Preview unavailable/)),
  ).toBeVisible({ timeout: 15_000 });
  await page.getByText("View DOT source").click();
  await expect(page.getByText(/digraph/)).toBeVisible();

  // Audio Summary
  await page.getByRole("radio", { name: "Audio" }).click();
  await page.getByRole("button", { name: "Generate Audio Script" }).click();
  await expect(page.getByText(/mock audio summary script/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/Word count: \d+/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Download script (.txt)" })).toBeVisible();

  // Compare Sources -- defaults to comparing the two uploaded sources directly.
  await page.getByRole("radio", { name: "Compare" }).click();
  await page.getByRole("button", { name: "Compare Sources" }).click();
  await expect(page.getByRole("heading", { name: "Source Comparison" })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/Mock overview of 'storage-review-with-references\.txt'/)).toBeVisible();

  // Knowledge Graph
  await page.getByRole("radio", { name: "Graph" }).click();
  await page.getByRole("button", { name: "Extract Knowledge Graph" }).click();
  await expect(
    page.getByAltText("Knowledge graph").or(page.getByText(/Preview unavailable/)),
  ).toBeVisible({ timeout: 15_000 });

  // Citation Timeline -- only the references-bearing fixture (source 1) yields
  // entries; battery-storage-briefing.txt (source 2) has no references section
  // and is silently skipped by extract_citation_timeline().
  await page.getByRole("radio", { name: "Citation Timeline" }).click();
  await page.getByRole("button", { name: "Extract Citation Timeline" }).click();
  // Scoped to table cells (not getByText) since each title also appears as a
  // substring of its own gist cell ("Mock one-line gist for 'Mock Cited Work
  // 1-1' ..."), which would otherwise trip Playwright's strict-mode check.
  await expect(page.getByRole("cell", { name: "Mock Cited Work 1-1", exact: true })).toBeVisible({
    timeout: 15_000,
  });
  await expect(page.getByRole("cell", { name: "Mock Cited Work 1-2", exact: true })).toBeVisible();
  await expect(page.getByText(/No citation timeline was generated/)).toHaveCount(0);

  // Study Comparison
  await page.getByRole("radio", { name: "Study Table" }).click();
  await page.getByRole("button", { name: "Generate Study Comparison" }).click();
  await expect(page.getByText(/mock synthesis paragraph/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("button", { name: "Download .docx" })).toBeVisible();
});
