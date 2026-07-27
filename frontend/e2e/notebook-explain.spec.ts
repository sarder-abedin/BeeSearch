import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/**
 * Runs against the REAL FastAPI backend (with BEESEARCH_MOCK_LLM=1, see
 * playwright.config.ts's webServer) and the production Vite build (`vite
 * preview`), exercising the full Mode 2 Phase D pipeline: React -> HTTP ->
 * FastAPI -> agents.story_graph -> all 7 storyteller nodes.
 *
 * Reuses the Phase A/B/C fixture (battery-storage-briefing.txt) -- the
 * storyteller's mock response is keyed off prompt *shape* (BEESEARCH_MOCK_LLM=1,
 * see backend/app/mock_llm.py), not the document's actual text, so the
 * fixture's exact paragraph mix doesn't matter here; it's reused only to
 * avoid a second near-duplicate fixture file (same rationale as
 * notebook-pipeline.spec.ts / notebook-advanced.spec.ts).
 *
 * Concept-map (Pyvis) rendering depends on the optional `pyvis` package
 * (agents/story_nodes.py::concept_visualizer_node) -- not guaranteed in
 * every sandbox/CI image, and the feature already degrades gracefully (any
 * extraction/render failure -> concept_visual_html="" -> the tab simply
 * omits the iframe) when missing, so that assertion accepts either outcome.
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

test("answers a grounded question with citations and a coverage caption, then rotates style on a detected repeat", async ({
  page,
}) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("Explain Notebook"));

  await page.getByLabel("Upload sources").setInputFiles(FIXTURE_PATH);
  await expect(page.getByText("battery-storage-briefing.txt")).toBeVisible();

  await page.getByRole("tab", { name: "Explain" }).click();
  await expect(page.getByText(/Type your first question/)).toBeVisible();

  // Keep this selection unchanged across both turns below -- repetition_tracker_node
  // only rotates to the next style in _STYLE_ROTATION when the *requested* style on
  // the repeat turn still matches what was actually used last time (see story_nodes.py).
  await page.getByRole("radio", { name: "Step-by-Step" }).click();
  await page.getByRole("radio", { name: "Expert" }).click();

  await page.getByLabel("Message").fill("What is grid-scale battery storage?");
  await page.getByRole("button", { name: "Send" }).click();

  // Mid-run: at least one of the 7 nodes' progress labels should be observable
  // over real network polling (700ms interval) before the run completes.
  await expect(
    page.getByText(/Assessing document coverage|Composing explanation|Checking for repeated questions/),
  ).toBeVisible({ timeout: 15_000 });

  await expect(page.getByText("Done.")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/^Failed:/)).toHaveCount(0);

  await expect(
    page.getByText(/This is a mock Explain answer to "What is grid-scale battery storage\?" \[1\]/),
  ).toBeVisible();

  await page.getByRole("button", { name: "Sources (1)" }).click();
  const citationItem = page.locator(".notebook-citations__item").first();
  await expect(citationItem).toContainText("battery-storage-briefing.txt");
  await expect(citationItem).toContainText("p. 1");

  await expect(page.getByText("Answered from your documents (coverage 8/10)")).toBeVisible();

  await expect(page.getByRole("button", { name: "Can you explain that with a different analogy?" })).toBeVisible();

  // Repeated clarification: turn 1 used "walkthrough" (Step-by-Step, requested as-is
  // since there's no prior turn to rotate away from). The style radio above is left
  // untouched, so on this repeat _next_explanation_strategy rotates walkthrough ->
  // debate -- the frontend should then call out that mismatch between what was
  // actually used and what's still selected.
  await page.getByLabel("Message").fill("I don't understand, can you explain again?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("Done.")).toBeVisible({ timeout: 20_000 });

  await expect(
    page.getByText(
      'This looked like a repeat of an earlier question, so this answer uses "For vs. Against" instead of your selected style — explaining it differently, not just rewording it.',
    ),
  ).toBeVisible();

  // Optional, environment-dependent enhancement -- see file header comment.
  const conceptFrame = page.getByTitle("Concept map");
  if ((await conceptFrame.count()) > 0) {
    await expect(conceptFrame).toBeVisible();
  }
});

test("falls back to online search and cites web/academic sources for a notebook with no uploaded documents", async ({
  page,
}) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("Explain No Docs Notebook"));

  await page.getByRole("tab", { name: "Explain" }).click();
  await expect(page.getByText(/Type your first question/)).toBeVisible();

  await page.getByLabel("Message").fill("What is X?");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("Done.")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/^Failed:/)).toHaveCount(0);

  await expect(page.getByText(/This fills the gap using \[Source 1\]/)).toBeVisible();
  await expect(page.getByText(/Document coverage: 0\/10 — No documents uploaded/)).toBeVisible();
  await expect(page.getByText(/3 source\(s\) from arXiv \/ Semantic Scholar \+ web/)).toBeVisible();

  await page.getByRole("button", { name: "Sources (1)" }).click();
  const link = page.getByRole("link", { name: "A Survey of What is X" });
  await expect(link).toBeVisible();
  await expect(link).toHaveAttribute("href", "https://example.org/mock-paper-1");
});
