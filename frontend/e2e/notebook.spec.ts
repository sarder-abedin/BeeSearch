import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/**
 * Runs against the REAL FastAPI backend (with BEESEARCH_MOCK_LLM=1, see
 * playwright.config.ts's webServer) and the production Vite build (`vite
 * preview`), so this exercises the full React -> HTTP -> FastAPI -> agents
 * pipeline -- no mocked fetch, no fake timers. Source upload runs through the
 * real DocumentProcessor/HybridStore (BM25-only, since no Ollama embedding
 * server is reachable here); only the chat LLM calls are stubbed, by
 * backend/app/mock_llm.py.
 *
 * The fixture file (fixtures/battery-storage-briefing.txt) has 6 short
 * paragraphs separated by blank lines, which DocumentProcessor splits into 6
 * "pages" / chunks: the first two are about battery storage (what the test
 * asks about); the other four are unrelated decoy topics. This isn't just
 * padding -- BM25's classic idf formula gives *every* term a non-positive
 * score in a 2-document corpus (a term in both docs scores negative, a term
 * in only one scores exactly zero), so with only the 2 real paragraphs,
 * HybridStore._search_sparse's "score > 0" filter discards everything and
 * retrieval comes back empty. Padding with 4 more documents makes "battery
 * storage capacity" genuinely rare corpus-wide, which gives it a positive
 * score, while the decoys (chosen to share no words with the question, not
 * even stopwords) score exactly 0 and get filtered out. Net effect: the two
 * battery-storage chunks are always the only ones retrieved and always rank
 * 1-2 by score, so the mock LLM always cites them as "[1] [2]", and
 * self-reflective RAG's relevance grader (which also runs through the mock
 * LLM and gets no parseable JSON back) always falls back to "all relevant"
 * -- i.e. citations (2), RAG pass rate 100%.
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

test("renders the header and an empty-state notebook selector", async ({ page }) => {
  await page.goto("/?mode=mode2");

  await expect(page.getByRole("heading", { name: "Mode 2 — Research Notebook" })).toBeVisible();
  await expect(page.getByLabel("Select a notebook")).toHaveValue("");
  await expect(page.getByLabel("Notebook name")).toHaveValue("");
  await expect(page.getByText("Create or select a notebook on the left to begin.")).toBeVisible();
});

test("warns instead of sending when a fresh notebook has no sources and web search is off", async ({ page }) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("No Sources Notebook"));

  await expect(page.getByText(/No sources yet/)).toBeVisible();
  await page.getByLabel("Message").fill("What does this notebook cover?");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(
    page.getByText(/Add at least one source before asking questions/),
  ).toBeVisible();
});

test("uploads a source and asks a grounded question end-to-end, with citations, RAG reflection, and a follow-up", async ({
  page,
}) => {
  await page.goto("/?mode=mode2");
  await createNotebookViaUi(page, uniqueName("Battery Storage Notebook"));

  await page.getByLabel("Upload sources").setInputFiles(FIXTURE_PATH);
  await expect(page.getByText("battery-storage-briefing.txt")).toBeVisible();

  await page.getByLabel("Message").fill("What happened to battery storage capacity in 2024?");
  await page.getByRole("button", { name: "Send" }).click();

  // The retrieve stage runs two mock-LLM calls (relevance grading + ReAct
  // "should we search again?" reasoning) before answer/save/eval, each with
  // a deliberate ~0.4s delay -- so the running status is reliably observable
  // over real network polling (700ms interval) before "done". The default
  // 5s timeout is too tight here: a cold backend process (first request that
  // touches HybridStore/BM25, or first run alongside other e2e specs sharing
  // this webServer) can comfortably push end-to-end latency past 5s even
  // though the steady-state pipeline finishes in ~2-3s, so this uses the same
  // generous timeout as the mock-answer assertion below.
  await expect(
    page.getByText(
      /Retrieving relevant sources|Composing grounded answer|Saving conversation turn|Evaluating answer quality/,
    ),
  ).toBeVisible({ timeout: 15_000 });

  await expect(page.getByText(/This is a mock grounded notebook answer/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/Failed:/)).toHaveCount(0);

  // Only the 2 battery-storage chunks are ever retrieved/cited, never the
  // 4 decoy chunks (see file header comment for why that's deterministic).
  const citationsToggle = page.getByRole("button", { name: "Sources (2)" });
  await expect(citationsToggle).toBeVisible();
  await citationsToggle.click();
  await expect(page.getByText("[1] battery-storage-briefing.txt")).toBeVisible();
  await expect(page.getByText("[2] battery-storage-briefing.txt")).toBeVisible();
  await expect(page.getByText("· p. 1")).toBeVisible();
  await expect(page.getByText("· p. 2")).toBeVisible();

  await expect(
    page.getByText("Self-Reflective RAG — 2/2 items passed grading (100%)"),
  ).toBeVisible();

  const followup = page.getByRole("button", { name: "What other sources discuss this topic?" });
  await expect(followup).toBeVisible();
  await followup.click();

  await expect(
    page.getByText(/mock grounded notebook answer to "What other sources discuss this topic\?"/),
  ).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/Failed:/)).toHaveCount(0);
});

test("renames and deletes a notebook from the Rename / Delete panel", async ({ page }) => {
  await page.goto("/?mode=mode2");
  const originalName = uniqueName("Rename Me");
  await createNotebookViaUi(page, originalName);

  await page.getByText("Rename / Delete").click();
  const renameInput = page.getByLabel("Notebook name");
  await renameInput.fill(uniqueName("Renamed Notebook"));
  await page.getByRole("button", { name: "Save name" }).click();
  await expect(page.getByRole("heading", { name: originalName })).toHaveCount(0);

  await page.getByRole("button", { name: "Delete notebook" }).click();
  await expect(page.getByText("Create or select a notebook on the left to begin.")).toBeVisible();
});
