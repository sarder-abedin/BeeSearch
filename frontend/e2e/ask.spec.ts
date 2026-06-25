import { expect, test } from "@playwright/test";

/**
 * Runs against the REAL FastAPI backend (with BEESEARCH_MOCK_LLM=1, see
 * playwright.config.ts's webServer) and the production Vite build (`vite
 * preview`), so this exercises the full React -> HTTP -> FastAPI -> agents
 * pipeline -- no mocked fetch, no fake timers, just the dev-only mock LLM
 * (backend/app/mock_llm.py) and mock search (backend/app/mock_search.py)
 * standing in for Ollama and the real search backends.
 */

test("renders the initial controls", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Mode 3 — AI Research Assistant" })).toBeVisible();
  await expect(page.getByLabel("Research question")).toHaveValue("");
  await expect(page.getByRole("checkbox", { name: "Also search the web (DuckDuckGo)" })).toBeChecked();
  await expect(page.getByRole("button", { name: "Ask" })).toBeEnabled();
});

test("shows a validation warning for a blank question without contacting the backend", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.getByText("Please enter a research question.")).toBeVisible();

  // Typing clears the warning immediately (matches the Streamlit original's
  // rerun-driven behavior -- see AskPage.tsx's onChange handlers).
  await page.getByLabel("Research question").fill("a");
  await expect(page.getByText("Please enter a research question.")).toHaveCount(0);
});

test("asks a question end-to-end and renders a grounded answer with citations", async ({ page }) => {
  await page.goto("/");

  await page.getByLabel("Research question").fill("Does sleep help memory?");
  await page.getByRole("button", { name: "Ask" }).click();

  // The mock search has a deliberate ~0.5-0.8s delay so the real polled
  // "searching" stage text is observable over real network polling, not
  // just in fake-timer unit tests.
  await expect(
    page.getByText("Searching Google Scholar · arXiv · Semantic Scholar · web…"),
  ).toBeVisible();

  await expect(page.getByText(/This is a mock grounded answer/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("heading", { name: "Citations (2)" })).toBeVisible();
  await expect(page.getByText("[1] 📄 paper — A Survey of Does sleep help memory (2023)")).toBeVisible();
  await expect(
    page.getByText("Searched 2 paper(s) and 1 web result(s); 3 used as context, 2 cited in the answer."),
  ).toBeVisible();
  await expect(page.getByText(/No published sources could be retrieved/)).toHaveCount(0);

  // Citation cards start collapsed and reveal snippet/APA/link on click.
  const firstCitation = page.getByRole("button", { name: /A Survey of Does sleep help memory/ });
  await expect(firstCitation).toHaveAttribute("aria-expanded", "false");
  await firstCitation.click();
  await expect(firstCitation).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByRole("link", { name: "Open source" }).first()).toHaveAttribute(
    "href",
    "https://example.org/mock-paper-1",
  );

  // Suggested follow-up questions render as clickable buttons.
  await expect(
    page.getByRole("button", { name: "What do other studies find on this topic?" }),
  ).toBeVisible();
});

test("forwards include_web=false and never shows a Failed status", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("checkbox", { name: "Also search the web (DuckDuckGo)" }).uncheck();
  await page.getByLabel("Research question").fill("How does caffeine affect sleep?");
  await page.getByRole("button", { name: "Ask" }).click();

  await expect(page.getByText(/This is a mock grounded answer/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/Failed:/)).toHaveCount(0);
  // include_web=false -> no web source in the mix, just the two mock papers.
  await expect(
    page.getByText("Searched 2 paper(s) and 0 web result(s); 2 used as context, 2 cited in the answer."),
  ).toBeVisible();
});

test("clicking a follow-up question re-asks immediately with the new question", async ({ page }) => {
  await page.goto("/");

  await page.getByLabel("Research question").fill("Does sleep help memory?");
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.getByText(/This is a mock grounded answer/)).toBeVisible({ timeout: 15_000 });

  const followup = page.getByRole("button", { name: "What do other studies find on this topic?" });
  await followup.click();

  await expect(page.getByLabel("Research question")).toHaveValue(
    "What do other studies find on this topic?",
  );
  await expect(page.getByText(/This is a mock grounded answer/)).toBeVisible({ timeout: 15_000 });
});
