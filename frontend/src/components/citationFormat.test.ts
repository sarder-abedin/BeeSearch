import { describe, expect, it } from "vitest";
import type { SourceItem } from "../api/types";
import { citationHeader } from "./citationFormat";

function makeCitation(overrides: Partial<SourceItem> = {}): SourceItem {
  return {
    n: 1,
    kind: "academic",
    title: "Sleep & Memory",
    authors: ["A Smith"],
    year: 2020,
    url: "http://p1",
    snippet: "sleep helps memory",
    apa: "Smith, A. (2020). Sleep & Memory.",
    source: "arxiv",
    ...overrides,
  };
}

describe("citationHeader", () => {
  it("renders a paper badge with year suffix", () => {
    expect(citationHeader(makeCitation())).toBe("[1] 📄 paper — Sleep & Memory (2020)");
  });

  it("renders a web badge with no year suffix", () => {
    const c = makeCitation({ n: 2, kind: "web", year: null, title: "Some Web Page" });
    expect(citationHeader(c)).toBe("[2] 🌐 web — Some Web Page");
  });

  it("omits the year suffix when year is null", () => {
    const c = makeCitation({ year: null });
    expect(citationHeader(c)).toBe("[1] 📄 paper — Sleep & Memory");
  });

  it("truncates titles longer than 80 characters", () => {
    const longTitle = "A".repeat(120);
    const c = makeCitation({ title: longTitle, year: null });
    expect(citationHeader(c)).toBe(`[1] 📄 paper — ${"A".repeat(80)}`);
  });

  it("falls back to Untitled when title is blank", () => {
    const c = makeCitation({ title: "", year: null });
    expect(citationHeader(c)).toBe("[1] 📄 paper — Untitled");
  });
});
