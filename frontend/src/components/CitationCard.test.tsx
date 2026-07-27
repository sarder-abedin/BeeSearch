import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { SourceItem } from "../api/types";
import CitationCard from "./CitationCard";

const fullCitation: SourceItem = {
  n: 1,
  kind: "academic",
  title: "Sleep & Memory",
  authors: ["A Smith"],
  year: 2020,
  url: "http://example.org/p1",
  snippet: "sleep helps memory",
  apa: "Smith, A. (2020). Sleep & Memory.",
  source: "arxiv",
};

describe("CitationCard", () => {
  it("shows the header but hides snippet/apa/link until expanded", () => {
    render(<CitationCard citation={fullCitation} />);

    expect(screen.getByText("[1] 📄 paper — Sleep & Memory (2020)")).toBeInTheDocument();
    expect(screen.queryByText("sleep helps memory")).not.toBeInTheDocument();
    expect(screen.queryByText(fullCitation.apa)).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Open source" })).not.toBeInTheDocument();
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "false");
  });

  it("reveals snippet, APA caption, and link on click; hides them again on a second click", async () => {
    const user = userEvent.setup();
    render(<CitationCard citation={fullCitation} />);

    await user.click(screen.getByRole("button"));

    expect(screen.getByText("sleep helps memory")).toBeInTheDocument();
    expect(screen.getByText(fullCitation.apa)).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "Open source" });
    expect(link).toHaveAttribute("href", "http://example.org/p1");
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "true");

    await user.click(screen.getByRole("button"));
    expect(screen.queryByText("sleep helps memory")).not.toBeInTheDocument();
  });

  it("omits snippet/apa/link sections that are empty", async () => {
    const user = userEvent.setup();
    const sparse: SourceItem = {
      ...fullCitation,
      kind: "web",
      snippet: "",
      apa: "",
      url: "",
    };
    render(<CitationCard citation={sparse} />);

    await user.click(screen.getByRole("button"));

    expect(screen.queryByRole("link", { name: "Open source" })).not.toBeInTheDocument();
    expect(screen.getByText("[1] 🌐 web — Sleep & Memory (2020)")).toBeInTheDocument();
  });
});
