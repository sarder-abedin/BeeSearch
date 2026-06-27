import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { CitationItem } from "../api/notebookTypes";
import NotebookCitations from "./NotebookCitations";

const DOC_CITATION: CitationItem = {
  n: 1,
  doc_name: "climate-paper.pdf",
  page: 3,
  page_label: "p. 3",
  snippet: "Average temperatures rose by 1.5°C.",
  url: "",
};

const WEB_CITATION: CitationItem = {
  n: 2,
  doc_name: "NOAA Climate Report",
  page: 0,
  page_label: "",
  snippet: "",
  url: "https://example.org/noaa-report",
};

describe("NotebookCitations", () => {
  it("renders nothing when there are no citations", () => {
    const { container } = render(<NotebookCitations citations={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the toggle but hides citation details until expanded", () => {
    render(<NotebookCitations citations={[DOC_CITATION]} />);

    expect(screen.getByText("Sources (1)")).toBeInTheDocument();
    expect(screen.queryByText(/climate-paper\.pdf/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Average temperatures rose/)).not.toBeInTheDocument();
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "false");
  });

  it("reveals doc-based citation details on click; hides them again on a second click", async () => {
    const user = userEvent.setup();
    render(<NotebookCitations citations={[DOC_CITATION]} />);

    const toggle = screen.getByRole("button");
    await user.click(toggle);

    expect(screen.getByText("[1] climate-paper.pdf")).toBeInTheDocument();
    expect(screen.getByText("· p. 3")).toBeInTheDocument();
    expect(screen.getByText("Average temperatures rose by 1.5°C.")).toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-expanded", "true");

    await user.click(toggle);
    expect(screen.queryByText("[1] climate-paper.pdf")).not.toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-expanded", "false");
  });

  it("renders a link (not a page label) when the citation has a url", async () => {
    const user = userEvent.setup();
    render(<NotebookCitations citations={[WEB_CITATION]} />);

    await user.click(screen.getByRole("button"));

    const link = screen.getByRole("link", { name: "NOAA Climate Report" });
    expect(link).toHaveAttribute("href", "https://example.org/noaa-report");
    expect(screen.getByText("[2]")).toBeInTheDocument();
    expect(screen.queryByText(/p\. /)).not.toBeInTheDocument();
  });

  it("renders multiple citations under a single toggle with the correct count", async () => {
    const user = userEvent.setup();
    render(<NotebookCitations citations={[DOC_CITATION, WEB_CITATION]} />);

    expect(screen.getByText("Sources (2)")).toBeInTheDocument();

    await user.click(screen.getByRole("button"));

    expect(screen.getByText("[1] climate-paper.pdf")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "NOAA Climate Report" })).toBeInTheDocument();
  });
});
