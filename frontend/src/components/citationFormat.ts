import type { SourceItem } from "../api/types";

/** Mirrors ui/tabs/research_assistant.py::_render_citation's header f-string exactly. */
export function citationHeader(citation: SourceItem): string {
  const badge = citation.kind === "web" ? "🌐 web" : "📄 paper";
  const title = citation.title || "Untitled";
  const yearSuffix = citation.year ? ` (${citation.year})` : "";
  return `[${citation.n}] ${badge} — ${title.slice(0, 80)}${yearSuffix}`;
}
