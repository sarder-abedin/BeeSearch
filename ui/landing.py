"""ui/landing.py — Professional landing page with mode selection cards."""
from __future__ import annotations
import streamlit as st

_PROJECTS = [
    {
        "id": "mode1",
        "label": "01",
        "name": "Literature Search",
        "description": "Search arXiv and Semantic Scholar for peer-reviewed papers on any topic. Results are synthesised into a structured report with APA citations.",
        "tags": ["arXiv", "Semantic Scholar", "CrossRef"],
    },
    {
        "id": "mode2",
        "label": "02",
        "name": "ProposalGPT",
        "description": "Upload a funding call (Horizon Europe, NSF, Vinnova…) and generate a complete, publication-quality proposal with 9 AI agents: call analysis, strategy, literature review, 20 proposal sections, budget, compliance check, reviewer simulation, and iterative improvement.",
        "tags": ["9-Agent Pipeline", "Horizon Europe", "Budget Generator", "Reviewer Simulation"],
    },
    {
        "id": "mode3",
        "label": "03",
        "name": "Wisdom Mode",
        "description": "Ask any research or life question and receive validated wisdom synthesised from scientific literature, with confidence ratings and a devil's advocate critique.",
        "tags": ["Evidence synthesis", "Claim validation", "Scientific literature"],
    },
    {
        "id": "mode4",
        "label": "04",
        "name": "Systematic Review",
        "description": "Run a simplified PRISMA systematic review: search, screen by inclusion/exclusion criteria, extract structured evidence, and synthesise findings.",
        "tags": ["PRISMA", "Evidence extraction", "Narrative synthesis"],
    },
    {
        "id": "mode5",
        "label": "05",
        "name": "Research Notebook",
        "description": "A NotebookLM-style workspace: build a notebook from your files and web pages, chat with sources, generate research reports, explain concepts with analogies, and run a full 7-agent analysis pipeline.",
        "tags": ["Grounded Q&A", "Research Report", "Explain", "7-Agent Pipeline"],
    },
    {
        "id": "mode6",
        "label": "06",
        "name": "Grammar Proofreading",
        "description": "Upload or paste any text for professional proofreading — academic papers, professional emails, formal documents, or informal writing. Get a polished rewrite, per-error explanations, style tips, and iterative feedback refinement.",
        "tags": ["Grammar", "Proofreading", "Academic", "Professional Email", "Formal", "Informal"],
    },
]

_CARD_CSS = """
<style>
.mode-card {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem 1.25rem 1rem;
    background: #FFFFFF;
    height: 100%;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.mode-card:hover {
    border-color: #93C5FD;
    box-shadow: 0 2px 12px rgba(37,99,235,0.09);
}
.mode-label {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}
.mode-title {
    font-size: 1rem;
    font-weight: 600;
    color: #0F172A;
    margin: 0.2rem 0 0.5rem;
    line-height: 1.3;
}
.mode-desc {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.55;
    margin-bottom: 0.75rem;
}
.mode-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 0.75rem;
}
.mode-tag {
    background: #F1F5F9;
    color: #475569;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid #E2E8F0;
}
</style>
"""


def _card_html(project: dict) -> str:
    tags_html = "".join(f'<span class="mode-tag">{t}</span>' for t in project["tags"])
    return f"""
<div class="mode-card">
  <div class="mode-label">MODE {project["label"]}</div>
  <div class="mode-title">{project["name"]}</div>
  <div class="mode-desc">{project["description"]}</div>
  <div class="mode-tags">{tags_html}</div>
</div>
"""


def render_landing() -> None:
    """Render the professional landing page with clickable mode cards."""
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    st.markdown(
        "<h1 style='margin-bottom:0.15rem'>Agentic Research Assistant</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#64748B;font-size:0.9rem;margin-bottom:1.5rem'>"
        "Local AI research workflows — Ollama · LangGraph · Hybrid RAG · arXiv · Semantic Scholar"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("##### Select a research mode to get started")
    st.divider()

    rows = [_PROJECTS[:3], _PROJECTS[3:]]
    for row_projects in rows:
        cols = st.columns(len(row_projects), gap="medium")
        for col, project in zip(cols, row_projects):
            with col:
                st.markdown(_card_html(project), unsafe_allow_html=True)
                if st.button(
                    f"Open {project['name']}",
                    key=f"launch_{project['id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["active_project"] = project["id"]
                    st.rerun()
        st.divider()

    st.markdown(
        "<p style='color:#64748B;font-size:0.85rem'>"
        "<strong>Writing Style Profiles</strong> are available below — upload writing samples "
        "to create a named style that shapes AI-generated prose across every mode."
        "</p>",
        unsafe_allow_html=True,
    )
