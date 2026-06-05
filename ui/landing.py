"""ui/landing.py — ResearchBuddy landing page with two mode cards."""
from __future__ import annotations
import streamlit as st

_PROJECTS = [
    {
        "id": "systematic_review",
        "label": "01",
        "name": "Systematic Review",
        "description": "Run a simplified PRISMA systematic review: search, screen by inclusion/exclusion criteria, extract structured evidence, and synthesise findings with narrative synthesis.",
        "tags": ["PRISMA", "Evidence extraction", "Narrative synthesis", "Quality scoring"],
    },
    {
        "id": "notebook",
        "label": "02",
        "name": "Research Notebook",
        "description": "A NotebookLM-style workspace: build a notebook from your files and web pages, chat with sources, generate research reports, explain concepts with analogies, and run a full 7-agent analysis pipeline.",
        "tags": ["Grounded Q&A", "Research Report", "Explain", "7-Agent Pipeline"],
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
    """Render the ResearchBuddy landing page with two mode cards."""
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    st.markdown(
        "<h1 style='margin-bottom:0.15rem'>ResearchBuddy</h1>",
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

    cols = st.columns(len(_PROJECTS), gap="medium")
    for col, project in zip(cols, _PROJECTS):
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
