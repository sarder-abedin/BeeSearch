"""
ui/theme.py
───────────
Global CSS theme injection for the Agentic Research Assistant.

Call apply_theme() once at app startup (top of app.py).
"""
from __future__ import annotations
import streamlit as st

_CSS = """
<style>
/* ── Base typography ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}

h1 { font-size: 1.75rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em; }
h2 { font-size: 1.35rem; font-weight: 600; color: #1E293B; letter-spacing: -0.01em; }
h3 { font-size: 1.1rem;  font-weight: 600; color: #334155; }

/* ── Force a light app background + dark body text ───────────────────── */
/* Always pin light/dark regardless of the visitor's OS/browser dark-mode
   preference — Streamlit otherwise auto-detects and falls back to its
   near-black dark theme for the main content area (the .streamlit/config.toml
   [theme] block is the primary fix; these rules are defense-in-depth so the
   app still looks right even if that config is ever overridden upstream). */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stBottomBlockContainer"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0) !important;
}
.stApp p, .stApp span, .stApp label, .stApp li,
.stApp .stMarkdown, [data-testid="stMarkdownContainer"] {
    color: #1E293B;
}

/* ── Main content area ──────────────────────────────────────────────── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #F8FAFC !important;
    border-right: 1px solid #E2E8F0;
}
/* Force all sidebar text to be dark — prevents white-on-light issues */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([class*="badge"]),
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div:not([class*="badge"]),
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stText"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #1E293B !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #0F172A !important;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
/* Sidebar select/slider labels */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #334155 !important;
    font-size: 0.85rem !important;
}
/* Sidebar expander text */
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: #1E293B !important;
}
/* Sidebar caption / helper text */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #64748B !important;
}
/* Metric labels in sidebar */
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #64748B !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #0F172A !important;
}
/* Sidebar slider value */
[data-testid="stSidebar"] [data-testid="stSlider"] p {
    color: #334155 !important;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
[data-testid="baseButton-primary"] {
    background-color: #B45309 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    transition: background-color 0.15s ease !important;
}
[data-testid="baseButton-primary"]:hover {
    background-color: #92400E !important;
    color: #FFFFFF !important;
}
[data-testid="baseButton-secondary"] {
    background-color: transparent !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    color: #334155 !important;
    font-weight: 500 !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: #94A3B8 !important;
    background-color: #F8FAFC !important;
}

/* ── Metric widgets ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #64748B !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

/* ── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
}
[data-testid="stExpander"] summary {
    font-weight: 500;
    color: #1E293B;
}

/* ── Tabs ───────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 500;
    font-size: 0.875rem;
    color: #64748B;
    padding: 0.5rem 1rem;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #B45309;
    border-bottom-color: #B45309;
}

/* ── Alerts / info boxes ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    border-left-width: 3px !important;
}

/* ── Input widgets ──────────────────────────────────────────────────── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    border-radius: 6px !important;
    border-color: #CBD5E1 !important;
    font-size: 0.9rem !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #B45309 !important;
    box-shadow: 0 0 0 2px rgba(180,83,9,0.12) !important;
}

/* ── Select boxes ───────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 6px !important;
    border-color: #CBD5E1 !important;
}

/* ── Containers with border ─────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

/* ── Progress bar ───────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background-color: #B45309 !important;
    border-radius: 4px !important;
}

/* ── Divider ────────────────────────────────────────────────────────── */
hr {
    border-color: #E2E8F0 !important;
    margin: 1.25rem 0 !important;
}

/* ── Code blocks ────────────────────────────────────────────────────── */
code {
    background: #F1F5F9 !important;
    color: #1E293B !important;
    border-radius: 4px !important;
    font-size: 0.85em !important;
    padding: 0.15em 0.35em !important;
}

/* ── Caption / small text ───────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    color: #64748B !important;
    font-size: 0.8rem !important;
}

/* ── Chat messages ──────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 8px !important;
    border: 1px solid #E2E8F0 !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
    background: #FFFFFF !important;
}

/* ── Download buttons ───────────────────────────────────────────────── */
[data-testid="baseButton-secondary"][kind="secondary"] {
    font-size: 0.85rem !important;
}

/* ── Quality badge helpers (used in st.markdown) ─────────────────────── */
.badge-high   { background:#DCFCE7; color:#166534; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; }
.badge-medium { background:#FEF9C3; color:#854D0E; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; }
.badge-low    { background:#FEE2E2; color:#991B1B; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; }
.badge-info   { background:#E2E8F0; color:#334155; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; }

/* ── Mode cards on landing page ─────────────────────────────────────── */
.mode-number {
    display: inline-block;
    background: #B45309;
    color: #FFFFFF;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 2px 7px;
    border-radius: 4px;
    margin-bottom: 0.4rem;
}
</style>
"""


def apply_theme() -> None:
    """Inject global professional CSS into the Streamlit app."""
    st.markdown(_CSS, unsafe_allow_html=True)


def quality_badge(level: str) -> str:
    """Return an HTML badge string for a quality level (High/Medium/Low)."""
    mapping = {
        "High":   ('<span class="badge-high">High</span>', "High"),
        "Medium": ('<span class="badge-medium">Medium</span>', "Medium"),
        "Low":    ('<span class="badge-low">Low</span>', "Low"),
    }
    html, _ = mapping.get(level, ('<span class="badge-info">Unknown</span>', "Unknown"))
    return html


def score_badge(score: int) -> str:
    """Return an HTML badge for a 1–5 integer score."""
    if score >= 4:
        cls = "badge-high"
    elif score >= 3:
        cls = "badge-medium"
    else:
        cls = "badge-low"
    return f'<span class="{cls}">{score}/5</span>'
