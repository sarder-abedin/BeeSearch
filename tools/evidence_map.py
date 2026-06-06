"""
tools/evidence_map.py
─────────────────────
Evidence map: a 2D bubble chart of Population × Intervention evidence density.

  Bubble size  = number of studies in that cell
  Bubble color = average quality (High=green, Medium=amber, Low=red)

Primary renderer: Plotly (interactive HTML for Streamlit).
Fallback:         Matplotlib PNG bytes when Plotly is unavailable.

If PICO fields are missing from the evidence table, study_design is used
as a proxy for the Intervention axis.
"""

from __future__ import annotations

import io
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_QUALITY_SCORE = {"High": 3.0, "Medium": 2.0, "Low": 1.0}


def _pico_labels(evidence_table: List[Dict]) -> Tuple[List[str], List[str]]:
    """Extract unique Population and Intervention labels (max 10 each)."""
    populations: List[str] = []
    interventions: List[str] = []
    for row in evidence_table:
        pop = row.get("population") or row.get("study_design", "Unknown")
        inter = row.get("intervention") or row.get("study_design", "Unknown")
        if pop not in populations:
            populations.append(pop)
        if inter not in interventions:
            interventions.append(inter)
    return populations[:10], interventions[:10]


def build_evidence_map_data(evidence_table: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate the evidence table into a grid of (population, intervention) cells.

    Returns a dict with lists suitable for both Plotly scatter and Matplotlib scatter.
    """
    if not evidence_table:
        return {
            "x_vals": [], "y_vals": [], "sizes": [], "colors": [],
            "texts": [], "x_labels": [], "y_labels": [],
            "total_cells": 0, "total_studies": 0,
        }

    populations, interventions = _pico_labels(evidence_table)

    cells: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in evidence_table:
        pop = row.get("population") or row.get("study_design", "Unknown")
        inter = row.get("intervention") or row.get("study_design", "Unknown")
        pop = pop if pop in populations else (populations[0] if populations else "Unknown")
        inter = inter if inter in interventions else (interventions[0] if interventions else "Unknown")
        cells[(pop, inter)].append(_QUALITY_SCORE.get(row.get("quality", "Medium"), 2.0))

    x_vals, y_vals, sizes, colors, texts = [], [], [], [], []
    for (pop, inter), scores in cells.items():
        n = len(scores)
        avg_q = sum(scores) / n
        x_vals.append(interventions.index(inter) if inter in interventions else 0)
        y_vals.append(populations.index(pop) if pop in populations else 0)
        sizes.append(n)
        colors.append(avg_q)
        texts.append(f"{pop} × {inter}<br>{n} studies<br>Avg quality: {avg_q:.1f}/3")

    return {
        "x_vals": x_vals,
        "y_vals": y_vals,
        "sizes": sizes,
        "colors": colors,
        "texts": texts,
        "x_labels": interventions,
        "y_labels": populations,
        "total_cells": len(cells),
        "total_studies": len(evidence_table),
    }


def evidence_map_to_plotly_html(map_data: Dict[str, Any]) -> str:
    """Render evidence map as a self-contained Plotly HTML fragment (no full_html)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("pip install plotly")

    x_labels = map_data["x_labels"]
    y_labels = map_data["y_labels"]

    fig = go.Figure(go.Scatter(
        x=map_data["x_vals"],
        y=map_data["y_vals"],
        mode="markers",
        marker=dict(
            size=[max(s * 18, 12) for s in map_data["sizes"]],
            color=map_data["colors"],
            colorscale=[[0, "#e74c3c"], [0.5, "#f39c12"], [1, "#2ecc71"]],
            cmin=1, cmax=3,
            colorbar=dict(
                title="Quality",
                tickvals=[1, 2, 3],
                ticktext=["Low", "Medium", "High"],
            ),
            showscale=True,
            opacity=0.85,
            line=dict(width=1, color="white"),
        ),
        text=map_data["texts"],
        hoverinfo="text",
    ))

    fig.update_layout(
        title="Evidence Map: Population × Intervention",
        xaxis=dict(
            title="Intervention / Study Design",
            tickvals=list(range(len(x_labels))),
            ticktext=x_labels,
            showgrid=True,
            gridcolor="#333333",
        ),
        yaxis=dict(
            title="Population / Study Type",
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
            showgrid=True,
            gridcolor="#333333",
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        height=460,
        margin=dict(l=150, r=20, t=60, b=120),
    )

    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def evidence_map_to_png(map_data: Dict[str, Any]) -> bytes:
    """Render the evidence map as a PNG (matplotlib fallback)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1a2e")

    sc = ax.scatter(
        map_data["x_vals"],
        map_data["y_vals"],
        s=[max(n * 200, 100) for n in map_data["sizes"]],
        c=map_data["colors"],
        cmap="RdYlGn",
        vmin=1, vmax=3,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )

    ax.set_xticks(range(len(map_data["x_labels"])))
    ax.set_xticklabels(map_data["x_labels"], rotation=30, ha="right", color="white", fontsize=8)
    ax.set_yticks(range(len(map_data["y_labels"])))
    ax.set_yticklabels(map_data["y_labels"], color="white", fontsize=8)
    ax.set_title("Evidence Map: Population × Intervention", color="white")
    ax.set_xlabel("Intervention / Study Design", color="white")
    ax.set_ylabel("Population / Study Type", color="white")
    ax.tick_params(colors="white")
    plt.colorbar(sc, ax=ax, label="Avg Quality (1=Low, 2=Medium, 3=High)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
