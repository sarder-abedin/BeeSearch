"""
tools/meta_analysis.py
──────────────────────
Statistical meta-analysis: pool effect sizes across studies and visualise
them as a forest plot — the standard "combine the evidence" step that
follows a systematic review's evidence extraction.

Supported effect measures
  • Ratio measures   — Odds Ratio (OR), Risk Ratio (RR), Hazard Ratio (HR)
                       pooled in log space (their sampling distribution is
                       approximately normal on the log scale); null = 1
  • Difference measures — Mean Difference (MD), Standardized Mean
                       Difference (SMD), Risk Difference (RD) pooled on
                       their native linear scale; null = 0

Each study contributes its published point estimate + 95% confidence
interval (the "generic inverse-variance" approach — the only inputs an
abstract reliably reports). From these we derive a per-study estimate (yi)
and variance (vi), then pool with:

  • Fixed-effect   — classic inverse-variance weighting (Σ wi·yi / Σ wi)
  • Random-effects — DerSimonian–Laird (adds between-study variance τ² to
                     each study's weight, so noisier evidence pools more
                     conservatively)

Heterogeneity is summarised with Cochran's Q, I², and τ² — pure NumPy/
stdlib (no scipy/statsmodels: they aren't in requirements.txt), matching
the dependency-light approach used throughout tools/concept_drift.py.

Rendering follows tools/evidence_map.py's pattern: Plotly HTML primary,
Matplotlib PNG fallback.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

# z-score for a 95% confidence interval (two-tailed, α = 0.05)
Z_95 = 1.959963984540054

RATIO_MEASURES = {"OR", "RR", "HR"}
DIFFERENCE_MEASURES = {"MD", "SMD", "RD"}

MEASURE_LABELS: Dict[str, str] = {
    "OR": "Odds Ratio (OR)",
    "RR": "Risk Ratio / Relative Risk (RR)",
    "HR": "Hazard Ratio (HR)",
    "MD": "Mean Difference (MD)",
    "SMD": "Standardized Mean Difference (SMD)",
    "RD": "Risk Difference (RD)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Scale conversions
# ─────────────────────────────────────────────────────────────────────────────

def _to_estimate_and_variance(effect: float, ci_low: float, ci_high: float, measure: str) -> tuple[float, float]:
    """
    Convert a study's reported effect + 95% CI into (yi, vi) on the scale
    the pooling math operates on.

    Ratio measures are log-transformed first (a CI that's symmetric on the
    ratio scale is symmetric on the log scale, which is what makes inverse-
    variance pooling valid for them); difference measures are used as-is.
    """
    if measure in RATIO_MEASURES:
        if effect <= 0 or ci_low <= 0 or ci_high <= 0:
            raise ValueError("ratio measures must be positive")
        yi = math.log(effect)
        se = (math.log(ci_high) - math.log(ci_low)) / (2 * Z_95)
    else:
        yi = effect
        se = (ci_high - ci_low) / (2 * Z_95)
    if se <= 0:
        raise ValueError("confidence interval requires ci_high > ci_low")
    return yi, se * se


def _back_transform(value: float, measure: str) -> float:
    """Undo the log transform for ratio measures so results display on their native scale."""
    return math.exp(value) if measure in RATIO_MEASURES else value


def _interpret_i_squared(i2: float) -> str:
    """Rough-guide interpretation following the commonly cited Cochrane thresholds."""
    if i2 < 25:
        return "Low heterogeneity — studies broadly agree."
    if i2 < 50:
        return "Moderate heterogeneity — some disagreement between studies."
    if i2 < 75:
        return "Substantial heterogeneity — interpret the pooled estimate with care."
    return "Considerable heterogeneity — studies disagree a lot; pooling may not be meaningful."


# ─────────────────────────────────────────────────────────────────────────────
# Pooling
# ─────────────────────────────────────────────────────────────────────────────

def run_meta_analysis(studies: List[Dict[str, Any]], measure: str = "OR") -> Dict[str, Any]:
    """
    Pool per-study effect sizes with inverse-variance weighting and report
    fixed-effect + random-effects (DerSimonian–Laird) estimates plus
    heterogeneity statistics.

    Parameters
    ----------
    studies : list of {label, effect, ci_low, ci_high, n (optional)}
        Each study's point estimate and 95% CI on the *native* scale
        (e.g. an odds ratio of 1.45 with CI 1.02–2.06 — exactly what a
        results section or abstract typically reports).
    measure : "OR" | "RR" | "HR" | "MD" | "SMD" | "RD"
        The effect type every row in `studies` is expressed in.

    Returns
    -------
    On success: {ok: True, measure, measure_label, null_value, log_scale, k,
                 studies: [...original rows plus yi/vi/weight_fe_pct/weight_re_pct...],
                 fixed_effect: {estimate, ci_low, ci_high, ...},
                 random_effects: {estimate, ci_low, ci_high, ...},
                 heterogeneity: {q, df, i_squared, tau_squared, interpretation}}
    On failure (fewer than 2 usable studies): {ok: False, reason, studies}
    """
    measure = (measure or "OR").upper()
    if measure not in MEASURE_LABELS:
        measure = "OR"

    rows: List[Dict[str, Any]] = []
    for s in studies:
        raw_label = s.get("label")
        label = raw_label.strip() if isinstance(raw_label, str) and raw_label.strip() else f"Study {len(rows) + 1}"
        try:
            effect = float(s["effect"])
            ci_low = float(s["ci_low"])
            ci_high = float(s["ci_high"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (ci_low <= effect <= ci_high):
            continue  # malformed: a correctly reported CI always brackets its point estimate
        try:
            yi, vi = _to_estimate_and_variance(effect, ci_low, ci_high, measure)
        except ValueError:
            continue
        rows.append({
            "label": label,
            "n": s.get("n"),
            "effect": effect, "ci_low": ci_low, "ci_high": ci_high,
            "yi": yi, "vi": vi,
        })

    if len(rows) < 2:
        return {
            "ok": False,
            "reason": "Need at least 2 studies with a valid effect size + 95% CI to pool.",
            "studies": rows,
        }

    yi = np.array([r["yi"] for r in rows], dtype=float)
    vi = np.array([r["vi"] for r in rows], dtype=float)
    k = len(rows)

    # ── Fixed-effect (inverse-variance) pooled estimate ──────────────────
    wi_fe = 1.0 / vi
    sum_w_fe = float(np.sum(wi_fe))
    pooled_fe_y = float(np.sum(wi_fe * yi) / sum_w_fe)
    pooled_fe_v = 1.0 / sum_w_fe

    # ── Heterogeneity: Cochran's Q, I², DerSimonian–Laird τ² ─────────────
    q = float(np.sum(wi_fe * (yi - pooled_fe_y) ** 2))
    df = k - 1
    i_squared = max(0.0, (q - df) / q * 100.0) if q > 0 else 0.0
    c = sum_w_fe - float(np.sum(wi_fe ** 2)) / sum_w_fe
    tau_squared = max(0.0, (q - df) / c) if c > 0 else 0.0

    # ── Random-effects (DerSimonian–Laird) pooled estimate ───────────────
    wi_re = 1.0 / (vi + tau_squared)
    sum_w_re = float(np.sum(wi_re))
    pooled_re_y = float(np.sum(wi_re * yi) / sum_w_re)
    pooled_re_v = 1.0 / sum_w_re

    for i, r in enumerate(rows):
        r["weight_fe_pct"] = 100.0 * float(wi_fe[i]) / sum_w_fe
        r["weight_re_pct"] = 100.0 * float(wi_re[i]) / sum_w_re
        # NOTE: plots show the study's *reported* effect/CI (r["effect"], r["ci_low"],
        # r["ci_high"]) verbatim — not a back-transform of yi — so users always see
        # exactly the numbers they entered, even when a reported CI is slightly
        # asymmetric in log space (yi's symmetric SE is an internal pooling device).

    def _summary(y: float, v: float) -> Dict[str, float]:
        se = math.sqrt(v)
        return {
            "estimate": _back_transform(y, measure),
            "ci_low": _back_transform(y - Z_95 * se, measure),
            "ci_high": _back_transform(y + Z_95 * se, measure),
            "log_estimate": y,
            "se": se,
        }

    return {
        "ok": True,
        "measure": measure,
        "measure_label": MEASURE_LABELS[measure],
        "null_value": 1.0 if measure in RATIO_MEASURES else 0.0,
        "log_scale": measure in RATIO_MEASURES,
        "k": k,
        "studies": rows,
        "fixed_effect": _summary(pooled_fe_y, pooled_fe_v),
        "random_effects": _summary(pooled_re_y, pooled_re_v),
        "heterogeneity": {
            "q": q,
            "df": df,
            "i_squared": i_squared,
            "tau_squared": tau_squared,
            "interpretation": _interpret_i_squared(i_squared),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM-assisted extraction (best-effort drafts — humans review before pooling)
# ─────────────────────────────────────────────────────────────────────────────

def _llm(model_name: str, num_ctx: int) -> ChatOllama:
    import httpx
    return ChatOllama(
        model=model_name or cfg.ollama_model, base_url=cfg.ollama_base_url,
        temperature=0.1, num_predict=320, num_ctx=num_ctx or cfg.num_ctx,
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
    )


def extract_effect_size_row(paper: Dict[str, Any], measure: str, model_name: str, num_ctx: int) -> Dict[str, Any]:
    """
    Best-effort extraction of one study's overall effect size + 95% CI for
    `measure` from its title/abstract.

    Abstracts often omit exact statistics, so this is explicitly a *draft*:
    returns found=False (numeric fields = None) rather than guessing when
    nothing usable is reported. Callers should present results in an
    editable table for human review, never feed them straight into pooling.
    """
    label_authors = paper.get("authors", [])
    author_str = label_authors[0].split(",")[0] if label_authors else "Unknown"
    year = paper.get("year", "n.d.")
    label = f"{author_str} et al. ({year})" if len(label_authors) > 1 else f"{author_str} ({year})"

    measure_label = MEASURE_LABELS.get(measure.upper(), measure)
    try:
        raw = _llm(model_name, num_ctx).invoke([
            SystemMessage(content=(
                f"You extract quantitative results from research abstracts for a meta-analysis. "
                f"The target effect measure is {measure_label}. Find the ONE overall {measure} "
                f"value with its 95% confidence interval reported for the primary outcome. "
                f"If the abstract doesn't report a usable {measure} + CI, say so honestly — "
                f"never invent or estimate numbers.\n"
                f'Return ONLY valid JSON: {{"found": true/false, "effect": <number or null>, '
                f'"ci_low": <number or null>, "ci_high": <number or null>, "n": <integer or null>}}'
            )),
            HumanMessage(content=f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')[:1000]}"),
        ]).content.strip()
    except Exception as e:
        logger.warning("Effect-size extraction failed for '%s': %s", paper.get("title", "")[:40], e)
        raw = ""

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
    except Exception:
        parsed = {}

    def _num(key: str) -> Optional[float]:
        v = parsed.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    found = bool(parsed.get("found")) and all(_num(k) is not None for k in ("effect", "ci_low", "ci_high"))
    n_val = parsed.get("n")
    return {
        "label": label,
        "effect": _num("effect") if found else None,
        "ci_low": _num("ci_low") if found else None,
        "ci_high": _num("ci_high") if found else None,
        "n": int(n_val) if isinstance(n_val, (int, float)) else None,
        "found": found,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Forest plot rendering — Plotly primary, Matplotlib PNG fallback
# ─────────────────────────────────────────────────────────────────────────────

def _forest_rows(result: Dict[str, Any], model: str) -> tuple[list, dict, str, str]:
    studies = result["studies"]
    weight_key = "weight_re_pct" if model == "random" else "weight_fe_pct"
    pooled = result["random_effects"] if model == "random" else result["fixed_effect"]
    pooled_label = "Pooled — Random-Effects" if model == "random" else "Pooled — Fixed-Effect"
    return studies, pooled, pooled_label, weight_key


def meta_analysis_to_forest_plotly(result: Dict[str, Any], model: str = "random") -> str:
    """Render a forest plot as a self-contained Plotly HTML fragment (no full_html)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("pip install plotly")

    studies, pooled, pooled_label, weight_key = _forest_rows(result, model)
    measure = result["measure"]
    labels = [s["label"] for s in studies] + [pooled_label]
    y_pos = list(range(len(labels)))[::-1]
    study_y, pooled_y = y_pos[:-1], y_pos[-1]

    fig = go.Figure()

    for y, s in zip(study_y, studies):
        fig.add_trace(go.Scatter(
            x=[s["ci_low"], s["ci_high"]], y=[y, y],
            mode="lines", line=dict(color="#5dade2", width=2),
            hoverinfo="skip", showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[s["effect"] for s in studies], y=study_y,
        mode="markers",
        marker=dict(
            symbol="square",
            size=[max(8, min(30, s[weight_key] * 0.7)) for s in studies],
            color="#2e86c1", line=dict(width=1, color="white"),
        ),
        text=[
            f"{s['label']}<br>{measure} = {s['effect']:.2f} "
            f"[{s['ci_low']:.2f}, {s['ci_high']:.2f}]<br>"
            f"Weight: {s[weight_key]:.1f}%"
            for s in studies
        ],
        hoverinfo="text", name="Studies", showlegend=False,
    ))

    half_h = 0.22
    fig.add_trace(go.Scatter(
        x=[pooled["ci_low"], pooled["estimate"], pooled["ci_high"], pooled["estimate"], pooled["ci_low"]],
        y=[pooled_y, pooled_y + half_h, pooled_y, pooled_y - half_h, pooled_y],
        mode="lines", fill="toself", fillcolor="#e67e22", line=dict(color="#e67e22"),
        hoverinfo="text",
        text=f"{pooled_label}<br>{measure} = {pooled['estimate']:.2f} [{pooled['ci_low']:.2f}, {pooled['ci_high']:.2f}]",
        name=pooled_label, showlegend=False,
    ))

    fig.add_vline(x=result["null_value"], line=dict(color="#aaaaaa", dash="dash", width=1))

    xaxis: Dict[str, Any] = dict(title=f"{measure} (95% CI)", gridcolor="#333333", zeroline=False)
    if result["log_scale"]:
        xaxis["type"] = "log"

    fig.update_layout(
        title=f"Forest Plot — {result['measure_label']}",
        xaxis=xaxis,
        yaxis=dict(tickvals=y_pos, ticktext=labels, showgrid=False, zeroline=False, range=[-1, len(labels)]),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        height=130 + 42 * len(labels),
        margin=dict(l=10, r=20, t=60, b=60),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def meta_analysis_to_forest_png(result: Dict[str, Any], model: str = "random") -> bytes:
    """Render the forest plot as a PNG (matplotlib fallback)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib")

    studies, pooled, pooled_label, weight_key = _forest_rows(result, model)
    measure = result["measure"]
    labels = [s["label"] for s in studies] + [pooled_label]
    n = len(labels)
    y_pos = list(range(n))[::-1]
    study_y, pooled_y = y_pos[:-1], y_pos[-1]

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.5 * n))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    for y, s in zip(study_y, studies):
        ax.plot([s["ci_low"], s["ci_high"]], [y, y],
                color="#5dade2", linewidth=1.6, zorder=2)
        size = max(30, min(260, s[weight_key] * 9))
        ax.scatter([s["effect"]], [y], s=size, marker="s", color="#2e86c1",
                   edgecolors="white", linewidths=0.6, zorder=3)

    half_h = 0.22
    diamond = plt.Polygon(
        [(pooled["ci_low"], pooled_y), (pooled["estimate"], pooled_y + half_h),
         (pooled["ci_high"], pooled_y), (pooled["estimate"], pooled_y - half_h)],
        closed=True, facecolor="#e67e22", edgecolor="#e67e22", zorder=3,
    )
    ax.add_patch(diamond)
    ax.axvline(result["null_value"], color="#aaaaaa", linestyle="--", linewidth=1, zorder=1)
    if result["log_scale"]:
        ax.set_xscale("log")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.set_xlabel(f"{measure} (95% CI)", color="white")
    ax.set_title(f"Forest Plot — {result['measure_label']}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_ylim(-1, n)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
