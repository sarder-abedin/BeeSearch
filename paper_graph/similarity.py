"""paper_graph/similarity.py
────────────────────────────
Pure-function similarity measures for the Connected-Papers–style graph.

References
──────────
  Small, H. (1973). Co-citation in the scientific literature: A new measure
    of the relationship between two documents. JASIS, 24(4), 265–269.
  Kessler, M. M. (1963). Bibliographic coupling between scientific papers.
    American Documentation, 14(1), 10–25.
  Fruchterman, T. M. J., & Reingold, E. M. (1991). Graph drawing by
    force-directed placement. Software: Practice and Experience, 21(11).
    (Layout is handled in the frontend via d3-force / react-force-graph-2d.)

No I/O in this module — all functions take plain Python objects and return
numbers.  This makes them trivially unit-testable without network mocking.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

# ── Tuneable weight constants ─────────────────────────────────────────────────
# Exposed as module-level constants so callers can override them per-request
# without touching defaults here.  The 0.5/0.5 equal split is not justified
# by a specific empirical ratio — it is a neutral starting point.  Adjust via
# the API's bc_weight / cc_weight parameters.

BC_WEIGHT: float = 0.5   # bibliographic coupling weight
CC_WEIGHT: float = 0.5   # co-citation weight


# ── Core measures ─────────────────────────────────────────────────────────────

def bibliographic_coupling(
    origin_refs: FrozenSet[str],
    candidate_refs: FrozenSet[str],
) -> int:
    """Count shared references between the origin paper and a candidate.

    A higher count means both papers draw from the same prior work, which
    is a proxy for topical relatedness (Kessler, 1963).

    Parameters
    ----------
    origin_refs:    frozen set of S2 paperIds cited by the origin paper
    candidate_refs: frozen set of S2 paperIds cited by the candidate
    """
    return len(origin_refs & candidate_refs)


def co_citation(
    origin_id: str,
    candidate_id: str,
    citing_index: Dict[str, FrozenSet[str]],
) -> int:
    """Count how many papers in the corpus cite both origin and candidate.

    A higher count means the two papers are frequently discussed together,
    suggesting topical overlap (Small, 1973).

    Parameters
    ----------
    origin_id:     S2 paperId of the origin paper
    candidate_id:  S2 paperId of the candidate paper
    citing_index:  maps each paper in the candidate pool to the set of S2
                   paperIds that cite it (built from the citation lists
                   fetched during graph construction)
    """
    citing_origin = citing_index.get(origin_id, frozenset())
    citing_candidate = citing_index.get(candidate_id, frozenset())
    return len(citing_origin & citing_candidate)


def combined_score(
    bc: int,
    cc: int,
    bc_max: float,
    cc_max: float,
    bc_weight: float = BC_WEIGHT,
    cc_weight: float = CC_WEIGHT,
) -> float:
    """Weighted, min-max-normalised combination of the two similarity measures.

    Each raw score is normalised to [0, 1] against the maximum value seen
    across all candidates (bc_max / cc_max), then combined with the given
    weights.  If both maxima are zero (no overlap found at all), returns 0.0.

    Parameters
    ----------
    bc, cc:              raw bibliographic-coupling / co-citation counts
    bc_max, cc_max:      maximum values seen across the full candidate pool
                         (used as the normalisation denominator)
    bc_weight, cc_weight: relative weights; need not sum to 1.0, though
                          equal-weight 0.5/0.5 is the default.
    """
    norm_bc = (bc / bc_max) if bc_max > 0 else 0.0
    norm_cc = (cc / cc_max) if cc_max > 0 else 0.0
    return bc_weight * norm_bc + cc_weight * norm_cc


def rank_candidates(
    origin_id: str,
    candidate_ids: List[str],
    refs_index: Dict[str, FrozenSet[str]],
    citing_index: Dict[str, FrozenSet[str]],
    top_n: int = 50,
    bc_weight: float = BC_WEIGHT,
    cc_weight: float = CC_WEIGHT,
) -> List[Tuple[str, float]]:
    """Score and rank a pool of candidate papers against an origin paper.

    Parameters
    ----------
    origin_id:      S2 paperId of the seed / origin paper
    candidate_ids:  pool of S2 paperIds to evaluate (deduplicated, origin excluded)
    refs_index:     maps each paper ID to its frozen set of reference IDs
    citing_index:   maps each paper ID to the set of IDs that cite it
    top_n:          return at most this many candidates, ranked by score
    bc_weight, cc_weight: forwarded to combined_score()

    Returns
    -------
    List of (paper_id, score) tuples, highest score first, length ≤ top_n.
    """
    origin_refs = refs_index.get(origin_id, frozenset())

    # First pass: compute raw scores
    raw: List[Tuple[str, int, int]] = []
    for cid in candidate_ids:
        if cid == origin_id:
            continue
        bc = bibliographic_coupling(origin_refs, refs_index.get(cid, frozenset()))
        cc = co_citation(origin_id, cid, citing_index)
        raw.append((cid, bc, cc))

    if not raw:
        return []

    bc_max = float(max(r[1] for r in raw)) or 1.0
    cc_max = float(max(r[2] for r in raw)) or 1.0

    # Second pass: normalise and combine
    scored: List[Tuple[str, float]] = [
        (cid, combined_score(bc, cc, bc_max, cc_max, bc_weight, cc_weight))
        for cid, bc, cc in raw
    ]

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_n]
