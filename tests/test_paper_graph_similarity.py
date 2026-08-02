"""tests/test_paper_graph_similarity.py
──────────────────────────────────────
Unit tests for paper_graph/similarity.py.

The citation graph is hand-constructed so every expected count can be
verified by inspection:

  Papers: A (origin), B, C, D

  References (what each paper cites):
    A → {X, Y, Z}      (X, Y, Z are papers outside the candidate pool)
    B → {X, Y, W}      (W is also outside the candidate pool)
    C → {X, V}         (V is outside the candidate pool)
    D → {P, Q}         (no overlap with A)

  Citations (who cites each paper in our pool):
    A is cited by: {B, C}     (B and C both cite A... wait, A is origin not in candidate pool)

  Let's set up the citing_index for co-citation:
    For co_citation(A, B): count papers that cite BOTH A and B.
      Papers citing A: {M, N}   (M and N are outside the pool, just index entries)
      Papers citing B: {M, O}
      Intersection: {M} → co_citation = 1

    For co_citation(A, C):
      Papers citing A: {M, N}
      Papers citing C: {N, O}
      Intersection: {N} → co_citation = 1

    For co_citation(A, D):
      Papers citing A: {M, N}
      Papers citing D: {P, Q}
      Intersection: {} → co_citation = 0

  Bibliographic coupling (shared references with A = {X, Y, Z}):
    bc(A, B) = |{X,Y,Z} ∩ {X,Y,W}| = |{X,Y}| = 2
    bc(A, C) = |{X,Y,Z} ∩ {X,V}|   = |{X}|   = 1
    bc(A, D) = |{X,Y,Z} ∩ {P,Q}|   = |{}|    = 0

  Expected ranking (bc_weight=0.5, cc_weight=0.5):
    bc_max = 2 (paper B), cc_max = 1 (papers B and C both at 1)
    score(B) = 0.5*(2/2) + 0.5*(1/1) = 0.5 + 0.5 = 1.0
    score(C) = 0.5*(1/2) + 0.5*(1/1) = 0.25 + 0.5 = 0.75
    score(D) = 0.5*(0/2) + 0.5*(0/1) = 0.0

  So: B > C > D.
"""

from __future__ import annotations

import pytest

from paper_graph.similarity import (
    BC_WEIGHT,
    CC_WEIGHT,
    bibliographic_coupling,
    co_citation,
    combined_score,
    rank_candidates,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

ORIGIN_ID = "A"
CANDIDATE_IDS = ["B", "C", "D"]

REFS_INDEX = {
    "A": frozenset(["X", "Y", "Z"]),
    "B": frozenset(["X", "Y", "W"]),
    "C": frozenset(["X", "V"]),
    "D": frozenset(["P", "Q"]),
}

CITING_INDEX = {
    "A": frozenset(["M", "N"]),
    "B": frozenset(["M", "O"]),
    "C": frozenset(["N", "O"]),
    "D": frozenset(["P", "Q"]),
}


# ── bibliographic_coupling ────────────────────────────────────────────────────

def test_bc_b_has_two_shared_refs():
    bc = bibliographic_coupling(REFS_INDEX["A"], REFS_INDEX["B"])
    assert bc == 2, f"Expected 2 shared refs between A and B, got {bc}"


def test_bc_c_has_one_shared_ref():
    bc = bibliographic_coupling(REFS_INDEX["A"], REFS_INDEX["C"])
    assert bc == 1, f"Expected 1 shared ref between A and C, got {bc}"


def test_bc_d_has_no_shared_refs():
    bc = bibliographic_coupling(REFS_INDEX["A"], REFS_INDEX["D"])
    assert bc == 0, f"Expected 0 shared refs between A and D, got {bc}"


def test_bc_identical_ref_sets():
    refs = frozenset(["X", "Y", "Z"])
    assert bibliographic_coupling(refs, refs) == 3


def test_bc_empty_refs():
    assert bibliographic_coupling(frozenset(), frozenset(["X", "Y"])) == 0
    assert bibliographic_coupling(frozenset(["X"]), frozenset()) == 0


# ── co_citation ───────────────────────────────────────────────────────────────

def test_cc_b_is_one():
    cc = co_citation(ORIGIN_ID, "B", CITING_INDEX)
    assert cc == 1, f"Expected 1 co-citer of A and B (M), got {cc}"


def test_cc_c_is_one():
    cc = co_citation(ORIGIN_ID, "C", CITING_INDEX)
    assert cc == 1, f"Expected 1 co-citer of A and C (N), got {cc}"


def test_cc_d_is_zero():
    cc = co_citation(ORIGIN_ID, "D", CITING_INDEX)
    assert cc == 0, f"Expected 0 co-citers of A and D, got {cc}"


def test_cc_missing_origin_in_index():
    cc = co_citation("UNKNOWN", "B", CITING_INDEX)
    assert cc == 0


def test_cc_missing_candidate_in_index():
    cc = co_citation(ORIGIN_ID, "UNKNOWN", CITING_INDEX)
    assert cc == 0


# ── combined_score ────────────────────────────────────────────────────────────

def test_combined_score_b():
    score = combined_score(bc=2, cc=1, bc_max=2.0, cc_max=1.0)
    assert abs(score - 1.0) < 1e-9, f"Expected 1.0, got {score}"


def test_combined_score_c():
    score = combined_score(bc=1, cc=1, bc_max=2.0, cc_max=1.0)
    assert abs(score - 0.75) < 1e-9, f"Expected 0.75, got {score}"


def test_combined_score_d():
    score = combined_score(bc=0, cc=0, bc_max=2.0, cc_max=1.0)
    assert score == 0.0


def test_combined_score_zero_maxima():
    # If no overlap found at all, should not divide by zero
    score = combined_score(bc=0, cc=0, bc_max=0.0, cc_max=0.0)
    assert score == 0.0


def test_combined_score_custom_weights():
    # All weight on bc: only bibliographic coupling matters
    score = combined_score(bc=2, cc=0, bc_max=2.0, cc_max=1.0, bc_weight=1.0, cc_weight=0.0)
    assert abs(score - 1.0) < 1e-9


def test_combined_score_all_cc_weight():
    score = combined_score(bc=0, cc=1, bc_max=2.0, cc_max=1.0, bc_weight=0.0, cc_weight=1.0)
    assert abs(score - 1.0) < 1e-9


# ── rank_candidates ───────────────────────────────────────────────────────────

def test_rank_candidates_order():
    ranked = rank_candidates(
        origin_id=ORIGIN_ID,
        candidate_ids=CANDIDATE_IDS,
        refs_index=REFS_INDEX,
        citing_index=CITING_INDEX,
    )
    ids = [pid for pid, _ in ranked]
    assert ids[0] == "B", f"B should rank first, got {ids}"
    assert ids[1] == "C", f"C should rank second, got {ids}"
    assert ids[2] == "D", f"D should rank last, got {ids}"


def test_rank_candidates_scores():
    ranked = rank_candidates(
        origin_id=ORIGIN_ID,
        candidate_ids=CANDIDATE_IDS,
        refs_index=REFS_INDEX,
        citing_index=CITING_INDEX,
    )
    scores = {pid: score for pid, score in ranked}
    assert abs(scores["B"] - 1.0) < 1e-9
    assert abs(scores["C"] - 0.75) < 1e-9
    assert scores["D"] == 0.0


def test_rank_candidates_excludes_origin():
    all_ids = CANDIDATE_IDS + [ORIGIN_ID]
    ranked = rank_candidates(
        origin_id=ORIGIN_ID,
        candidate_ids=all_ids,
        refs_index=REFS_INDEX,
        citing_index=CITING_INDEX,
    )
    result_ids = [pid for pid, _ in ranked]
    assert ORIGIN_ID not in result_ids, "Origin should be excluded from ranking"


def test_rank_candidates_top_n():
    ranked = rank_candidates(
        origin_id=ORIGIN_ID,
        candidate_ids=CANDIDATE_IDS,
        refs_index=REFS_INDEX,
        citing_index=CITING_INDEX,
        top_n=2,
    )
    assert len(ranked) == 2
    assert ranked[0][0] == "B"
    assert ranked[1][0] == "C"


def test_rank_candidates_empty_pool():
    ranked = rank_candidates(
        origin_id=ORIGIN_ID,
        candidate_ids=[],
        refs_index=REFS_INDEX,
        citing_index=CITING_INDEX,
    )
    assert ranked == []


def test_rank_candidates_only_origin_in_pool():
    ranked = rank_candidates(
        origin_id=ORIGIN_ID,
        candidate_ids=[ORIGIN_ID],
        refs_index=REFS_INDEX,
        citing_index=CITING_INDEX,
    )
    assert ranked == []
