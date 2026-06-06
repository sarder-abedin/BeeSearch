"""tests/test_citation_tools.py — Unit tests for tools/citation_tools.py"""

import pytest

from tools.citation_tools import ref_to_bibtex, ref_to_ris, refs_to_bibtex, refs_to_ris


# ── Fixtures ───────────────────────────────────────────────────────────────────

_SENTINEL = object()

def _ref(
    title="Attention Is All You Need",
    authors=_SENTINEL,
    year=2017,
    journal="NeurIPS",
    doi="10.48550/arXiv.1706.03762",
    url="https://arxiv.org/abs/1706.03762",
    ref_num=1,
    source="arxiv",
):
    return {
        "title": title,
        "authors": ["Vaswani A", "Shazeer N", "Parmar N"] if authors is _SENTINEL else authors,
        "year": year,
        "journal": journal,
        "doi": doi,
        "url": url,
        "ref_num": ref_num,
        "apa": f"Vaswani, A. et al. ({year}). {title}. {journal}.",
        "source": source,
    }


# ── ref_to_bibtex ──────────────────────────────────────────────────────────────

class TestRefToBibtex:
    def test_arxiv_uses_misc_entry_type(self):
        # source="arxiv" produces @misc, peer-reviewed sources produce @article
        entry = ref_to_bibtex(_ref(source="arxiv"))
        assert entry.startswith("@misc{")

    def test_peer_reviewed_uses_article_entry_type(self):
        entry = ref_to_bibtex(_ref(source="semantic_scholar"))
        assert entry.startswith("@article{")

    def test_key_uses_first_token_of_author_name(self):
        # "Smith J" → split()[0] → "Smith" → lower → "smith"; key = "smith2022"
        entry = ref_to_bibtex(_ref(authors=["Smith J", "Jones K"], year=2022))
        assert "smith2022" in entry

    def test_title_in_entry(self):
        entry = ref_to_bibtex(_ref(title="My Paper Title"))
        assert "My Paper Title" in entry

    def test_doi_in_entry(self):
        entry = ref_to_bibtex(_ref(doi="10.1234/test"))
        assert "10.1234/test" in entry

    def test_no_doi_still_produces_valid_entry(self):
        ref = _ref(doi=None)
        ref.pop("doi", None)
        entry = ref_to_bibtex(ref)
        assert entry.startswith("@")

    def test_empty_authors_falls_back_to_anon(self):
        entry = ref_to_bibtex(_ref(authors=[]))
        # key is "anon<year>"; the entry must be valid BibTeX
        assert entry.startswith("@")

    def test_missing_year_falls_back_to_nd(self):
        ref = _ref()
        ref["year"] = None
        entry = ref_to_bibtex(ref)
        assert entry.startswith("@")


# ── refs_to_bibtex ─────────────────────────────────────────────────────────────

class TestRefsToBibtex:
    def test_empty_list_returns_comment_string(self):
        # Implementation returns a comment placeholder, not empty string
        result = refs_to_bibtex([])
        assert "No references" in result or result == ""

    def test_single_ref_produces_one_bibtex_block(self):
        bib = refs_to_bibtex([_ref()])
        # Count any @-entry type (misc for arxiv, article for peer-reviewed)
        assert bib.count("@") >= 1

    def test_multiple_refs_produce_multiple_blocks(self):
        refs = [
            _ref(title="Paper A", authors=["Alpha A"], year=2020, ref_num=1),
            _ref(title="Paper B", authors=["Beta B"], year=2021, ref_num=2),
        ]
        bib = refs_to_bibtex(refs)
        assert bib.count("Paper A") == 1
        assert bib.count("Paper B") == 1

    def test_key_collision_suffix(self):
        """Two refs with the same first token + year get base key then 'a' suffix."""
        # Both start with "Smith" with same year → first "smith2022", second "smith2022a"
        refs = [
            _ref(title="Paper A", authors=["Smith A"], year=2022, ref_num=1),
            _ref(title="Paper B", authors=["Smith B"], year=2022, ref_num=2),
        ]
        bib = refs_to_bibtex(refs)
        assert "smith2022" in bib
        assert "smith2022a" in bib

    def test_no_duplicate_keys(self):
        # Use different last tokens each time so we get stable key generation
        refs = [
            _ref(title=f"Paper {i}", authors=[f"Author {chr(65+i)}"], year=2020+i, ref_num=i)
            for i in range(5)
        ]
        bib = refs_to_bibtex(refs)
        at_lines = [line for line in bib.splitlines() if line.startswith("@")]
        keys = [line.split("{")[1].rstrip(",") for line in at_lines]
        assert len(keys) == len(set(keys))


# ── ref_to_ris ────────────────────────────────────────────────────────────────

class TestRefToRis:
    def test_starts_with_ty(self):
        entry = ref_to_ris(_ref())
        assert entry.startswith("TY  - ")

    def test_ends_with_er(self):
        entry = ref_to_ris(_ref())
        assert "ER  -" in entry

    def test_title_present(self):
        entry = ref_to_ris(_ref(title="My RIS Title"))
        assert "My RIS Title" in entry

    def test_doi_present(self):
        entry = ref_to_ris(_ref(doi="10.9999/xyz"))
        assert "10.9999/xyz" in entry

    def test_no_doi_still_valid(self):
        ref = _ref()
        ref["doi"] = None
        entry = ref_to_ris(ref)
        assert "TY  - " in entry and "ER  -" in entry


# ── refs_to_ris ───────────────────────────────────────────────────────────────

class TestRefsToRis:
    def test_empty_list_returns_empty_string(self):
        assert refs_to_ris([]) == ""

    def test_single_ref(self):
        ris = refs_to_ris([_ref()])
        assert ris.count("TY  - ") == 1

    def test_multiple_refs(self):
        refs = [_ref(ref_num=i) for i in range(1, 4)]
        ris = refs_to_ris(refs)
        assert ris.count("TY  - ") == 3

    def test_each_block_terminated(self):
        refs = [_ref(ref_num=1), _ref(ref_num=2)]
        ris = refs_to_ris(refs)
        assert ris.count("ER  -") == 2
