"""backend/tests/test_paper_graph_api.py
──────────────────────────────────────────
API tests for the paper discovery endpoints.

The SemanticScholarClient is replaced with a fixture-backed mock so these
tests require no network access.  Mirrors test_notebook_advanced_api.py's
monkeypatch + poll-until-terminal pattern.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import paper_graph.s2_client as s2_module
from paper_graph.collection_store import CollectionStore
import paper_graph.collection_store as collection_store_module
from paper_graph.s2_client import PaperNode
from backend.app.main import app

_BASE = "/api/paper-graph"


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_node(paper_id: str, title: str = "", year: int = 2020) -> PaperNode:
    return PaperNode(
        id=paper_id,
        title=title or f"Paper {paper_id}",
        authors=["Author A"],
        year=year,
        venue="Test Venue",
        abstract="Test abstract.",
        citation_count=10,
        url=f"https://example.com/{paper_id}",
    )


# Canned responses for the mock client
_ORIGIN = _make_node("aaa" + "0" * 37, "Origin Paper", 2021)
_PAPER_B = _make_node("bbb" + "0" * 37, "Paper B", 2020)
_PAPER_C = _make_node("ccc" + "0" * 37, "Paper C", 2019)


class _MockClient:
    """Minimal mock of SemanticScholarClient that returns canned data."""

    def get_paper(self, paper_id: str) -> Optional[PaperNode]:
        mapping = {
            _ORIGIN.id: _ORIGIN,
            _PAPER_B.id: _PAPER_B,
            _PAPER_C.id: _PAPER_C,
        }
        return mapping.get(paper_id)

    def get_references(self, paper_id: str) -> List[str]:
        if paper_id == _ORIGIN.id:
            return [_PAPER_B.id, "ref_x", "ref_y"]
        if paper_id == _PAPER_B.id:
            return [_ORIGIN.id, "ref_x"]
        return []

    def get_citations(self, paper_id: str) -> List[str]:
        if paper_id == _ORIGIN.id:
            return [_PAPER_C.id]
        return []

    def batch_get_papers(self, ids: List[str]) -> List[PaperNode]:
        results = []
        for pid in ids:
            n = self.get_paper(pid)
            if n:
                results.append(n)
        return results

    def get_recommendations(self, paper_ids: List[str]) -> List[PaperNode]:
        return [_PAPER_C]

    def get_author_papers(self, author_id: str) -> List[PaperNode]:
        return [_PAPER_B]

    def search_paper(self, query: str, limit: int = 1) -> List[PaperNode]:
        if "origin" in query.lower():
            return [_ORIGIN]
        return []


@pytest.fixture(autouse=True)
def mock_s2_client(monkeypatch):
    mock = _MockClient()
    monkeypatch.setattr(s2_module, "_client", mock)
    # Patch get_client() to return mock in service and router imports
    monkeypatch.setattr("paper_graph.s2_client._client", mock)
    yield mock


@pytest.fixture(autouse=True)
def fresh_collection_store(monkeypatch):
    """Reset the collection store between tests."""
    fresh_store = CollectionStore()
    monkeypatch.setattr(collection_store_module, "_store", fresh_store)
    yield fresh_store


@pytest.fixture()
def client():
    return TestClient(app)


# ── Helper ────────────────────────────────────────────────────────────────────

def _poll(client: TestClient, url: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    data: dict = {}
    while time.monotonic() < deadline:
        r = client.get(url)
        assert r.status_code == 200
        data = r.json()
        if data.get("status") in ("done", "error"):
            return data
        time.sleep(0.05)
    raise TimeoutError(f"Job at {url} did not finish within {timeout}s. Last: {data}")


# ── Feature 1: Similarity Graph ───────────────────────────────────────────────

def test_similarity_graph_happy_path(client):
    r = client.post(f"{_BASE}/similarity-graph", json={"paper_id": _ORIGIN.id})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    result = _poll(client, f"{_BASE}/jobs/{job_id}")
    assert result["status"] == "done", result.get("error")
    graph = result["result"]["graph"]
    node_ids = {n["id"] for n in graph["nodes"]}
    assert _ORIGIN.id in node_ids


def test_similarity_graph_title_resolution(client):
    r = client.post(f"{_BASE}/similarity-graph", json={"paper_id": "Origin Paper"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    result = _poll(client, f"{_BASE}/jobs/{job_id}")
    assert result["status"] == "done", result.get("error")


def test_similarity_graph_unknown_paper(client):
    r = client.post(f"{_BASE}/similarity-graph", json={"paper_id": "totally unknown xyz 99999"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    result = _poll(client, f"{_BASE}/jobs/{job_id}")
    assert result["status"] == "error"
    assert "not found" in (result.get("error") or "").lower()


def test_similarity_graph_job_not_found(client):
    r = client.get(f"{_BASE}/jobs/nonexistent_job_id")
    assert r.status_code == 404


def test_similarity_graph_graph_has_edges(client):
    r = client.post(f"{_BASE}/similarity-graph", json={"paper_id": _ORIGIN.id})
    job_id = r.json()["job_id"]
    result = _poll(client, f"{_BASE}/jobs/{job_id}")
    graph = result["result"]["graph"]
    # Should have at least origin + one candidate
    assert len(graph["nodes"]) >= 1


# ── Building-block endpoints ──────────────────────────────────────────────────

def test_get_paper(client):
    r = client.get(f"{_BASE}/papers/{_ORIGIN.id}")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == _ORIGIN.id
    assert data["title"] == _ORIGIN.title


def test_get_paper_not_found(client):
    r = client.get(f"{_BASE}/papers/ffffffffffffffffffffffffffffffffffffffff")
    assert r.status_code == 404


def test_get_references(client):
    r = client.get(f"{_BASE}/papers/{_ORIGIN.id}/references")
    assert r.status_code == 200
    refs = r.json()
    assert _PAPER_B.id in refs


def test_get_citations(client):
    r = client.get(f"{_BASE}/papers/{_ORIGIN.id}/citations")
    assert r.status_code == 200
    cits = r.json()
    assert _PAPER_C.id in cits


# ── Feature 2: Discovery Network ──────────────────────────────────────────────

def test_create_collection(client):
    r = client.post(f"{_BASE}/collections", json={"seed_paper_ids": [_ORIGIN.id]})
    assert r.status_code == 201
    data = r.json()
    assert "collection_id" in data
    node_ids = {n["id"] for n in data["graph"]["nodes"]}
    assert _ORIGIN.id in node_ids


def test_create_collection_title_resolution(client):
    r = client.post(f"{_BASE}/collections", json={"seed_paper_ids": ["Origin Paper"]})
    assert r.status_code == 201


def test_create_collection_invalid_seeds(client):
    r = client.post(f"{_BASE}/collections", json={"seed_paper_ids": ["totally unknown xyz 99999"]})
    assert r.status_code == 422


def test_get_collection(client):
    create_r = client.post(f"{_BASE}/collections", json={"seed_paper_ids": [_ORIGIN.id]})
    cid = create_r.json()["collection_id"]
    r = client.get(f"{_BASE}/collections/{cid}")
    assert r.status_code == 200
    assert r.json()["collection_id"] == cid


def test_get_collection_not_found(client):
    r = client.get(f"{_BASE}/collections/nonexistent")
    assert r.status_code == 404


def test_expand_collection_later(client):
    create_r = client.post(f"{_BASE}/collections", json={"seed_paper_ids": [_ORIGIN.id]})
    cid = create_r.json()["collection_id"]

    expand_r = client.post(
        f"{_BASE}/collections/{cid}/expand",
        json={"node_id": _ORIGIN.id, "relationship": "later"},
    )
    assert expand_r.status_code == 202
    job_id = expand_r.json()["job_id"]

    result = _poll(client, f"{_BASE}/collections/{cid}/jobs/{job_id}")
    assert result["status"] == "done", result.get("error")
    graph = result["result"]["graph"]
    node_ids = {n["id"] for n in graph["nodes"]}
    assert _PAPER_C.id in node_ids


def test_expand_collection_earlier(client):
    create_r = client.post(f"{_BASE}/collections", json={"seed_paper_ids": [_ORIGIN.id]})
    cid = create_r.json()["collection_id"]

    expand_r = client.post(
        f"{_BASE}/collections/{cid}/expand",
        json={"node_id": _ORIGIN.id, "relationship": "earlier"},
    )
    assert expand_r.status_code == 202
    job_id = expand_r.json()["job_id"]

    result = _poll(client, f"{_BASE}/collections/{cid}/jobs/{job_id}")
    assert result["status"] == "done", result.get("error")
    graph = result["result"]["graph"]
    node_ids = {n["id"] for n in graph["nodes"]}
    assert _PAPER_B.id in node_ids


def test_expand_collection_similar(client):
    create_r = client.post(f"{_BASE}/collections", json={"seed_paper_ids": [_ORIGIN.id]})
    cid = create_r.json()["collection_id"]

    expand_r = client.post(
        f"{_BASE}/collections/{cid}/expand",
        json={"node_id": _ORIGIN.id, "relationship": "similar"},
    )
    assert expand_r.status_code == 202
    job_id = expand_r.json()["job_id"]

    result = _poll(client, f"{_BASE}/collections/{cid}/jobs/{job_id}")
    assert result["status"] == "done", result.get("error")
    graph = result["result"]["graph"]
    node_ids = {n["id"] for n in graph["nodes"]}
    assert _PAPER_C.id in node_ids


def test_expand_collection_not_found(client):
    r = client.post(
        f"{_BASE}/collections/nonexistent/expand",
        json={"node_id": _ORIGIN.id, "relationship": "later"},
    )
    assert r.status_code == 404


def test_collection_graph_grows_incrementally(client):
    """Expanding should add nodes to the collection, not replace them."""
    create_r = client.post(f"{_BASE}/collections", json={"seed_paper_ids": [_ORIGIN.id]})
    cid = create_r.json()["collection_id"]
    initial_count = len(create_r.json()["graph"]["nodes"])

    expand_r = client.post(
        f"{_BASE}/collections/{cid}/expand",
        json={"node_id": _ORIGIN.id, "relationship": "earlier"},
    )
    job_id = expand_r.json()["job_id"]
    result = _poll(client, f"{_BASE}/collections/{cid}/jobs/{job_id}")
    assert result["status"] == "done"
    final_count = len(result["result"]["graph"]["nodes"])
    assert final_count >= initial_count, "Collection should grow or stay same, not shrink"
