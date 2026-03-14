"""Tests for the GitDB web dashboard server."""

import json
import threading
import time
import urllib.request
import urllib.error

import pytest
import torch

from gitdb import GitDB
from gitdb.server import GitDBHandler, DASHBOARD_HTML
from http.server import HTTPServer


@pytest.fixture
def db(tmp_path):
    """Create a GitDB with test data and a couple of commits."""
    store = GitDB(str(tmp_path / "store"), dim=8, device="cpu")
    store.add(
        torch.randn(5, 8),
        documents=[f"document_{i}" for i in range(5)],
        metadata=[{"idx": i, "type": "test"} for i in range(5)],
    )
    store.commit("Initial ingest")
    store.add(
        torch.randn(3, 8),
        documents=[f"extra_{i}" for i in range(3)],
        metadata=[{"idx": i + 5, "type": "extra"} for i in range(3)],
    )
    store.commit("Add more vectors")
    store.branch("experiment")
    store.tag("v0.1")
    return store


@pytest.fixture
def server(db):
    """Run the dashboard server in a background thread."""
    GitDBHandler.db = db
    httpd = HTTPServer(("127.0.0.1", 0), GitDBHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    yield base, httpd
    httpd.shutdown()


def get(url):
    """GET request, return parsed JSON or raw bytes."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = resp.read()
        ct = resp.headers.get("Content-Type", "")
        if "json" in ct:
            return json.loads(data)
        return data


def post(url, body):
    """POST JSON, return parsed JSON."""
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


# ── HTML ──

class TestDashboardPage:
    def test_serves_html(self, server):
        base, _ = server
        data = get(base + "/")
        assert b"<!DOCTYPE html>" in data
        assert b"GitDB" in data

    def test_html_contains_search(self, server):
        base, _ = server
        data = get(base + "/")
        assert b"search-input" in data

    def test_html_contains_heatmap(self, server):
        base, _ = server
        data = get(base + "/")
        assert b"ac-heatmap" in data


# ── Status ──

class TestStatusAPI:
    def test_status_returns_json(self, server):
        base, _ = server
        data = get(base + "/api/status")
        assert isinstance(data, dict)
        assert "branch" in data

    def test_status_branch(self, server):
        base, _ = server
        data = get(base + "/api/status")
        assert data["branch"] == "main"

    def test_status_has_rows(self, server):
        base, _ = server
        data = get(base + "/api/status")
        assert data["active_rows"] == 8

    def test_status_has_dim(self, server):
        base, _ = server
        data = get(base + "/api/status")
        assert data["dim"] == 8

    def test_status_has_path(self, server):
        base, _ = server
        data = get(base + "/api/status")
        assert "path" in data
        assert "store" in data["path"]


# ── Log ──

class TestLogAPI:
    def test_log_returns_commits(self, server):
        base, _ = server
        data = get(base + "/api/log?limit=10")
        assert "commits" in data
        assert len(data["commits"]) == 2

    def test_log_commit_fields(self, server):
        base, _ = server
        data = get(base + "/api/log?limit=1")
        c = data["commits"][0]
        assert "hash" in c
        assert "message" in c
        assert "timestamp" in c
        assert "stats" in c

    def test_log_respects_limit(self, server):
        base, _ = server
        data = get(base + "/api/log?limit=1")
        assert len(data["commits"]) == 1


# ── Branches ──

class TestBranchesAPI:
    def test_branches_returns_list(self, server):
        base, _ = server
        data = get(base + "/api/branches")
        assert "branches" in data
        assert "main" in data["branches"]

    def test_branches_current(self, server):
        base, _ = server
        data = get(base + "/api/branches")
        assert data["current"] == "main"

    def test_branches_includes_experiment(self, server):
        base, _ = server
        data = get(base + "/api/branches")
        assert "experiment" in data["branches"]


# ── Tags ──

class TestTagsAPI:
    def test_tags_returns_list(self, server):
        base, _ = server
        data = get(base + "/api/tags")
        assert "tags" in data
        assert "v0.1" in data["tags"]


# ── Select ──

class TestSelectAPI:
    def test_select_all(self, server):
        base, _ = server
        data = post(base + "/api/select", {"limit": 5})
        assert "rows" in data
        assert len(data["rows"]) == 5

    def test_select_with_where(self, server):
        base, _ = server
        data = post(base + "/api/select", {"where": {"type": "extra"}, "limit": 100})
        assert data["count"] == 3


# ── Show ──

class TestShowAPI:
    def test_show_head(self, server):
        base, _ = server
        data = get(base + "/api/show?ref=HEAD")
        assert "hash" in data
        assert "message" in data
        assert data["message"] == "Add more vectors"

    def test_show_invalid_ref(self, server):
        base, _ = server
        try:
            get(base + "/api/show?ref=nonexistent")
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 400


# ── AC ──

class TestACAPI:
    def test_ac_status(self, server):
        base, _ = server
        data = get(base + "/api/ac/status")
        assert "running" in data

    def test_ac_heatmap(self, server):
        base, _ = server
        data = get(base + "/api/ac/heatmap")
        assert "activations" in data
        assert "stats" in data


# ── 404 ──

class TestNotFound:
    def test_unknown_route(self, server):
        base, _ = server
        try:
            get(base + "/api/nonexistent")
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 404
