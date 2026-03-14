"""Tests for WatchManager — change subscriptions."""

import pytest
import torch

from gitdb import GitDB


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_store")


@pytest.fixture
def db(db_path):
    return GitDB(db_path, dim=8, device="cpu")


def random_vectors(n, dim=8):
    return torch.randn(n, dim)


class TestWatchManager:
    def test_watch_where_returns_id(self, db):
        wid = db.watch(where={"tag": "test"}, on_change=lambda e, c: None)
        assert isinstance(wid, int)
        assert wid > 0

    def test_watch_branch_returns_id(self, db):
        wid = db.watch(branch="main", on_change=lambda e, c: None)
        assert isinstance(wid, int)
        assert wid > 0

    def test_unwatch(self, db):
        wid = db.watch(where={"tag": "x"}, on_change=lambda e, c: None)
        db.unwatch(wid)
        assert len(db.watches_list()) == 0

    def test_unwatch_invalid_raises(self, db):
        with pytest.raises(ValueError):
            db.unwatch(9999)

    def test_watches_list(self, db):
        db.watch(where={"tag": "a"}, on_change=lambda e, c: None)
        db.watch(branch="main", on_change=lambda e, c: None)
        watches = db.watches_list()
        assert len(watches) == 2
        types = {w["type"] for w in watches}
        assert "where" in types
        assert "branch" in types

    def test_watch_fires_on_commit(self, db):
        fired = []
        db.watch(where={"tag": "important"}, on_change=lambda e, c: fired.append((e, c)))
        db.add(
            random_vectors(2),
            documents=["a", "b"],
            metadata=[{"tag": "important"}, {"tag": "other"}],
        )
        db.commit("test commit")
        assert len(fired) == 1
        assert fired[0][0] == "commit"
        assert "commit_hash" in fired[0][1]

    def test_watch_does_not_fire_on_no_match(self, db):
        fired = []
        db.watch(where={"tag": "nonexistent"}, on_change=lambda e, c: fired.append(c))
        db.add(random_vectors(2), metadata=[{"tag": "a"}, {"tag": "b"}])
        db.commit("test")
        assert len(fired) == 0

    def test_watch_branch_fires_on_commit(self, db):
        fired = []
        db.watch(branch="main", on_change=lambda e, c: fired.append((e, c)))
        db.add(random_vectors(2))
        db.commit("branch watch test")
        assert len(fired) == 1
        assert fired[0][0] == "commit"

    def test_watch_branch_wrong_branch_no_fire(self, db):
        fired = []
        db.watch(branch="other", on_change=lambda e, c: fired.append(c))
        db.add(random_vectors(2))
        db.commit("main commit")
        assert len(fired) == 0

    def test_multiple_watches_fire_independently(self, db):
        fired_a = []
        fired_b = []
        db.watch(where={"tag": "a"}, on_change=lambda e, c: fired_a.append(c))
        db.watch(where={"tag": "b"}, on_change=lambda e, c: fired_b.append(c))
        db.add(random_vectors(1), metadata=[{"tag": "a"}])
        db.commit("add a")
        assert len(fired_a) == 1
        assert len(fired_b) == 0

    def test_watch_survives_bad_callback(self, db):
        good_fired = []

        def bad_callback(e, c):
            raise RuntimeError("boom")

        db.watch(where={"tag": "x"}, on_change=bad_callback)
        db.watch(where={"tag": "x"}, on_change=lambda e, c: good_fired.append(c))
        db.add(random_vectors(1), metadata=[{"tag": "x"}])
        db.commit("test")
        # Good callback still fires despite bad one
        assert len(good_fired) == 1

    def test_watch_requires_where_or_branch(self, db):
        with pytest.raises(ValueError):
            db.watch(on_change=lambda e, c: None)

    def test_watch_unique_ids(self, db):
        id1 = db.watch(where={"a": 1}, on_change=lambda e, c: None)
        id2 = db.watch(where={"b": 2}, on_change=lambda e, c: None)
        assert id1 != id2
