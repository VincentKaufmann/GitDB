"""Tests for Snapshots — cheap read-only frozen views."""

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


class TestSnapshot:
    def test_create_snapshot(self, db):
        vecs = random_vectors(5)
        db.add(vecs, documents=[f"doc{i}" for i in range(5)])
        snap = db.snapshot("v1")
        assert snap.name == "v1"
        assert snap.size == 5

    def test_snapshot_list(self, db):
        db.add(random_vectors(3))
        db.snapshot("a")
        db.snapshot("b")
        snaps = db.snapshots()
        assert len(snaps) == 2
        names = {s["name"] for s in snaps}
        assert names == {"a", "b"}

    def test_snapshot_duplicate_name_raises(self, db):
        db.add(random_vectors(2))
        db.snapshot("x")
        with pytest.raises(ValueError):
            db.snapshot("x")

    def test_snapshot_query(self, db):
        vecs = random_vectors(10)
        db.add(vecs, documents=[f"doc{i}" for i in range(10)])
        snap = db.snapshot("q1")
        results = snap.query(vecs[0], k=3)
        assert len(results) == 3
        assert results.scores[0] > results.scores[1]

    def test_snapshot_query_with_where(self, db):
        vecs = random_vectors(6)
        db.add(
            vecs,
            documents=[f"doc{i}" for i in range(6)],
            metadata=[{"group": "a"}, {"group": "b"}, {"group": "a"},
                      {"group": "b"}, {"group": "a"}, {"group": "b"}],
        )
        snap = db.snapshot("filtered")
        results = snap.query(vecs[0], k=10, where={"group": "a"})
        assert len(results) <= 3
        for meta in results.metadata:
            assert meta["group"] == "a"

    def test_snapshot_frozen_after_db_change(self, db):
        vecs = random_vectors(5)
        db.add(vecs, documents=[f"doc{i}" for i in range(5)])
        snap = db.snapshot("frozen")
        assert snap.size == 5

        # Add more to the db
        db.add(random_vectors(3))
        # Snapshot should still be 5
        assert snap.size == 5
        assert db.tree.size == 8

    def test_snapshot_frozen_after_remove(self, db):
        vecs = random_vectors(5)
        db.add(vecs, metadata=[{"tag": "keep"}] * 3 + [{"tag": "remove"}] * 2)
        snap = db.snapshot("before_remove")
        db.remove(where={"tag": "remove"})
        assert snap.size == 5
        assert db.tree.size == 3

    def test_snapshot_select(self, db):
        db.add(
            random_vectors(3),
            documents=["a", "b", "c"],
            metadata=[{"x": 1}, {"x": 2}, {"x": 3}],
        )
        snap = db.snapshot("sel")
        rows = snap.select(where={"x": {"$gt": 1}})
        assert len(rows) == 2

    def test_snapshot_select_with_fields(self, db):
        db.add(
            random_vectors(2),
            documents=["hello", "world"],
            metadata=[{"x": 1}, {"x": 2}],
        )
        snap = db.snapshot("proj")
        rows = snap.select(fields=["document"])
        assert len(rows) == 2
        assert "document" in rows[0]

    def test_snapshot_empty_db(self, db):
        snap = db.snapshot("empty")
        assert snap.size == 0
        results = snap.query(torch.randn(8), k=5)
        assert len(results) == 0

    def test_get_snapshot(self, db):
        db.add(random_vectors(2))
        db.snapshot("test")
        snap = db.get_snapshot("test")
        assert snap.name == "test"

    def test_get_snapshot_missing_raises(self, db):
        with pytest.raises(ValueError):
            db.get_snapshot("nonexistent")

    def test_snapshot_repr(self, db):
        db.add(random_vectors(3))
        snap = db.snapshot("mysnap")
        r = repr(snap)
        assert "mysnap" in r
        assert "3 vectors" in r

    def test_snapshot_with_tombstones(self, db):
        """Snapshot should only include active (non-tombstoned) rows."""
        vecs = random_vectors(5)
        db.add(vecs)
        db.remove(ids=[1, 3])
        snap = db.snapshot("after_delete")
        assert snap.size == 3

    def test_snapshot_tensor_is_clone(self, db):
        """Modifying db tensor should not affect snapshot."""
        vecs = random_vectors(3)
        db.add(vecs)
        snap = db.snapshot("clone_test")
        # Modify db embeddings in place
        db.tree.embeddings[0] = torch.zeros(8)
        # Snapshot should still have original
        assert not torch.equal(snap.embeddings[0], torch.zeros(8))

    def test_multiple_snapshots_independent(self, db):
        db.add(random_vectors(3))
        s1 = db.snapshot("s1")
        db.add(random_vectors(2))
        s2 = db.snapshot("s2")
        assert s1.size == 3
        assert s2.size == 5
