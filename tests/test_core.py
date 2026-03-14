"""Core GitDB tests — the 7 invariants from the design doc."""

import shutil
import tempfile
from pathlib import Path

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


def random_query(dim=8):
    return torch.randn(dim)


class TestBasicOperations:
    def test_init_creates_structure(self, db, db_path):
        p = Path(db_path) / ".gitdb"
        assert p.exists()
        assert (p / "HEAD").exists()
        assert (p / "config").exists()

    def test_add_vectors(self, db):
        vecs = random_vectors(5)
        indices = db.add(vecs, documents=["a", "b", "c", "d", "e"])
        assert len(indices) == 5
        assert db.tree.size == 5

    def test_add_with_metadata(self, db):
        vecs = random_vectors(3)
        db.add(
            vecs,
            documents=["doc1", "doc2", "doc3"],
            metadata=[{"source": "web"}, {"source": "manual"}, {"source": "web"}],
        )
        assert db.tree.size == 3
        assert db.tree.metadata[0].metadata["source"] == "web"

    def test_remove_by_ids(self, db):
        vecs = random_vectors(5)
        db.add(vecs)
        db.remove(ids=[1, 3])
        assert db.tree.size == 3

    def test_remove_by_where(self, db):
        vecs = random_vectors(4)
        db.add(vecs, metadata=[{"tag": "a"}, {"tag": "b"}, {"tag": "a"}, {"tag": "c"}])
        removed = db.remove(where={"tag": "a"})
        assert len(removed) == 2
        assert db.tree.size == 2

    def test_query_basic(self, db):
        vecs = random_vectors(10)
        db.add(vecs, documents=[f"doc{i}" for i in range(10)])
        results = db.query(vecs[0], k=3)
        assert len(results) == 3
        assert results.scores[0] > results.scores[1]  # sorted by score
        assert results.ids[0] == db.tree.metadata[0].id  # top match is self

    def test_query_with_where(self, db):
        vecs = random_vectors(6)
        db.add(
            vecs,
            documents=[f"doc{i}" for i in range(6)],
            metadata=[{"group": "a"}, {"group": "b"}, {"group": "a"},
                      {"group": "b"}, {"group": "a"}, {"group": "b"}],
        )
        results = db.query(vecs[0], k=10, where={"group": "a"})
        assert len(results) <= 3  # only 3 in group "a"
        for meta in results.metadata:
            assert meta["group"] == "a"

    def test_query_empty(self, db):
        results = db.query(random_query(), k=5)
        assert len(results) == 0

    def test_update_embeddings(self, db):
        vecs = random_vectors(3)
        db.add(vecs)
        new_vec = random_vectors(1)
        db.update_embeddings([1], new_vec)
        assert torch.equal(db.tree.embeddings[1].cpu(), new_vec[0].cpu())


class TestVersionControl:
    """Invariant 1: Commit → checkout → query returns same results."""

    def test_commit_and_log(self, db):
        db.add(random_vectors(5))
        h = db.commit("First commit")
        assert len(h) == 64  # SHA-256 hex
        log = db.log()
        assert len(log) == 1
        assert log[0].message == "First commit"
        assert log[0].stats.added == 5

    def test_multiple_commits(self, db):
        db.add(random_vectors(3))
        db.commit("Add 3")
        db.add(random_vectors(2))
        db.commit("Add 2 more")
        log = db.log()
        assert len(log) == 2
        assert log[0].message == "Add 2 more"  # newest first
        assert log[1].message == "Add 3"

    def test_commit_checkout_roundtrip(self, db):
        """Invariant 1: state is preserved through commit/checkout."""
        vecs = random_vectors(10)
        db.add(vecs, documents=[f"d{i}" for i in range(10)])
        h1 = db.commit("Snapshot 1")

        # Query before checkout
        q = random_query()
        results_before = db.query(q, k=5)

        # Add more, commit
        db.add(random_vectors(5))
        db.commit("Snapshot 2")

        # Checkout back to first commit
        db.checkout(h1)

        # Query should return same results
        results_after = db.query(q, k=5)
        assert results_before.ids == results_after.ids
        assert len(results_before.scores) == len(results_after.scores)
        for s1, s2 in zip(results_before.scores, results_after.scores):
            assert abs(s1 - s2) < 1e-5

    def test_commit_with_deletions(self, db):
        vecs = random_vectors(5)
        db.add(vecs)
        db.commit("Add 5")
        db.remove(ids=[1, 3])
        h2 = db.commit("Remove 2")
        log = db.log()
        assert log[0].stats.removed == 2
        assert db.tree.size == 3

    def test_commit_with_modifications(self, db):
        vecs = random_vectors(5)
        db.add(vecs)
        db.commit("Add 5")
        new_emb = random_vectors(1)
        db.update_embeddings([2], new_emb)
        db.commit("Modify row 2")
        log = db.log()
        assert log[0].stats.modified == 1

    def test_nothing_to_commit(self, db):
        db.add(random_vectors(3))
        db.commit("First")
        with pytest.raises(ValueError, match="Nothing to commit"):
            db.commit("Empty")


class TestDelta:
    """Invariant 3: Delta round-trip preserves tensor equality."""

    def test_delta_serialize_roundtrip(self):
        from gitdb.delta import Delta
        from gitdb.types import VectorMeta

        dim = 8
        delta = Delta(
            add_indices=[0, 1],
            add_values=torch.randn(2, dim),
            add_metadata=[
                VectorMeta(id="aaa", document="doc1", metadata={"k": "v"}),
                VectorMeta(id="bbb", document="doc2", metadata={}),
            ],
            del_indices=[5],
            del_metadata=[VectorMeta(id="ccc")],
            mod_indices=[3],
            mod_old_values=torch.randn(1, dim),
            mod_new_values=torch.randn(1, dim),
        )

        data = delta.serialize()
        restored = Delta.deserialize(data, dim)

        assert restored.add_indices == [0, 1]
        assert restored.del_indices == [5]
        assert restored.mod_indices == [3]
        assert torch.allclose(delta.add_values, restored.add_values, atol=1e-6)
        assert torch.allclose(delta.mod_old_values, restored.mod_old_values, atol=1e-6)
        assert torch.allclose(delta.mod_new_values, restored.mod_new_values, atol=1e-6)
        assert restored.add_metadata[0].id == "aaa"
        assert restored.add_metadata[0].document == "doc1"

    def test_delta_compression(self):
        """Invariant 4: Sparse delta << full tensor for small changes."""
        from gitdb.delta import Delta
        from gitdb.types import VectorMeta

        dim = 1024
        # Small change: 5 additions to a 10000-row store
        delta = Delta(
            add_indices=list(range(5)),
            add_values=torch.randn(5, dim),
            add_metadata=[VectorMeta(id=f"id{i}") for i in range(5)],
        )
        delta_bytes = delta.serialize()
        full_tensor_bytes = 10000 * dim * 4  # float32
        assert len(delta_bytes) < full_tensor_bytes * 0.01  # <1% of full tensor

    def test_empty_delta(self):
        from gitdb.delta import Delta
        d = Delta()
        assert d.is_empty


class TestDiff:
    def test_diff_added(self, db):
        db.add(random_vectors(3))
        h1 = db.commit("Base")
        db.add(random_vectors(2))
        h2 = db.commit("Add 2")
        diff = db.diff(h1, h2)
        assert diff.added_count == 2
        assert diff.removed_count == 0

    def test_diff_removed(self, db):
        vecs = random_vectors(5)
        db.add(vecs)
        h1 = db.commit("Base")
        db.remove(ids=[0, 1])
        h2 = db.commit("Remove 2")
        diff = db.diff(h1, h2)
        assert diff.removed_count == 2

    def test_diff_same(self, db):
        db.add(random_vectors(3))
        h = db.commit("Same")
        diff = db.diff(h, h)
        assert diff.added_count == 0
        assert diff.removed_count == 0


class TestTimeTravel:
    def test_query_at_historical_commit(self, db):
        vecs = random_vectors(5)
        db.add(vecs, documents=[f"old{i}" for i in range(5)])
        h1 = db.commit("Old state")

        db.add(random_vectors(5), documents=[f"new{i}" for i in range(5)])
        db.commit("New state")

        # Query at old commit
        results = db.query(vecs[0], k=3, at=h1)
        assert len(results) == 3
        assert all("old" in d for d in results.documents)


class TestStatus:
    def test_status(self, db):
        s = db.status()
        assert s["branch"] == "main"
        assert s["staged_additions"] == 0

        db.add(random_vectors(3))
        s = db.status()
        assert s["staged_additions"] == 3

    def test_repr(self, db):
        r = repr(db)
        assert "GitDB" in r
        assert "main" in r


class TestPersistence:
    def test_reopen_preserves_state(self, db_path):
        """Open, add, commit, close, reopen, verify."""
        db = GitDB(db_path, dim=8, device="cpu")
        vecs = random_vectors(10)
        db.add(vecs, documents=[f"doc{i}" for i in range(10)])
        db.commit("Persisted")
        del db

        # Reopen
        db2 = GitDB(db_path, dim=8, device="cpu")
        assert db2.tree.size == 10
        results = db2.query(vecs[0], k=1)
        assert len(results) == 1
        assert results.documents[0] == "doc0"


class TestTag:
    def test_create_tag(self, db):
        db.add(random_vectors(3))
        h = db.commit("Tagged")
        db.tag("v1.0")
        assert db.refs.resolve("v1.0") == h

    def test_duplicate_tag_raises(self, db):
        db.add(random_vectors(3))
        db.commit("Tagged")
        db.tag("v1.0")
        with pytest.raises(ValueError, match="Tag already exists"):
            db.tag("v1.0")


class TestReset:
    def test_reset_discards_staged(self, db):
        db.add(random_vectors(5))
        db.commit("Base")
        db.add(random_vectors(3))
        assert db.status()["staged_additions"] == 3
        db.reset()
        assert db.status()["staged_additions"] == 0
        assert db.tree.size == 5
