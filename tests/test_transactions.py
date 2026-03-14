"""Tests for the transaction system."""

import pytest
import torch

from gitdb import GitDB


@pytest.fixture
def db(tmp_path):
    return GitDB(str(tmp_path / "test_store"), dim=8, device="cpu")


def rvec(n=1, dim=8):
    return torch.randn(n, dim)


class TestTransactionSuccess:
    def test_basic_add(self, db):
        with db.transaction() as tx:
            tx.add(embeddings=rvec(3), documents=["a", "b", "c"])
        assert db.tree.size == 3

    def test_add_and_remove(self, db):
        db.add(rvec(5), metadata=[{"keep": True}] * 3 + [{"keep": False}] * 2)
        db.commit("base")
        with db.transaction() as tx:
            tx.add(embeddings=rvec(2), documents=["new1", "new2"])
            tx.remove(where={"keep": False})
        # 5 - 2 removed + 2 added = 5
        assert db.tree.size == 5

    def test_update_embeddings(self, db):
        db.add(rvec(3))
        db.commit("base")
        new_vec = rvec(1)
        with db.transaction() as tx:
            tx.update_embeddings([0], new_vec)
        assert torch.allclose(db.tree.embeddings[0].cpu(), new_vec[0].cpu())

    def test_transaction_then_commit(self, db):
        with db.transaction() as tx:
            tx.add(embeddings=rvec(3), documents=["a", "b", "c"])
        h = db.commit("after transaction")
        assert len(h) == 64
        log = db.log()
        assert log[0].stats.added == 3


class TestTransactionRollback:
    def test_rollback_on_exception(self, db):
        db.add(rvec(3), documents=["a", "b", "c"])
        db.commit("base")
        initial_size = db.tree.size

        with pytest.raises(RuntimeError):
            with db.transaction() as tx:
                tx.add(embeddings=rvec(2), documents=["x", "y"])
                assert db.tree.size == 5  # added during transaction
                raise RuntimeError("something went wrong")

        # Rolled back
        assert db.tree.size == initial_size

    def test_rollback_preserves_staging(self, db):
        db.add(rvec(3))
        db.commit("base")
        db.add(rvec(1))  # stage one addition before transaction

        with pytest.raises(RuntimeError):
            with db.transaction() as tx:
                tx.add(embeddings=rvec(2))
                raise RuntimeError("fail")

        # Original staging is preserved
        assert db.status()["staged_additions"] == 1

    def test_rollback_restores_removals(self, db):
        db.add(rvec(5), documents=["a", "b", "c", "d", "e"])
        db.commit("base")

        with pytest.raises(RuntimeError):
            with db.transaction() as tx:
                tx.remove(ids=[0, 1])
                assert db.tree.size == 3
                raise RuntimeError("fail")

        assert db.tree.size == 5

    def test_rollback_on_empty_db(self, db):
        """Transaction rollback on a db with no data should not crash."""
        with pytest.raises(RuntimeError):
            with db.transaction() as tx:
                tx.add(embeddings=rvec(2), documents=["a", "b"])
                raise RuntimeError("fail")
        assert db.tree.size == 0

    def test_rollback_does_not_suppress_exception(self, db):
        """Exception should propagate after rollback."""
        with pytest.raises(ValueError, match="custom error"):
            with db.transaction() as tx:
                tx.add(embeddings=rvec(1))
                raise ValueError("custom error")


class TestTransactionEdgeCases:
    def test_nested_is_flat(self, db):
        """Nesting transactions is just two context managers (no special nesting)."""
        with db.transaction() as tx1:
            tx1.add(embeddings=rvec(2))
            # Inner transaction is a separate snapshot
            with db.transaction() as tx2:
                tx2.add(embeddings=rvec(1))
        assert db.tree.size == 3

    def test_empty_transaction(self, db):
        """Empty transaction is a no-op."""
        db.add(rvec(3))
        db.commit("base")
        with db.transaction():
            pass
        assert db.tree.size == 3

    def test_multiple_sequential_transactions(self, db):
        with db.transaction() as tx:
            tx.add(embeddings=rvec(2))
        with db.transaction() as tx:
            tx.add(embeddings=rvec(3))
        assert db.tree.size == 5
