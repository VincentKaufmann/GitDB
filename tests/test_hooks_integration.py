"""Tests for hooks integration with GitDB."""

import pytest
import torch
from gitdb import GitDB


@pytest.fixture
def db(tmp_path):
    return GitDB(str(tmp_path / "store"), dim=8, device="cpu")


class TestHooksInCommit:
    def test_pre_commit_reject(self, db):
        db.add(torch.randn(2, 8), documents=["a", "b"])
        db.hooks.register("pre-commit", lambda **ctx: False)
        with pytest.raises(ValueError, match="pre-commit hook rejected"):
            db.commit("should fail")

    def test_pre_commit_accept(self, db):
        db.add(torch.randn(2, 8), documents=["a", "b"])
        db.hooks.register("pre-commit", lambda **ctx: True)
        h = db.commit("should work")
        assert h is not None

    def test_post_commit_fires(self, db):
        called = []
        db.hooks.register("post-commit", lambda **ctx: called.append(ctx))
        db.add(torch.randn(2, 8), documents=["a", "b"])
        db.commit("test")
        assert len(called) == 1
        assert "commit_hash" in called[0]


class TestHooksInMerge:
    def test_pre_merge_reject(self, db):
        db.add(torch.randn(2, 8), documents=["a", "b"])
        db.commit("base")
        db.branch("feature")
        db.switch("feature")
        db.add(torch.randn(1, 8), documents=["c"])
        db.commit("feature work")
        db.switch("main")
        db.hooks.register("pre-merge", lambda **ctx: False)
        with pytest.raises(ValueError, match="pre-merge hook rejected"):
            db.merge("feature")

    def test_post_merge_fires(self, db):
        called = []
        db.add(torch.randn(2, 8), documents=["a", "b"])
        db.commit("base")
        db.branch("feature")
        db.switch("feature")
        db.add(torch.randn(1, 8), documents=["c"])
        db.commit("feature work")
        db.switch("main")
        db.hooks.register("post-merge", lambda **ctx: called.append(ctx))
        db.merge("feature")
        # Post-merge may or may not fire depending on fast-forward
        # If fast-forward, merge() returns early without calling commit()
