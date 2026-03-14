"""Tests for the hooks system."""

import pytest
import torch

from gitdb import GitDB
from gitdb.hooks import HookManager


@pytest.fixture
def db(tmp_path):
    return GitDB(str(tmp_path / "test_store"), dim=8, device="cpu")


def rvec(n=1, dim=8):
    return torch.randn(n, dim)


class TestHookManager:
    def test_register_and_fire(self):
        hm = HookManager()
        log = []
        hm.register("pre-commit", lambda **ctx: log.append("fired"))
        hm.fire("pre-commit")
        assert log == ["fired"]

    def test_fire_passes_context(self):
        hm = HookManager()
        captured = {}
        hm.register("post-commit", lambda **ctx: captured.update(ctx))
        hm.fire("post-commit", commit_hash="abc123", message="test")
        assert captured["commit_hash"] == "abc123"
        assert captured["message"] == "test"

    def test_pre_hook_rejects(self):
        hm = HookManager()
        hm.register("pre-commit", lambda **ctx: False)
        assert hm.fire("pre-commit") is False

    def test_pre_hook_accepts(self):
        hm = HookManager()
        hm.register("pre-commit", lambda **ctx: True)
        assert hm.fire("pre-commit") is True

    def test_pre_hook_none_is_accept(self):
        """Returning None (no explicit return) should not reject."""
        hm = HookManager()
        hm.register("pre-commit", lambda **ctx: None)
        assert hm.fire("pre-commit") is True

    def test_multiple_pre_hooks_one_rejects(self):
        hm = HookManager()
        hm.register("pre-commit", lambda **ctx: True)
        hm.register("pre-commit", lambda **ctx: False)
        assert hm.fire("pre-commit") is False

    def test_unregister(self):
        hm = HookManager()
        log = []
        fn = lambda **ctx: log.append("fired")
        hm.register("pre-commit", fn)
        hm.unregister("pre-commit", fn)
        hm.fire("pre-commit")
        assert log == []

    def test_unregister_nonexistent_is_noop(self):
        hm = HookManager()
        hm.unregister("pre-commit", lambda **ctx: None)  # no error

    def test_invalid_event_register(self):
        hm = HookManager()
        with pytest.raises(ValueError, match="Unknown event"):
            hm.register("pre-foo", lambda **ctx: None)

    def test_invalid_event_fire(self):
        hm = HookManager()
        with pytest.raises(ValueError, match="Unknown event"):
            hm.fire("pre-foo")

    def test_list_hooks(self):
        hm = HookManager()

        def my_hook(**ctx):
            pass

        hm.register("pre-commit", my_hook)
        hooks = hm.list_hooks()
        assert "pre-commit" in hooks
        assert "my_hook" in hooks["pre-commit"][0]

    def test_clear_event(self):
        hm = HookManager()
        hm.register("pre-commit", lambda **ctx: None)
        hm.register("post-commit", lambda **ctx: None)
        hm.clear("pre-commit")
        assert "pre-commit" not in hm.list_hooks()
        assert "post-commit" in hm.list_hooks()

    def test_clear_all(self):
        hm = HookManager()
        hm.register("pre-commit", lambda **ctx: None)
        hm.register("post-commit", lambda **ctx: None)
        hm.clear()
        assert hm.list_hooks() == {}

    def test_idempotent_register(self):
        hm = HookManager()
        fn = lambda **ctx: None
        hm.register("pre-commit", fn)
        hm.register("pre-commit", fn)
        assert len(hm._hooks["pre-commit"]) == 1

    def test_post_hook_always_returns_true(self):
        hm = HookManager()
        hm.register("post-commit", lambda **ctx: False)
        assert hm.fire("post-commit") is True

    def test_on_drift_event(self):
        hm = HookManager()
        log = []
        hm.register("on-drift", lambda **ctx: log.append(ctx.get("magnitude")))
        hm.fire("on-drift", magnitude=0.42)
        assert log == [0.42]


class TestHooksIntegration:
    def test_pre_commit_rejects(self, db):
        """Pre-commit returning False should prevent commit."""
        db.add(rvec(3), documents=["a", "b", "c"])
        db.hook("pre-commit", lambda **ctx: False)
        with pytest.raises(ValueError, match="pre-commit hook rejected"):
            db.commit("should fail")
        # Data is still staged
        assert db.status()["staged_additions"] == 3

    def test_post_commit_fires(self, db):
        log = []
        db.hook("post-commit", lambda **ctx: log.append(ctx["commit_hash"]))
        db.add(rvec(3), documents=["a", "b", "c"])
        h = db.commit("test")
        assert log == [h]

    def test_pre_commit_accepts(self, db):
        db.hook("pre-commit", lambda **ctx: True)
        db.add(rvec(3))
        h = db.commit("should pass")
        assert len(h) == 64

    def test_hook_unhook(self, db):
        log = []
        fn = lambda **ctx: log.append("x")
        db.hook("post-commit", fn)
        db.add(rvec(3))
        db.commit("first")
        assert len(log) == 1
        db.unhook("post-commit", fn)
        db.add(rvec(3))
        db.commit("second")
        assert len(log) == 1  # no new append

    def test_pre_merge_rejects(self, db):
        db.add(rvec(5))
        db.commit("base")
        db.branch("feature")
        db.switch("feature")
        db.add(rvec(3))
        db.commit("feature work")
        db.switch("main")
        db.hook("pre-merge", lambda **ctx: False)
        with pytest.raises(ValueError, match="pre-merge hook rejected"):
            db.merge("feature")

    def test_post_merge_fires(self, db):
        log = []
        db.hook("post-merge", lambda **ctx: log.append(ctx.get("branch")))
        db.add(rvec(5))
        db.commit("base")
        db.branch("feature")
        db.switch("feature")
        db.add(rvec(3))
        db.commit("feature work")
        db.switch("main")
        result = db.merge("feature")
        assert log == ["feature"]
