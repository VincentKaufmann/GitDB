"""Tests for remotes, push/pull/fetch, PRs, and comments."""

import pytest
import torch

from gitdb import GitDB


@pytest.fixture
def origin(tmp_path):
    """A 'remote' GitDB store."""
    db = GitDB(str(tmp_path / "origin"), dim=8, device="cpu")
    db.add(torch.randn(5, 8), documents=[f"orig{i}" for i in range(5)])
    db.commit("Origin initial")
    return db


@pytest.fixture
def local(tmp_path, origin):
    """A local clone with origin configured."""
    db = origin.clone(str(tmp_path / "local"))
    db.remote_add("origin", str(origin.path))
    return db


def rvec(n=1, dim=8):
    return torch.randn(n, dim)


# ═══════════════════════════════════════════════════════════════
#  Remotes
# ═══════════════════════════════════════════════════════════════

class TestRemotes:
    def test_add_remote(self, tmp_path):
        db = GitDB(str(tmp_path / "store"), dim=8, device="cpu")
        db.add(rvec(1))
        db.commit("x")
        db.remote_add("origin", "/some/path")
        assert "origin" in db.remotes()

    def test_remove_remote(self, tmp_path):
        db = GitDB(str(tmp_path / "store"), dim=8, device="cpu")
        db.add(rvec(1))
        db.commit("x")
        db.remote_add("origin", "/some/path")
        db.remote_remove("origin")
        assert "origin" not in db.remotes()

    def test_duplicate_remote(self, tmp_path):
        db = GitDB(str(tmp_path / "store"), dim=8, device="cpu")
        db.add(rvec(1))
        db.commit("x")
        db.remote_add("origin", "/a")
        with pytest.raises(ValueError, match="already exists"):
            db.remote_add("origin", "/b")


# ═══════════════════════════════════════════════════════════════
#  Push / Pull / Fetch (local remotes)
# ═══════════════════════════════════════════════════════════════

class TestPushPull:
    def test_push_new_commits(self, local, origin):
        local.add(rvec(3), documents=["new1", "new2", "new3"])
        local.commit("Local work")

        result = local.push("origin", "main")
        assert result["status"] == "pushed"
        assert result["objects_pushed"] > 0

        # Origin should now have the new commit
        origin2 = GitDB(str(origin.path), dim=8, device="cpu")
        assert origin2.tree.size == 8  # 5 original + 3 new

    def test_push_up_to_date(self, local):
        result = local.push("origin", "main")
        assert result["status"] == "up-to-date"

    def test_pull_new_commits(self, local, origin):
        # Add to origin directly
        origin.add(rvec(2), documents=["remote1", "remote2"])
        origin.commit("Remote work")

        result = local.pull("origin", "main")
        assert result["objects_fetched"] > 0
        assert local.tree.size == 7  # 5 + 2

    def test_pull_up_to_date(self, local):
        result = local.pull("origin", "main")
        assert result["status"] == "up-to-date"

    def test_fetch_without_merge(self, local, origin):
        origin.add(rvec(2))
        origin.commit("Remote")

        result = local.fetch("origin", "main")
        assert result["objects_fetched"] > 0
        # Local tree should NOT be updated (fetch only)
        assert local.tree.size == 5

    def test_push_branch(self, local, origin):
        local.branch("feature")
        local.switch("feature")
        local.add(rvec(3))
        local.commit("Feature work")

        result = local.push("origin", "feature")
        assert result["status"] == "pushed"

        # Origin should have the branch
        origin2 = GitDB(str(origin.path), dim=8, device="cpu")
        assert "feature" in origin2.branches()


# ═══════════════════════════════════════════════════════════════
#  Pull Requests
# ═══════════════════════════════════════════════════════════════

class TestPullRequests:
    def test_create_pr(self, local):
        local.branch("feature")
        local.switch("feature")
        local.add(rvec(3), documents=["f1", "f2", "f3"])
        local.commit("Feature")

        pr = local.pr_create(
            title="Add feature vectors",
            source_branch="feature",
            target_branch="main",
            description="3 new vectors from the feature branch",
            author="Vincent",
        )
        assert pr.id == 1
        assert pr.status == "open"
        assert pr.title == "Add feature vectors"

    def test_list_prs(self, local):
        local.branch("feat1")
        local.branch("feat2")
        local.pr_create("First PR", source_branch="feat1")
        local.pr_create("Second PR", source_branch="feat2")
        prs = local.pr_list()
        assert len(prs) == 2

    def test_pr_comment(self, local):
        local.branch("feat")
        pr = local.pr_create("Test PR", source_branch="feat")
        local.pr_comment(pr.id, "Slick", "Looks good to me")
        local.pr_comment(pr.id, "Vincent", "Ship it!")

        pr = local.pr_show(pr.id)
        assert len(pr.comments) == 2
        assert pr.comments[0].author == "Slick"
        assert pr.comments[1].message == "Ship it!"

    def test_pr_merge(self, local):
        local.branch("feature")
        local.switch("feature")
        local.add(rvec(3))
        local.commit("Feature vectors")
        local.switch("main")

        pr = local.pr_create("Merge feature", source_branch="feature")
        result = local.pr_merge(pr.id)

        pr = local.pr_show(pr.id)
        assert pr.status == "merged"
        assert pr.merge_commit is not None
        assert local.tree.size == 8

    def test_pr_close(self, local):
        local.branch("wontfix")
        pr = local.pr_create("Bad idea", source_branch="wontfix")
        local.pr_close(pr.id)
        pr = local.pr_show(pr.id)
        assert pr.status == "closed"

    def test_pr_filter_by_status(self, local):
        local.branch("a")
        local.branch("b")
        local.pr_create("Open one", source_branch="a")
        pr2 = local.pr_create("Will close", source_branch="b")
        local.pr_close(pr2.id)

        open_prs = local.pr_list(status="open")
        assert len(open_prs) == 1
        closed_prs = local.pr_list(status="closed")
        assert len(closed_prs) == 1

    def test_merge_closed_pr_fails(self, local):
        local.branch("x")
        pr = local.pr_create("Test", source_branch="x")
        local.pr_close(pr.id)
        with pytest.raises(ValueError, match="closed"):
            local.pr_merge(pr.id)


# ═══════════════════════════════════════════════════════════════
#  Comments on Commits
# ═══════════════════════════════════════════════════════════════

class TestComments:
    def test_comment_on_commit(self, local):
        h = local.HEAD
        local.comment(h, "Vincent", "This ingest looks clean")
        local.comment(h, "Slick", "Agreed, all vectors normalized")

        comments = local.comments(h)
        assert len(comments) == 2
        assert "Vincent" in comments[0]["message"]

    def test_comment_on_head(self, local):
        local.comment("HEAD", "reviewer", "LGTM")
        comments = local.comments()
        assert len(comments) == 1
