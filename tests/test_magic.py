"""Tests for Phase 2: Magic, Black Magic, and Hidden Black Magic commands."""

import pytest
import torch
import torch.nn.functional as F

from gitdb import GitDB, MergeResult, BlameEntry, BisectResult, StashEntry


@pytest.fixture
def db(tmp_path):
    return GitDB(str(tmp_path / "store"), dim=8, device="cpu")


def rvec(n=1, dim=8):
    return torch.randn(n, dim)


def qvec(dim=8):
    return torch.randn(dim)


# ═══════════════════════════════════════════════════════════════
#  MAGIC: Cherry-pick
# ═══════════════════════════════════════════════════════════════

class TestCherryPick:
    def test_cherry_pick_additions(self, db):
        db.add(rvec(5), documents=[f"base{i}" for i in range(5)])
        h1 = db.commit("Base")

        # Create a branch with new vectors
        db.branch("feature")
        db.switch("feature")
        db.add(rvec(3), documents=["feat0", "feat1", "feat2"])
        h_feat = db.commit("Feature work")

        # Switch back to main
        db.switch("main")
        assert db.tree.size == 5

        # Cherry-pick the feature commit
        h_cp = db.cherry_pick(h_feat)
        assert db.tree.size == 8
        log = db.log()
        assert "cherry-pick" in log[0].message

    def test_cherry_pick_preserves_data(self, db):
        db.add(rvec(3), documents=["a", "b", "c"])
        db.commit("Base")

        db.branch("other")
        db.switch("other")
        special_vec = rvec(1)
        db.add(special_vec, documents=["special"])
        h_other = db.commit("Special")

        db.switch("main")
        db.cherry_pick(h_other)

        # Should be able to find the special vector
        results = db.query(special_vec.squeeze(), k=1)
        assert results.documents[0] == "special"


# ═══════════════════════════════════════════════════════════════
#  MAGIC: Revert
# ═══════════════════════════════════════════════════════════════

class TestRevert:
    def test_revert_additions(self, db):
        db.add(rvec(5), documents=[f"d{i}" for i in range(5)])
        h1 = db.commit("Base 5")

        db.add(rvec(3), documents=["bad0", "bad1", "bad2"])
        h2 = db.commit("Bad batch")
        assert db.tree.size == 8

        # Revert the bad batch
        db.revert(h2)
        assert db.tree.size == 5

    def test_revert_creates_new_commit(self, db):
        db.add(rvec(3))
        h1 = db.commit("First")
        db.add(rvec(2))
        h2 = db.commit("Second")

        h_revert = db.revert(h2)
        log = db.log()
        assert len(log) == 3
        assert "revert" in log[0].message


# ═══════════════════════════════════════════════════════════════
#  MAGIC: Branch + Merge
# ═══════════════════════════════════════════════════════════════

class TestBranching:
    def test_create_branch(self, db):
        db.add(rvec(3))
        db.commit("Init")
        db.branch("feature")
        branches = db.branches()
        assert "main" in branches
        assert "feature" in branches

    def test_branch_already_exists(self, db):
        db.add(rvec(3))
        db.commit("Init")
        db.branch("feature")
        with pytest.raises(ValueError, match="already exists"):
            db.branch("feature")

    def test_switch_branch(self, db):
        db.add(rvec(3), documents=["a", "b", "c"])
        db.commit("Main work")

        db.branch("feature")
        db.switch("feature")
        assert db.refs.current_branch == "feature"

        db.add(rvec(2), documents=["feat1", "feat2"])
        db.commit("Feature work")
        assert db.tree.size == 5

        db.switch("main")
        assert db.tree.size == 3

    def test_switch_nonexistent(self, db):
        db.add(rvec(1))
        db.commit("x")
        with pytest.raises(ValueError, match="not found"):
            db.switch("nope")


class TestMerge:
    def test_union_merge(self, db):
        db.add(rvec(3), documents=["m1", "m2", "m3"])
        db.commit("Main base")

        db.branch("feature")
        db.switch("feature")
        db.add(rvec(2), documents=["f1", "f2"])
        db.commit("Feature adds")

        db.switch("main")
        result = db.merge("feature")
        assert db.tree.size == 5  # 3 main + 2 feature

    def test_merge_ours_strategy(self, db):
        db.add(rvec(3))
        db.commit("Base")

        db.branch("other")
        db.switch("other")
        db.add(rvec(5))
        db.commit("Other adds 5")

        db.switch("main")
        result = db.merge("other", strategy="ours")
        assert db.tree.size == 3  # unchanged

    def test_merge_theirs_strategy(self, db):
        db.add(rvec(3))
        db.commit("Base")

        db.branch("other")
        db.switch("other")
        db.add(rvec(5))
        db.commit("Other adds 5")

        db.switch("main")
        result = db.merge("other", strategy="theirs")
        assert db.tree.size == 8  # theirs has 3+5

    def test_merge_same_branch(self, db):
        db.add(rvec(3))
        db.commit("Base")
        db.branch("same")
        result = db.merge("same")
        assert db.tree.size == 3

    def test_fast_forward_merge(self, db):
        db.add(rvec(3))
        db.commit("Base")

        db.branch("feature")
        db.switch("feature")
        db.add(rvec(2))
        db.commit("Ahead")

        db.switch("main")
        result = db.merge("feature")
        assert result.strategy == "fast-forward"
        assert db.tree.size == 5


# ═══════════════════════════════════════════════════════════════
#  MAGIC: Stash
# ═══════════════════════════════════════════════════════════════

class TestStash:
    def test_stash_and_pop(self, db):
        db.add(rvec(5), documents=[f"d{i}" for i in range(5)])
        db.commit("Base")

        # Add more (unstaged from HEAD's perspective)
        db.add(rvec(3), documents=["wip1", "wip2", "wip3"])
        assert db.tree.size == 8

        db.stash("My WIP")
        assert db.tree.size == 5  # back to HEAD

        entry = db.stash_pop()
        assert entry.message == "My WIP"
        assert db.tree.size == 8  # restored

    def test_stash_list(self, db):
        db.add(rvec(3))
        db.commit("Base")

        db.add(rvec(1))
        db.stash("First")

        db.add(rvec(2))
        db.stash("Second")

        stashes = db.stash_list()
        assert len(stashes) == 2
        assert stashes[0].message == "First"
        assert stashes[1].message == "Second"

    def test_stash_nothing(self, db):
        db.add(rvec(3))
        db.commit("Base")
        with pytest.raises(ValueError, match="Nothing to stash"):
            db.stash()


# ═══════════════════════════════════════════════════════════════
#  BLACK MAGIC: Blame
# ═══════════════════════════════════════════════════════════════

class TestBlame:
    def test_blame_tracks_origin(self, db):
        v1 = rvec(3)
        db.add(v1, documents=["batch1_a", "batch1_b", "batch1_c"])
        h1 = db.commit("Batch 1 ingest")

        v2 = rvec(2)
        db.add(v2, documents=["batch2_a", "batch2_b"])
        h2 = db.commit("Batch 2 ingest")

        blame = db.blame()
        assert len(blame) == 5

        # First 3 should trace to h1
        batch1_entries = [b for b in blame if "Batch 1" in b.commit_message]
        assert len(batch1_entries) == 3

        # Last 2 should trace to h2
        batch2_entries = [b for b in blame if "Batch 2" in b.commit_message]
        assert len(batch2_entries) == 2

    def test_blame_specific_ids(self, db):
        db.add(rvec(5), ids=["aaa", "bbb", "ccc", "ddd", "eee"])
        db.commit("All")

        blame = db.blame(ids=["bbb", "ddd"])
        assert len(blame) == 2
        assert {b.vector_id for b in blame} == {"bbb", "ddd"}


# ═══════════════════════════════════════════════════════════════
#  BLACK MAGIC: Bisect
# ═══════════════════════════════════════════════════════════════

class TestBisect:
    def test_bisect_finds_bad_commit(self, db):
        # Build a history where quality degrades at commit 5
        poison_vec = torch.ones(1, 8) * 999  # obviously bad

        for i in range(10):
            if i == 5:
                db.add(poison_vec, documents=["POISON"])
            else:
                db.add(rvec(3), documents=[f"good{i}a", f"good{i}b", f"good{i}c"])
            db.commit(f"Commit {i}")

        # Test function: bad if poison is in top results
        query = torch.ones(8) * 999
        def is_good(snap):
            results = snap.query(query, k=1)
            if len(results) == 0:
                return True
            return results.scores[0] < 0.99

        result = db.bisect(is_good)
        assert result.bad_message == "Commit 5"
        assert result.steps < 10  # binary search is efficient


# ═══════════════════════════════════════════════════════════════
#  BLACK MAGIC: Rebase
# ═══════════════════════════════════════════════════════════════

class TestRebase:
    def test_rebase_replays_commits(self, db):
        db.add(rvec(3), documents=["base1", "base2", "base3"])
        db.commit("Base")

        # Create feature branch with 2 commits
        db.branch("feature")
        db.switch("feature")
        db.add(rvec(2), documents=["feat1", "feat2"])
        db.commit("Feature 1")
        db.add(rvec(1), documents=["feat3"])
        db.commit("Feature 2")

        # Meanwhile, main advances
        db.switch("main")
        db.add(rvec(2), documents=["main1", "main2"])
        db.commit("Main advance")

        # Rebase feature onto main
        db.switch("feature")
        new_hashes = db.rebase("main")
        assert len(new_hashes) == 2  # replayed 2 commits
        assert db.tree.size == 8  # 3 base + 2 main + 2 feat + 1 feat


# ═══════════════════════════════════════════════════════════════
#  HIDDEN BLACK MAGIC: GC
# ═══════════════════════════════════════════════════════════════

class TestGC:
    def test_gc_creates_checkpoints(self, db):
        for i in range(15):
            db.add(rvec(1))
            db.commit(f"Commit {i}")

        db.gc(keep_last=5)
        # Cache should have entries
        cache_files = list(db._cache_dir.glob("*.pt"))
        assert len(cache_files) > 0


# ═══════════════════════════════════════════════════════════════
#  HIDDEN BLACK MAGIC: Reflog
# ═══════════════════════════════════════════════════════════════

class TestReflog:
    def test_reflog_tracks_commits(self, db):
        db.add(rvec(3))
        db.commit("First")
        db.add(rvec(2))
        db.commit("Second")

        entries = db.reflog()
        assert len(entries) == 2
        assert "First" in entries[1]["action"]
        assert "Second" in entries[0]["action"]

    def test_reflog_tracks_checkout(self, db):
        db.add(rvec(3))
        h1 = db.commit("First")
        db.add(rvec(2))
        h2 = db.commit("Second")
        db.checkout(h1)

        entries = db.reflog()
        assert any("checkout" in e["action"] for e in entries)

    def test_reflog_empty(self, db):
        assert db.reflog() == []


# ═══════════════════════════════════════════════════════════════
#  HIDDEN BLACK MAGIC: Filter-Branch
# ═══════════════════════════════════════════════════════════════

class TestFilterBranch:
    def test_normalize_all_embeddings(self, db):
        db.add(torch.randn(10, 8) * 5)  # unnormalized
        db.commit("Raw embeddings")

        # Check they're NOT normalized
        norms = db.tree.embeddings.norm(dim=1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=0.01)

        # Filter-branch to normalize
        db.filter_branch(lambda t: F.normalize(t, dim=1), "normalize")

        # Now they should be unit vectors
        norms = db.tree.embeddings.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_filter_branch_creates_commit(self, db):
        db.add(rvec(5))
        db.commit("Base")
        db.filter_branch(lambda t: t * 2, "scale")
        log = db.log()
        assert len(log) == 2
        assert "filter-branch" in log[0].message or "scale" in log[0].message

    def test_filter_branch_shape_mismatch(self, db):
        db.add(rvec(5))
        db.commit("Base")
        with pytest.raises(ValueError, match="preserve shape"):
            db.filter_branch(lambda t: t[:3])  # wrong shape


# ═══════════════════════════════════════════════════════════════
#  Relative Refs (HEAD~N, main~N)
# ═══════════════════════════════════════════════════════════════

class TestRelativeRefs:
    def test_head_tilde(self, db):
        db.add(rvec(3), documents=["a", "b", "c"])
        h1 = db.commit("First")
        db.add(rvec(2))
        h2 = db.commit("Second")
        db.add(rvec(1))
        h3 = db.commit("Third")

        # HEAD~1 should be h2
        resolved = db._resolve("HEAD~1")
        assert resolved == h2

        # HEAD~2 should be h1
        resolved = db._resolve("HEAD~2")
        assert resolved == h1

    def test_query_at_relative_ref(self, db):
        vecs = rvec(5)
        db.add(vecs, documents=[f"old{i}" for i in range(5)])
        h1 = db.commit("Old")

        db.add(rvec(5), documents=[f"new{i}" for i in range(5)])
        db.commit("New")

        results = db.query(vecs[0].squeeze(), k=3, at="HEAD~1")
        assert all("old" in d for d in results.documents)


# ═══════════════════════════════════════════════════════════════
#  FORBIDDEN BLACK MAGIC: Purge (History Rewrite)
# ═══════════════════════════════════════════════════════════════

class TestPurge:
    def test_purge_by_where(self, db):
        """Remove all vectors matching a filter from ALL history."""
        db.add(rvec(3), documents=["good1", "good2", "good3"],
               metadata=[{"author": "human"}, {"author": "human"}, {"author": "human"}])
        db.commit("Human batch")

        db.add(rvec(2), documents=["ai1", "ai2"],
               metadata=[{"author": "claude"}, {"author": "claude"}])
        db.commit("AI batch")

        db.add(rvec(1), documents=["good4"],
               metadata=[{"author": "human"}])
        db.commit("More human")

        assert db.tree.size == 6

        # Purge all claude vectors from history
        result = db.purge(where={"author": "claude"}, reason="attribution cleanup")
        assert result["vectors_purged"] == 2
        assert db.tree.size == 4

        # Verify they're gone from ALL history, not just HEAD
        # Check every commit in the log
        for commit_info in db.log():
            tensor, meta = db._reconstruct(commit_info.hash)
            for m in meta:
                assert m.metadata.get("author") != "claude", \
                    f"Found purged vector in commit {commit_info.hash[:8]}"

    def test_purge_by_ids(self, db):
        db.add(rvec(5), ids=["keep1", "purge1", "keep2", "purge2", "keep3"])
        db.commit("Mixed")

        result = db.purge(ids=["purge1", "purge2"])
        assert result["vectors_purged"] == 2
        assert db.tree.size == 3

        # Verify purged IDs are gone
        current_ids = {m.id for m in db.tree.metadata}
        assert "purge1" not in current_ids
        assert "purge2" not in current_ids
        assert "keep1" in current_ids

    def test_purge_rewrites_all_commits(self, db):
        db.add(rvec(3), metadata=[{"src": "a"}, {"src": "b"}, {"src": "a"}])
        h1 = db.commit("First")
        db.add(rvec(2), metadata=[{"src": "a"}, {"src": "b"}])
        h2 = db.commit("Second")

        result = db.purge(where={"src": "b"})
        assert result["commits_rewritten"] >= 2

        # Old hashes should no longer be the head
        assert db.refs.get_head_commit() != h2

    def test_purge_nothing_matches(self, db):
        db.add(rvec(3))
        db.commit("Base")
        result = db.purge(where={"nonexistent": "value"})
        assert result["vectors_purged"] == 0

    def test_purge_preserves_non_target_data(self, db):
        good_vecs = rvec(3)
        db.add(good_vecs, documents=["safe1", "safe2", "safe3"],
               metadata=[{"keep": True}] * 3)
        db.commit("Safe data")

        db.add(rvec(2), documents=["toxic1", "toxic2"],
               metadata=[{"keep": False}] * 2)
        db.commit("Toxic data")

        db.purge(where={"keep": False})

        # Safe data should be perfectly intact and queryable
        results = db.query(good_vecs[0].squeeze(), k=1)
        assert results.documents[0] == "safe1"
        assert len(db) == 3

    def test_purge_shows_in_reflog(self, db):
        db.add(rvec(3), metadata=[{"x": 1}] * 3)
        db.commit("Base")
        db.purge(where={"x": 1}, reason="test cleanup")
        entries = db.reflog()
        assert any("purge" in e["action"] for e in entries)

    def test_purge_requires_ids_or_where(self, db):
        db.add(rvec(1))
        db.commit("x")
        with pytest.raises(ValueError, match="Must specify"):
            db.purge()


# ═══════════════════════════════════════════════════════════════
#  HEAD, Show, Amend, Squash, Fork, Notes, Branch Mgmt
# ═══════════════════════════════════════════════════════════════

class TestHEAD:
    def test_head_property(self, db):
        assert db.HEAD is None
        db.add(rvec(3))
        h = db.commit("First")
        assert db.HEAD == h

    def test_current_branch(self, db):
        assert db.current_branch == "main"


class TestShow:
    def test_show_commit(self, db):
        db.add(rvec(5), documents=["a", "b", "c", "d", "e"])
        h = db.commit("Five vectors")
        info = db.show(h)
        assert info["hash"] == h
        assert info["message"] == "Five vectors"
        assert info["stats"]["added"] == 5
        assert info["tensor_rows"] == 5
        assert info["delta_size_bytes"] > 0

    def test_show_head(self, db):
        db.add(rvec(3))
        db.commit("Test")
        info = db.show()  # defaults to HEAD
        assert info["message"] == "Test"


class TestAmend:
    def test_amend_message(self, db):
        db.add(rvec(3))
        h1 = db.commit("Typo msg")
        h2 = db.amend(message="Fixed message")
        assert h2 != h1
        log = db.log()
        assert len(log) == 1
        assert log[0].message == "Fixed message"

    def test_amend_with_staged(self, db):
        db.add(rvec(3))
        db.commit("Base")
        db.add(rvec(2))  # staged but not committed
        h = db.amend(message="Base + extras")
        log = db.log()
        assert len(log) == 1
        assert db.tree.size == 5

    def test_amend_nothing(self, db):
        db.add(rvec(3))
        db.commit("Fine")
        with pytest.raises(ValueError, match="Nothing to amend"):
            db.amend()


class TestSquash:
    def test_squash_commits(self, db):
        db.add(rvec(2))
        db.commit("First")
        db.add(rvec(3))
        db.commit("Second")
        db.add(rvec(1))
        db.commit("Third")
        assert len(db.log()) == 3

        h = db.squash(3, message="All in one")
        log = db.log()
        assert len(log) == 1
        assert log[0].message == "All in one"
        assert db.tree.size == 6

    def test_squash_auto_message(self, db):
        db.add(rvec(1))
        db.commit("A")
        db.add(rvec(1))
        db.commit("B")
        db.squash(2)
        log = db.log()
        assert "A" in log[0].message and "B" in log[0].message

    def test_squash_too_few(self, db):
        db.add(rvec(1))
        db.commit("Only one")
        with pytest.raises(ValueError):
            db.squash(1)


class TestFork:
    def test_fork_copies_everything(self, db, tmp_path):
        db.add(rvec(5), documents=[f"d{i}" for i in range(5)])
        db.commit("Source data")
        db.tag("v1.0")

        fork_path = str(tmp_path / "forked")
        forked = db.fork(fork_path)
        assert forked.tree.size == 5
        assert forked.HEAD is not None
        assert forked.refs.resolve("v1.0") is not None

    def test_clone_alias(self, db, tmp_path):
        db.add(rvec(3))
        db.commit("Data")
        clone_path = str(tmp_path / "cloned")
        cloned = db.clone(clone_path)
        assert cloned.tree.size == 3

    def test_fork_isolation(self, db, tmp_path):
        db.add(rvec(3))
        db.commit("Base")
        forked = db.fork(str(tmp_path / "fork"))

        # Add to original — fork should not see it
        db.add(rvec(2))
        db.commit("Original only")
        assert db.tree.size == 5
        assert forked.tree.size == 3


class TestNotes:
    def test_add_and_read_note(self, db):
        db.add(rvec(3))
        h = db.commit("Annotated")
        db.note(h, "This commit needs review")
        db.note(h, "Approved by Vincent")
        notes = db.notes(h)
        assert len(notes) == 2
        assert notes[0]["message"] == "This commit needs review"
        assert notes[1]["message"] == "Approved by Vincent"

    def test_notes_empty(self, db):
        db.add(rvec(1))
        h = db.commit("No notes")
        assert db.notes(h) == []


class TestBranchManagement:
    def test_delete_branch(self, db):
        db.add(rvec(3))
        db.commit("Base")
        db.branch("temp")
        assert "temp" in db.branches()
        db.delete_branch("temp")
        assert "temp" not in db.branches()

    def test_delete_current_branch_fails(self, db):
        db.add(rvec(1))
        db.commit("x")
        with pytest.raises(ValueError, match="Cannot delete current"):
            db.delete_branch("main")

    def test_rename_branch(self, db):
        db.add(rvec(3))
        db.commit("Base")
        db.branch("old-name")
        db.rename_branch("old-name", "new-name")
        assert "new-name" in db.branches()
        assert "old-name" not in db.branches()

    def test_rename_current_branch(self, db):
        db.add(rvec(3))
        db.commit("Base")
        db.rename_branch("main", "primary")
        assert db.current_branch == "primary"
