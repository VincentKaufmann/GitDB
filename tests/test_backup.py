"""Tests for native backup and restore."""

import json
import pytest
import torch

from gitdb import GitDB


@pytest.fixture
def db(tmp_path):
    db = GitDB(str(tmp_path / "store"), dim=8, device="cpu")
    db.add(torch.randn(5, 8), documents=[f"doc_{i}" for i in range(5)],
           metadata=[{"idx": i, "category": "A"} for i in range(5)])
    db.commit("Initial data")
    return db


class TestFullBackup:
    def test_backup_creates_file(self, db, tmp_path):
        out = str(tmp_path / "test.gitdb-backup")
        manifest = db.backup(out)
        assert (tmp_path / "test.gitdb-backup").exists()
        assert manifest["type"] == "full"
        assert manifest["object_count"] > 0

    def test_backup_manifest_sidecar(self, db, tmp_path):
        out = str(tmp_path / "test.gitdb-backup")
        db.backup(out)
        sidecar = tmp_path / "test.manifest.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["type"] == "full"

    def test_backup_records_history(self, db, tmp_path):
        out = str(tmp_path / "test.gitdb-backup")
        db.backup(out)
        backups = db.backup_list()
        assert len(backups) == 1
        assert backups[0]["type"] == "full"


class TestIncrementalBackup:
    def test_incremental_backup(self, db, tmp_path):
        # First full backup
        db.backup(str(tmp_path / "full.gitdb-backup"))
        # Add more data
        db.add(torch.randn(3, 8), documents=["new_0", "new_1", "new_2"])
        db.commit("More data")
        # Incremental
        out = str(tmp_path / "incr.gitdb-incr")
        manifest = db.backup_incremental(out)
        assert manifest["type"] == "incremental"
        assert (tmp_path / "incr.gitdb-incr").exists()

    def test_incremental_history(self, db, tmp_path):
        db.backup(str(tmp_path / "full.gitdb-backup"))
        db.add(torch.randn(2, 8), documents=["x", "y"])
        db.commit("Add")
        db.backup_incremental(str(tmp_path / "incr.gitdb-incr"))
        backups = db.backup_list()
        assert len(backups) == 2
        assert backups[1]["type"] == "incremental"


class TestRestore:
    def test_restore_full(self, db, tmp_path):
        out = str(tmp_path / "test.gitdb-backup")
        db.backup(out)
        # Restore to new location
        dest = tmp_path / "restored"
        from gitdb.backup import restore
        manifest = restore(out, str(dest))
        assert (dest / ".gitdb").exists()
        assert manifest["type"] == "full"

    def test_restore_and_query(self, db, tmp_path):
        out = str(tmp_path / "test.gitdb-backup")
        db.backup(out)
        dest = tmp_path / "restored"
        from gitdb.backup import restore
        restore(out, str(dest))
        db2 = GitDB(str(dest), dim=8, device="cpu")
        assert db2.tree.size == 5
        results = db2.query(torch.randn(8), k=3)
        assert len(results.ids) == 3

    def test_restore_overwrite_needed(self, db, tmp_path):
        out = str(tmp_path / "test.gitdb-backup")
        db.backup(out)
        dest = tmp_path / "restored"
        from gitdb.backup import restore
        restore(out, str(dest))
        # Second restore without overwrite should fail
        with pytest.raises(ValueError, match="overwrite"):
            restore(out, str(dest))
        # With overwrite should work
        restore(out, str(dest), overwrite=True)


class TestVerify:
    def test_verify_valid_store(self, db):
        result = db.backup_verify()
        assert result["valid"] is True
        assert result["commits_verified"] >= 1

    def test_verify_reports_issues(self, db):
        # Corrupt a ref by pointing to nonexistent object
        import os
        heads_dir = db._gitdb_dir / "refs" / "heads"
        branch_file = heads_dir / db.refs.current_branch
        branch_file.write_text("deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
        result = db.backup_verify()
        assert result["valid"] is False
        assert len(result["issues"]) > 0
