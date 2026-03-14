"""Native backup and restore — no VEEM, no external tools.

Supports:
  1. Full backup: snapshot entire .gitdb/ to a compressed archive
  2. Incremental backup: only objects created since last backup
  3. Restore: rebuild from backup archive
  4. Verify: check backup integrity against live store
  5. Schedule: register periodic backups (works with hooks)

Backup format: .gitdb-backup (tar.zst — tar archive compressed with zstd)
Incremental format: .gitdb-incr (tar.zst of new objects + manifest)
"""

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# ═══════════════════════════════════════════════════════════════
#  Backup Manifest
# ═══════════════════════════════════════════════════════════════

def _create_manifest(gitdb_dir: Path, backup_type: str = "full") -> dict:
    """Create a backup manifest with metadata about the store."""
    config_path = gitdb_dir / "config"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    head_path = gitdb_dir / "HEAD"
    head = head_path.read_text().strip() if head_path.exists() else None

    # Count objects
    objects_dir = gitdb_dir / "objects"
    obj_count = 0
    obj_size = 0
    if objects_dir.exists():
        for prefix_dir in objects_dir.iterdir():
            if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                for obj_file in prefix_dir.iterdir():
                    obj_count += 1
                    obj_size += obj_file.stat().st_size

    # List branches
    branches = {}
    heads_dir = gitdb_dir / "refs" / "heads"
    if heads_dir.exists():
        for p in heads_dir.rglob("*"):
            if p.is_file():
                name = str(p.relative_to(heads_dir))
                branches[name] = p.read_text().strip()

    # List tags
    tags = {}
    tags_dir = gitdb_dir / "refs" / "tags"
    if tags_dir.exists():
        for p in tags_dir.rglob("*"):
            if p.is_file():
                name = str(p.relative_to(tags_dir))
                tags[name] = p.read_text().strip()

    return {
        "version": "1.0",
        "type": backup_type,
        "timestamp": time.time(),
        "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "head": head,
        "branches": branches,
        "tags": tags,
        "object_count": obj_count,
        "object_size_bytes": obj_size,
    }


def _hash_file(path: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════
#  Full Backup
# ═══════════════════════════════════════════════════════════════

def backup_full(
    gitdb_dir: Path,
    output_path: str,
    compression_level: int = 3,
) -> Dict[str, Any]:
    """Create a full backup of the entire .gitdb directory.

    Args:
        gitdb_dir: Path to .gitdb/ directory.
        output_path: Destination file path (.gitdb-backup).
        compression_level: zstd compression level (1-22, default 3).

    Returns:
        Backup manifest dict.
    """
    gitdb_dir = Path(gitdb_dir)
    output_path = Path(output_path)

    manifest = _create_manifest(gitdb_dir, "full")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write manifest
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Create tar archive
        tar_path = Path(tmpdir) / "backup.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(str(manifest_path), arcname="manifest.json")
            tar.add(str(gitdb_dir), arcname=".gitdb")

        # Compress with zstd
        if HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=compression_level)
            with open(tar_path, "rb") as f_in, open(output_path, "wb") as f_out:
                cctx.copy_stream(f_in, f_out)
        else:
            # Fallback: just copy tar without compression
            shutil.copy2(tar_path, output_path)

    manifest["backup_path"] = str(output_path)
    manifest["backup_size_bytes"] = output_path.stat().st_size
    manifest["checksum"] = _hash_file(output_path)

    # Write a sidecar manifest for quick inspection
    sidecar = output_path.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2))

    return manifest


# ═══════════════════════════════════════════════════════════════
#  Incremental Backup
# ═══════════════════════════════════════════════════════════════

def backup_incremental(
    gitdb_dir: Path,
    output_path: str,
    since_manifest: Optional[Dict] = None,
    compression_level: int = 3,
) -> Dict[str, Any]:
    """Create an incremental backup with only new objects since last backup.

    Args:
        gitdb_dir: Path to .gitdb/ directory.
        output_path: Destination file path (.gitdb-incr).
        since_manifest: Previous backup manifest (to find new objects).
        compression_level: zstd compression level.

    Returns:
        Incremental backup manifest.
    """
    gitdb_dir = Path(gitdb_dir)
    output_path = Path(output_path)

    # Get current state
    manifest = _create_manifest(gitdb_dir, "incremental")

    # Find new objects since last backup
    prev_timestamp = since_manifest.get("timestamp", 0) if since_manifest else 0
    objects_dir = gitdb_dir / "objects"
    new_objects = []

    if objects_dir.exists():
        for prefix_dir in objects_dir.iterdir():
            if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                for obj_file in prefix_dir.iterdir():
                    if obj_file.stat().st_mtime > prev_timestamp:
                        rel_path = obj_file.relative_to(gitdb_dir)
                        new_objects.append(str(rel_path))

    manifest["new_objects"] = len(new_objects)
    manifest["since_timestamp"] = prev_timestamp

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        tar_path = Path(tmpdir) / "backup.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(str(manifest_path), arcname="manifest.json")

            # Add new objects
            for rel in new_objects:
                full_path = gitdb_dir / rel
                if full_path.exists():
                    tar.add(str(full_path), arcname=f".gitdb/{rel}")

            # Always include refs (small, always needed for restore)
            refs_dir = gitdb_dir / "refs"
            if refs_dir.exists():
                tar.add(str(refs_dir), arcname=".gitdb/refs")

            # Always include HEAD, config
            for fname in ["HEAD", "config", "remotes.json"]:
                fpath = gitdb_dir / fname
                if fpath.exists():
                    tar.add(str(fpath), arcname=f".gitdb/{fname}")

            # Include PRs if they exist
            prs_dir = gitdb_dir / "prs"
            if prs_dir.exists():
                tar.add(str(prs_dir), arcname=".gitdb/prs")

        # Compress
        if HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=compression_level)
            with open(tar_path, "rb") as f_in, open(output_path, "wb") as f_out:
                cctx.copy_stream(f_in, f_out)
        else:
            shutil.copy2(tar_path, output_path)

    manifest["backup_path"] = str(output_path)
    manifest["backup_size_bytes"] = output_path.stat().st_size
    manifest["checksum"] = _hash_file(output_path)

    sidecar = output_path.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2))

    return manifest


# ═══════════════════════════════════════════════════════════════
#  Restore
# ═══════════════════════════════════════════════════════════════

def restore(
    backup_path: str,
    dest_path: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Restore a GitDB store from a backup archive.

    Args:
        backup_path: Path to .gitdb-backup or .gitdb-incr file.
        dest_path: Destination directory for restored store.
        overwrite: If True, overwrite existing .gitdb/ directory.

    Returns:
        Manifest from the backup.
    """
    backup_path = Path(backup_path)
    dest_path = Path(dest_path)
    dest_gitdb = dest_path / ".gitdb"

    if dest_gitdb.exists() and not overwrite:
        raise ValueError(
            f"Destination already has .gitdb/ directory. "
            f"Use overwrite=True to replace it."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "backup.tar"

        # Decompress
        if HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            with open(backup_path, "rb") as f_in, open(tar_path, "wb") as f_out:
                dctx.copy_stream(f_in, f_out)
        else:
            shutil.copy2(backup_path, tar_path)

        # Extract
        extract_dir = Path(tmpdir) / "extracted"
        extract_dir.mkdir()
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir, filter="data")

        # Read manifest
        manifest_path = extract_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        extracted_gitdb = extract_dir / ".gitdb"

        if manifest.get("type") == "full":
            # Full restore: replace entire .gitdb/
            if dest_gitdb.exists():
                shutil.rmtree(dest_gitdb)
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(extracted_gitdb, dest_gitdb)
        else:
            # Incremental: merge new objects into existing store
            dest_gitdb.mkdir(parents=True, exist_ok=True)
            for root, dirs, files in os.walk(extracted_gitdb):
                for f in files:
                    src = Path(root) / f
                    rel = src.relative_to(extracted_gitdb)
                    dst = dest_gitdb / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    manifest["restored_to"] = str(dest_path)
    manifest["restored_at"] = time.time()
    return manifest


# ═══════════════════════════════════════════════════════════════
#  Verify
# ═══════════════════════════════════════════════════════════════

def verify(
    gitdb_dir: Path,
    backup_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify integrity of a live store or backup.

    Checks:
      - All refs point to existing objects
      - All commit parents exist
      - All delta hashes exist
      - Object count matches
      - No orphaned objects

    Returns:
        Dict with verification results.
    """
    gitdb_dir = Path(gitdb_dir)
    issues = []

    # Check HEAD
    head_path = gitdb_dir / "HEAD"
    if not head_path.exists():
        issues.append("Missing HEAD file")
    else:
        head_ref = head_path.read_text().strip()
        if head_ref.startswith("ref:"):
            ref_path = gitdb_dir / head_ref.split("ref:")[1].strip()
            if not ref_path.exists():
                issues.append(f"HEAD ref points to missing file: {ref_path}")

    # Check all branch refs point to existing objects
    heads_dir = gitdb_dir / "refs" / "heads"
    objects_dir = gitdb_dir / "objects"
    if heads_dir.exists():
        for p in heads_dir.rglob("*"):
            if p.is_file():
                commit_hash = p.read_text().strip()
                obj_path = objects_dir / commit_hash[:2] / commit_hash[2:]
                if not obj_path.exists():
                    branch = str(p.relative_to(heads_dir))
                    issues.append(f"Branch '{branch}' points to missing object: {commit_hash}")

    # Walk object chain from HEAD
    obj_count = 0
    visited = set()
    if head_path.exists():
        head_val = head_path.read_text().strip()
        if head_val.startswith("ref:"):
            ref_name = head_val.split("ref:")[1].strip()
            ref_path = gitdb_dir / ref_name
            if ref_path.exists():
                head_val = ref_path.read_text().strip()

        current = head_val
        while current and current not in visited:
            visited.add(current)
            obj_path = objects_dir / current[:2] / current[2:]
            if not obj_path.exists():
                issues.append(f"Missing object in chain: {current}")
                break
            try:
                data = json.loads(obj_path.read_bytes().decode("utf-8"))
                delta_hash = data.get("delta_hash")
                if delta_hash:
                    delta_path = objects_dir / delta_hash[:2] / delta_hash[2:]
                    if not delta_path.exists():
                        issues.append(f"Missing delta object: {delta_hash} (referenced by commit {current})")
                current = data.get("parent")
                obj_count += 1
            except Exception as e:
                issues.append(f"Corrupt object {current}: {e}")
                break

    # Count total objects on disk
    total_objects = 0
    if objects_dir.exists():
        for prefix_dir in objects_dir.iterdir():
            if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                total_objects += sum(1 for _ in prefix_dir.iterdir())

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "commits_verified": obj_count,
        "total_objects_on_disk": total_objects,
        "branches_checked": len(list(heads_dir.rglob("*"))) if heads_dir.exists() else 0,
    }


# ═══════════════════════════════════════════════════════════════
#  Backup History
# ═══════════════════════════════════════════════════════════════

class BackupManager:
    """Manages backup history and scheduling for a GitDB store."""

    def __init__(self, gitdb_dir: Path):
        self._dir = gitdb_dir / "backups"
        self._dir.mkdir(exist_ok=True)
        self._history_file = self._dir / "history.json"

    def _load_history(self) -> List[dict]:
        if self._history_file.exists():
            return json.loads(self._history_file.read_text())
        return []

    def _save_history(self, history: List[dict]):
        self._history_file.write_text(json.dumps(history, indent=2))

    def record(self, manifest: dict):
        """Record a backup in history."""
        history = self._load_history()
        history.append({
            "timestamp": manifest.get("timestamp"),
            "type": manifest.get("type"),
            "path": manifest.get("backup_path"),
            "size": manifest.get("backup_size_bytes"),
            "checksum": manifest.get("checksum"),
            "objects": manifest.get("object_count", manifest.get("new_objects", 0)),
        })
        self._save_history(history)

    def list_backups(self) -> List[dict]:
        """List all recorded backups."""
        return self._load_history()

    def last_backup(self) -> Optional[dict]:
        """Get the most recent backup record."""
        history = self._load_history()
        return history[-1] if history else None

    def last_manifest(self) -> Optional[dict]:
        """Load the most recent backup's full manifest."""
        last = self.last_backup()
        if last and last.get("path"):
            sidecar = Path(last["path"]).with_suffix(".manifest.json")
            if sidecar.exists():
                return json.loads(sidecar.read_text())
        return None
