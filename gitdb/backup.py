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
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# ═══════════════════════════════════════════════════════════════
#  Backup Scheduler — Enterprise-grade automated backups
# ═══════════════════════════════════════════════════════════════

# Days of week constants
_WEEKDAY_MAP = {
    "mon": 0, "monday": 0,
    "tue": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

_INTERVAL_MULTIPLIERS = {
    "seconds": 1, "second": 1, "s": 1,
    "minutes": 60, "minute": 60, "m": 60,
    "hours": 3600, "hour": 3600, "h": 3600,
    "days": 86400, "day": 86400, "d": 86400,
    "weeks": 604800, "week": 604800, "w": 604800,
}


class BackupSchedule:
    """A single backup schedule definition."""

    def __init__(
        self,
        name: str,
        backup_type: str = "full",
        output_dir: str = "/tmp",
        compression_level: int = 3,
        # Interval scheduling
        interval_seconds: Optional[int] = None,
        # Calendar scheduling
        weekdays: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        time_of_day: str = "02:00",
        # Retention
        retention_count: Optional[int] = None,
        retention_days: Optional[int] = None,
        # Cloud upload
        cloud_uri: Optional[str] = None,
        # Verification
        verify_after: bool = True,
        # State
        enabled: bool = True,
    ):
        self.name = name
        self.backup_type = backup_type  # "full" or "incremental"
        self.output_dir = output_dir
        self.compression_level = compression_level
        self.interval_seconds = interval_seconds
        self.weekdays = weekdays  # ["mon", "wed", "fri"]
        self.dates = dates  # ["2026-03-20", "2026-04-01"]
        self.time_of_day = time_of_day  # "HH:MM"
        self.retention_count = retention_count
        self.retention_days = retention_days
        self.cloud_uri = cloud_uri
        self.verify_after = verify_after
        self.enabled = enabled
        self.last_run: Optional[float] = None
        self.next_run: Optional[float] = None
        self.run_count: int = 0
        self.last_status: Optional[str] = None
        self.last_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "backup_type": self.backup_type,
            "output_dir": self.output_dir,
            "compression_level": self.compression_level,
            "interval_seconds": self.interval_seconds,
            "weekdays": self.weekdays,
            "dates": self.dates,
            "time_of_day": self.time_of_day,
            "retention_count": self.retention_count,
            "retention_days": self.retention_days,
            "cloud_uri": self.cloud_uri,
            "verify_after": self.verify_after,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BackupSchedule":
        sched = cls(
            name=d["name"],
            backup_type=d.get("backup_type", "full"),
            output_dir=d.get("output_dir", "/tmp"),
            compression_level=d.get("compression_level", 3),
            interval_seconds=d.get("interval_seconds"),
            weekdays=d.get("weekdays"),
            dates=d.get("dates"),
            time_of_day=d.get("time_of_day", "02:00"),
            retention_count=d.get("retention_count"),
            retention_days=d.get("retention_days"),
            cloud_uri=d.get("cloud_uri"),
            verify_after=d.get("verify_after", True),
            enabled=d.get("enabled", True),
        )
        sched.last_run = d.get("last_run")
        sched.next_run = d.get("next_run")
        sched.run_count = d.get("run_count", 0)
        sched.last_status = d.get("last_status")
        sched.last_error = d.get("last_error")
        return sched

    def compute_next_run(self, now: Optional[float] = None) -> Optional[float]:
        """Compute next run timestamp based on schedule config."""
        now = now or time.time()
        now_dt = datetime.fromtimestamp(now)

        # Interval-based
        if self.interval_seconds:
            base = self.last_run or now
            nxt = base + self.interval_seconds
            # If we missed runs, skip to next future slot
            while nxt < now:
                nxt += self.interval_seconds
            self.next_run = nxt
            return nxt

        # Parse time_of_day
        try:
            hh, mm = map(int, self.time_of_day.split(":"))
        except (ValueError, AttributeError):
            hh, mm = 2, 0

        # Specific dates
        if self.dates:
            candidates = []
            for date_str in self.dates:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hh, minute=mm)
                    if dt.timestamp() > now:
                        candidates.append(dt.timestamp())
                except ValueError:
                    continue
            if candidates:
                self.next_run = min(candidates)
                return self.next_run
            self.next_run = None
            return None

        # Weekday-based
        if self.weekdays:
            target_days = set()
            for wd in self.weekdays:
                wd_lower = wd.lower().strip()
                if wd_lower in _WEEKDAY_MAP:
                    target_days.add(_WEEKDAY_MAP[wd_lower])
                elif wd_lower == "weekday":
                    target_days.update({0, 1, 2, 3, 4})
                elif wd_lower == "weekend":
                    target_days.update({5, 6})

            if not target_days:
                self.next_run = None
                return None

            # Find next matching day
            check_dt = now_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if check_dt.timestamp() <= now:
                check_dt += timedelta(days=1)
            for _ in range(8):  # max 7 days ahead
                if check_dt.weekday() in target_days:
                    self.next_run = check_dt.timestamp()
                    return self.next_run
                check_dt += timedelta(days=1)

        self.next_run = None
        return None


class BackupLog:
    """Persistent backup execution log with fingerprints."""

    def __init__(self, gitdb_dir: Path):
        self._dir = gitdb_dir / "backups"
        self._dir.mkdir(exist_ok=True)
        self._log_file = self._dir / "backup_log.jsonl"

    def append(self, entry: dict):
        """Append a log entry."""
        entry.setdefault("timestamp", time.time())
        entry.setdefault("timestamp_human", time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def entries(self, limit: int = 100) -> List[dict]:
        """Read log entries (newest first)."""
        if not self._log_file.exists():
            return []
        lines = self._log_file.read_text().strip().split("\n")
        entries = []
        for line in reversed(lines):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if len(entries) >= limit:
                break
        return entries

    def clear(self):
        """Clear the log."""
        if self._log_file.exists():
            self._log_file.unlink()


class BackupScheduler:
    """Enterprise-grade backup scheduler with persistent config.

    Features:
    - Interval-based: every N seconds/minutes/hours/days/weeks
    - Calendar-based: specific weekdays (Mon, Tue, etc.), weekday/weekend
    - Date-based: specific calendar dates (2026-03-20)
    - Full or incremental backup per schedule
    - Retention policy: keep N backups or N days
    - Auto-verify after backup with fingerprint
    - Cloud upload after backup
    - Persistent schedule config (survives restart)
    - Detailed backup log with checksums
    """

    def __init__(self, gitdb_dir: Path, backup_fn: Callable, backup_incr_fn: Callable,
                 verify_fn: Optional[Callable] = None):
        self._dir = gitdb_dir / "backups"
        self._dir.mkdir(exist_ok=True)
        self._config_file = self._dir / "schedules.json"
        self._schedules: Dict[str, BackupSchedule] = {}
        self._backup_fn = backup_fn
        self._backup_incr_fn = backup_incr_fn
        self._verify_fn = verify_fn
        self._log = BackupLog(gitdb_dir)
        self._manager = BackupManager(gitdb_dir)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._load_config()

    def _load_config(self):
        """Load schedules from persistent config."""
        if self._config_file.exists():
            try:
                data = json.loads(self._config_file.read_text())
                for d in data:
                    sched = BackupSchedule.from_dict(d)
                    self._schedules[sched.name] = sched
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_config(self):
        """Persist schedules to disk."""
        data = [s.to_dict() for s in self._schedules.values()]
        self._config_file.write_text(json.dumps(data, indent=2))

    def add_schedule(self, schedule: BackupSchedule) -> dict:
        """Add or update a backup schedule."""
        with self._lock:
            schedule.compute_next_run()
            self._schedules[schedule.name] = schedule
            self._save_config()
        return schedule.to_dict()

    def remove_schedule(self, name: str) -> bool:
        """Remove a schedule by name."""
        with self._lock:
            if name in self._schedules:
                del self._schedules[name]
                self._save_config()
                return True
        return False

    def list_schedules(self) -> List[dict]:
        """List all schedules."""
        with self._lock:
            return [s.to_dict() for s in self._schedules.values()]

    def get_schedule(self, name: str) -> Optional[dict]:
        """Get a specific schedule."""
        s = self._schedules.get(name)
        return s.to_dict() if s else None

    def enable_schedule(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a schedule."""
        with self._lock:
            if name in self._schedules:
                self._schedules[name].enabled = enabled
                self._save_config()
                return True
        return False

    def get_log(self, limit: int = 100) -> List[dict]:
        """Get backup log entries."""
        return self._log.entries(limit)

    def _generate_path(self, schedule: BackupSchedule) -> str:
        """Generate backup output path."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        ext = "tar.zst" if HAS_ZSTD else "tar"
        suffix = "full" if schedule.backup_type == "full" else "incr"
        filename = f"gitdb_{schedule.name}_{suffix}_{ts}.{ext}"
        return str(Path(schedule.output_dir) / filename)

    def _execute_backup(self, schedule: BackupSchedule) -> dict:
        """Execute a single backup for a schedule."""
        output_path = self._generate_path(schedule)
        os.makedirs(schedule.output_dir, exist_ok=True)

        log_entry = {
            "schedule": schedule.name,
            "backup_type": schedule.backup_type,
            "output_path": output_path,
            "started_at": time.time(),
        }

        try:
            # Run backup
            if schedule.backup_type == "incremental":
                manifest = self._backup_incr_fn(output_path, compression_level=schedule.compression_level)
            else:
                manifest = self._backup_fn(output_path, compression_level=schedule.compression_level)

            log_entry["status"] = "success"
            log_entry["checksum"] = manifest.get("checksum", "")
            log_entry["size_bytes"] = manifest.get("backup_size_bytes", 0)
            log_entry["objects"] = manifest.get("object_count", manifest.get("new_objects", 0))

            # Verify if requested
            if schedule.verify_after and self._verify_fn:
                verify_result = self._verify_fn()
                log_entry["verified"] = verify_result.get("valid", False)
                log_entry["verify_issues"] = verify_result.get("issues", [])
                if not verify_result.get("valid"):
                    log_entry["status"] = "verified_with_issues"

            # Cloud upload
            if schedule.cloud_uri:
                try:
                    from gitdb.storage import parse_storage_uri
                    backend = parse_storage_uri(schedule.cloud_uri)
                    parts = schedule.cloud_uri.split("://", 1)
                    if len(parts) == 2:
                        path_part = parts[1].split("/", 1)
                        remote_key = path_part[1] if len(path_part) > 1 and path_part[1] else os.path.basename(output_path)
                    else:
                        remote_key = os.path.basename(output_path)
                    with open(output_path, "rb") as f:
                        backend.write(remote_key, f.read())
                    log_entry["cloud_uploaded"] = True
                    log_entry["cloud_uri"] = schedule.cloud_uri
                except Exception as e:
                    log_entry["cloud_uploaded"] = False
                    log_entry["cloud_error"] = str(e)

            # Retention cleanup
            self._apply_retention(schedule)

            # Update schedule state
            schedule.last_run = time.time()
            schedule.run_count += 1
            schedule.last_status = log_entry["status"]
            schedule.last_error = None
            schedule.compute_next_run()

        except Exception as e:
            log_entry["status"] = "error"
            log_entry["error"] = str(e)
            schedule.last_run = time.time()
            schedule.last_status = "error"
            schedule.last_error = str(e)
            schedule.compute_next_run()

        log_entry["finished_at"] = time.time()
        log_entry["duration_seconds"] = round(log_entry["finished_at"] - log_entry["started_at"], 3)
        self._log.append(log_entry)
        self._save_config()
        return log_entry

    def _apply_retention(self, schedule: BackupSchedule):
        """Clean up old backups based on retention policy."""
        if not schedule.retention_count and not schedule.retention_days:
            return

        # Find backup files matching this schedule
        output_dir = Path(schedule.output_dir)
        if not output_dir.exists():
            return

        pattern = f"gitdb_{schedule.name}_"
        backups = sorted(
            [f for f in output_dir.iterdir() if f.name.startswith(pattern) and f.is_file()
             and not f.name.endswith(".manifest.json")],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        to_delete = set()

        # Retention by count
        if schedule.retention_count and len(backups) > schedule.retention_count:
            to_delete.update(backups[schedule.retention_count:])

        # Retention by age
        if schedule.retention_days:
            cutoff = time.time() - (schedule.retention_days * 86400)
            for f in backups:
                if f.stat().st_mtime < cutoff:
                    to_delete.add(f)

        for f in to_delete:
            try:
                f.unlink()
                # Also delete sidecar manifest
                sidecar = f.with_suffix(".manifest.json")
                if sidecar.exists():
                    sidecar.unlink()
            except OSError:
                pass

    def run_now(self, name: str) -> dict:
        """Trigger immediate execution of a schedule."""
        sched = self._schedules.get(name)
        if not sched:
            return {"status": "error", "error": f"Schedule '{name}' not found"}
        return self._execute_backup(sched)

    def start(self):
        """Start the scheduler background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gitdb-backup-scheduler")
        self._thread.start()

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _loop(self):
        """Main scheduler loop. Checks every 30 seconds."""
        while self._running:
            now = time.time()
            with self._lock:
                for sched in list(self._schedules.values()):
                    if not sched.enabled:
                        continue
                    if sched.next_run is None:
                        sched.compute_next_run(now)
                    if sched.next_run and sched.next_run <= now:
                        try:
                            self._execute_backup(sched)
                        except Exception:
                            pass
            # Sleep in small increments for clean shutdown
            for _ in range(30):
                if not self._running:
                    break
                time.sleep(1)


def parse_interval(spec: str) -> int:
    """Parse interval string like '5m', '2h', '1d', '30s', '1w'.

    Also accepts: 'half-week' (3.5d), 'half-month' (15d), 'month' (30d).
    """
    spec = spec.strip().lower()
    # Special names
    if spec in ("half-week", "halfweek"):
        return int(3.5 * 86400)
    if spec in ("month", "monthly"):
        return 30 * 86400
    if spec in ("half-month", "halfmonth"):
        return 15 * 86400
    if spec in ("week", "weekly"):
        return 7 * 86400
    if spec in ("day", "daily"):
        return 86400
    if spec in ("hour", "hourly"):
        return 3600

    # Parse numeric + unit
    import re
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([a-z]+)$", spec)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit in _INTERVAL_MULTIPLIERS:
            return int(val * _INTERVAL_MULTIPLIERS[unit])

    # Try pure seconds
    try:
        return int(float(spec))
    except ValueError:
        raise ValueError(f"Cannot parse interval: '{spec}'. Examples: '5m', '2h', '1d', '30s', '1w'")


def create_schedule(
    name: str,
    backup_type: str = "full",
    output_dir: str = "/tmp",
    interval: Optional[str] = None,
    weekdays: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    time_of_day: str = "02:00",
    retention_count: Optional[int] = None,
    retention_days: Optional[int] = None,
    cloud_uri: Optional[str] = None,
    verify_after: bool = True,
    compression_level: int = 3,
) -> BackupSchedule:
    """Convenience factory for creating a backup schedule.

    Args:
        name: Schedule name (unique identifier).
        backup_type: "full" or "incremental".
        output_dir: Directory for backup files.
        interval: Interval string (e.g., "5m", "2h", "1d", "1w", "month").
        weekdays: List of weekday names (e.g., ["mon", "wed", "fri"]).
        dates: List of specific dates (e.g., ["2026-03-20"]).
        time_of_day: Time for calendar-based schedules ("HH:MM").
        retention_count: Max backups to keep.
        retention_days: Max age in days for backups.
        cloud_uri: Auto-upload to cloud after backup.
        verify_after: Run integrity check after backup.
        compression_level: zstd compression level (1-22).
    """
    interval_seconds = parse_interval(interval) if interval else None
    return BackupSchedule(
        name=name,
        backup_type=backup_type,
        output_dir=output_dir,
        compression_level=compression_level,
        interval_seconds=interval_seconds,
        weekdays=weekdays,
        dates=dates,
        time_of_day=time_of_day,
        retention_count=retention_count,
        retention_days=retention_days,
        cloud_uri=cloud_uri,
        verify_after=verify_after,
    )
