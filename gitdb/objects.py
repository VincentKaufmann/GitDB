"""Content-addressed object store — commits and deltas stored by SHA-256."""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from gitdb.types import CommitStats


@dataclass
class Commit:
    """A snapshot in the history."""

    hash: str = ""                     # SHA-256, computed on save
    parent: Optional[str] = None       # parent commit hash
    parent2: Optional[str] = None      # second parent (merge commits)
    delta_hash: str = ""               # hash of the Delta object
    timestamp: float = 0.0
    message: str = ""
    stats: CommitStats = field(default_factory=CommitStats)

    # Snapshot sizes for fast reconstruction decisions
    tensor_rows: int = 0               # number of rows after this commit

    @property
    def is_merge(self) -> bool:
        return self.parent2 is not None

    @property
    def short_hash(self) -> str:
        return self.hash[:8] if self.hash else "????????"

    def to_bytes(self) -> bytes:
        """Serialize commit to deterministic JSON bytes."""
        d = {
            "parent": self.parent,
            "parent2": self.parent2,
            "delta_hash": self.delta_hash,
            "timestamp": self.timestamp,
            "message": self.message,
            "stats": {
                "added": self.stats.added,
                "removed": self.stats.removed,
                "modified": self.stats.modified,
            },
            "tensor_rows": self.tensor_rows,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes, commit_hash: str) -> "Commit":
        """Deserialize commit from JSON bytes."""
        d = json.loads(data.decode("utf-8"))
        stats = CommitStats(**d["stats"])
        return Commit(
            hash=commit_hash,
            parent=d["parent"],
            parent2=d.get("parent2"),
            delta_hash=d["delta_hash"],
            timestamp=d["timestamp"],
            message=d["message"],
            stats=stats,
            tensor_rows=d.get("tensor_rows", 0),
        )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash from content."""
        return hashlib.sha256(self.to_bytes()).hexdigest()


class ObjectStore:
    """Content-addressed storage for commits and deltas.

    Layout:
        .gitdb/objects/ab/c123def456...  → object bytes

    Optional encryption: pass an EncryptionManager to encrypt at rest.
    """

    def __init__(self, root: Path, encryption=None):
        self.root = root / "objects"
        self.root.mkdir(parents=True, exist_ok=True)
        self._encryption = encryption

    def _path(self, obj_hash: str) -> Path:
        """Two-character prefix directory (like git)."""
        return self.root / obj_hash[:2] / obj_hash[2:]

    def _encrypt(self, data: bytes) -> bytes:
        if self._encryption and self._encryption.enabled:
            return self._encryption.encrypt(data)
        return data

    def _decrypt(self, data: bytes) -> bytes:
        if self._encryption and self._encryption.enabled:
            return self._encryption.decrypt(data)
        return data

    def has(self, obj_hash: str) -> bool:
        return self._path(obj_hash).exists()

    def read(self, obj_hash: str) -> bytes:
        p = self._path(obj_hash)
        if not p.exists():
            raise KeyError(f"Object not found: {obj_hash[:12]}...")
        return self._decrypt(p.read_bytes())

    def write(self, data: bytes) -> str:
        """Write data, return its SHA-256 hash."""
        h = hashlib.sha256(data).hexdigest()
        p = self._path(h)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write via temp file
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(self._encrypt(data))
            tmp.rename(p)
        return h

    def write_commit(self, commit: Commit) -> str:
        """Write a commit object, setting its hash."""
        commit.hash = commit.compute_hash()
        data = commit.to_bytes()
        stored_hash = self.write(data)
        # The stored hash is of the bytes, commit.hash is of the content.
        # We store by commit.hash for lookup by commit hash.
        p = self._path(commit.hash)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(self._encrypt(data))
            tmp.rename(p)
        return commit.hash

    def read_commit(self, commit_hash: str) -> Commit:
        """Read and deserialize a commit object."""
        data = self.read(commit_hash)
        return Commit.from_bytes(data, commit_hash)

    def count(self) -> int:
        """Count total objects."""
        n = 0
        if self.root.exists():
            for prefix_dir in self.root.iterdir():
                if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                    n += sum(1 for _ in prefix_dir.iterdir() if not _.name.endswith(".tmp"))
        return n


class RefStore:
    """Manages branches (heads), tags, and HEAD.

    Layout:
        .gitdb/HEAD                     → "ref: refs/heads/main"
        .gitdb/refs/heads/main          → commit_hash
        .gitdb/refs/tags/v1.0           → commit_hash
    """

    def __init__(self, root: Path):
        self.root = root
        self.refs_dir = root / "refs"
        self.heads_dir = self.refs_dir / "heads"
        self.tags_dir = self.refs_dir / "tags"
        self.head_file = root / "HEAD"

        self.heads_dir.mkdir(parents=True, exist_ok=True)
        self.tags_dir.mkdir(parents=True, exist_ok=True)

    def init(self, default_branch: str = "main"):
        """Initialize HEAD pointing to default branch."""
        if not self.head_file.exists():
            self.head_file.write_text(f"ref: refs/heads/{default_branch}\n")

    @property
    def current_branch(self) -> str:
        """Get current branch name from HEAD."""
        content = self.head_file.read_text().strip()
        if content.startswith("ref: refs/heads/"):
            return content[len("ref: refs/heads/"):]
        raise ValueError(f"Detached HEAD not supported in MVP: {content}")

    def get_head_commit(self) -> Optional[str]:
        """Get the commit hash that HEAD points to."""
        branch = self.current_branch
        return self.get_branch(branch)

    def get_branch(self, name: str) -> Optional[str]:
        """Get commit hash for a branch, or None if it doesn't exist."""
        p = self.heads_dir / name
        if p.exists():
            return p.read_text().strip()
        return None

    def set_branch(self, name: str, commit_hash: str):
        """Set a branch to point to a commit."""
        p = self.heads_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(commit_hash + "\n")

    def list_branches(self) -> Dict[str, str]:
        """Return all branches and their commit hashes."""
        branches = {}
        if self.heads_dir.exists():
            for p in self.heads_dir.rglob("*"):
                if p.is_file():
                    name = str(p.relative_to(self.heads_dir))
                    branches[name] = p.read_text().strip()
        return branches

    def set_tag(self, name: str, commit_hash: str):
        """Create a tag pointing to a commit."""
        p = self.tags_dir / name
        if p.exists():
            raise ValueError(f"Tag already exists: {name}")
        p.write_text(commit_hash + "\n")

    def delete_tag(self, name: str):
        """Delete a tag."""
        p = self.tags_dir / name
        if not p.exists():
            raise ValueError(f"Tag not found: {name}")
        p.unlink()

    def set_head(self, branch_name: str):
        """Point HEAD to a branch."""
        self.head_file.write_text(f"ref: refs/heads/{branch_name}\n")

    def delete_branch(self, name: str):
        """Delete a branch ref."""
        p = self.heads_dir / name
        if not p.exists():
            raise ValueError(f"Branch not found: {name}")
        if name == self.current_branch:
            raise ValueError(f"Cannot delete current branch: {name}")
        p.unlink()

    def resolve(self, ref: str, object_store: Optional["ObjectStore"] = None) -> Optional[str]:
        """Resolve a ref string to a commit hash.

        Supports: branch names, tag names, raw commit hashes,
        and relative refs like main~3 or HEAD~1.
        """
        # Handle relative refs
        if "~" in ref:
            base, n_str = ref.split("~", 1)
            n = int(n_str)
            commit_hash = self.resolve(base, object_store)
            if commit_hash and object_store and n > 0:
                current = commit_hash
                for _ in range(n):
                    commit = object_store.read_commit(current)
                    if commit.parent is None:
                        return current  # Can't go further back
                    current = commit.parent
                return current
            return commit_hash

        # HEAD
        if ref == "HEAD":
            return self.get_head_commit()

        # Branch name
        branch_hash = self.get_branch(ref)
        if branch_hash:
            return branch_hash

        # Tag name
        tag_file = self.tags_dir / ref
        if tag_file.exists():
            return tag_file.read_text().strip()

        # Raw commit hash (if it looks like one)
        if len(ref) >= 7 and all(c in "0123456789abcdef" for c in ref):
            return ref

        return None
