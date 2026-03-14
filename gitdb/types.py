"""Shared data types for GitDB."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch


@dataclass
class VectorMeta:
    """Per-vector metadata."""
    id: str                          # content hash
    document: Optional[str] = None   # raw text
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Results:
    """Query results."""
    ids: List[str]
    scores: List[float]
    documents: List[Optional[str]]
    metadata: List[Dict[str, Any]]

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return f"Results(count={len(self)}, top_score={self.scores[0]:.4f})" if self.scores else "Results(empty)"


@dataclass
class CommitStats:
    """Summary stats for a commit."""
    added: int = 0
    removed: int = 0
    modified: int = 0

    def __repr__(self):
        parts = []
        if self.added: parts.append(f"+{self.added}")
        if self.removed: parts.append(f"-{self.removed}")
        if self.modified: parts.append(f"~{self.modified}")
        return " ".join(parts) or "no changes"


@dataclass
class CommitInfo:
    """Public-facing commit information."""
    hash: str
    parent: Optional[str]
    message: str
    timestamp: float
    stats: CommitStats

    def __repr__(self):
        short = self.hash[:8]
        return f"Commit({short} | {self.message!r} | {self.stats})"


@dataclass
class DiffResult:
    """Result of comparing two refs."""
    added_count: int
    removed_count: int
    modified_count: int
    added_ids: List[str] = field(default_factory=list)
    removed_ids: List[str] = field(default_factory=list)
    modified_ids: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"Diff(+{self.added_count} -{self.removed_count} ~{self.modified_count})"


@dataclass
class MergeResult:
    """Result of a merge operation."""
    commit_hash: str
    conflicts: List[str]       # IDs of vectors modified in both branches
    strategy: str
    added: int = 0
    removed: int = 0

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    def __repr__(self):
        c = f", {len(self.conflicts)} conflicts" if self.conflicts else ""
        return f"Merge({self.commit_hash[:8]} | {self.strategy}{c})"


@dataclass
class BlameEntry:
    """Provenance info for a single vector."""
    vector_id: str
    commit_hash: str
    commit_message: str
    timestamp: float
    row_index: int

    def __repr__(self):
        return f"Blame({self.vector_id[:8]} ← {self.commit_hash[:8]} | {self.commit_message!r})"


@dataclass
class BisectResult:
    """Result of a bisect operation."""
    bad_commit: str
    bad_message: str
    steps: int
    total_commits: int

    def __repr__(self):
        return f"Bisect(found {self.bad_commit[:8]} in {self.steps}/{self.total_commits} steps)"


@dataclass
class StashEntry:
    """A stashed state."""
    index: int
    message: str
    timestamp: float
    tensor_file: str
    meta_file: str

    def __repr__(self):
        return f"Stash({self.index}: {self.message!r})"
