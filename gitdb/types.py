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
class DiffEntry:
    """A single changed vector in a diff."""
    id: str
    change: str                          # "added", "removed", "modified"
    document: Optional[str] = None       # current document text
    document_before: Optional[str] = None  # previous document text (modified only)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metadata_before: Dict[str, Any] = field(default_factory=dict)
    similarity: Optional[float] = None   # cosine sim between old/new (modified only)

    def __repr__(self):
        sim = f" sim={self.similarity:.4f}" if self.similarity is not None else ""
        doc = f" {self.document[:40]}..." if self.document and len(self.document) > 40 else f" {self.document}" if self.document else ""
        return f"DiffEntry({self.change} {self.id[:8]}{sim}{doc})"


@dataclass
class DiffResult:
    """Result of comparing two refs."""
    added_count: int
    removed_count: int
    modified_count: int
    added_ids: List[str] = field(default_factory=list)
    removed_ids: List[str] = field(default_factory=list)
    modified_ids: List[str] = field(default_factory=list)
    entries: List[DiffEntry] = field(default_factory=list)

    def __repr__(self):
        return f"Diff(+{self.added_count} -{self.removed_count} ~{self.modified_count})"

    def show(self) -> str:
        """Human-readable diff output."""
        lines = []
        for e in self.entries:
            if e.change == "added":
                doc = e.document or "(no document)"
                lines.append(f"+ {e.id[:12]}  {doc}")
                if e.metadata:
                    lines.append(f"  metadata: {e.metadata}")
            elif e.change == "removed":
                doc = e.document or "(no document)"
                lines.append(f"- {e.id[:12]}  {doc}")
                if e.metadata:
                    lines.append(f"  metadata: {e.metadata}")
            elif e.change == "modified":
                sim = f"  (similarity: {e.similarity:.4f})" if e.similarity is not None else ""
                lines.append(f"~ {e.id[:12]}{sim}")
                if e.document_before != e.document:
                    if e.document_before:
                        lines.append(f"  - {e.document_before}")
                    if e.document:
                        lines.append(f"  + {e.document}")
                if e.metadata_before != e.metadata:
                    lines.append(f"  - metadata: {e.metadata_before}")
                    lines.append(f"  + metadata: {e.metadata}")
        return "\n".join(lines)

    def unified(self, ref_a: str = "a", ref_b: str = "b") -> str:
        """Git-style unified diff format."""
        import json as _json
        lines = []
        for e in self.entries:
            lines.append(f"diff --gitdb {ref_a}/{e.id[:12]} {ref_b}/{e.id[:12]}")
            if e.change == "added":
                lines.append(f"new vector {e.id}")
                lines.append(f"--- /dev/null")
                lines.append(f"+++ {ref_b}/{e.id[:12]}")
                lines.append(f"@@ -0,0 +1 @@")
                lines.append(f"+document: {e.document or '(none)'}")
                if e.metadata:
                    lines.append(f"+metadata: {_json.dumps(e.metadata, separators=(',', ':'))}")
            elif e.change == "removed":
                lines.append(f"deleted vector {e.id}")
                lines.append(f"--- {ref_a}/{e.id[:12]}")
                lines.append(f"+++ /dev/null")
                lines.append(f"@@ -1 +0,0 @@")
                lines.append(f"-document: {e.document or '(none)'}")
                if e.metadata:
                    lines.append(f"-metadata: {_json.dumps(e.metadata, separators=(',', ':'))}")
            elif e.change == "modified":
                sim = f" (cosine similarity: {e.similarity:.4f})" if e.similarity is not None else ""
                lines.append(f"modified vector {e.id}{sim}")
                lines.append(f"--- {ref_a}/{e.id[:12]}")
                lines.append(f"+++ {ref_b}/{e.id[:12]}")
                # Document diff
                doc_a = e.document_before or "(none)"
                doc_b = e.document or "(none)"
                meta_a = _json.dumps(e.metadata_before, separators=(',', ':')) if e.metadata_before else "{}"
                meta_b = _json.dumps(e.metadata, separators=(',', ':')) if e.metadata else "{}"
                hunk_lines = 0
                changes = []
                if doc_a != doc_b:
                    changes.append(f"-document: {doc_a}")
                    changes.append(f"+document: {doc_b}")
                    hunk_lines += 1
                else:
                    changes.append(f" document: {doc_a}")
                    hunk_lines += 1
                if meta_a != meta_b:
                    changes.append(f"-metadata: {meta_a}")
                    changes.append(f"+metadata: {meta_b}")
                    hunk_lines += 1
                else:
                    changes.append(f" metadata: {meta_a}")
                    hunk_lines += 1
                lines.append(f"@@ -1,{hunk_lines} +1,{hunk_lines} @@")
                lines.extend(changes)
            lines.append("")  # blank line between entries
        return "\n".join(lines)


@dataclass
class MergeResult:
    """Result of a merge operation."""
    commit_hash: str
    conflicts: List[str]       # IDs of vectors modified in both branches
    strategy: str
    added: int = 0
    removed: int = 0
    diff: Optional['DiffResult'] = None  # Full diff of what the merge changed

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    def __repr__(self):
        c = f", {len(self.conflicts)} conflicts" if self.conflicts else ""
        return f"Merge({self.commit_hash[:8]} | {self.strategy} +{self.added} -{self.removed}{c})"

    def show(self) -> str:
        """Human-readable merge summary."""
        lines = [f"Merge commit {self.commit_hash[:12]} ({self.strategy})"]
        if self.added:
            lines.append(f"  +{self.added} vectors added")
        if self.removed:
            lines.append(f"  -{self.removed} vectors removed")
        if self.conflicts:
            lines.append(f"  {len(self.conflicts)} conflicts: {', '.join(c[:8] for c in self.conflicts)}")
        if self.diff and self.diff.entries:
            lines.append("")
            lines.append(self.diff.unified())
        return "\n".join(lines)


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
