"""Pull Requests — propose, review, and merge changes across branches."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Comment:
    """A comment on a PR or commit."""
    author: str
    message: str
    timestamp: float
    reply_to: Optional[int] = None  # parent comment index

    def to_dict(self) -> dict:
        return {
            "author": self.author,
            "message": self.message,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
        }

    @staticmethod
    def from_dict(d: dict) -> "Comment":
        return Comment(
            author=d["author"],
            message=d["message"],
            timestamp=d["timestamp"],
            reply_to=d.get("reply_to"),
        )


@dataclass
class PullRequest:
    """A pull request proposing to merge one branch into another."""
    id: int
    title: str
    description: str
    source_branch: str
    target_branch: str
    author: str
    status: str = "open"           # open, merged, closed
    created_at: float = 0.0
    updated_at: float = 0.0
    merged_at: Optional[float] = None
    merge_commit: Optional[str] = None
    comments: List[Comment] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "author": self.author,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "merged_at": self.merged_at,
            "merge_commit": self.merge_commit,
            "comments": [c.to_dict() for c in self.comments],
            "labels": self.labels,
        }

    @staticmethod
    def from_dict(d: dict) -> "PullRequest":
        return PullRequest(
            id=d["id"],
            title=d["title"],
            description=d["description"],
            source_branch=d["source_branch"],
            target_branch=d["target_branch"],
            author=d["author"],
            status=d["status"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            merged_at=d.get("merged_at"),
            merge_commit=d.get("merge_commit"),
            comments=[Comment.from_dict(c) for c in d.get("comments", [])],
            labels=d.get("labels", []),
        )

    def __repr__(self):
        return f"PR #{self.id}: {self.title} ({self.source_branch} → {self.target_branch}) [{self.status}]"


class PRStore:
    """Manages pull requests for a GitDB store."""

    def __init__(self, gitdb_dir: Path):
        self._dir = gitdb_dir / "prs"
        self._dir.mkdir(exist_ok=True)
        self._index_file = self._dir / "index.json"

    def _load_index(self) -> List[dict]:
        if self._index_file.exists():
            return json.loads(self._index_file.read_text())
        return []

    def _save_index(self, index: List[dict]):
        self._index_file.write_text(json.dumps(index, indent=2))

    def create(
        self,
        title: str,
        source_branch: str,
        target_branch: str,
        description: str = "",
        author: str = "anonymous",
        labels: Optional[List[str]] = None,
    ) -> PullRequest:
        """Create a new pull request."""
        index = self._load_index()
        pr_id = len(index) + 1
        now = time.time()
        pr = PullRequest(
            id=pr_id,
            title=title,
            description=description,
            source_branch=source_branch,
            target_branch=target_branch,
            author=author,
            status="open",
            created_at=now,
            updated_at=now,
            labels=labels or [],
        )
        index.append(pr.to_dict())
        self._save_index(index)
        return pr

    def get(self, pr_id: int) -> PullRequest:
        """Get a PR by ID."""
        index = self._load_index()
        for d in index:
            if d["id"] == pr_id:
                return PullRequest.from_dict(d)
        raise ValueError(f"PR #{pr_id} not found")

    def list(self, status: Optional[str] = None) -> List[PullRequest]:
        """List all PRs, optionally filtered by status."""
        index = self._load_index()
        prs = [PullRequest.from_dict(d) for d in index]
        if status:
            prs = [pr for pr in prs if pr.status == status]
        return prs

    def comment(self, pr_id: int, author: str, message: str, reply_to: Optional[int] = None):
        """Add a comment to a PR."""
        index = self._load_index()
        for d in index:
            if d["id"] == pr_id:
                d["comments"].append(Comment(
                    author=author,
                    message=message,
                    timestamp=time.time(),
                    reply_to=reply_to,
                ).to_dict())
                d["updated_at"] = time.time()
                self._save_index(index)
                return
        raise ValueError(f"PR #{pr_id} not found")

    def update_status(self, pr_id: int, status: str, merge_commit: Optional[str] = None):
        """Update PR status (open → merged/closed)."""
        index = self._load_index()
        for d in index:
            if d["id"] == pr_id:
                d["status"] = status
                d["updated_at"] = time.time()
                if status == "merged":
                    d["merged_at"] = time.time()
                    d["merge_commit"] = merge_commit
                self._save_index(index)
                return
        raise ValueError(f"PR #{pr_id} not found")

    def close(self, pr_id: int):
        """Close a PR without merging."""
        self.update_status(pr_id, "closed")

    def add_label(self, pr_id: int, label: str):
        """Add a label to a PR."""
        index = self._load_index()
        for d in index:
            if d["id"] == pr_id:
                if label not in d["labels"]:
                    d["labels"].append(label)
                    d["updated_at"] = time.time()
                    self._save_index(index)
                return
        raise ValueError(f"PR #{pr_id} not found")
