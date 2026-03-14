"""GitDB — GPU-accelerated version-controlled vector database."""

from gitdb.core import GitDB
from gitdb.types import (
    Results, CommitInfo, DiffResult, MergeResult,
    BlameEntry, BisectResult, StashEntry,
)

__version__ = "0.3.0"
__all__ = [
    "GitDB", "Results", "CommitInfo", "DiffResult",
    "MergeResult", "BlameEntry", "BisectResult", "StashEntry",
]
