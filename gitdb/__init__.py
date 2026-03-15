"""GitDB — GPU-accelerated version-controlled vector database."""

from gitdb.core import GitDB, Transaction
from gitdb.hooks import HookManager
from gitdb.schema import Schema, SchemaError
from gitdb.types import (
    Results, CommitInfo, DiffEntry, DiffResult, MergeResult,
    BlameEntry, BisectResult, StashEntry,
)

__version__ = "0.8.0"
__all__ = [
    "GitDB", "Transaction", "HookManager", "Schema", "SchemaError",
    "Results", "CommitInfo", "DiffEntry", "DiffResult",
    "MergeResult", "BlameEntry", "BisectResult", "StashEntry",
]
