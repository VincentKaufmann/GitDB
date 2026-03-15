"""GitDB — GPU-accelerated version-controlled database."""

from gitdb.core import GitDB, Transaction
from gitdb.documents import DocumentStore
from gitdb.hooks import HookManager
from gitdb.schema import Schema, SchemaError
from gitdb.types import (
    Results, CommitInfo, DiffEntry, DiffResult, MergeResult,
    BlameEntry, BisectResult, StashEntry,
)

__version__ = "0.9.0"
__all__ = [
    "GitDB", "Transaction", "DocumentStore",
    "HookManager", "Schema", "SchemaError",
    "Results", "CommitInfo", "DiffEntry", "DiffResult",
    "MergeResult", "BlameEntry", "BisectResult", "StashEntry",
]
