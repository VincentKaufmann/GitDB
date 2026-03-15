"""GitDB — GPU-accelerated version-controlled database."""

from gitdb.core import GitDB, Transaction
from gitdb.documents import DocumentStore, Table, TableStore
from gitdb.encryption import EncryptionManager, EncryptionError
from gitdb.hooks import HookManager
from gitdb.streaming import StreamIngest, ShardStream, ChunkMeta, MerkleTree
from gitdb.schema import Schema, SchemaError
from gitdb.storage import (
    StorageBackend, LocalStorage, S3Storage, GCSStorage,
    AzureStorage, SFTPStorage, parse_storage_uri, copy_between,
)
from gitdb.types import (
    Results, CommitInfo, DiffEntry, DiffResult, MergeResult,
    BlameEntry, BisectResult, StashEntry,
)

__version__ = "0.11.0"
__all__ = [
    "GitDB", "Transaction", "DocumentStore", "Table", "TableStore",
    "EncryptionManager", "EncryptionError",
    "StorageBackend", "LocalStorage", "S3Storage", "GCSStorage",
    "AzureStorage", "SFTPStorage", "parse_storage_uri", "copy_between",
    "StreamIngest", "ShardStream", "ChunkMeta", "MerkleTree",
    "HookManager", "Schema", "SchemaError",
    "Results", "CommitInfo", "DiffEntry", "DiffResult",
    "MergeResult", "BlameEntry", "BisectResult", "StashEntry",
]
