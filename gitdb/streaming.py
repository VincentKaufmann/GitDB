"""Streaming ingest for GitDB — sharded WAL-backed chunk ingestion with Merkle integrity.

Each shard operates on its own branch (stream/{shard_id}), buffers chunks in memory,
and batch-commits to GitDB with Merkle root verification. Thread-safe, crash-recoverable.
"""

import base64
import hashlib
import json
import os
import struct
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from gitdb.core import GitDB
from gitdb.types import MergeResult


# ═══════════════════════════════════════════════════════════════
#  ChunkMeta — metadata for a single ingested chunk
# ═══════════════════════════════════════════════════════════════

@dataclass
class ChunkMeta:
    """Metadata for a single ingested chunk."""
    chunk_hash: str                    # SHA-256 of chunk content
    source: str                        # provenance: who/what produced this
    shard_id: str                      # which shard stream
    timestamp: float                   # arrival time
    sequence: int                      # monotonic sequence number within shard
    parent_hash: Optional[str]         # previous chunk hash (hash chain)
    size_bytes: int                    # raw chunk size
    chunk_type: str                    # "records", "embeddings", "documents", "raw_bytes"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkMeta":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ═══════════════════════════════════════════════════════════════
#  WAL — append-only write-ahead log per shard
# ═══════════════════════════════════════════════════════════════

class WAL:
    """Append-only write-ahead log for crash recovery.

    Format per entry: [4B entry_length][JSON metadata][newline][chunk_data_bytes]
    entry_length covers everything after the 4-byte header (meta JSON + newline + data).

    If an EncryptionManager is provided, chunk data is encrypted in the WAL.
    """

    def __init__(self, wal_path: Path, encryption=None):
        self._path = Path(wal_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._encryption = encryption
        self._fp = open(self._path, "ab+")

    def append(self, chunk_meta: ChunkMeta, chunk_data: bytes) -> int:
        """Write an entry, fsync, return sequence number."""
        with self._lock:
            meta_json = json.dumps(chunk_meta.to_dict(), sort_keys=True).encode("utf-8")
            data = chunk_data
            if self._encryption and self._encryption.enabled:
                data = self._encryption.encrypt(data)
            payload = meta_json + b"\n" + data
            length = len(payload)
            entry = struct.pack("<I", length) + payload
            self._fp.write(entry)
            self._fp.flush()
            os.fsync(self._fp.fileno())
            return chunk_meta.sequence

    def replay(self, from_sequence: int = 0) -> Iterator[Tuple[ChunkMeta, bytes]]:
        """Replay WAL entries from a given sequence for crash recovery."""
        with self._lock:
            yield from self._replay_unlocked(from_sequence)

    def _replay_unlocked(self, from_sequence: int = 0) -> Iterator[Tuple[ChunkMeta, bytes]]:
        """Internal replay without lock (for use inside locked context)."""
        try:
            with open(self._path, "rb") as f:
                while True:
                    header = f.read(4)
                    if len(header) < 4:
                        break
                    length = struct.unpack("<I", header)[0]
                    payload = f.read(length)
                    if len(payload) < length:
                        # Truncated entry — skip (crash recovery)
                        break
                    try:
                        newline_pos = payload.index(b"\n")
                    except ValueError:
                        break
                    meta_json = payload[:newline_pos]
                    data = payload[newline_pos + 1:]
                    try:
                        meta_dict = json.loads(meta_json.decode("utf-8"))
                        meta = ChunkMeta.from_dict(meta_dict)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        break
                    if self._encryption and self._encryption.enabled:
                        try:
                            data = self._encryption.decrypt(data)
                        except Exception:
                            break
                    if meta.sequence >= from_sequence:
                        yield (meta, data)
        except FileNotFoundError:
            return

    def truncate(self, up_to_sequence: int):
        """Remove committed entries up to (and including) up_to_sequence."""
        with self._lock:
            # Read remaining entries
            remaining = []
            try:
                with open(self._path, "rb") as f:
                    while True:
                        header = f.read(4)
                        if len(header) < 4:
                            break
                        length = struct.unpack("<I", header)[0]
                        payload = f.read(length)
                        if len(payload) < length:
                            break
                        try:
                            newline_pos = payload.index(b"\n")
                        except ValueError:
                            break
                        meta_json = payload[:newline_pos]
                        try:
                            meta_dict = json.loads(meta_json.decode("utf-8"))
                            seq = meta_dict.get("sequence", 0)
                        except (json.JSONDecodeError, KeyError):
                            break
                        if seq > up_to_sequence:
                            remaining.append(struct.pack("<I", length) + payload)
            except FileNotFoundError:
                return

            # Rewrite file with only remaining entries
            self._fp.close()
            with open(self._path, "wb") as f:
                for entry in remaining:
                    f.write(entry)
            self._fp = open(self._path, "ab+")

    def close(self):
        """Flush and close the WAL file."""
        with self._lock:
            self._fp.flush()
            self._fp.close()


# ═══════════════════════════════════════════════════════════════
#  MerkleTree — binary hash tree for integrity verification
# ═══════════════════════════════════════════════════════════════

class MerkleTree:
    """Binary Merkle tree of SHA-256 hashes, computed lazily."""

    def __init__(self):
        self._leaves: List[str] = []
        self._dirty = True
        self._cached_root: Optional[str] = None
        self._cached_tree: List[List[str]] = []

    def add_leaf(self, chunk_hash: str):
        """Add a leaf node."""
        self._leaves.append(chunk_hash)
        self._dirty = True

    @staticmethod
    def _hash_pair(left: str, right: str) -> str:
        return hashlib.sha256((left + right).encode("utf-8")).hexdigest()

    def _build(self):
        """Build the full tree from leaves."""
        if not self._leaves:
            self._cached_root = hashlib.sha256(b"").hexdigest()
            self._cached_tree = [[]]
            self._dirty = False
            return

        level = list(self._leaves)
        self._cached_tree = [level[:]]

        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                next_level.append(self._hash_pair(left, right))
            level = next_level
            self._cached_tree.append(level[:])

        self._cached_root = level[0]
        self._dirty = False

    def root_hash(self) -> str:
        """Compute and return the Merkle root hash."""
        if self._dirty:
            self._build()
        return self._cached_root

    def proof(self, leaf_index: int) -> List[Tuple[str, str]]:
        """Generate an inclusion proof for a leaf.

        Returns list of (hash, side) tuples where side is "L" or "R"
        indicating the sibling's position relative to the path node.
        """
        if self._dirty:
            self._build()
        if leaf_index < 0 or leaf_index >= len(self._leaves):
            raise IndexError(f"Leaf index {leaf_index} out of range")

        proof_path = []
        idx = leaf_index

        for level in self._cached_tree[:-1]:  # skip root level
            if idx % 2 == 0:
                # We're on the left, sibling is on the right
                sibling_idx = idx + 1
                if sibling_idx < len(level):
                    proof_path.append((level[sibling_idx], "R"))
                else:
                    proof_path.append((level[idx], "R"))  # duplicate
            else:
                # We're on the right, sibling is on the left
                proof_path.append((level[idx - 1], "L"))
            idx //= 2

        return proof_path

    @staticmethod
    def verify_proof(chunk_hash: str, proof: List[Tuple[str, str]], root: str) -> bool:
        """Verify a Merkle inclusion proof."""
        current = chunk_hash
        for sibling_hash, side in proof:
            if side == "L":
                current = MerkleTree._hash_pair(sibling_hash, current)
            else:
                current = MerkleTree._hash_pair(current, sibling_hash)
        return current == root


# ═══════════════════════════════════════════════════════════════
#  ChunkBuffer — thread-safe bounded buffer with backpressure
# ═══════════════════════════════════════════════════════════════

class ChunkBuffer:
    """Thread-safe bounded buffer with backpressure via Condition."""

    def __init__(self, max_size: int = 10000):
        self._buffer: List[Tuple[ChunkMeta, bytes]] = []
        self._max_size = max_size
        self._cond = threading.Condition()

    def put(self, meta: ChunkMeta, data: bytes, timeout: Optional[float] = None) -> bool:
        """Add an item to the buffer.

        Returns False if the buffer is full after timeout.
        """
        with self._cond:
            if len(self._buffer) >= self._max_size:
                if timeout is not None and timeout <= 0:
                    return False
                if not self._cond.wait_for(
                    lambda: len(self._buffer) < self._max_size,
                    timeout=timeout,
                ):
                    return False
            self._buffer.append((meta, data))
            self._cond.notify_all()
            return True

    def drain(self, n: Optional[int] = None) -> List[Tuple[ChunkMeta, bytes]]:
        """Take up to n items from the buffer (all if n is None)."""
        with self._cond:
            if n is None:
                items = list(self._buffer)
                self._buffer.clear()
            else:
                items = self._buffer[:n]
                self._buffer = self._buffer[n:]
            self._cond.notify_all()
            return items

    @property
    def size(self) -> int:
        with self._cond:
            return len(self._buffer)


# ═══════════════════════════════════════════════════════════════
#  ShardStream — per-shard ingest pipeline
# ═══════════════════════════════════════════════════════════════

class ShardStream:
    """A single shard stream operating on branch stream/{shard_id}.

    Buffers chunks in memory, writes WAL for durability, and batch-commits
    to GitDB with Merkle root in the commit message.
    """

    def __init__(
        self,
        stream_ingest: "StreamIngest",
        shard_id: str,
        commit_every_n: int = 100,
        commit_every_t: float = 30.0,
        buffer_size: int = 10000,
    ):
        self._ingest = stream_ingest
        self.shard_id = shard_id
        self.branch_name = f"stream/{shard_id}"
        self._commit_every_n = commit_every_n
        self._commit_every_t = commit_every_t
        self._lock = threading.Lock()

        # WAL
        wal_dir = Path(stream_ingest._db.path) / ".gitdb" / "wal"
        self._wal = WAL(wal_dir / f"{shard_id}.wal", encryption=stream_ingest._db.encryption)

        # Buffer
        self._buffer = ChunkBuffer(max_size=buffer_size)

        # Merkle tree for current batch
        self._merkle = MerkleTree()

        # Dedup
        self._seen_hashes: Set[str] = set()

        # Counters
        self._sequence = 0
        self._chunk_count = 0
        self._last_hash: Optional[str] = None
        self._committed_sequence = -1
        self._last_commit_time = time.time()
        self._total_committed = 0

        # Ensure branch exists
        self._ensure_branch()

    def _ensure_branch(self):
        """Create stream branch if it doesn't exist."""
        db = self._ingest._db
        with self._ingest._db_lock:
            if db.refs.get_branch(self.branch_name) is None:
                head = db.refs.get_head_commit()
                if head is not None:
                    db.refs.set_branch(self.branch_name, head)
                # If no commits yet, branch will be created on first commit

    def ingest(
        self,
        data: bytes,
        source: str = "unknown",
        chunk_type: str = "raw_bytes",
        metadata: Optional[dict] = None,
    ) -> ChunkMeta:
        """Ingest a chunk of data.

        Returns ChunkMeta for the ingested chunk.
        Deduplicates by content hash.
        """
        chunk_hash = hashlib.sha256(data).hexdigest()

        with self._lock:
            # Dedup check
            if chunk_hash in self._seen_hashes:
                # Return meta for the duplicate without storing
                return ChunkMeta(
                    chunk_hash=chunk_hash,
                    source=source,
                    shard_id=self.shard_id,
                    timestamp=time.time(),
                    sequence=-1,  # not stored
                    parent_hash=self._last_hash,
                    size_bytes=len(data),
                    chunk_type=chunk_type,
                )

            self._sequence += 1
            seq = self._sequence
            parent = self._last_hash

            meta = ChunkMeta(
                chunk_hash=chunk_hash,
                source=source,
                shard_id=self.shard_id,
                timestamp=time.time(),
                sequence=seq,
                parent_hash=parent,
                size_bytes=len(data),
                chunk_type=chunk_type,
            )

            # Write to WAL
            self._wal.append(meta, data)

            # Add to buffer
            self._buffer.put(meta, data)

            # Track
            self._seen_hashes.add(chunk_hash)
            self._last_hash = chunk_hash
            self._chunk_count += 1
            self._merkle.add_leaf(chunk_hash)

            # Check auto-commit threshold
            should_commit = self._chunk_count >= self._commit_every_n

        if should_commit:
            self._do_commit()

        return meta

    def _do_commit(self):
        """Drain buffer and commit chunks to GitDB."""
        items = self._buffer.drain()
        if not items:
            return

        db = self._ingest._db
        merkle_root = self._merkle.root_hash()

        with self._ingest._db_lock:
            # Save current branch state
            original_branch = db.refs.current_branch

            # Switch to shard branch
            if original_branch != self.branch_name:
                # Ensure branch exists for first commit
                if db.refs.get_branch(self.branch_name) is None:
                    head = db.refs.get_head_commit()
                    if head is not None:
                        db.refs.set_branch(self.branch_name, head)

                # Direct branch switch without staged-changes check
                commit_hash = db.refs.get_branch(self.branch_name)
                if commit_hash is not None:
                    tensor, metadata = db._reconstruct(commit_hash)
                    db.tree.load_state(tensor, metadata)
                    db._load_data_state(commit_hash)
                else:
                    db.tree.load_state(
                        __import__("torch").zeros(0, db.dim),
                        [],
                    )
                    db.docs = __import__("gitdb.documents", fromlist=["DocumentStore"]).DocumentStore()
                db.refs.set_head(self.branch_name)
                db._clear_staging()

            # Store chunks as documents
            max_seq = -1
            for meta, data in items:
                self._store_chunk(db, meta, data)
                if meta.sequence > max_seq:
                    max_seq = meta.sequence

            # Commit
            try:
                commit_hash = db.commit(
                    f"stream/{self.shard_id}: {len(items)} chunks, merkle={merkle_root[:12]}"
                )
            except ValueError:
                # Nothing to commit (all deduped away)
                commit_hash = db.refs.get_head_commit()

            # Switch back to original branch
            if original_branch != self.branch_name:
                orig_commit = db.refs.get_branch(original_branch)
                if orig_commit is not None:
                    tensor, metadata = db._reconstruct(orig_commit)
                    db.tree.load_state(tensor, metadata)
                    db._load_data_state(orig_commit)
                else:
                    db.tree.load_state(
                        __import__("torch").zeros(0, db.dim),
                        [],
                    )
                    db.docs = __import__("gitdb.documents", fromlist=["DocumentStore"]).DocumentStore()
                db.refs.set_head(original_branch)
                db._clear_staging()

        # Update tracking
        with self._lock:
            self._chunk_count = 0
            self._committed_sequence = max_seq
            self._last_commit_time = time.time()
            self._total_committed += len(items)
            self._merkle = MerkleTree()  # reset for next batch

        # Truncate WAL up to committed sequence
        if max_seq >= 0:
            self._wal.truncate(max_seq)

    def _store_chunk(self, db: GitDB, meta: ChunkMeta, data: bytes):
        """Store a single chunk in GitDB's document store."""
        chunk_meta_fields = {
            f"_chunk_{k}": v for k, v in meta.to_dict().items()
        }

        if meta.chunk_type == "records":
            try:
                records = json.loads(data.decode("utf-8"))
                if isinstance(records, list):
                    for record in records:
                        doc = dict(record)
                        doc.update(chunk_meta_fields)
                        db.insert(doc)
                    return
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # For raw_bytes, embeddings, documents, or failed records parse
        doc = {
            "_data": base64.b64encode(data).decode("ascii"),
        }
        doc.update(chunk_meta_fields)
        db.insert(doc)

    def flush(self):
        """Force commit of all buffered chunks."""
        self._do_commit()

    def close(self):
        """Flush remaining data and close WAL."""
        self.flush()
        self._wal.close()

    def recover(self):
        """Replay WAL from last committed sequence for crash recovery."""
        recovered = 0
        for meta, data in self._wal.replay(from_sequence=self._committed_sequence + 1):
            with self._lock:
                if meta.chunk_hash not in self._seen_hashes:
                    self._buffer.put(meta, data)
                    self._seen_hashes.add(meta.chunk_hash)
                    self._merkle.add_leaf(meta.chunk_hash)
                    self._chunk_count += 1
                    if meta.sequence > self._sequence:
                        self._sequence = meta.sequence
                    self._last_hash = meta.chunk_hash
                    recovered += 1

        if recovered > 0:
            self._do_commit()
        return recovered

    @property
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "shard_id": self.shard_id,
                "branch": self.branch_name,
                "sequence": self._sequence,
                "buffered": self._buffer.size,
                "committed_sequence": self._committed_sequence,
                "total_committed": self._total_committed,
                "dedup_set_size": len(self._seen_hashes),
            }


# ═══════════════════════════════════════════════════════════════
#  StreamIngest — orchestrator across shards
# ═══════════════════════════════════════════════════════════════

class StreamIngest:
    """Orchestrates streaming ingest across multiple shards.

    Each shard gets its own branch, WAL, and buffer. The orchestrator
    provides convenience methods for ingest, merge, and verification.

    Usage:
        with StreamIngest("my_db", dim=1024) as si:
            si.ingest(b'data', source="sensor_1", shard_id="s1")
            si.merge_all()
    """

    def __init__(
        self,
        db_path: str,
        dim: int = 1024,
        device: str = "cpu",
        commit_every_n: int = 100,
        commit_every_t: float = 30.0,
        buffer_size: int = 10000,
    ):
        self._db = GitDB(db_path, dim=dim, device=device)
        self._shards: Dict[str, ShardStream] = {}
        self._db_lock = threading.RLock()
        self._commit_every_n = commit_every_n
        self._commit_every_t = commit_every_t
        self._buffer_size = buffer_size

    @property
    def db(self) -> GitDB:
        """Access the underlying GitDB instance."""
        return self._db

    def shard(self, shard_id: str) -> ShardStream:
        """Get or create a shard stream."""
        if shard_id not in self._shards:
            self._shards[shard_id] = ShardStream(
                stream_ingest=self,
                shard_id=shard_id,
                commit_every_n=self._commit_every_n,
                commit_every_t=self._commit_every_t,
                buffer_size=self._buffer_size,
            )
        return self._shards[shard_id]

    def ingest(
        self,
        data: bytes,
        source: str = "unknown",
        shard_id: str = "default",
        **kw,
    ) -> ChunkMeta:
        """Convenience: ingest data into a shard."""
        return self.shard(shard_id).ingest(data, source=source, **kw)

    def merge(self, shard_id: str, target: str = "main") -> MergeResult:
        """Merge a shard branch into the target branch."""
        s = self.shard(shard_id)
        s.flush()

        with self._db_lock:
            # Switch to target branch
            current = self._db.refs.current_branch
            if current != target:
                target_hash = self._db.refs.get_branch(target)
                if target_hash is not None:
                    tensor, metadata = self._db._reconstruct(target_hash)
                    self._db.tree.load_state(tensor, metadata)
                    self._db._load_data_state(target_hash)
                self._db.refs.set_head(target)
                self._db._clear_staging()

            result = self._db.merge(s.branch_name)
            return result

    def merge_all(self, target: str = "main") -> List[MergeResult]:
        """Merge all shard branches into the target."""
        results = []
        for shard_id in list(self._shards.keys()):
            result = self.merge(shard_id, target)
            results.append(result)
        return results

    def verify(self, ref: str = "HEAD") -> bool:
        """Verify Merkle integrity for commits on a ref.

        Checks that all chunks referenced in commit messages have valid
        Merkle roots by rebuilding the tree from stored chunk hashes.
        """
        with self._db_lock:
            commit_hash = self._db.refs.resolve(ref, self._db.objects)
            if commit_hash is None:
                return True  # No commits, vacuously true

            # Load docs at this commit and rebuild Merkle tree
            doc_file = self._db._docs_dir / f"{commit_hash}.jsonl"
            if not doc_file.exists():
                return True

            data = self._db._decrypt_bytes(doc_file.read_bytes())
            docs = []
            for line in data.decode().strip().split("\n"):
                if line.strip():
                    docs.append(json.loads(line))

            # Extract chunk hashes and rebuild tree
            chunk_hashes = []
            for doc in docs:
                ch = doc.get("_chunk_chunk_hash")
                if ch:
                    chunk_hashes.append(ch)

            if not chunk_hashes:
                return True

            # Rebuild Merkle tree
            tree = MerkleTree()
            for ch in chunk_hashes:
                tree.add_leaf(ch)

            # The root should match what's in the commit message
            commit = self._db.objects.read_commit(commit_hash)
            expected_prefix = "merkle="
            msg = commit.message
            if expected_prefix in msg:
                stored_root_prefix = msg.split("merkle=")[1][:12]
                computed_root = tree.root_hash()[:12]
                return stored_root_prefix == computed_root

            return True

    def status(self) -> Dict[str, Any]:
        """Status of all shards."""
        return {
            "db_path": str(self._db.path),
            "shards": {sid: s.status for sid, s in self._shards.items()},
            "total_shards": len(self._shards),
        }

    def close(self):
        """Close all shards."""
        for s in self._shards.values():
            s.close()
        self._shards.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
