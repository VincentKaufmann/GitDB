"""Tests for the streaming ingest module."""

import json
import struct
import threading
import time

import pytest
import torch

from gitdb import GitDB
from gitdb.streaming import (
    ChunkBuffer,
    ChunkMeta,
    MerkleTree,
    ShardStream,
    StreamIngest,
    WAL,
)


# ═══════════════════════════════════════════════════════════════
#  TestChunkMeta
# ═══════════════════════════════════════════════════════════════

class TestChunkMeta:

    def test_create_meta(self):
        meta = ChunkMeta(
            chunk_hash="abc123",
            source="sensor_1",
            shard_id="s1",
            timestamp=1000.0,
            sequence=1,
            parent_hash=None,
            size_bytes=256,
            chunk_type="raw_bytes",
        )
        assert meta.chunk_hash == "abc123"
        assert meta.source == "sensor_1"
        assert meta.shard_id == "s1"
        assert meta.timestamp == 1000.0
        assert meta.sequence == 1
        assert meta.parent_hash is None
        assert meta.size_bytes == 256
        assert meta.chunk_type == "raw_bytes"

    def test_hash_chain(self):
        m1 = ChunkMeta("hash1", "src", "s1", 1.0, 1, None, 10, "raw_bytes")
        m2 = ChunkMeta("hash2", "src", "s1", 2.0, 2, m1.chunk_hash, 20, "raw_bytes")
        m3 = ChunkMeta("hash3", "src", "s1", 3.0, 3, m2.chunk_hash, 30, "raw_bytes")

        assert m1.parent_hash is None
        assert m2.parent_hash == "hash1"
        assert m3.parent_hash == "hash2"

    def test_to_dict_from_dict(self):
        meta = ChunkMeta(
            chunk_hash="abc",
            source="test",
            shard_id="s1",
            timestamp=42.0,
            sequence=7,
            parent_hash="prev",
            size_bytes=100,
            chunk_type="records",
        )
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["chunk_hash"] == "abc"
        assert d["parent_hash"] == "prev"

        restored = ChunkMeta.from_dict(d)
        assert restored.chunk_hash == meta.chunk_hash
        assert restored.source == meta.source
        assert restored.shard_id == meta.shard_id
        assert restored.timestamp == meta.timestamp
        assert restored.sequence == meta.sequence
        assert restored.parent_hash == meta.parent_hash
        assert restored.size_bytes == meta.size_bytes
        assert restored.chunk_type == meta.chunk_type


# ═══════════════════════════════════════════════════════════════
#  TestWAL
# ═══════════════════════════════════════════════════════════════

class TestWAL:

    def _make_meta(self, seq, shard="test"):
        return ChunkMeta(
            chunk_hash=f"hash_{seq}",
            source="test",
            shard_id=shard,
            timestamp=float(seq),
            sequence=seq,
            parent_hash=f"hash_{seq-1}" if seq > 1 else None,
            size_bytes=10,
            chunk_type="raw_bytes",
        )

    def test_append_and_replay(self, tmp_path):
        wal = WAL(tmp_path / "test.wal")
        data1 = b"hello world"
        data2 = b"second entry"

        wal.append(self._make_meta(1), data1)
        wal.append(self._make_meta(2), data2)
        wal.close()

        wal2 = WAL(tmp_path / "test.wal")
        entries = list(wal2.replay())
        assert len(entries) == 2
        assert entries[0][0].sequence == 1
        assert entries[0][1] == data1
        assert entries[1][0].sequence == 2
        assert entries[1][1] == data2
        wal2.close()

    def test_truncate(self, tmp_path):
        wal = WAL(tmp_path / "test.wal")
        for i in range(1, 6):
            wal.append(self._make_meta(i), f"data_{i}".encode())

        wal.truncate(3)
        entries = list(wal.replay())
        assert len(entries) == 2
        assert entries[0][0].sequence == 4
        assert entries[1][0].sequence == 5
        wal.close()

    def test_empty_replay(self, tmp_path):
        wal = WAL(tmp_path / "test.wal")
        entries = list(wal.replay())
        assert entries == []
        wal.close()

    def test_crash_recovery(self, tmp_path):
        wal_path = tmp_path / "test.wal"
        wal = WAL(wal_path)
        wal.append(self._make_meta(1), b"good_entry")
        wal.close()

        # Append a partial entry — simulate crash mid-write
        with open(wal_path, "ab") as f:
            # Write a length header claiming 1000 bytes, but only write 10
            f.write(struct.pack("<I", 1000))
            f.write(b"incomplete!")

        wal2 = WAL(wal_path)
        entries = list(wal2.replay())
        assert len(entries) == 1
        assert entries[0][0].sequence == 1
        assert entries[0][1] == b"good_entry"
        wal2.close()

    def test_concurrent_append(self, tmp_path):
        wal = WAL(tmp_path / "test.wal")
        errors = []

        def writer(start, count):
            try:
                for i in range(start, start + count):
                    wal.append(self._make_meta(i, shard=f"t{start}"), f"data_{i}".encode())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 100, 50)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        entries = list(wal.replay())
        assert len(entries) == 200  # 4 threads * 50 entries
        wal.close()


# ═══════════════════════════════════════════════════════════════
#  TestMerkleTree
# ═══════════════════════════════════════════════════════════════

class TestMerkleTree:

    def test_single_leaf(self):
        tree = MerkleTree()
        tree.add_leaf("abc123")
        assert tree.root_hash() == "abc123"

    def test_two_leaves(self):
        import hashlib
        tree = MerkleTree()
        tree.add_leaf("aaa")
        tree.add_leaf("bbb")
        expected = hashlib.sha256(("aaa" + "bbb").encode("utf-8")).hexdigest()
        assert tree.root_hash() == expected

    def test_root_deterministic(self):
        t1 = MerkleTree()
        t2 = MerkleTree()
        leaves = ["hash1", "hash2", "hash3", "hash4"]
        for leaf in leaves:
            t1.add_leaf(leaf)
            t2.add_leaf(leaf)
        assert t1.root_hash() == t2.root_hash()

    def test_proof_and_verify(self):
        tree = MerkleTree()
        leaves = ["a", "b", "c", "d"]
        for leaf in leaves:
            tree.add_leaf(leaf)

        root = tree.root_hash()

        for i in range(len(leaves)):
            p = tree.proof(i)
            assert MerkleTree.verify_proof(leaves[i], p, root)

    def test_tamper_detection(self):
        tree = MerkleTree()
        leaves = ["a", "b", "c", "d"]
        for leaf in leaves:
            tree.add_leaf(leaf)

        root = tree.root_hash()
        proof = tree.proof(1)

        # Tamper: use wrong leaf hash
        assert not MerkleTree.verify_proof("tampered", proof, root)


# ═══════════════════════════════════════════════════════════════
#  TestChunkBuffer
# ═══════════════════════════════════════════════════════════════

class TestChunkBuffer:

    def _meta(self, seq):
        return ChunkMeta(f"h{seq}", "src", "s1", 0.0, seq, None, 1, "raw_bytes")

    def test_put_drain(self):
        buf = ChunkBuffer(max_size=100)
        buf.put(self._meta(1), b"a")
        buf.put(self._meta(2), b"b")
        items = buf.drain()
        assert len(items) == 2
        assert items[0][1] == b"a"
        assert items[1][1] == b"b"
        assert buf.size == 0

    def test_backpressure(self):
        buf = ChunkBuffer(max_size=2)
        buf.put(self._meta(1), b"a")
        buf.put(self._meta(2), b"b")
        # Buffer is full, should return False with timeout=0
        result = buf.put(self._meta(3), b"c", timeout=0)
        assert result is False
        assert buf.size == 2

    def test_concurrent_put_drain(self):
        buf = ChunkBuffer(max_size=1000)
        produced = []
        consumed = []

        def producer(start, n):
            for i in range(start, start + n):
                buf.put(self._meta(i), f"d{i}".encode())
                produced.append(i)

        def consumer():
            time.sleep(0.05)
            while True:
                items = buf.drain(50)
                if not items:
                    break
                consumed.extend(items)
                time.sleep(0.01)

        threads = [
            threading.Thread(target=producer, args=(0, 100)),
            threading.Thread(target=producer, args=(100, 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Drain everything remaining
        remaining = buf.drain()
        consumed.extend(remaining)

        assert len(consumed) == 200


# ═══════════════════════════════════════════════════════════════
#  TestShardStream
# ═══════════════════════════════════════════════════════════════

class TestShardStream:

    def test_ingest_single(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4) as si:
            meta = si.shard("s1").ingest(b"hello", source="test")
            assert meta.chunk_hash is not None
            assert meta.source == "test"
            assert meta.shard_id == "s1"
            assert meta.size_bytes == 5
            si.shard("s1").flush()

    def test_dedup(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4) as si:
            s = si.shard("s1")
            m1 = s.ingest(b"same_data", source="a")
            m2 = s.ingest(b"same_data", source="b")  # duplicate
            assert m1.chunk_hash == m2.chunk_hash
            assert m1.sequence == 1
            assert m2.sequence == -1  # not stored

    def test_auto_commit_on_count(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4, commit_every_n=3) as si:
            s = si.shard("s1")
            for i in range(3):
                s.ingest(f"chunk_{i}".encode(), source="test")

            # After 3 chunks, auto-commit should have fired
            # Verify data is on the shard branch
            db = si._db
            shard_commit = db.refs.get_branch("stream/s1")
            assert shard_commit is not None

    def test_hash_chain_integrity(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4, commit_every_n=100) as si:
            s = si.shard("s1")
            metas = []
            for i in range(5):
                m = s.ingest(f"chunk_{i}".encode(), source="test")
                metas.append(m)

            assert metas[0].parent_hash is None
            for i in range(1, 5):
                assert metas[i].parent_hash == metas[i - 1].chunk_hash

    def test_flush(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4, commit_every_n=1000) as si:
            s = si.shard("s1")
            s.ingest(b"data1", source="test")
            s.ingest(b"data2", source="test")

            # Not auto-committed yet (commit_every_n=1000)
            s.flush()

            # After flush, data should be committed
            shard_commit = si._db.refs.get_branch("stream/s1")
            assert shard_commit is not None

    def test_recover(self, tmp_path):
        db_path = str(tmp_path / "db")

        # Phase 1: ingest data but don't commit (simulate crash)
        si1 = StreamIngest(db_path, dim=4, commit_every_n=1000)
        s1 = si1.shard("s1")
        s1.ingest(b"recover_me_1", source="test")
        s1.ingest(b"recover_me_2", source="test")
        # Close WAL without flushing
        s1._wal.close()

        # Phase 2: create new StreamIngest and recover
        si2 = StreamIngest(db_path, dim=4, commit_every_n=1000)
        s2 = si2.shard("s1")
        recovered = s2.recover()
        assert recovered == 2

        # Data should now be committed
        shard_commit = si2._db.refs.get_branch("stream/s1")
        assert shard_commit is not None
        si2.close()


# ═══════════════════════════════════════════════════════════════
#  TestStreamIngest
# ═══════════════════════════════════════════════════════════════

class TestStreamIngest:

    def test_single_shard_ingest(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4) as si:
            meta = si.ingest(b"test_data", source="sensor", shard_id="s1")
            si.shard("s1").flush()
            assert meta.chunk_hash is not None
            assert meta.shard_id == "s1"

    def test_multi_shard(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4, commit_every_n=5) as si:
            for i in range(5):
                si.ingest(f"s1_{i}".encode(), source="a", shard_id="s1")
                si.ingest(f"s2_{i}".encode(), source="b", shard_id="s2")

            # Both shards should have committed
            assert si._db.refs.get_branch("stream/s1") is not None
            assert si._db.refs.get_branch("stream/s2") is not None

    def test_merge_shard(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4, commit_every_n=3) as si:
            # Make initial commit on main so merge has something to work with
            si._db.insert({"_bootstrap": True})
            si._db.commit("bootstrap")

            for i in range(3):
                si.ingest(f"chunk_{i}".encode(), source="test", shard_id="s1")

            result = si.merge("s1", target="main")
            assert result.commit_hash is not None

            # Verify data is on main
            si._db.switch("main")
            docs = si._db.find()
            chunk_docs = [d for d in docs if "_chunk_chunk_hash" in d]
            assert len(chunk_docs) >= 3

    def test_status(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4) as si:
            si.ingest(b"x", shard_id="alpha")
            si.ingest(b"y", shard_id="beta")

            status = si.status()
            assert status["total_shards"] == 2
            assert "alpha" in status["shards"]
            assert "beta" in status["shards"]
            assert status["shards"]["alpha"]["shard_id"] == "alpha"

    def test_context_manager(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4) as si:
            si.ingest(b"data", shard_id="s1")
        # After exiting context, shards should be closed
        assert len(si._shards) == 0

    def test_verify_integrity(self, tmp_path):
        with StreamIngest(str(tmp_path / "db"), dim=4, commit_every_n=3) as si:
            for i in range(3):
                si.ingest(f"v_{i}".encode(), source="test", shard_id="s1")

            # Switch to shard branch and verify
            si._db.switch("stream/s1")
            assert si.verify(ref="HEAD") is True
