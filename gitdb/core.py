"""GitDB — the main interface class."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from gitdb.delta import Delta
from gitdb.objects import Commit, ObjectStore, RefStore
from gitdb.pullrequest import PRStore, PullRequest, Comment
from gitdb.remote import RemoteManager, push as _push, pull as _pull, fetch as _fetch
from gitdb.types import (
    BlameEntry, BisectResult, CommitInfo, CommitStats, DiffResult,
    MergeResult, Results, StashEntry, VectorMeta,
)
from gitdb.ambient import EmiratiAC
from gitdb.documents import DocumentStore, TableStore, Table
from gitdb.hooks import HookManager
from gitdb.indexes import IndexManager
from gitdb.schema import Schema, SchemaError
from gitdb.snapshots import Snapshot
from gitdb.watches import WatchManager
from gitdb.backup import backup_full, backup_incremental, restore, verify, BackupManager
from gitdb.encryption import EncryptionManager
from gitdb.working_tree import WorkingTree


class GitDB:
    """GPU-accelerated version-controlled vector database.

    Usage:
        db = GitDB("my_store", dim=1024, device="mps")
        db.add(embeddings=tensor, documents=["doc1", "doc2"])
        db.commit("Initial ingest")
        results = db.query(vector=query_vec, k=10)
    """

    def __init__(self, path: str, dim: int = 1024, device: str = "cpu",
                 encryption: Optional["EncryptionManager"] = None):
        self.path = Path(path)
        self.dim = dim
        self.device = device

        # Encryption — auto-detect from env if not explicitly provided
        if encryption is None:
            encryption = EncryptionManager.from_env()
        self.encryption = encryption

        self._gitdb_dir = self.path / ".gitdb"
        self._is_new = not self._gitdb_dir.exists()

        # Initialize storage
        self._gitdb_dir.mkdir(parents=True, exist_ok=True)
        self.objects = ObjectStore(self._gitdb_dir, encryption=self.encryption)
        self.refs = RefStore(self._gitdb_dir)

        # Write config
        config_path = self._gitdb_dir / "config"
        if self._is_new:
            config_path.write_text(json.dumps({
                "dim": dim,
                "device": device,
                "version": "0.1.0",
            }, indent=2))
            self.refs.init()
        else:
            cfg = json.loads(config_path.read_text())
            self.dim = cfg["dim"]
            # Device can be overridden at open time
            if device == "cpu" and cfg.get("device") != "cpu":
                pass  # User explicitly chose cpu, that's fine

        # Working tree (GPU-resident state)
        self.tree = WorkingTree(dim=self.dim, device=self.device)

        # Staging area — tracks what changed since last commit
        self._staged_additions: List[int] = []       # row indices of added vectors
        self._staged_deletions: List[int] = []       # row indices before compaction
        self._staged_del_embeddings: List[torch.Tensor] = []  # old values for reverse
        self._staged_del_metadata: List[VectorMeta] = []
        self._staged_modifications: List[tuple] = [] # (idx, old_emb, new_emb)

        # Tensor cache for reconstructed historical states
        self._cache_dir = self._gitdb_dir / "cache" / "tensors"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Emirati AC — spreading activation engine
        self.ac = EmiratiAC(self)

        # Watches — change subscriptions
        self.watches = WatchManager()

        # Secondary indexes on metadata fields
        self.indexes = IndexManager()

        # In-memory snapshots
        self._snapshots: Dict[str, Snapshot] = {}

        # Hooks — pre/post event callbacks
        self.hooks = HookManager()

        # Schema enforcement on metadata
        self._schema: Optional[Schema] = None
        schema_path = self._gitdb_dir / "schema.json"
        if schema_path.exists():
            self._schema = Schema(json.loads(schema_path.read_text()))

        # Document store (MongoDB replacement — no vectors needed)
        self.docs = DocumentStore()
        self._staged_doc_inserts: int = 0
        self._staged_doc_updates: int = 0
        self._staged_doc_deletes: int = 0
        self._docs_dir = self._gitdb_dir / "docs"
        self._docs_dir.mkdir(exist_ok=True)

        # Table store (SQLite replacement — named tables with schemas)
        self.tables = TableStore()
        self._staged_table_changes: int = 0
        self._tables_dir = self._gitdb_dir / "tables"
        self._tables_dir.mkdir(exist_ok=True)

        # If existing repo, load HEAD state
        if not self._is_new:
            self._load_head()

    def _encrypt_bytes(self, data: bytes) -> bytes:
        if self.encryption and self.encryption.enabled:
            return self.encryption.encrypt(data)
        return data

    def _decrypt_bytes(self, data: bytes) -> bytes:
        if self.encryption and self.encryption.enabled:
            return self.encryption.decrypt(data)
        return data

    def _save_data_state(self, commit_hash: str):
        """Save document and table state for a commit."""
        self.docs.compact()
        doc_snapshot = self.docs.snapshot()
        if doc_snapshot.strip():
            (self._docs_dir / f"{commit_hash}.jsonl").write_bytes(
                self._encrypt_bytes(doc_snapshot))
        table_snapshot = self.tables.snapshot()
        if table_snapshot.strip() and table_snapshot != b"{}":
            (self._tables_dir / f"{commit_hash}.json").write_bytes(
                self._encrypt_bytes(table_snapshot))

    def _load_data_state(self, commit_hash: Optional[str]):
        """Load document and table state for a commit (or clear if None)."""
        if commit_hash is None:
            self.docs = DocumentStore()
            self.tables = TableStore()
            return
        doc_file = self._docs_dir / f"{commit_hash}.jsonl"
        if doc_file.exists():
            self.docs.restore(self._decrypt_bytes(doc_file.read_bytes()))
        else:
            self.docs = DocumentStore()
        table_file = self._tables_dir / f"{commit_hash}.json"
        if table_file.exists():
            self.tables.restore(self._decrypt_bytes(table_file.read_bytes()))
        else:
            self.tables = TableStore()

    def _load_head(self):
        """Reconstruct working tree from HEAD commit."""
        head_hash = self.refs.get_head_commit()
        if head_hash is None:
            return  # Empty repo, no commits yet

        tensor, metadata = self._reconstruct(head_hash)
        self.tree.load_state(tensor, metadata)
        self.indexes.rebuild(self.tree.metadata)
        self._load_data_state(head_hash)

    def _reconstruct(self, commit_hash: str):
        """Reconstruct tensor + metadata at a given commit by replaying deltas."""
        # Check cache first
        cache_file = self._cache_dir / f"{commit_hash}.pt"
        meta_cache = self._cache_dir / f"{commit_hash}.meta.json"
        if cache_file.exists() and meta_cache.exists():
            tensor = torch.load(cache_file, weights_only=True)
            meta_raw = json.loads(meta_cache.read_text())
            metadata = [
                VectorMeta(id=m["id"], document=m.get("document"), metadata=m.get("metadata", {}))
                for m in meta_raw
            ]
            return tensor, metadata

        # Walk back to root, collecting deltas
        chain = []
        current = commit_hash
        while current is not None:
            commit = self.objects.read_commit(current)
            chain.append(commit)
            current = commit.parent

        # Replay forward from root
        chain.reverse()
        tensor = torch.zeros(0, self.dim)
        metadata: List[VectorMeta] = []

        for commit in chain:
            delta_data = self.objects.read(commit.delta_hash)
            delta = Delta.deserialize(delta_data, self.dim)
            tensor, metadata = delta.apply_forward(tensor, metadata)

        # Cache the result
        torch.save(tensor, cache_file)
        meta_raw = [{"id": m.id, "document": m.document, "metadata": m.metadata} for m in metadata]
        meta_cache.write_text(json.dumps(meta_raw))

        return tensor, metadata

    # ─── Data Operations ─────────────────────────────────────

    def add(
        self,
        embeddings: Optional[torch.Tensor] = None,
        documents: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        embed_model: Optional[str] = None,
        embed_dim: Optional[int] = None,
    ) -> List[int]:
        """Add vectors with optional documents and metadata.

        If `texts` is given instead of `embeddings`, auto-embed using the
        specified model (default: arctic). Documents are set to the texts.
        """
        if texts is not None and embeddings is None:
            from gitdb.embed import embed, DEFAULT_MODEL
            model = embed_model or DEFAULT_MODEL
            embeddings = embed(texts, model_name=model, dim=embed_dim or self.dim)
            if documents is None:
                documents = texts
        if embeddings is None:
            raise ValueError("Must provide embeddings or texts")
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        # Schema validation
        self._validate_metadata(metadata)
        indices = self.tree.add(embeddings, documents, metadata, ids)
        self._staged_additions.extend(indices)
        # Update secondary indexes
        for idx in indices:
            self.indexes.update(idx, self.tree.metadata[idx].metadata)
        # Feed AC
        if self.ac.running:
            self.ac.feed_vectors(embeddings)
            self.ac.track_addition(embeddings)
        return indices

    def remove(
        self,
        ids: Optional[List[int]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Remove vectors by index or metadata filter."""
        if ids is not None:
            # Capture old values before deletion
            for idx in ids:
                self._staged_deletions.append(idx)
                self._staged_del_embeddings.append(self.tree.embeddings[idx].cpu().clone())
                self._staged_del_metadata.append(self.tree.metadata[idx])
            self.tree.remove(ids)
            return ids
        elif where is not None:
            removed = self.tree.remove_where(where)
            for idx in removed:
                self._staged_deletions.append(idx)
                self._staged_del_embeddings.append(self.tree.embeddings[idx].cpu().clone())
                self._staged_del_metadata.append(self.tree.metadata[idx])
            return removed
        else:
            raise ValueError("Must specify ids or where")

    def update_embeddings(self, ids: List[int], embeddings: torch.Tensor):
        """Update embeddings for specific rows."""
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        for i, idx in enumerate(ids):
            old_emb = self.tree.embeddings[idx].cpu().clone()
            self._staged_modifications.append((idx, old_emb, embeddings[i].cpu().clone()))
        self.tree.update_embeddings(ids, embeddings)

    # ─── Document Operations (MongoDB/SQLite replacement) ────
    # Use db.docs for direct access, or these convenience methods.
    # These work independently of vectors — pure document storage.

    def insert(self, doc: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[str, List[str]]:
        """Insert document(s). Returns _id or list of _ids.

        Usage:
            db.insert({"name": "Alice", "age": 30})
            db.insert([{"name": "Bob"}, {"name": "Charlie"}])
        """
        if isinstance(doc, list):
            ids = self.docs.insert_many(doc)
            self._staged_doc_inserts += len(doc)
            return ids
        _id = self.docs.insert(doc)
        self._staged_doc_inserts += 1
        return _id

    def find(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Find documents matching a query (MongoDB-style).

        Usage:
            db.find(where={"age": {"$gt": 25}})
            db.find(where={"role": "engineer"}, limit=10)
        """
        return self.docs.find(where=where, limit=limit, skip=skip)

    def find_one(self, where: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Find a single matching document."""
        return self.docs.find_one(where=where)

    def update_docs(self, where: Dict[str, Any], set_fields: Dict[str, Any]) -> int:
        """Update documents matching query. Returns count updated.

        Usage:
            db.update_docs({"name": "Alice"}, {"age": 31})
        """
        count = self.docs.update(where, set_fields)
        self._staged_doc_updates += count
        return count

    def delete_docs(self, where: Dict[str, Any]) -> int:
        """Delete documents matching query. Returns count deleted."""
        count = self.docs.delete(where)
        self._staged_doc_deletes += count
        return count

    def count_docs(self, where: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching query."""
        return self.docs.count(where=where)

    def distinct(self, field: str, where: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Get distinct values for a field across documents."""
        return self.docs.distinct(field, where=where)

    def aggregate_docs(
        self,
        group_by: str,
        agg_field: Optional[str] = None,
        agg_fn: str = "count",
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[Any, Any]:
        """Group documents and aggregate (count, sum, avg, min, max)."""
        return self.docs.aggregate(group_by, agg_field, agg_fn, where=where)

    # ─── Table Operations (SQLite replacement) ────────────────

    def create_table(self, name: str, columns: Optional[Dict[str, str]] = None) -> "Table":
        """Create a named table with optional column schema.

        Usage:
            db.create_table("users", {"name": "text", "age": "integer", "email": "text"})
            db.create_table("logs")  # schemaless
        """
        t = self.tables.create(name, columns)
        self._staged_table_changes += 1
        return t

    def table(self, name: str) -> "Table":
        """Get a table by name. Raises if not found."""
        return self.tables.get(name)

    def collection(self, name: str) -> "Table":
        """Get or create a schemaless collection (MongoDB-style alias for table)."""
        t = self.tables.get_or_create(name)
        return t

    def drop_table(self, name: str):
        """Drop a table."""
        self.tables.drop(name)
        self._staged_table_changes += 1

    def list_tables(self) -> List[str]:
        """List all table names."""
        return self.tables.list_tables()

    def insert_into(self, table: str, row: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[int, List[int]]:
        """Insert row(s) into a named table.

        Usage:
            db.insert_into("users", {"name": "Alice", "age": 30})
            db.insert_into("users", [{"name": "Bob"}, {"name": "Charlie"}])
        """
        t = self.tables.get(table)
        if isinstance(row, list):
            ids = t.insert_many(row)
            self._staged_table_changes += len(row)
            return ids
        _id = t.insert(row)
        self._staged_table_changes += 1
        return _id

    def select_from(
        self, table: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """SQL-style SELECT from a named table."""
        return self.tables.get(table).select(
            columns=columns, where=where,
            order_by=order_by, desc=desc,
            limit=limit, offset=offset,
        )

    def update_table(self, table: str, where: Dict[str, Any], set_fields: Dict[str, Any]) -> int:
        """UPDATE rows in a table."""
        count = self.tables.get(table).update(where, set_fields)
        self._staged_table_changes += count
        return count

    def delete_from(self, table: str, where: Dict[str, Any]) -> int:
        """DELETE rows from a table."""
        count = self.tables.get(table).delete(where)
        self._staged_table_changes += count
        return count

    # ─── Query ────────────────────────────────────────────────

    def query(
        self,
        vector: torch.Tensor,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        at: Optional[str] = None,
        ac_boost: bool = True,
    ) -> Results:
        """Similarity search. Use `at` for time-travel queries.

        If AC is running and ac_boost=True, results are boosted by activation level.
        """
        if at is not None:
            return self._query_at(vector, k, where, at)
        # Feed AC with query vector
        if self.ac.running:
            self.ac.feed_vectors(vector)
        results = self.tree.query(vector, k, where)
        # Boost with AC if running
        if self.ac.running and ac_boost:
            results = self.ac.boost_results(results)
        return results

    def _resolve(self, ref: str) -> Optional[str]:
        """Resolve a ref, passing object store for relative refs."""
        return self.refs.resolve(ref, self.objects)

    def _query_at(self, vector, k, where, ref):
        """Query a historical state."""
        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")
        tensor, metadata = self._reconstruct(commit_hash)
        # Build a temporary working tree
        tmp = WorkingTree(dim=self.dim, device=self.device)
        tmp.load_state(tensor, metadata)
        return tmp.query(vector, k, where)

    def query_text(
        self,
        text: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        at: Optional[str] = None,
        embed_model: Optional[str] = None,
        embed_dim: Optional[int] = None,
    ) -> "Results":
        """Semantic query — embed text then search."""
        from gitdb.embed import embed_query, DEFAULT_MODEL
        model = embed_model or DEFAULT_MODEL
        if self.ac.running:
            self.ac.feed("query", text)
        vector = embed_query(text, model_name=model, dim=embed_dim or self.dim)
        return self.query(vector, k=k, where=where, at=at)

    def semantic_cherry_pick(
        self,
        source_branch: str,
        query: str,
        threshold: float = 0.5,
        embed_model: Optional[str] = None,
    ) -> Optional[str]:
        """Cherry-pick commits from source_branch whose added vectors match query semantically.

        Walks source_branch history, embeds query, checks each delta's added vectors.
        Cherry-picks the first commit where any added vector exceeds threshold.
        """
        from gitdb.embed import embed_query, similarity, DEFAULT_MODEL
        model = embed_model or DEFAULT_MODEL
        q = embed_query(query, model_name=model, dim=self.dim)

        # Walk source branch
        source_head = self.refs.resolve(source_branch, self.objects)
        if source_head is None:
            raise ValueError(f"Cannot resolve branch: {source_branch}")

        current = source_head
        picked = []
        while current is not None:
            commit = self.objects.read_commit(current)
            delta_data = self.objects.read(commit.delta_hash)
            delta = Delta.deserialize(delta_data, self.dim)

            if delta.add_embeddings is not None and delta.add_embeddings.shape[0] > 0:
                sims = similarity(q, delta.add_embeddings).squeeze(0)
                if sims.max().item() >= threshold:
                    picked.append(current)

            current = commit.parent

        # Cherry-pick all matching commits (oldest first)
        result = None
        for commit_hash in reversed(picked):
            result = self.cherry_pick(commit_hash)
        return result

    def semantic_blame(
        self,
        query: str,
        threshold: float = 0.5,
        embed_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Blame by concept — find which commits introduced vectors matching the query.

        Returns list of {commit, author, message, vector_idx, similarity, document}.
        """
        from gitdb.embed import embed_query, similarity, DEFAULT_MODEL
        model = embed_model or DEFAULT_MODEL
        q = embed_query(query, model_name=model, dim=self.dim)

        blame_entries = self.blame()
        results = []

        for entry in blame_entries:
            idx = entry.row_index
            if idx < self.tree.embeddings.shape[0]:
                sim = similarity(q, self.tree.embeddings[idx].unsqueeze(0).cpu()).item()
                if sim >= threshold:
                    results.append({
                        "commit": entry.commit_hash,
                        "message": entry.commit_message,
                        "vector_idx": idx,
                        "similarity": round(sim, 4),
                        "document": self.tree.metadata[idx].document if idx < len(self.tree.metadata) else None,
                    })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def semantic_diff(
        self,
        ref_a: str,
        ref_b: str,
        embed_model: Optional[str] = None,
        n_clusters: int = 5,
    ) -> Dict[str, Any]:
        """Semantic diff — cluster added/removed vectors by meaning.

        Returns dict with added/removed/modified counts plus document clusters.
        """
        diff = self.diff(ref_a, ref_b)

        def _cluster_entries(entries):
            """Group diff entries by first word heuristic."""
            if not entries:
                return []
            groups = {}
            for e in entries:
                doc = e.document or e.id[:8]
                key = doc.split()[0] if doc.split() else "unknown"
                groups.setdefault(key, []).append(doc)
            return [{"label": k, "count": len(v), "samples": v[:3]} for k, v in groups.items()]

        added_entries = [e for e in diff.entries if e.change == "added"]
        removed_entries = [e for e in diff.entries if e.change == "removed"]

        return {
            "added_count": diff.added_count,
            "removed_count": diff.removed_count,
            "modified_count": diff.modified_count,
            "added_summary": _cluster_entries(added_entries),
            "removed_summary": _cluster_entries(removed_entries),
        }

    def re_embed(
        self,
        embed_model: Optional[str] = None,
        embed_dim: Optional[int] = None,
    ) -> str:
        """Re-embed all vectors using their document text. Returns commit hash.

        Walks all vectors, re-embeds from document text using the specified model,
        commits the result as a new state.
        """
        from gitdb.embed import embed, DEFAULT_MODEL
        model = embed_model or DEFAULT_MODEL

        texts = []
        for meta in self.tree.metadata:
            if meta.document:
                texts.append(meta.document)
            else:
                texts.append("")  # No text, will get zero-ish vector

        if not texts:
            raise ValueError("No documents to re-embed")

        has_text = [i for i, t in enumerate(texts) if t]
        if not has_text:
            raise ValueError("No vectors have document text for re-embedding")

        # Embed only non-empty texts
        text_list = [texts[i] for i in has_text]
        new_embeddings = embed(text_list, model_name=model, dim=embed_dim or self.dim)

        # Update each vector
        for j, idx in enumerate(has_text):
            old_emb = self.tree.embeddings[idx].cpu().clone()
            self._staged_modifications.append((idx, old_emb, new_embeddings[j].cpu()))

        self.tree.embeddings[has_text] = new_embeddings.to(self.tree.embeddings.device)
        return f"re-embedded {len(has_text)} vectors with {model}"

    # ─── Structured Data Operations ───────────────────────────

    def select(
        self,
        fields: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        reverse: bool = False,
    ) -> List[Dict[str, Any]]:
        """SQL-style SELECT over metadata.

        Args:
            fields: Fields to return (None = all). Supports dotted paths.
            where: Structured filter with operators ($gt, $in, $regex, etc).
            order_by: Field to sort by.
            limit: Max rows to return.
            reverse: Sort descending.

        Returns:
            List of dicts with projected fields.

        Examples:
            db.select(fields=["document", "category"], where={"score": {"$gt": 0.8}})
            db.select(where={"$or": [{"type": "finance"}, {"type": "legal"}]})
            db.select(where={"tags": {"$contains": "urgent"}}, order_by="priority")
        """
        from gitdb.structured import matches, project, sort_by, to_table

        # Filter
        active = [m for i, m in enumerate(self.tree.metadata) if i not in self.tree.tombstones]
        if where:
            active = [m for m in active if matches(m, where)]

        # Sort
        if order_by:
            active = sort_by(active, order_by, reverse=reverse)

        # Limit
        if limit:
            active = active[:limit]

        # Project
        if fields:
            return project(active, fields)
        return to_table(active, fields)

    def group_by(
        self,
        field: str,
        agg_field: Optional[str] = None,
        agg_fn: str = "count",
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[Any, Any]:
        """GROUP BY aggregation over metadata.

        Args:
            field: Field to group by.
            agg_field: Field to aggregate (for sum/avg/min/max).
            agg_fn: count, sum, avg, min, max, collect.
            where: Optional pre-filter.

        Examples:
            db.group_by("category")  # → {"finance": 15, "legal": 8}
            db.group_by("author", agg_field="score", agg_fn="avg")
        """
        from gitdb.structured import matches, aggregate

        active = [m for i, m in enumerate(self.tree.metadata) if i not in self.tree.tombstones]
        if where:
            active = [m for m in active if matches(m, where)]

        return aggregate(active, field, agg_field, agg_fn)

    def export_jsonl(self, output_path: str, include_embeddings: bool = False):
        """Export entire store to JSONL."""
        from gitdb.structured import export_jsonl
        active_meta = [m for i, m in enumerate(self.tree.metadata) if i not in self.tree.tombstones]
        emb = self.tree.snapshot_tensor() if include_embeddings else None
        export_jsonl(active_meta, output_path, include_embeddings, emb)

    def import_jsonl(self, input_path: str, embed_texts: bool = False, embed_model: Optional[str] = None):
        """Import from JSONL. If rows have embeddings, use them. Otherwise optionally embed."""
        import torch as _torch
        from gitdb.structured import import_jsonl

        ids, docs, metas, embs = import_jsonl(input_path)

        has_embs = any(e is not None for e in embs)
        if has_embs:
            embeddings = _torch.tensor([e for e in embs if e is not None], dtype=_torch.float32)
            return self.add(embeddings=embeddings, documents=docs, metadata=metas, ids=ids)
        elif embed_texts and docs:
            return self.add(texts=[d or "" for d in docs], embed_model=embed_model, metadata=metas, ids=ids)
        else:
            raise ValueError("JSONL has no embeddings. Use embed_texts=True to auto-embed from documents.")

    # ─── Version Control ──────────────────────────────────────

    def commit(self, message: str) -> str:
        """Commit staged changes. Returns commit hash."""
        if not self._has_staged_changes():
            raise ValueError("Nothing to commit")

        # Fire pre-commit hook
        if not self.hooks.fire("pre-commit", message=message, db=self):
            raise ValueError("pre-commit hook rejected")

        # Build delta from staged changes
        delta = Delta()
        stats = CommitStats()

        # Additions
        if self._staged_additions:
            delta.add_indices = list(self._staged_additions)
            add_tensors = []
            for idx in self._staged_additions:
                add_tensors.append(self.tree.embeddings[idx].cpu())
            delta.add_values = torch.stack(add_tensors)
            delta.add_metadata = [self.tree.metadata[idx] for idx in self._staged_additions]
            stats.added = len(self._staged_additions)

        # Deletions
        if self._staged_deletions:
            delta.del_indices = list(self._staged_deletions)
            delta.del_values = torch.stack(self._staged_del_embeddings) if self._staged_del_embeddings else None
            delta.del_metadata = list(self._staged_del_metadata)
            stats.removed = len(self._staged_deletions)

        # Modifications
        if self._staged_modifications:
            delta.mod_indices = [m[0] for m in self._staged_modifications]
            delta.mod_old_values = torch.stack([m[1] for m in self._staged_modifications])
            delta.mod_new_values = torch.stack([m[2] for m in self._staged_modifications])
            stats.modified = len(self._staged_modifications)

        # Compact tombstones
        self.tree.compact()

        # Serialize and store delta
        delta_bytes = delta.serialize()
        delta_hash = self.objects.write(delta_bytes)

        # Create commit
        parent = self.refs.get_head_commit()
        commit = Commit(
            parent=parent,
            delta_hash=delta_hash,
            timestamp=time.time(),
            message=message,
            stats=stats,
            tensor_rows=self.tree.total_rows,
        )
        commit_hash = self.objects.write_commit(commit)

        # Update branch ref
        branch = self.refs.current_branch
        self.refs.set_branch(branch, commit_hash)

        # Cache current state
        cache_file = self._cache_dir / f"{commit_hash}.pt"
        meta_cache = self._cache_dir / f"{commit_hash}.meta.json"
        if self.tree.embeddings is not None:
            torch.save(self.tree.embeddings.cpu(), cache_file)
        else:
            torch.save(torch.zeros(0, self.dim), cache_file)
        meta_raw = [{"id": m.id, "document": m.document, "metadata": m.metadata} for m in self.tree.metadata]
        meta_cache.write_text(json.dumps(meta_raw))

        # Save document and table state
        self._save_data_state(commit_hash)

        # Reflog
        self._reflog_append(f"commit: {message}", commit_hash)

        # Fire watches
        self.watches.check(
            "commit",
            branch=branch,
            commit_hash=commit_hash,
            message=message,
            added_metadata=delta.add_metadata or [],
            removed_metadata=delta.del_metadata or [],
            modified_count=stats.modified,
        )

        # Clear staging
        self._clear_staging()

        # Fire post-commit hook
        self.hooks.fire("post-commit", commit_hash=commit_hash, message=message, db=self)

        return commit_hash

    def log(self, limit: int = 50, source: Optional[str] = None) -> List[CommitInfo]:
        """Commit history for current branch."""
        head = self.refs.get_head_commit()
        if head is None:
            return []

        commits = []
        current = head
        while current is not None and len(commits) < limit:
            commit = self.objects.read_commit(current)
            info = CommitInfo(
                hash=commit.hash,
                parent=commit.parent,
                message=commit.message,
                timestamp=commit.timestamp,
                stats=commit.stats,
            )

            if source is not None:
                # Filter: only show commits that mention source in message
                if source.lower() in commit.message.lower():
                    commits.append(info)
            else:
                commits.append(info)

            current = commit.parent

        return commits

    def diff(self, ref_a: str, ref_b: str) -> DiffResult:
        """Compare two refs (commits, branches, tags).

        Returns a DiffResult with full content: documents, metadata,
        and cosine similarity for modified vectors.
        """
        from gitdb.types import DiffEntry

        hash_a = self._resolve(ref_a)
        hash_b = self._resolve(ref_b)
        if hash_a is None:
            raise ValueError(f"Cannot resolve ref: {ref_a}")
        if hash_b is None:
            raise ValueError(f"Cannot resolve ref: {ref_b}")

        tensor_a, meta_a = self._reconstruct(hash_a)
        tensor_b, meta_b = self._reconstruct(hash_b)

        ids_a = {m.id for m in meta_a}
        ids_b = {m.id for m in meta_b}

        added = ids_b - ids_a
        removed = ids_a - ids_b
        common = ids_a & ids_b

        id_to_idx_a = {m.id: i for i, m in enumerate(meta_a)}
        id_to_idx_b = {m.id: i for i, m in enumerate(meta_b)}
        meta_by_id_a = {m.id: m for m in meta_a}
        meta_by_id_b = {m.id: m for m in meta_b}

        modified = set()
        entries = []

        # Added vectors
        for vid in sorted(added):
            m = meta_by_id_b[vid]
            entries.append(DiffEntry(
                id=vid, change="added",
                document=m.document, metadata=dict(m.metadata),
            ))

        # Removed vectors
        for vid in sorted(removed):
            m = meta_by_id_a[vid]
            entries.append(DiffEntry(
                id=vid, change="removed",
                document=m.document, metadata=dict(m.metadata),
            ))

        # Modified vectors
        for vid in sorted(common):
            ia, ib = id_to_idx_a[vid], id_to_idx_b[vid]
            if not torch.equal(tensor_a[ia], tensor_b[ib]):
                modified.add(vid)
                ma, mb = meta_by_id_a[vid], meta_by_id_b[vid]
                # Cosine similarity between old and new embedding
                sim = torch.nn.functional.cosine_similarity(
                    tensor_a[ia].unsqueeze(0), tensor_b[ib].unsqueeze(0)
                ).item()
                entries.append(DiffEntry(
                    id=vid, change="modified",
                    document=mb.document, document_before=ma.document,
                    metadata=dict(mb.metadata), metadata_before=dict(ma.metadata),
                    similarity=sim,
                ))

        return DiffResult(
            added_count=len(added),
            removed_count=len(removed),
            modified_count=len(modified),
            added_ids=sorted(added),
            removed_ids=sorted(removed),
            modified_ids=sorted(modified),
            entries=entries,
        )

    def checkout(self, ref: str):
        """Switch working tree to a specific ref (branch, tag, or commit)."""
        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")

        if self._has_staged_changes():
            raise ValueError("Cannot checkout with staged changes — commit or reset first")

        tensor, metadata = self._reconstruct(commit_hash)
        self.tree.load_state(tensor, metadata)
        self._load_data_state(commit_hash)
        self._reflog_append(f"checkout: {ref}", commit_hash)
        self._clear_staging()

    def reset(self, ref: str = "HEAD"):
        """Discard staged changes and reset working tree to ref."""
        commit_hash = self._resolve(ref)
        if commit_hash is None:
            self.tree = WorkingTree(dim=self.dim, device=self.device)
            self._load_data_state(None)
        else:
            tensor, metadata = self._reconstruct(commit_hash)
            self.tree.load_state(tensor, metadata)
            self._load_data_state(commit_hash)
            self._reflog_append(f"reset: {ref}", commit_hash)
        self._clear_staging()

    def tag(self, name: str, ref: str = "HEAD"):
        """Create a tag at the given ref."""
        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")
        self.refs.set_tag(name, commit_hash)

    def delete_tag(self, name: str):
        """Delete a tag."""
        self.refs.delete_tag(name)

    def check_integrity(self) -> dict:
        """Validate document store and table integrity.

        Checks:
          - JSON docs parse correctly
          - Table rows match column schema
          - No orphaned tombstones
          - Document _id uniqueness

        Returns dict with {valid, issues, repaired}.
        """
        issues = []
        repaired = 0

        # Check document store
        bad_doc_indices = []
        seen_ids = {}
        for i, doc in enumerate(self.docs._docs):
            if i in self.docs._tombstones:
                continue
            # Check JSON serializable
            try:
                json.dumps(doc, default=str)
            except (TypeError, ValueError) as e:
                issues.append(f"Doc index {i}: invalid JSON — {e}")
                bad_doc_indices.append(i)
                continue
            # Check _id
            _id = doc.get("_id")
            if _id is None:
                issues.append(f"Doc index {i}: missing _id")
            elif _id in seen_ids:
                issues.append(f"Doc index {i}: duplicate _id '{_id}' (first at {seen_ids[_id]})")
            else:
                seen_ids[_id] = i

        # Check orphaned tombstones
        max_idx = len(self.docs._docs)
        orphaned = [t for t in self.docs._tombstones if t >= max_idx]
        if orphaned:
            issues.append(f"Doc store: {len(orphaned)} orphaned tombstones (indices beyond doc list)")

        # Check tables
        for name in self.tables.list_tables():
            table = self.tables._tables.get(name)
            if table is None:
                continue
            cols = table.columns or {}
            for row_i, row in enumerate(table._rows):
                if row_i in getattr(table, '_tombstones', set()):
                    continue
                # Check row matches schema
                for col_name, col_type in cols.items():
                    val = row.get(col_name)
                    if val is not None:
                        try:
                            json.dumps(val, default=str)
                        except (TypeError, ValueError):
                            issues.append(f"Table '{name}' row {row_i}, col '{col_name}': non-serializable value")
                # Check for extra columns (exclude internal fields)
                extra = set(row.keys()) - set(cols.keys()) - {'_id', '_rowid'}
                if extra:
                    issues.append(f"Table '{name}' row {row_i}: extra columns {extra}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "docs_checked": self.docs.size,
            "tables_checked": len(self.tables.list_tables()),
        }

    def repair(self) -> dict:
        """Auto-repair corrupt documents and table rows.

        - Removes docs that can't serialize to JSON
        - Fixes missing _id fields
        - Removes orphaned tombstones
        - Removes extra table columns

        Returns dict with repair actions taken.
        """
        actions = []

        # Fix docs that can't serialize
        bad_indices = []
        for i, doc in enumerate(self.docs._docs):
            if i in self.docs._tombstones:
                continue
            try:
                json.dumps(doc, default=str)
            except (TypeError, ValueError):
                bad_indices.append(i)
        if bad_indices:
            for i in bad_indices:
                self.docs._tombstones.add(i)
            actions.append(f"Tombstoned {len(bad_indices)} corrupt docs")

        # Fix missing _ids
        import uuid as _uuid
        for i, doc in enumerate(self.docs._docs):
            if i in self.docs._tombstones:
                continue
            if "_id" not in doc:
                doc["_id"] = hashlib.sha256(
                    json.dumps(doc, sort_keys=True, default=str).encode()
                    + _uuid.uuid4().bytes
                ).hexdigest()[:16]
                actions.append(f"Generated _id for doc index {i}")

        # Fix orphaned tombstones
        max_idx = len(self.docs._docs)
        orphaned = {t for t in self.docs._tombstones if t >= max_idx}
        if orphaned:
            self.docs._tombstones -= orphaned
            actions.append(f"Removed {len(orphaned)} orphaned tombstones")

        # Fix table rows with extra columns
        for name in self.tables.list_tables():
            table = self.tables._tables.get(name)
            if table is None:
                continue
            cols = table.columns or {}
            fixed = 0
            for row in table._rows:
                extra = set(row.keys()) - set(cols.keys()) - {'_id', '_rowid'}
                if extra:
                    for k in extra:
                        del row[k]
                    fixed += 1
            if fixed:
                actions.append(f"Table '{name}': removed extra columns from {fixed} rows")

        return {
            "repaired": len(actions) > 0,
            "actions": actions,
        }

    def status(self) -> Dict[str, Any]:
        """Show staged changes summary."""
        return {
            "branch": self.refs.current_branch,
            "head": self.refs.get_head_commit(),
            "staged_additions": len(self._staged_additions),
            "staged_deletions": len(self._staged_deletions),
            "staged_modifications": len(self._staged_modifications),
            "total_rows": self.tree.total_rows,
            "active_rows": self.tree.size,
            "documents": self.docs.size,
            "staged_doc_inserts": self._staged_doc_inserts,
            "staged_doc_updates": self._staged_doc_updates,
            "staged_doc_deletes": self._staged_doc_deletes,
            "tables": self.tables.list_tables(),
            "table_rows": self.tables.size,
            "staged_table_changes": self._staged_table_changes,
        }

    # ─── Properties ─────────────────────────────────────────

    @property
    def HEAD(self) -> Optional[str]:
        """Current HEAD commit hash."""
        return self.refs.get_head_commit()

    @property
    def current_branch(self) -> str:
        """Current branch name."""
        return self.refs.current_branch

    # ─── Show / Inspect ───────────────────────────────────────

    def show(self, ref: str = "HEAD") -> Dict[str, Any]:
        """Show details of a commit (like git show)."""
        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")
        commit = self.objects.read_commit(commit_hash)
        delta_data = self.objects.read(commit.delta_hash)
        delta = Delta.deserialize(delta_data, self.dim)
        return {
            "hash": commit.hash,
            "parent": commit.parent,
            "parent2": commit.parent2,
            "message": commit.message,
            "timestamp": commit.timestamp,
            "is_merge": commit.is_merge,
            "stats": {
                "added": commit.stats.added,
                "removed": commit.stats.removed,
                "modified": commit.stats.modified,
            },
            "tensor_rows": commit.tensor_rows,
            "delta_size_bytes": len(delta_data),
            "additions": len(delta.add_indices),
            "deletions": len(delta.del_indices),
            "modifications": len(delta.mod_indices),
            "meta_changes": len(delta.meta_changes),
        }

    # ─── Amend / Squash ───────────────────────────────────────

    def amend(self, message: Optional[str] = None) -> str:
        """Amend the last commit (like git commit --amend).

        If there are staged changes, they're folded into the last commit.
        If only message is given, rewrites the commit message.
        """
        head = self.refs.get_head_commit()
        if head is None:
            raise ValueError("No commits to amend")

        old_commit = self.objects.read_commit(head)

        if not self._has_staged_changes() and message is None:
            raise ValueError("Nothing to amend — no staged changes and no new message")

        if self._has_staged_changes():
            # Reset to parent, then re-commit with combined changes
            parent = old_commit.parent
            if parent:
                parent_tensor, parent_meta = self._reconstruct(parent)
            else:
                parent_tensor = torch.zeros(0, self.dim)
                parent_meta = []

            # Current tree already has the new changes applied
            # We need to build a delta from parent → current tree
            current_tensor = self.tree.embeddings.cpu().clone() if self.tree.embeddings is not None else torch.zeros(0, self.dim)
            current_meta = list(self.tree.metadata)

            # Compact tombstones
            self.tree.compact()

            # Build combined delta
            delta = self._build_diff_delta(parent_tensor, parent_meta, self.tree.embeddings.cpu() if self.tree.embeddings is not None else torch.zeros(0, self.dim), self.tree.metadata)

            delta_bytes = delta.serialize()
            delta_hash = self.objects.write(delta_bytes)

            new_commit = Commit(
                parent=parent,
                parent2=old_commit.parent2,
                delta_hash=delta_hash,
                timestamp=time.time(),
                message=message or old_commit.message,
                stats=CommitStats(
                    added=len(delta.add_indices),
                    removed=len(delta.del_indices),
                    modified=len(delta.mod_indices),
                ),
                tensor_rows=self.tree.total_rows,
            )
            new_hash = self.objects.write_commit(new_commit)
        else:
            # Message-only amend
            new_commit = Commit(
                parent=old_commit.parent,
                parent2=old_commit.parent2,
                delta_hash=old_commit.delta_hash,
                timestamp=old_commit.timestamp,
                message=message,
                stats=old_commit.stats,
                tensor_rows=old_commit.tensor_rows,
            )
            new_hash = self.objects.write_commit(new_commit)

        branch = self.refs.current_branch
        self.refs.set_branch(branch, new_hash)

        # Invalidate old cache
        for suffix in [".pt", ".meta.json"]:
            old_cache = self._cache_dir / f"{head}{suffix}"
            if old_cache.exists():
                old_cache.unlink()

        # Cache new state
        self._cache_state(new_hash)
        self._save_data_state(new_hash)
        self._reflog_append(f"amend: {new_commit.message}", new_hash)
        self._clear_staging()
        return new_hash

    def squash(self, n: int, message: Optional[str] = None) -> str:
        """Squash the last N commits into one (like interactive rebase squash).

        Collapses N commits into a single commit with combined changes.
        """
        if n < 2:
            raise ValueError("Must squash at least 2 commits")
        if self._has_staged_changes():
            raise ValueError("Cannot squash with staged changes")

        head = self.refs.get_head_commit()
        if head is None:
            raise ValueError("No commits to squash")

        # Walk back N commits
        chain = []
        current = head
        for _ in range(n):
            if current is None:
                raise ValueError(f"Only {len(chain)} commits available, cannot squash {n}")
            commit = self.objects.read_commit(current)
            chain.append(commit)
            current = commit.parent

        # The parent of the squashed commit
        new_parent = current  # commit before the N we're squashing

        # Collect messages if no override
        if message is None:
            messages = [c.message for c in reversed(chain)]
            message = "squash: " + " | ".join(messages)

        # Reconstruct state at new_parent and at HEAD
        if new_parent:
            base_tensor, base_meta = self._reconstruct(new_parent)
        else:
            base_tensor = torch.zeros(0, self.dim)
            base_meta = []

        head_tensor, head_meta = self._reconstruct(head)

        # Build single delta from base → HEAD
        delta = self._build_diff_delta(base_tensor, base_meta, head_tensor, head_meta)
        delta_bytes = delta.serialize()
        delta_hash = self.objects.write(delta_bytes)

        new_commit = Commit(
            parent=new_parent,
            delta_hash=delta_hash,
            timestamp=time.time(),
            message=message,
            stats=CommitStats(
                added=len(delta.add_indices),
                removed=len(delta.del_indices),
                modified=len(delta.mod_indices),
            ),
            tensor_rows=len(head_meta),
        )
        new_hash = self.objects.write_commit(new_commit)

        branch = self.refs.current_branch
        self.refs.set_branch(branch, new_hash)

        # Invalidate old caches
        for c in chain:
            for suffix in [".pt", ".meta.json"]:
                old_cache = self._cache_dir / f"{c.hash}{suffix}"
                if old_cache.exists():
                    old_cache.unlink()

        self._cache_state(new_hash)
        self._save_data_state(new_hash)
        self._reflog_append(f"squash {n}: {message}", new_hash)
        return new_hash

    # ─── Fork / Clone ─────────────────────────────────────────

    def fork(self, dest_path: str, branch: Optional[str] = None) -> "GitDB":
        """Fork the entire database to a new location (like git clone).

        Copies the full object store, refs, and optionally checks out
        a specific branch.
        """
        import shutil
        dest = Path(dest_path)
        if dest.exists():
            raise ValueError(f"Destination already exists: {dest}")

        # Copy the entire .gitdb directory
        shutil.copytree(self._gitdb_dir, dest / ".gitdb")

        # Open the forked copy
        forked = GitDB(dest_path, dim=self.dim, device=self.device)
        if branch:
            forked.switch(branch)
        return forked

    def clone(self, dest_path: str) -> "GitDB":
        """Alias for fork (like git clone)."""
        return self.fork(dest_path)

    # ─── Branch Management ────────────────────────────────────

    def delete_branch(self, name: str):
        """Delete a branch (like git branch -d)."""
        self.refs.delete_branch(name)

    def rename_branch(self, old_name: str, new_name: str):
        """Rename a branch (like git branch -m)."""
        commit_hash = self.refs.get_branch(old_name)
        if commit_hash is None:
            raise ValueError(f"Branch not found: {old_name}")
        if self.refs.get_branch(new_name) is not None:
            raise ValueError(f"Branch already exists: {new_name}")
        self.refs.set_branch(new_name, commit_hash)
        if old_name == self.refs.current_branch:
            self.refs.set_head(new_name)
        self.refs.delete_branch(old_name)

    # ─── Notes / Comments ─────────────────────────────────────

    def note(self, ref: str, message: str):
        """Attach a note/comment to a commit (like git notes add)."""
        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")
        notes_dir = self._gitdb_dir / "notes"
        notes_dir.mkdir(exist_ok=True)
        notes_file = notes_dir / commit_hash
        # Append mode — multiple notes per commit
        with open(notes_file, "a") as f:
            f.write(json.dumps({"message": message, "timestamp": time.time()}) + "\n")

    def notes(self, ref: str = "HEAD") -> List[Dict[str, Any]]:
        """Read notes/comments on a commit (like git notes show)."""
        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")
        notes_file = self._gitdb_dir / "notes" / commit_hash
        if not notes_file.exists():
            return []
        entries = []
        for line in notes_file.read_text().strip().split("\n"):
            if line:
                entries.append(json.loads(line))
        return entries

    # ─── Remotes: Push / Pull / Fetch ─────────────────────────

    def remote_add(self, name: str, url: str):
        """Add a remote (like git remote add)."""
        remotes = RemoteManager(self._gitdb_dir)
        remotes.add(name, url)

    def remote_remove(self, name: str):
        """Remove a remote (like git remote remove)."""
        remotes = RemoteManager(self._gitdb_dir)
        remotes.remove(name)

    def remotes(self) -> Dict[str, str]:
        """List all remotes."""
        return RemoteManager(self._gitdb_dir).list()

    def push(self, remote_name: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """Push a branch to a remote (like git push)."""
        remotes = RemoteManager(self._gitdb_dir)
        remote = remotes.get(remote_name)
        branch = branch or self.refs.current_branch

        # Fire pre-push hook
        if not self.hooks.fire("pre-push", remote=remote_name, branch=branch, db=self):
            raise ValueError("pre-push hook rejected")

        result = _push(self.objects, self.refs, remote, branch)
        self._reflog_append(f"push: {remote_name}/{branch}", self.refs.get_head_commit() or "empty")
        self.watches.check("push", branch=branch)

        # Fire post-push hook
        self.hooks.fire("post-push", remote=remote_name, branch=branch, result=result, db=self)

        return result

    def pull(self, remote_name: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """Pull from a remote (like git pull = fetch + update ref)."""
        remotes = RemoteManager(self._gitdb_dir)
        remote = remotes.get(remote_name)
        branch = branch or self.refs.current_branch
        result = _pull(self.objects, self.refs, remote, branch)

        # Reload working tree if our branch was updated
        if result.get("status") == "pulled" and branch == self.refs.current_branch:
            head = self.refs.get_head_commit()
            if head:
                tensor, metadata = self._reconstruct(head)
                self.tree.load_state(tensor, metadata)
                self._clear_staging()

        self._reflog_append(f"pull: {remote_name}/{branch}", self.refs.get_head_commit() or "empty")
        self.watches.check("pull", branch=branch)
        return result

    def fetch(self, remote_name: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """Fetch objects from remote without updating local refs (like git fetch)."""
        remotes = RemoteManager(self._gitdb_dir)
        remote = remotes.get(remote_name)
        branch = branch or self.refs.current_branch
        return _fetch(self.objects, remote, branch)

    # ─── Pull Requests ────────────────────────────────────────

    def pr_create(
        self,
        title: str,
        source_branch: Optional[str] = None,
        target_branch: str = "main",
        description: str = "",
        author: str = "anonymous",
    ) -> PullRequest:
        """Create a pull request (like gh pr create)."""
        prs = PRStore(self._gitdb_dir)
        source = source_branch or self.refs.current_branch
        if self.refs.get_branch(source) is None:
            raise ValueError(f"Source branch not found: {source}")
        if self.refs.get_branch(target_branch) is None:
            raise ValueError(f"Target branch not found: {target_branch}")
        return prs.create(title, source, target_branch, description, author)

    def pr_list(self, status: Optional[str] = None) -> List[PullRequest]:
        """List pull requests."""
        return PRStore(self._gitdb_dir).list(status)

    def pr_show(self, pr_id: int) -> PullRequest:
        """Show a pull request."""
        return PRStore(self._gitdb_dir).get(pr_id)

    def pr_merge(self, pr_id: int, strategy: str = "union") -> MergeResult:
        """Merge a pull request."""
        prs = PRStore(self._gitdb_dir)
        pr = prs.get(pr_id)
        if pr.status != "open":
            raise ValueError(f"PR #{pr_id} is {pr.status}, cannot merge")

        # Switch to target, merge source
        current = self.refs.current_branch
        if current != pr.target_branch:
            self.switch(pr.target_branch)

        result = self.merge(pr.source_branch, strategy=strategy)
        prs.update_status(pr_id, "merged", merge_commit=result.commit_hash)
        return result

    def pr_close(self, pr_id: int):
        """Close a PR without merging."""
        PRStore(self._gitdb_dir).close(pr_id)

    def pr_comment(self, pr_id: int, author: str, message: str):
        """Add a comment to a PR."""
        PRStore(self._gitdb_dir).comment(pr_id, author, message)

    # ─── Comments on Commits ──────────────────────────────────

    def comment(self, ref: str, author: str, message: str):
        """Add a comment/review to a commit (like GitHub commit comments)."""
        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")
        self.note(commit_hash, f"[{author}] {message}")

    def comments(self, ref: str = "HEAD") -> List[Dict[str, Any]]:
        """Read comments on a commit."""
        return self.notes(ref)

    # ─── Cherry-pick & Revert ─────────────────────────────────

    def cherry_pick(self, ref: str) -> str:
        """Apply a specific commit's changes to the current branch.

        Like git cherry-pick: takes the delta from one commit and
        replays it on top of HEAD.
        """
        if self._has_staged_changes():
            raise ValueError("Cannot cherry-pick with staged changes")

        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")

        commit = self.objects.read_commit(commit_hash)
        delta_data = self.objects.read(commit.delta_hash)
        delta = Delta.deserialize(delta_data, self.dim)

        # Apply the delta to current working tree
        if self.tree.embeddings is None:
            tensor = torch.zeros(0, self.dim)
        else:
            tensor = self.tree.embeddings.cpu().clone()
        metadata = list(self.tree.metadata)

        tensor, metadata = delta.apply_forward(tensor, metadata)
        self.tree.load_state(tensor, metadata)

        # Auto-commit the cherry-pick
        # Build a new delta relative to current HEAD
        self._staged_additions = list(range(
            len(metadata) - (len(delta.add_indices) if delta.add_indices else 0),
            len(metadata)
        ))
        if delta.del_indices:
            self._staged_deletions = list(delta.del_indices)
            self._staged_del_embeddings = [delta.del_values[i].cpu() for i in range(len(delta.del_indices))] if delta.del_values is not None else []
            self._staged_del_metadata = list(delta.del_metadata)
        if delta.mod_indices:
            self._staged_modifications = [
                (idx, delta.mod_old_values[i].cpu(), delta.mod_new_values[i].cpu())
                for i, idx in enumerate(delta.mod_indices)
            ]

        # Cherry-pick document changes
        doc_file = self._docs_dir / f"{commit_hash}.jsonl"
        if doc_file.exists():
            picked_docs = []
            for line in doc_file.read_bytes().decode().strip().split("\n"):
                if line.strip():
                    picked_docs.append(json.loads(line))
            parent_docs = []
            if commit.parent:
                parent_doc_file = self._docs_dir / f"{commit.parent}.jsonl"
                if parent_doc_file.exists():
                    for line in parent_doc_file.read_bytes().decode().strip().split("\n"):
                        if line.strip():
                            parent_docs.append(json.loads(line))
            parent_ids = {d.get("_id") for d in parent_docs}
            new_docs = [d for d in picked_docs if d.get("_id") not in parent_ids]
            if new_docs:
                for d in new_docs:
                    self.docs.insert(d)
                self._staged_doc_inserts += len(new_docs)

        # Cherry-pick table changes
        table_file = self._tables_dir / f"{commit_hash}.json"
        if table_file.exists():
            picked_tables = TableStore()
            picked_tables.restore(table_file.read_bytes())
            parent_tables = TableStore()
            if commit.parent:
                parent_table_file = self._tables_dir / f"{commit.parent}.json"
                if parent_table_file.exists():
                    parent_tables.restore(parent_table_file.read_bytes())
            for tname in picked_tables.list_tables():
                pt = picked_tables.get(tname)
                if tname not in parent_tables.list_tables():
                    # Whole table is new — bring it in
                    new_t = self.tables.get_or_create(tname, pt.columns)
                    for row in pt.select():
                        new_t.insert(row)
                    self._staged_table_changes += pt.size
                else:
                    # Diff rows
                    parent_t = parent_tables.get(tname)
                    parent_ids = {r.get("_rowid") for r in parent_t.select()}
                    new_t = self.tables.get_or_create(tname, pt.columns)
                    for row in pt.select():
                        if row.get("_rowid") not in parent_ids:
                            new_t.insert(row)
                            self._staged_table_changes += 1

        msg = f"cherry-pick: {commit.message} (from {commit.short_hash})"
        return self.commit(msg)

    def revert(self, ref: str) -> str:
        """Undo a specific commit by applying its inverse delta.

        Like git revert: creates a new commit that undoes the changes.
        """
        if self._has_staged_changes():
            raise ValueError("Cannot revert with staged changes")

        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")

        commit = self.objects.read_commit(commit_hash)
        delta_data = self.objects.read(commit.delta_hash)
        delta = Delta.deserialize(delta_data, self.dim)

        # Apply the delta in REVERSE to current working tree
        if self.tree.embeddings is None:
            raise ValueError("Cannot revert on empty database")

        tensor = self.tree.embeddings.cpu().clone()
        metadata = list(self.tree.metadata)
        tensor, metadata = delta.apply_reverse(tensor, metadata)
        self.tree.load_state(tensor, metadata)

        # Stage the reverse changes for commit
        # Reverse of additions = deletions
        if delta.add_indices and delta.add_values is not None:
            n_reverted = len(delta.add_indices)
            # The additions were appended, so reversing removes from end
            # After apply_reverse these are gone, so we just record stats
            pass
        # Reverse of deletions = additions
        if delta.del_indices and delta.del_values is not None:
            n_restored = len(delta.del_indices)
            start = len(metadata) - n_restored
            self._staged_additions = list(range(start, len(metadata)))

        # Force a commit by marking something staged if delta wasn't empty
        if not self._has_staged_changes() and not delta.is_empty:
            # The state changed but our staging doesn't track it perfectly.
            # Record a synthetic modification on row 0 to force a commit.
            if len(metadata) > 0:
                self._staged_modifications.append(
                    (0, self.tree.embeddings[0].cpu(), self.tree.embeddings[0].cpu())
                )

        msg = f"revert: {commit.message} (undoing {commit.short_hash})"
        return self.commit(msg)

    # ─── Magic: Branching & Merging ───────────────────────────

    def branch(self, name: str, ref: str = "HEAD"):
        """Create a new branch at the given ref."""
        commit_hash = self._resolve(ref)
        if commit_hash is None:
            raise ValueError(f"Cannot resolve ref: {ref}")
        if self.refs.get_branch(name) is not None:
            raise ValueError(f"Branch already exists: {name}")
        self.refs.set_branch(name, commit_hash)

    def switch(self, branch_name: str):
        """Switch HEAD to a different branch (like git switch)."""
        if self._has_staged_changes():
            raise ValueError("Cannot switch with staged changes — commit or reset first")
        commit_hash = self.refs.get_branch(branch_name)
        if commit_hash is None:
            raise ValueError(f"Branch not found: {branch_name}")
        tensor, metadata = self._reconstruct(commit_hash)
        self.tree.load_state(tensor, metadata)
        self._load_data_state(commit_hash)
        self.refs.set_head(branch_name)
        self._clear_staging()

    def branches(self) -> Dict[str, str]:
        """List all branches with their commit hashes."""
        return self.refs.list_branches()

    def merge(self, branch: str, strategy: str = "union") -> MergeResult:
        """Merge another branch into the current branch.

        Strategies:
          - "union": take all unique vectors from both (default)
          - "ours": keep current branch as-is
          - "theirs": take the other branch entirely
        """
        if self._has_staged_changes():
            raise ValueError("Cannot merge with staged changes")

        # Fire pre-merge hook
        if not self.hooks.fire("pre-merge", branch=branch, strategy=strategy, db=self):
            raise ValueError("pre-merge hook rejected")

        current_branch = self.refs.current_branch
        ours_hash = self.refs.get_head_commit()
        theirs_hash = self.refs.get_branch(branch)
        if theirs_hash is None:
            raise ValueError(f"Branch not found: {branch}")
        if ours_hash == theirs_hash:
            result = MergeResult(commit_hash=ours_hash, conflicts=[], strategy=strategy)
            self.hooks.fire("post-merge", result=result, branch=branch, db=self)
            return result

        # Fast-forward: if ours is an ancestor of theirs and strategy is union, just advance
        if strategy == "union" and self._is_ancestor(ours_hash, theirs_hash):
            tensor, metadata = self._reconstruct(theirs_hash)
            self.tree.load_state(tensor, metadata)
            self._load_data_state(theirs_hash)
            self.refs.set_branch(current_branch, theirs_hash)
            self._reflog_append(f"merge (fast-forward): {branch}", theirs_hash)
            result = MergeResult(commit_hash=theirs_hash, conflicts=[], strategy="fast-forward")
            self.hooks.fire("post-merge", result=result, branch=branch, db=self)
            return result

        # Find common ancestor (merge base)
        base_hash = self._find_merge_base(ours_hash, theirs_hash)

        ours_tensor, ours_meta = self._reconstruct(ours_hash)
        theirs_tensor, theirs_meta = self._reconstruct(theirs_hash)

        if strategy == "ours":
            merged_tensor, merged_meta = ours_tensor, ours_meta
            conflicts = []
        elif strategy == "theirs":
            merged_tensor, merged_meta = theirs_tensor, theirs_meta
            conflicts = []
        else:  # union
            merged_tensor, merged_meta, conflicts = self._merge_union(
                base_hash, ours_tensor, ours_meta, theirs_tensor, theirs_meta
            )

        # Load merged state
        self.tree.load_state(merged_tensor, merged_meta)

        # Build delta from ours → merged
        n_added = len(merged_meta) - len(ours_meta)
        if n_added > 0:
            self._staged_additions = list(range(len(ours_meta), len(merged_meta)))
        elif n_added < 0 or conflicts:
            # Force commit
            if len(merged_meta) > 0:
                self._staged_modifications.append(
                    (0, self.tree.embeddings[0].cpu(), self.tree.embeddings[0].cpu())
                )

        # Merge documents and tables from other branch
        theirs_doc_file = self._docs_dir / f"{theirs_hash}.jsonl"
        if theirs_doc_file.exists():
            theirs_docs = []
            for line in theirs_doc_file.read_bytes().decode().strip().split("\n"):
                if line.strip():
                    theirs_docs.append(json.loads(line))
            our_ids = {d.get("_id") for d in self.docs.get_docs()}
            new_docs = [d for d in theirs_docs if d.get("_id") not in our_ids]
            if new_docs:
                for d in new_docs:
                    self.docs.insert(d)
                self._staged_doc_inserts += len(new_docs)

        # Merge tables
        theirs_table_file = self._tables_dir / f"{theirs_hash}.json"
        if theirs_table_file.exists():
            their_tables = TableStore()
            their_tables.restore(theirs_table_file.read_bytes())
            for tname in their_tables.list_tables():
                their_table = their_tables.get(tname)
                if tname not in self.tables.list_tables():
                    # New table — bring it in
                    our_table = self.tables.create(tname, their_table.columns)
                    for row in their_table.select():
                        our_table.insert(row)
                    self._staged_table_changes += their_table.size
                else:
                    # Existing table — merge rows by _rowid
                    our_table = self.tables.get(tname)
                    our_ids = {r.get("_rowid") for r in our_table.select()}
                    for row in their_table.select():
                        if row.get("_rowid") not in our_ids:
                            our_table.insert(row)
                            self._staged_table_changes += 1

        if not self._has_staged_changes():
            self.refs.set_branch(current_branch, theirs_hash)
            self._clear_staging()
            self._load_data_state(theirs_hash)
            result = MergeResult(commit_hash=theirs_hash, conflicts=[], strategy="fast-forward")
            self.hooks.fire("post-merge", result=result, branch=branch, db=self)
            return result

        msg = f"Merge branch '{branch}' into {current_branch}"
        commit_hash = self.commit(msg)

        # Set parent2 on the merge commit
        commit = self.objects.read_commit(commit_hash)
        commit.parent2 = theirs_hash
        # Re-write with parent2 (hash changes)
        new_hash = self.objects.write_commit(commit)
        self.refs.set_branch(current_branch, new_hash)

        # Compute diff showing what the merge brought in
        merge_diff = None
        try:
            merge_diff = self.diff(ours_hash, new_hash)
        except Exception:
            pass  # diff is best-effort

        n_removed = max(0, len(ours_meta) - len(merged_meta)) if len(ours_meta) > len(merged_meta) else 0

        result = MergeResult(
            commit_hash=new_hash,
            conflicts=conflicts,
            strategy=strategy,
            added=max(0, n_added),
            removed=n_removed,
            diff=merge_diff,
        )

        # Fire post-merge hook
        self.hooks.fire("post-merge", result=result, branch=branch, db=self)

        return result

    def _find_merge_base(self, hash_a: str, hash_b: str) -> Optional[str]:
        """Find common ancestor of two commits."""
        # Collect all ancestors of A
        ancestors_a = set()
        current = hash_a
        while current is not None:
            ancestors_a.add(current)
            commit = self.objects.read_commit(current)
            current = commit.parent

        # Walk B's history until we find a common ancestor
        current = hash_b
        while current is not None:
            if current in ancestors_a:
                return current
            commit = self.objects.read_commit(current)
            current = commit.parent

        return None  # No common ancestor (disjoint histories)

    def _is_ancestor(self, ancestor_hash: str, descendant_hash: str) -> bool:
        """Check if ancestor_hash is an ancestor of descendant_hash."""
        current = descendant_hash
        while current is not None:
            if current == ancestor_hash:
                return True
            commit = self.objects.read_commit(current)
            current = commit.parent
        return False

    def _merge_union(self, base_hash, ours_tensor, ours_meta, theirs_tensor, theirs_meta):
        """Union merge: combine all unique vectors from both branches."""
        conflicts = []

        if base_hash:
            base_tensor, base_meta = self._reconstruct(base_hash)
            base_ids = {m.id for m in base_meta}
        else:
            base_tensor = torch.zeros(0, self.dim)
            base_meta = []
            base_ids = set()

        ours_ids = {m.id: i for i, m in enumerate(ours_meta)}
        theirs_ids = {m.id: i for i, m in enumerate(theirs_meta)}

        # Detect conflicts: same ID modified differently in both branches
        if base_hash:
            base_id_to_idx = {m.id: i for i, m in enumerate(base_meta)}
            for vid in set(ours_ids) & set(theirs_ids) & base_ids:
                base_idx = base_id_to_idx.get(vid)
                ours_idx = ours_ids[vid]
                theirs_idx = theirs_ids[vid]
                if base_idx is not None:
                    ours_changed = not torch.equal(base_tensor[base_idx], ours_tensor[ours_idx])
                    theirs_changed = not torch.equal(base_tensor[base_idx], theirs_tensor[theirs_idx])
                    if ours_changed and theirs_changed:
                        conflicts.append(vid)

        # Start with ours, add vectors unique to theirs
        merged_tensor = ours_tensor.clone()
        merged_meta = list(ours_meta)

        for i, meta in enumerate(theirs_meta):
            if meta.id not in ours_ids:
                # New in theirs — add it
                merged_tensor = torch.cat([merged_tensor, theirs_tensor[i:i+1]], dim=0)
                merged_meta.append(meta)

        return merged_tensor, merged_meta, conflicts

    # ─── Magic: Stash ─────────────────────────────────────────

    def stash(self, message: str = "WIP"):
        """Save staged changes and restore clean HEAD state.

        Like git stash: saves your work-in-progress for later.
        """
        if not self._has_staged_changes() and self.tree.size == (
            self.objects.read_commit(self.refs.get_head_commit()).tensor_rows
            if self.refs.get_head_commit() else 0
        ):
            raise ValueError("Nothing to stash")

        stash_dir = self._gitdb_dir / "stash"
        stash_dir.mkdir(exist_ok=True)

        # Count existing stashes
        index_file = stash_dir / "index.json"
        if index_file.exists():
            stash_index = json.loads(index_file.read_text())
        else:
            stash_index = []

        idx = len(stash_index)
        tensor_file = f"stash_{idx}.pt"
        meta_file = f"stash_{idx}.meta.json"

        # Save current state
        if self.tree.embeddings is not None:
            torch.save(self.tree.embeddings.cpu(), stash_dir / tensor_file)
        else:
            torch.save(torch.zeros(0, self.dim), stash_dir / tensor_file)
        meta_raw = [{"id": m.id, "document": m.document, "metadata": m.metadata} for m in self.tree.metadata]
        (stash_dir / meta_file).write_text(json.dumps(meta_raw))

        # Save document and table state
        docs_file = f"stash_{idx}.docs.jsonl"
        tables_file = f"stash_{idx}.tables.json"
        (stash_dir / docs_file).write_bytes(self.docs.snapshot())
        (stash_dir / tables_file).write_bytes(self.tables.snapshot())

        stash_index.append({
            "index": idx,
            "message": message,
            "timestamp": time.time(),
            "tensor_file": tensor_file,
            "meta_file": meta_file,
            "docs_file": docs_file,
            "tables_file": tables_file,
        })
        index_file.write_text(json.dumps(stash_index, indent=2))

        # Restore HEAD state
        self.reset("HEAD")

    def stash_pop(self, index: int = -1) -> StashEntry:
        """Restore the most recent stash (or by index) and remove it."""
        stash_dir = self._gitdb_dir / "stash"
        index_file = stash_dir / "index.json"
        if not index_file.exists():
            raise ValueError("No stashes found")

        stash_index = json.loads(index_file.read_text())
        if not stash_index:
            raise ValueError("No stashes found")

        if index == -1:
            index = len(stash_index) - 1
        if index < 0 or index >= len(stash_index):
            raise IndexError(f"Stash index out of range: {index}")

        entry_data = stash_index[index]
        tensor = torch.load(stash_dir / entry_data["tensor_file"], weights_only=True)
        meta_raw = json.loads((stash_dir / entry_data["meta_file"]).read_text())
        metadata = [
            VectorMeta(id=m["id"], document=m.get("document"), metadata=m.get("metadata", {}))
            for m in meta_raw
        ]

        # Restore state
        self.tree.load_state(tensor, metadata)
        self._clear_staging()

        # Restore documents and tables if stashed
        docs_file = entry_data.get("docs_file")
        if docs_file and (stash_dir / docs_file).exists():
            self.docs.restore((stash_dir / docs_file).read_bytes())
        tables_file = entry_data.get("tables_file")
        if tables_file and (stash_dir / tables_file).exists():
            self.tables.restore((stash_dir / tables_file).read_bytes())

        # Figure out what's staged vs HEAD
        head_hash = self.refs.get_head_commit()
        if head_hash:
            head_tensor, head_meta = self._reconstruct(head_hash)
            head_ids = {m.id for m in head_meta}
            for i, m in enumerate(metadata):
                if m.id not in head_ids:
                    self._staged_additions.append(i)

        # Remove stash entry
        entry = StashEntry(
            index=entry_data["index"],
            message=entry_data["message"],
            timestamp=entry_data["timestamp"],
            tensor_file=entry_data["tensor_file"],
            meta_file=entry_data["meta_file"],
        )

        # Clean up files
        (stash_dir / entry_data["tensor_file"]).unlink(missing_ok=True)
        (stash_dir / entry_data["meta_file"]).unlink(missing_ok=True)
        if docs_file:
            (stash_dir / docs_file).unlink(missing_ok=True)
        if tables_file:
            (stash_dir / tables_file).unlink(missing_ok=True)
        stash_index.pop(index)
        index_file.write_text(json.dumps(stash_index, indent=2))

        return entry

    def stash_list(self) -> List[StashEntry]:
        """List all stash entries."""
        stash_dir = self._gitdb_dir / "stash"
        index_file = stash_dir / "index.json"
        if not index_file.exists():
            return []
        stash_index = json.loads(index_file.read_text())
        return [
            StashEntry(
                index=e["index"], message=e["message"], timestamp=e["timestamp"],
                tensor_file=e["tensor_file"], meta_file=e["meta_file"],
            )
            for e in stash_index
        ]

    # ─── Black Magic: Blame ───────────────────────────────────

    def blame(self, ids: Optional[List[str]] = None) -> List[BlameEntry]:
        """Trace each vector back to the commit that introduced it.

        Like git blame: "who added this garbage embedding and when?"
        """
        # Walk full history collecting additions
        head = self.refs.get_head_commit()
        if head is None:
            return []

        # Build chain of commits (oldest first)
        chain = []
        current = head
        while current is not None:
            commit = self.objects.read_commit(current)
            chain.append(commit)
            current = commit.parent
        chain.reverse()

        # Track which IDs were introduced by which commit
        introduced_by: Dict[str, tuple] = {}  # id → (commit_hash, message, timestamp)
        current_meta: List[VectorMeta] = []

        for commit in chain:
            delta_data = self.objects.read(commit.delta_hash)
            delta = Delta.deserialize(delta_data, self.dim)

            # New additions in this commit
            for meta in delta.add_metadata:
                introduced_by[meta.id] = (commit.hash, commit.message, commit.timestamp)

            # Track current state
            tensor = torch.zeros(0, self.dim)  # dummy
            current_meta_copy = list(current_meta)
            # Just track metadata evolution
            if delta.add_metadata:
                current_meta.extend(delta.add_metadata)
            if delta.del_indices:
                for idx in sorted(delta.del_indices, reverse=True):
                    if idx < len(current_meta):
                        current_meta.pop(idx)

        # Build blame entries for requested IDs (or all)
        target_ids = set(ids) if ids else {m.id for m in self.tree.metadata}
        entries = []
        for i, meta in enumerate(self.tree.metadata):
            if meta.id in target_ids and meta.id in introduced_by:
                ch, msg, ts = introduced_by[meta.id]
                entries.append(BlameEntry(
                    vector_id=meta.id,
                    commit_hash=ch,
                    commit_message=msg,
                    timestamp=ts,
                    row_index=i,
                ))

        return entries

    # ─── Black Magic: Bisect ──────────────────────────────────

    def bisect(self, test_fn, good_ref: str = None, bad_ref: str = "HEAD") -> BisectResult:
        """Binary search through history to find the commit that broke something.

        test_fn: callable(GitDB) → bool
            Returns True if the state is "good", False if "bad".
            Called with a temporary GitDB loaded at each test commit.

        Example:
            def quality_check(db):
                results = db.query(my_query, k=10)
                return results.scores[0] > 0.8  # Good if top score > 0.8

            result = db.bisect(quality_check)
        """
        bad_hash = self._resolve(bad_ref)
        if bad_hash is None:
            raise ValueError(f"Cannot resolve bad ref: {bad_ref}")

        # Collect commit chain
        chain = []
        current = bad_hash
        while current is not None:
            chain.append(current)
            commit = self.objects.read_commit(current)
            current = commit.parent
        chain.reverse()  # oldest first

        if good_ref:
            good_hash = self._resolve(good_ref)
            good_idx = chain.index(good_hash) if good_hash in chain else 0
        else:
            good_idx = 0

        bad_idx = len(chain) - 1
        total = len(chain)
        steps = 0

        while good_idx < bad_idx:
            mid = (good_idx + bad_idx) // 2
            steps += 1

            # Load state at midpoint
            mid_hash = chain[mid]
            tensor, metadata = self._reconstruct(mid_hash)
            tmp = WorkingTree(dim=self.dim, device=self.device)
            tmp.load_state(tensor, metadata)

            # Create a temporary GitDB-like object for the test
            class _Snapshot:
                pass
            snap = _Snapshot()
            snap.tree = tmp
            snap.query = lambda v, k=10, where=None, _t=tmp: _t.query(v, k, where)
            snap.dim = self.dim

            if test_fn(snap):
                good_idx = mid + 1
            else:
                bad_idx = mid

        bad_commit = self.objects.read_commit(chain[bad_idx])
        return BisectResult(
            bad_commit=chain[bad_idx],
            bad_message=bad_commit.message,
            steps=steps,
            total_commits=total,
        )

    # ─── Black Magic: Rebase ──────────────────────────────────

    def rebase(self, onto: str) -> List[str]:
        """Replay current branch's commits onto a different base.

        Like git rebase: re-applies your deltas on top of `onto`.
        Returns list of new commit hashes.
        """
        if self._has_staged_changes():
            raise ValueError("Cannot rebase with staged changes")

        current_branch = self.refs.current_branch
        head_hash = self.refs.get_head_commit()
        onto_hash = self._resolve(onto)
        if onto_hash is None:
            raise ValueError(f"Cannot resolve ref: {onto}")

        # Find the fork point (merge base)
        base_hash = self._find_merge_base(head_hash, onto_hash)
        if base_hash is None:
            raise ValueError("No common ancestor found")
        if base_hash == head_hash:
            return []  # Already up to date

        # Collect commits to replay (base..HEAD, exclusive of base)
        replay = []
        current = head_hash
        while current != base_hash and current is not None:
            replay.append(current)
            commit = self.objects.read_commit(current)
            current = commit.parent
        replay.reverse()  # oldest first

        # Switch to onto state
        tensor, metadata = self._reconstruct(onto_hash)
        self.tree.load_state(tensor, metadata)
        self._load_data_state(onto_hash)
        self.refs.set_branch(current_branch, onto_hash)

        # Replay each commit's delta
        new_hashes = []
        for old_hash in replay:
            commit = self.objects.read_commit(old_hash)
            delta_data = self.objects.read(commit.delta_hash)
            delta = Delta.deserialize(delta_data, self.dim)

            t = self.tree.embeddings.cpu().clone() if self.tree.embeddings is not None else torch.zeros(0, self.dim)
            meta = list(self.tree.metadata)
            t, meta = delta.apply_forward(t, meta)
            self.tree.load_state(t, meta)

            # Stage the changes
            if delta.add_indices:
                start = len(meta) - len(delta.add_indices)
                self._staged_additions = list(range(start, len(meta)))
            if delta.del_indices:
                self._staged_deletions = list(delta.del_indices)
                self._staged_del_metadata = list(delta.del_metadata)
                if delta.del_values is not None:
                    self._staged_del_embeddings = [delta.del_values[i].cpu() for i in range(len(delta.del_indices))]
            if delta.mod_indices:
                self._staged_modifications = [
                    (idx, delta.mod_old_values[i].cpu(), delta.mod_new_values[i].cpu())
                    for i, idx in enumerate(delta.mod_indices)
                ]

            # Carry forward doc/table changes from the original commit
            doc_file = self._docs_dir / f"{old_hash}.jsonl"
            if doc_file.exists():
                orig_docs = []
                for line in doc_file.read_bytes().decode().strip().split("\n"):
                    if line.strip():
                        orig_docs.append(json.loads(line))
                cur_ids = {d.get("_id") for d in self.docs.get_docs()}
                for d in orig_docs:
                    if d.get("_id") not in cur_ids:
                        self.docs.insert(d)
                        self._staged_doc_inserts += 1

            table_file = self._tables_dir / f"{old_hash}.json"
            if table_file.exists():
                orig_tables = TableStore()
                orig_tables.restore(table_file.read_bytes())
                for tname in orig_tables.list_tables():
                    ot = orig_tables.get(tname)
                    local = self.tables.get_or_create(tname, ot.columns)
                    local_ids = {r.get("_rowid") for r in local.select()}
                    for row in ot.select():
                        if row.get("_rowid") not in local_ids:
                            local.insert(row)
                            self._staged_table_changes += 1

            if self._has_staged_changes():
                new_hash = self.commit(commit.message)
                new_hashes.append(new_hash)
            else:
                self._clear_staging()

        return new_hashes

    # ─── Hidden Black Magic: GC ───────────────────────────────

    def gc(self, keep_last: int = 10):
        """Garbage collect: create checkpoint snapshots to speed up reconstruction.

        Compacts the delta chain by caching snapshots at regular intervals.
        Old cache entries beyond keep_last commits are evicted.
        """
        head = self.refs.get_head_commit()
        if head is None:
            return

        # Walk history collecting all commits
        chain = []
        current = head
        while current is not None:
            chain.append(current)
            commit = self.objects.read_commit(current)
            current = commit.parent
        chain.reverse()

        # Cache every `keep_last` commits
        for i, commit_hash in enumerate(chain):
            if i % keep_last == 0:
                cache_file = self._cache_dir / f"{commit_hash}.pt"
                if not cache_file.exists():
                    tensor, metadata = self._reconstruct(commit_hash)
                    # _reconstruct already caches, so this is a no-op for most

        # Evict old cache entries that aren't in the chain
        chain_set = set(chain)
        if self._cache_dir.exists():
            for f in self._cache_dir.iterdir():
                commit_hash = f.stem.replace(".meta", "")
                if commit_hash not in chain_set:
                    f.unlink()

    # ─── Hidden Black Magic: Reflog ───────────────────────────

    def _reflog_append(self, action: str, commit_hash: str):
        """Append an entry to the reflog."""
        reflog_file = self._gitdb_dir / "reflog"
        entry = json.dumps({
            "action": action,
            "commit": commit_hash,
            "branch": self.refs.current_branch,
            "timestamp": time.time(),
        })
        with open(reflog_file, "a") as f:
            f.write(entry + "\n")

    def reflog(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Show the reflog — every HEAD movement.

        Like git reflog: nothing is truly lost.
        """
        reflog_file = self._gitdb_dir / "reflog"
        if not reflog_file.exists():
            return []
        entries = []
        for line in reflog_file.read_text().strip().split("\n"):
            if line:
                entries.append(json.loads(line))
        return list(reversed(entries[:limit]))

    # ─── Hidden Black Magic: Filter-Branch ────────────────────

    def filter_branch(self, transform_fn, message_prefix: str = "filter-branch") -> str:
        """Rewrite every vector through a transform function.

        transform_fn: callable(torch.Tensor) → torch.Tensor
            Applied to the entire embedding tensor [N, dim].
            Must return same shape.

        Example — L2 normalize all embeddings across all history:
            db.filter_branch(lambda t: F.normalize(t, dim=1))
        """
        if self._has_staged_changes():
            raise ValueError("Cannot filter-branch with staged changes")

        # Apply transform to current working tree
        if self.tree.embeddings is None or self.tree.size == 0:
            raise ValueError("Nothing to transform")

        old_emb = self.tree.embeddings.cpu().clone()
        new_emb = transform_fn(old_emb)

        if new_emb.shape != old_emb.shape:
            raise ValueError(f"Transform must preserve shape: {old_emb.shape} → {new_emb.shape}")

        self.tree.embeddings = new_emb.to(self.device)

        # Stage all rows as modified
        for i in range(len(self.tree.metadata)):
            self._staged_modifications.append((i, old_emb[i], new_emb[i]))

        return self.commit(f"{message_prefix}: applied transform to {len(self.tree.metadata)} vectors")

    # ─── Forbidden Black Magic: Purge (History Rewrite) ─────

    def purge(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Purge vectors from ALL history — as if they were never added.

        Rewrites every commit and delta in the chain, stripping matching
        vectors. Old objects become orphans cleaned up by gc().

        This is the nuclear option. Like git filter-branch + BFG:
        every hash changes, every delta is rewritten, refs are updated.

        Args:
            ids: Vector IDs to purge
            where: Metadata filter (e.g. {"author": "claude"})
            reason: Audit log reason for the purge

        Returns:
            Stats dict: commits_rewritten, vectors_purged, old_hashes, new_hashes
        """
        if ids is None and where is None:
            raise ValueError("Must specify ids or where")
        if self._has_staged_changes():
            raise ValueError("Cannot purge with staged changes")

        from gitdb.working_tree import _matches_where

        # Build the set of vector IDs to purge
        purge_ids = set(ids) if ids else set()

        if where:
            # Walk all history to find every vector that ever matched
            head = self.refs.get_head_commit()
            if head is None:
                return {"commits_rewritten": 0, "vectors_purged": 0}

            chain = []
            current = head
            while current is not None:
                commit = self.objects.read_commit(current)
                chain.append(commit)
                current = commit.parent
            chain.reverse()

            for commit in chain:
                delta_data = self.objects.read(commit.delta_hash)
                delta = Delta.deserialize(delta_data, self.dim)
                for meta in delta.add_metadata:
                    if _matches_where(meta, where):
                        purge_ids.add(meta.id)

        if not purge_ids:
            return {"commits_rewritten": 0, "vectors_purged": 0}

        # Now rewrite the entire history
        head = self.refs.get_head_commit()
        chain = []
        current = head
        while current is not None:
            commit = self.objects.read_commit(current)
            chain.append(commit)
            current = commit.parent
        chain.reverse()

        old_hashes = [c.hash for c in chain]
        new_parent = None
        new_hashes = []
        commits_rewritten = 0
        total_purged = 0

        for commit in chain:
            delta_data = self.objects.read(commit.delta_hash)
            delta = Delta.deserialize(delta_data, self.dim)

            # Filter additions: remove purged vectors
            new_add_indices = []
            new_add_values = []
            new_add_metadata = []
            purged_in_commit = 0

            if delta.add_indices:
                for i, meta in enumerate(delta.add_metadata):
                    if meta.id in purge_ids:
                        purged_in_commit += 1
                        total_purged += 1
                    else:
                        new_add_indices.append(delta.add_indices[i])
                        if delta.add_values is not None:
                            new_add_values.append(delta.add_values[i])
                        new_add_metadata.append(meta)

            # Filter deletions: remove refs to purged vectors
            new_del_indices = []
            new_del_values = []
            new_del_metadata = []
            if delta.del_indices:
                for i, meta in enumerate(delta.del_metadata):
                    if meta.id not in purge_ids:
                        new_del_indices.append(delta.del_indices[i])
                        if delta.del_values is not None:
                            new_del_values.append(delta.del_values[i])
                        new_del_metadata.append(meta)

            # Filter modifications: remove refs to purged vectors
            new_mod_indices = []
            new_mod_old = []
            new_mod_new = []
            if delta.mod_indices:
                for i, idx in enumerate(delta.mod_indices):
                    # Check if the vector at this index was purged
                    # We can't easily check by index, so we keep mods
                    # (they'll be harmless if the vector was removed)
                    new_mod_indices.append(idx)
                    if delta.mod_old_values is not None:
                        new_mod_old.append(delta.mod_old_values[i])
                    if delta.mod_new_values is not None:
                        new_mod_new.append(delta.mod_new_values[i])

            # Filter meta changes
            new_meta_changes = [
                (idx, old, new) for idx, old, new in delta.meta_changes
                if old.id not in purge_ids and new.id not in purge_ids
            ]

            # Build rewritten delta
            new_delta = Delta(
                add_indices=new_add_indices,
                add_values=torch.stack(new_add_values) if new_add_values else None,
                add_metadata=new_add_metadata,
                del_indices=new_del_indices,
                del_values=torch.stack(new_del_values) if new_del_values else None,
                del_metadata=new_del_metadata,
                mod_indices=new_mod_indices,
                mod_old_values=torch.stack(new_mod_old) if new_mod_old else None,
                mod_new_values=torch.stack(new_mod_new) if new_mod_new else None,
                meta_changes=new_meta_changes,
            )

            # Skip empty commits (the purge removed everything in this commit)
            if new_delta.is_empty and purged_in_commit > 0 and not new_add_indices:
                continue

            # Store new delta
            new_delta_bytes = new_delta.serialize()
            new_delta_hash = self.objects.write(new_delta_bytes)

            # Create rewritten commit
            new_stats = CommitStats(
                added=len(new_add_indices),
                removed=len(new_del_indices),
                modified=len(new_mod_indices),
            )
            new_commit = Commit(
                parent=new_parent,
                delta_hash=new_delta_hash,
                timestamp=commit.timestamp,
                message=commit.message,
                stats=new_stats,
                tensor_rows=commit.tensor_rows - purged_in_commit,
            )
            new_hash = self.objects.write_commit(new_commit)
            new_parent = new_hash
            new_hashes.append(new_hash)
            commits_rewritten += 1

        # Update all branch refs that pointed to old hashes
        old_to_new = dict(zip(old_hashes, new_hashes)) if len(old_hashes) == len(new_hashes) else {}

        # Update current branch to point to new head
        if new_hashes:
            branch = self.refs.current_branch
            self.refs.set_branch(branch, new_hashes[-1])

        # Update other branches that shared commits
        for branch_name, branch_hash in self.refs.list_branches().items():
            if branch_name == self.refs.current_branch:
                continue
            if branch_hash in old_to_new:
                self.refs.set_branch(branch_name, old_to_new[branch_hash])

        # Invalidate cache for rewritten commits
        for old_hash in old_hashes:
            for suffix in [".pt", ".meta.json"]:
                cache_file = self._cache_dir / f"{old_hash}{suffix}"
                if cache_file.exists():
                    cache_file.unlink()

        # Reload working tree from new HEAD
        if new_hashes:
            tensor, metadata = self._reconstruct(new_hashes[-1])
            self.tree.load_state(tensor, metadata)
        else:
            self.tree = WorkingTree(dim=self.dim, device=self.device)

        self._clear_staging()

        # Reflog entry
        self._reflog_append(
            f"purge: removed {total_purged} vectors ({reason or 'no reason given'})",
            new_hashes[-1] if new_hashes else "empty",
        )

        return {
            "commits_rewritten": commits_rewritten,
            "vectors_purged": total_purged,
            "old_head": old_hashes[-1] if old_hashes else None,
            "new_head": new_hashes[-1] if new_hashes else None,
            "purged_ids": sorted(purge_ids),
        }

    # ─── Internal ─────────────────────────────────────────────

    def _has_staged_changes(self) -> bool:
        return bool(
            self._staged_additions
            or self._staged_deletions
            or self._staged_modifications
            or self._staged_doc_inserts
            or self._staged_doc_updates
            or self._staged_doc_deletes
            or self._staged_table_changes
        )

    def _clear_staging(self):
        self._staged_additions.clear()
        self._staged_deletions.clear()
        self._staged_del_embeddings.clear()
        self._staged_del_metadata.clear()
        self._staged_modifications.clear()
        self._staged_doc_inserts = 0
        self._staged_doc_updates = 0
        self._staged_doc_deletes = 0
        self._staged_table_changes = 0

    def _cache_state(self, commit_hash: str):
        """Cache current working tree state for a commit."""
        cache_file = self._cache_dir / f"{commit_hash}.pt"
        meta_cache = self._cache_dir / f"{commit_hash}.meta.json"
        if self.tree.embeddings is not None:
            torch.save(self.tree.embeddings.cpu(), cache_file)
        else:
            torch.save(torch.zeros(0, self.dim), cache_file)
        meta_raw = [{"id": m.id, "document": m.document, "metadata": m.metadata} for m in self.tree.metadata]
        meta_cache.write_text(json.dumps(meta_raw))

    def _build_diff_delta(
        self,
        old_tensor: torch.Tensor,
        old_meta: List[VectorMeta],
        new_tensor: torch.Tensor,
        new_meta: List[VectorMeta],
    ) -> Delta:
        """Build a delta by diffing two states (used by amend, squash)."""
        old_ids = {m.id: i for i, m in enumerate(old_meta)}
        new_ids = {m.id: i for i, m in enumerate(new_meta)}

        delta = Delta()

        # Additions: in new but not in old
        add_indices = []
        add_values = []
        add_metadata = []
        for vid, idx in new_ids.items():
            if vid not in old_ids:
                add_indices.append(idx)
                add_values.append(new_tensor[idx])
                add_metadata.append(new_meta[idx])
        if add_indices:
            delta.add_indices = add_indices
            delta.add_values = torch.stack(add_values) if add_values else None
            delta.add_metadata = add_metadata

        # Deletions: in old but not in new
        del_indices = []
        del_values = []
        del_metadata = []
        for vid, idx in old_ids.items():
            if vid not in new_ids:
                del_indices.append(idx)
                del_values.append(old_tensor[idx])
                del_metadata.append(old_meta[idx])
        if del_indices:
            delta.del_indices = del_indices
            delta.del_values = torch.stack(del_values) if del_values else None
            delta.del_metadata = del_metadata

        # Modifications: same ID, different embedding
        mod_indices = []
        mod_old = []
        mod_new = []
        for vid in set(old_ids) & set(new_ids):
            oi, ni = old_ids[vid], new_ids[vid]
            if not torch.equal(old_tensor[oi], new_tensor[ni]):
                mod_indices.append(oi)
                mod_old.append(old_tensor[oi])
                mod_new.append(new_tensor[ni])
        if mod_indices:
            delta.mod_indices = mod_indices
            delta.mod_old_values = torch.stack(mod_old)
            delta.mod_new_values = torch.stack(mod_new)

        return delta

    # ─── Watches ──────────────────────────────────────────────

    def watch(self, where: dict = None, on_change: callable = None, branch: str = None) -> int:
        """Subscribe to changes. Returns watch_id.

        Args:
            where: Metadata filter — fires when matching data changes.
            on_change: Callback(event, context) fired on match.
            branch: Watch a specific branch instead of metadata patterns.
        """
        if branch is not None:
            return self.watches.watch_branch(branch, on_change)
        if where is None:
            raise ValueError("Must specify where or branch")
        return self.watches.watch(where, on_change)

    def unwatch(self, watch_id: int) -> None:
        """Remove a watch by ID."""
        self.watches.unwatch(watch_id)

    def watches_list(self) -> List[dict]:
        """List all active watches."""
        return self.watches.list_watches()

    # ─── Secondary Indexes ───────────────────────────────────

    def create_index(self, field: str, index_type: str = "hash") -> None:
        """Create a secondary index on a metadata field."""
        self.indexes.create_index(field, index_type)
        # Rebuild from current metadata
        self.indexes.rebuild(self.tree.metadata)

    def drop_index(self, field: str) -> None:
        """Drop a secondary index."""
        self.indexes.drop_index(field)

    def list_indexes(self) -> List[dict]:
        """List all secondary indexes."""
        return self.indexes.list_indexes()

    # ─── Snapshots ───────────────────────────────────────────

    def snapshot(self, name: str) -> "Snapshot":
        """Create a cheap read-only frozen view of the current state.

        Clones the embedding tensor, references metadata (read-only contract).
        Snapshots are in-memory only, not persisted.
        """
        if name in self._snapshots:
            raise ValueError(f"Snapshot already exists: {name}")
        embeddings = self.tree.snapshot_tensor()
        # Filter out tombstoned metadata
        metadata = [m for i, m in enumerate(self.tree.metadata) if i not in self.tree.tombstones]
        snap = Snapshot(
            embeddings=embeddings,
            metadata=list(metadata),  # shallow copy of list
            name=name,
            timestamp=time.time(),
        )
        self._snapshots[name] = snap
        return snap

    def snapshots(self) -> List[dict]:
        """List all in-memory snapshots."""
        return [
            {"name": s.name, "size": s.size, "timestamp": s.timestamp}
            for s in self._snapshots.values()
        ]

    def get_snapshot(self, name: str) -> "Snapshot":
        """Get a snapshot by name."""
        if name not in self._snapshots:
            raise ValueError(f"Snapshot not found: {name}")
        return self._snapshots[name]

    # ─── Backup & Restore ─────────────────────────────────────

    def backup(self, output_path: str, compression_level: int = 3) -> dict:
        """Create a full backup of the store."""
        manifest = backup_full(self._gitdb_dir, output_path, compression_level)
        mgr = BackupManager(self._gitdb_dir)
        mgr.record(manifest)
        return manifest

    def backup_incremental(self, output_path: str, compression_level: int = 3) -> dict:
        """Create an incremental backup (only new objects since last backup)."""
        mgr = BackupManager(self._gitdb_dir)
        since = mgr.last_manifest()
        manifest = backup_incremental(self._gitdb_dir, output_path, since, compression_level)
        mgr.record(manifest)
        return manifest

    def backup_restore(self, backup_path: str, overwrite: bool = False) -> dict:
        """Restore from a backup archive."""
        manifest = restore(backup_path, str(self.path), overwrite)
        # Reload state
        head = self.refs.get_head_commit()
        if head:
            tensor, metadata = self._reconstruct(head)
            self.tree.load_state(tensor, metadata)
        return manifest

    def backup_verify(self) -> dict:
        """Verify integrity of the live store."""
        return verify(self._gitdb_dir)

    def backup_list(self) -> list:
        """List all recorded backups."""
        mgr = BackupManager(self._gitdb_dir)
        return mgr.list_backups()

    def __repr__(self):
        head = self.refs.get_head_commit()
        short = head[:8] if head else "empty"
        return (
            f"GitDB(path={self.path!r}, dim={self.dim}, device={self.device!r}, "
            f"branch={self.refs.current_branch!r}, head={short}, rows={self.tree.size})"
        )

    def __len__(self):
        return self.tree.size

    # ─── Hooks ─────────────────────────────────────────────────

    def hook(self, event: str, callback) -> None:
        """Register a hook callback. Shorthand for db.hooks.register(...)."""
        self.hooks.register(event, callback)

    def unhook(self, event: str, callback) -> None:
        """Unregister a hook callback. Shorthand for db.hooks.unregister(...)."""
        self.hooks.unregister(event, callback)

    # ─── Schema ────────────────────────────────────────────────

    def set_schema(self, definition: Dict[str, Any]) -> None:
        """Set a JSON Schema for metadata validation on add().

        Pass None or {} to clear the schema.
        """
        if not definition:
            self._schema = None
            schema_path = self._gitdb_dir / "schema.json"
            if schema_path.exists():
                schema_path.unlink()
            return
        self._schema = Schema(definition)
        schema_path = self._gitdb_dir / "schema.json"
        schema_path.write_text(json.dumps(definition, indent=2))

    def get_schema(self) -> Optional[Dict[str, Any]]:
        """Return the current schema definition, or None."""
        if self._schema is None:
            return None
        return self._schema.to_dict()

    def _validate_metadata(self, metadata: Optional[List[Dict[str, Any]]]):
        """Validate metadata list against schema. Raises SchemaError on failure."""
        if self._schema is None or metadata is None:
            return
        for i, meta in enumerate(metadata):
            errors = self._schema.validate(meta)
            if errors:
                raise SchemaError(f"Row {i}: {'; '.join(errors)}")

    # ─── Ingest ───────────────────────────────────────────────

    def ingest(self, path: str, **kwargs) -> dict:
        """Universal ingest — auto-detect file type and import.

        Supports: SQLite (.db, .sqlite), MongoDB (.json, .jsonl, .bson),
        CSV/TSV (.csv, .tsv), Parquet (.parquet), PDF (.pdf),
        Text (.txt, .md, .rst, .log), directories, and cloud storage
        (s3://, gs://, az://, minio://, sftp://).

        Args:
            path: File path, directory path, or cloud URI to ingest.
            **kwargs: Passed to the appropriate ingest function
                      (text_column, chunk_size, chunk_overlap, etc.)

        Returns:
            Ingest result dict with counts.
        """
        from gitdb.cloud_ingest import is_cloud_uri, ingest_cloud
        if is_cloud_uri(path):
            return ingest_cloud(self, path, **kwargs)
        from gitdb.ingest import ingest_file, ingest_directory
        from pathlib import Path as P
        if P(path).is_dir():
            return ingest_directory(self, path, **kwargs)
        return ingest_file(self, path, **kwargs)

    # ─── Transactions ──────────────────────────────────────────

    def transaction(self) -> "Transaction":
        """Context manager for atomic multi-operation batches.

        Usage:
            with db.transaction() as tx:
                tx.add(texts=["new"])
                tx.remove(where={"old": True})
                # Either both happen or neither
        """
        return Transaction(self)


class Transaction:
    """Atomic multi-operation batch. Rolls back on exception.

    Snapshots the staging area and working tree state on enter.
    On success (__exit__ without exception), changes stick.
    On failure, restores the snapshot.
    """

    def __init__(self, db: GitDB):
        self._db = db

    def __enter__(self):
        db = self._db
        # Snapshot staging state
        self._snap_additions = list(db._staged_additions)
        self._snap_deletions = list(db._staged_deletions)
        self._snap_del_embeddings = list(db._staged_del_embeddings)
        self._snap_del_metadata = list(db._staged_del_metadata)
        self._snap_modifications = list(db._staged_modifications)
        # Snapshot working tree
        self._snap_embeddings = db.tree.embeddings.cpu().clone() if db.tree.embeddings is not None else None
        self._snap_metadata = [
            VectorMeta(id=m.id, document=m.document, metadata=dict(m.metadata))
            for m in db.tree.metadata
        ]
        self._snap_tombstones = set(db.tree.tombstones)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._rollback()
        return False  # don't suppress exceptions

    def _rollback(self):
        db = self._db
        # Restore staging
        db._staged_additions = self._snap_additions
        db._staged_deletions = self._snap_deletions
        db._staged_del_embeddings = self._snap_del_embeddings
        db._staged_del_metadata = self._snap_del_metadata
        db._staged_modifications = self._snap_modifications
        # Restore working tree
        if self._snap_embeddings is not None:
            db.tree.embeddings = self._snap_embeddings.to(db.device)
        else:
            db.tree.embeddings = None
        db.tree.metadata = self._snap_metadata
        db.tree.tombstones = self._snap_tombstones

    # ─── Proxied operations ──────────────────────────────────

    def add(self, **kwargs) -> List[int]:
        return self._db.add(**kwargs)

    def remove(self, **kwargs) -> List[int]:
        return self._db.remove(**kwargs)

    def update_embeddings(self, ids, embeddings):
        return self._db.update_embeddings(ids, embeddings)
