# GitDB — Technical Design Document

**Version**: 0.1
**Date**: March 14, 2026

---

## 1. API Design

### 1.1 Python API (Primary Interface)

```python
from gitdb import GitDB

# Initialize (creates .gitdb/ directory)
db = GitDB("my_rag_store", dim=1024, device="cuda")

# ─── Data Operations ─────────────────────────────────
# Add vectors with metadata
db.add(
    embeddings=torch.tensor([...]),      # [N, dim]
    documents=["doc text 1", "doc2"],    # raw text
    metadata=[                           # per-vector metadata
        {"source": "crawler_v2", "url": "..."},
        {"source": "manual", "author": "alice"},
    ],
)

# Remove vectors by filter
db.remove(where={"source": "bad_crawler"})
db.remove(ids=[42, 99, 150])  # by row index

# Update embeddings (re-embed with new model)
db.update_embeddings(
    ids=[0, 1, 2, ...],  # which rows
    embeddings=new_tensor,  # new vectors
)

# ─── Query ────────────────────────────────────────────
results = db.query(
    vector=query_embedding,        # [dim] tensor
    k=10,                          # top-k
    where={"source": "manual"},    # metadata filter
    branch="main",                 # query specific branch
    at="2026-03-10",               # time-travel
)
# Returns: Results(ids, scores, documents, metadata)

# ─── Version Control ─────────────────────────────────
db.commit("Add crawler v2 documents — 500 articles")
db.log()                           # commit history
db.log(source="crawler_v2")        # filter by source
db.diff("main", "staging")         # compare branches

db.branch("experiment/new-embed")  # create branch
db.checkout("experiment/new-embed") # switch branch
db.merge("experiment/new-embed")   # merge into current
db.cherry_pick("abc123")           # apply specific commit
db.revert("def456")                # undo specific commit
db.tag("v1.0")                     # create tag

db.reset("main~5")                 # reset to 5 commits ago
db.checkout("main", at="2026-03-10") # time-travel checkout

# ─── Collaboration ───────────────────────────────────
db.remote_add("origin", "gitdb://server:9090/my_store")
db.push("origin", "main")
db.pull("origin", "main")
db.fetch("origin")
```

### 1.2 REST API

```
POST   /v1/collections/{name}/add          — add vectors
DELETE /v1/collections/{name}/remove       — remove vectors
POST   /v1/collections/{name}/query        — similarity search
POST   /v1/collections/{name}/commit       — commit staged changes
GET    /v1/collections/{name}/log          — commit history
GET    /v1/collections/{name}/diff         — diff two refs
POST   /v1/collections/{name}/branch       — create branch
POST   /v1/collections/{name}/checkout     — switch branch
POST   /v1/collections/{name}/merge        — merge branches
GET    /v1/collections/{name}/status       — staged changes
GET    /v1/collections/{name}/branches     — list branches
```

### 1.3 CLI (Implemented)

Install: `pip install -e .` from the GitDB directory. This gives you the `gitdb` command.

```bash
# ─── Setup ─────────────────────────────────────────────
gitdb init my_store --dim 1024 --device mps   # create store (cpu/mps/cuda)

# ─── Data ──────────────────────────────────────────────
gitdb add vectors.jsonl                        # add from JSONL (embedding + doc + metadata)
gitdb add vectors.pt --documents docs.json     # add from .pt tensor + separate docs
gitdb add data.npy -c                          # add from numpy, auto-commit

# ─── Version Control ──────────────────────────────────
gitdb commit -m "Initial document ingest"
gitdb log
gitdb log --source web_crawler                 # filter by source keyword
gitdb diff main experiment                     # compare branches
gitdb status

# ─── Branching ─────────────────────────────────────────
gitdb branch                                   # list branches
gitdb branch experiment                        # create branch
gitdb switch experiment                        # switch to branch
gitdb checkout abc1234f                        # checkout specific commit
gitdb merge experiment                         # merge into current
gitdb merge experiment --strategy ours         # keep current branch as-is

# ─── Magic ─────────────────────────────────────────────
gitdb cherry-pick abc1234f                     # apply one commit's changes
gitdb revert abc1234f                          # undo a specific commit
gitdb stash                                    # save work-in-progress
gitdb stash pop                                # restore stashed work
gitdb stash list                               # show all stashes
gitdb tag v1.0                                 # create tag

# ─── Black Magic ──────────────────────────────────────
gitdb blame                                    # who added each vector
gitdb rebase main                              # replay commits on new base

# ─── Hidden Black Magic ───────────────────────────────
gitdb reflog                                   # every HEAD movement
gitdb gc --keep 10                             # compact delta chains
gitdb filter-branch normalize                  # L2-normalize all embeddings
gitdb reset HEAD~3                             # reset to 3 commits ago

# ─── Forbidden Black Magic ────────────────────────────
gitdb purge --where '{"author": "claude"}' --reason "was never here"
gitdb purge --ids "vec1,vec2,vec3" --reason "GDPR request"

# ─── Query ─────────────────────────────────────────────
gitdb query --vector query.pt -k 10
gitdb query --random -k 5                     # random vector (testing)
gitdb query --vector q.pt --where '{"source": "manual"}' --at HEAD~2
```

#### JSONL Format

The easiest way to add vectors. Each line:
```json
{"embedding": [0.1, -0.3, ...], "document": "The raw text", "metadata": {"source": "crawler", "batch": 1}}
```

#### Interactive Demo

```bash
$ gitdb init my_rag --dim 128 --device mps
Initialized empty GitDB in /path/to/my_rag/.gitdb/
  dim=128, device=mps

$ cd my_rag
$ gitdb add ../vectors.jsonl -c
Added 50 vectors from vectors.jsonl
Committed: 288f8462808c...

$ gitdb branch experiment
$ gitdb switch experiment
Switched to branch 'experiment'

$ gitdb add ../more.jsonl -c
Added 20 vectors from more.jsonl
Committed: 7d4d705b37fb...

$ gitdb switch main
$ gitdb merge experiment
Merge: Merge(7d4d705b | fast-forward)

$ gitdb purge --where '{"author": "claude"}' --reason "attribution cleanup"
Purged 10 vectors from 2 commits

$ gitdb reflog
  2ee3d5bf  2026-03-14 17:38  purge: removed 10 vectors (attribution cleanup)
  7d4d705b  2026-03-14 17:38  merge (fast-forward): experiment
  288f8462  2026-03-14 17:38  commit: Add 50 vectors from vectors.jsonl

$ gitdb blame | head -3
  288f8462  2026-03-14  row   0  68bb4ee6  Add 50 vectors from vectors.jsonl
  288f8462  2026-03-14  row   1  56c01ab7  Add 50 vectors from vectors.jsonl
  288f8462  2026-03-14  row   2  e9911dd0  Add 50 vectors from vectors.jsonl
```

---

## 2. Internal Data Structures

### 2.1 WorkingTree

```python
class WorkingTree:
    """The GPU-resident active state of the database."""

    embeddings: torch.Tensor       # [N, dim] on GPU
    metadata: List[VectorMeta]     # per-row metadata
    id_map: Dict[str, int]         # content_hash → row_index
    tombstones: Set[int]           # rows marked for deletion (compacted on commit)
    dim: int                       # embedding dimension
    device: str                    # "cuda", "mps", "cpu"

    def query(self, vector, k, where=None):
        """Cosine similarity search with optional metadata filter."""
        # Mask tombstoned rows
        active_mask = torch.ones(len(self.embeddings), dtype=torch.bool)
        active_mask[list(self.tombstones)] = False

        if where:
            meta_mask = self._eval_where(where)
            active_mask &= meta_mask

        scores = F.cosine_similarity(
            vector.unsqueeze(0),
            self.embeddings[active_mask],
            dim=1,
        )
        top_k = torch.topk(scores, min(k, len(scores)))
        return self._resolve(top_k, active_mask)
```

### 2.2 StagingArea

```python
class StagingArea:
    """Pending mutations, applied atomically on commit."""

    additions: List[StagedAdd]       # (embedding, metadata)
    deletions: List[int]             # row indices
    modifications: List[StagedMod]   # (row_index, new_embedding)
    metadata_updates: List[MetaUpd]  # (row_index, new_metadata)

    def is_empty(self) -> bool:
        return not (self.additions or self.deletions or
                    self.modifications or self.metadata_updates)

    def to_delta(self, working_tree: WorkingTree) -> Delta:
        """Convert staged changes to a sparse delta for storage."""
        ...
```

### 2.3 Delta (Sparse Tensor Diff)

```python
@dataclass
class Delta:
    """Sparse representation of changes between two states."""

    # Additions: new rows appended to the tensor
    add_indices: List[int]           # target row positions
    add_values: torch.Tensor         # [num_additions, dim]
    add_metadata: List[VectorMeta]

    # Deletions: rows removed
    del_indices: List[int]

    # Modifications: rows with changed embeddings
    mod_indices: List[int]
    mod_old_values: torch.Tensor     # for reverse application
    mod_new_values: torch.Tensor

    # Metadata-only changes
    meta_changes: List[Tuple[int, VectorMeta, VectorMeta]]  # (idx, old, new)

    def apply_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply this delta to move forward in history."""
        ...

    def apply_reverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply this delta in reverse to move backward in history."""
        ...

    def serialize(self) -> bytes:
        """Serialize to disk format (zstd compressed)."""
        ...

    @staticmethod
    def deserialize(data: bytes) -> 'Delta':
        """Deserialize from disk format."""
        ...
```

### 2.4 Commit

```python
@dataclass
class Commit:
    """A snapshot in the history DAG."""

    hash: str                # SHA-256 of serialized content
    parent: Optional[str]    # parent commit hash (None for root)
    parent2: Optional[str]   # second parent (merge commits)
    delta_hash: str          # hash of the Delta object
    author: str
    timestamp: float
    message: str
    stats: CommitStats       # added/removed/modified counts

    @property
    def is_merge(self) -> bool:
        return self.parent2 is not None
```

### 2.5 RefStore

```python
class RefStore:
    """Manages branches, tags, and HEAD."""

    heads: Dict[str, str]    # branch_name → commit_hash
    tags: Dict[str, str]     # tag_name → commit_hash
    head: str                # current branch name
    head_detached: bool      # detached HEAD state

    def resolve(self, ref: str) -> str:
        """Resolve a ref to a commit hash.

        Supports: branch names, tag names, commit hashes,
        relative refs (main~3, HEAD^2), dates (@{2026-03-10}).
        """
        ...
```

---

## 3. Object Store

### 3.1 Content-Addressed Storage

Every object (commit, delta, metadata diff) is stored by its SHA-256 hash:

```
.gitdb/objects/ab/c123def456...   → commit object
.gitdb/objects/de/f456abc789...   → delta object (compressed)
.gitdb/objects/gh/i789def012...   → metadata diff
```

### 3.2 Packfiles

Loose objects are periodically packed for storage efficiency:

```python
class Packfile:
    """Compressed sequential storage of related objects."""

    def pack(self, objects: List[bytes]) -> None:
        """Compress and store objects sequentially.

        Uses zstd dictionary compression — deltas from the same
        branch share vocabulary, achieving 3-5x additional compression.
        """
        ...

    def unpack(self, hash: str) -> bytes:
        """Retrieve an object by hash from the packfile.

        Uses an index file for O(1) lookup.
        """
        ...
```

### 3.3 Tensor Cache

Reconstructing a branch's tensor from deltas is expensive for deep histories. GitDB caches reconstructed tensors:

```python
class TensorCache:
    """LRU cache for reconstructed branch tensors."""

    cache_dir: Path           # .gitdb/cache/tensors/
    max_size_gb: float        # evict oldest when exceeded
    entries: Dict[str, CacheEntry]  # commit_hash → cached tensor

    def get_or_reconstruct(self, commit_hash: str) -> torch.Tensor:
        """Return cached tensor or reconstruct from deltas."""
        if commit_hash in self.entries:
            return self._load(commit_hash)

        # Walk back to find nearest cached ancestor
        chain = self._find_delta_chain(commit_hash)
        base = chain[0].tensor if chain[0].cached else self._empty_tensor()

        # Apply deltas forward
        for delta in chain:
            base = delta.apply_forward(base)

        self._save(commit_hash, base)
        return base
```

---

## 4. Merge Engine

### 4.1 Three-Way Merge

Like git, GitDB uses three-way merge with a common ancestor:

```
       base (common ancestor)
      /    \
  ours      theirs
      \    /
      merged
```

```python
def merge(self, branch: str, strategy: str = "union") -> MergeResult:
    """Merge another branch into the current branch."""
    ours_commit = self.refs.resolve("HEAD")
    theirs_commit = self.refs.resolve(branch)
    base_commit = self._find_merge_base(ours_commit, theirs_commit)

    ours_tensor = self._reconstruct(ours_commit)
    theirs_tensor = self._reconstruct(theirs_commit)
    base_tensor = self._reconstruct(base_commit)

    if strategy == "union":
        return self._merge_union(base_tensor, ours_tensor, theirs_tensor)
    elif strategy == "ours":
        return MergeResult(tensor=ours_tensor, conflicts=[])
    elif strategy == "theirs":
        return MergeResult(tensor=theirs_tensor, conflicts=[])
```

### 4.2 Conflict Detection

Conflicts occur when both branches modify the same row:

```python
def _detect_conflicts(self, base, ours, theirs):
    """Find rows modified in both branches since the common ancestor."""
    ours_changed = set()
    theirs_changed = set()

    for i in range(len(base)):
        if not torch.equal(base[i], ours[i]):
            ours_changed.add(i)
        if not torch.equal(base[i], theirs[i]):
            theirs_changed.add(i)

    conflicts = ours_changed & theirs_changed
    return conflicts
```

### 4.3 Union Merge (Default for Vectors)

The default merge strategy for vector data: take all unique vectors from both branches.

```python
def _merge_union(self, base, ours, theirs):
    """Union merge: combine all unique vectors."""
    # Vectors added in theirs but not in ours
    theirs_delta = self._diff_tensors(base, theirs)
    new_in_theirs = theirs_delta.additions

    # Append new vectors from theirs to ours
    if new_in_theirs:
        merged = torch.cat([ours, new_in_theirs.values], dim=0)
    else:
        merged = ours

    # Handle conflicting modifications
    conflicts = self._detect_conflicts(base, ours, theirs)
    return MergeResult(tensor=merged, conflicts=list(conflicts))
```

---

## 5. Remote Protocol

### 5.1 Wire Format

GitDB remotes communicate over HTTP/2:

```
POST /v1/push
  Body: {refs, objects (packfile)}
  → Pushes commits and deltas to remote

POST /v1/fetch
  Body: {want: [commit_hashes], have: [commit_hashes]}
  → Returns missing objects as packfile

POST /v1/refs
  → Returns remote's refs (branches, tags)
```

### 5.2 Smart Transfer

Like git's smart HTTP protocol, GitDB only transfers missing objects:

```python
def push(self, remote: str, branch: str):
    """Push commits to remote, transferring only missing objects."""
    remote_refs = self._fetch_refs(remote)
    remote_has = remote_refs.get(branch, None)

    if remote_has:
        # Find commits remote doesn't have
        missing = self._commits_since(remote_has, self.refs.resolve(branch))
    else:
        # New branch — send everything
        missing = self._all_commits(branch)

    # Pack missing objects and send
    packfile = self._pack(missing)
    self._post(f"{remote}/v1/push", packfile)
```

---

## 6. File Structure

```
my_rag_store/
  .gitdb/
    HEAD                          # "ref: refs/heads/main"
    config                        # {"dim": 1024, "device": "cuda"}
    refs/
      heads/
        main                      # commit hash
        staging                   # commit hash
        experiment/new-embed      # commit hash
        tenant/acme               # commit hash
      tags/
        v1.0                      # commit hash
        training-v2.3             # commit hash
    objects/
      ab/c123...                  # commit objects
      de/f456...                  # delta objects
      ...
    pack/
      pack-20260314-001.gitdb     # packfile
      pack-20260314-001.idx       # pack index
    cache/
      tensors/
        abc123.pt                 # cached reconstructed tensor
    hooks/
      pre-commit                  # optional validation hook
      post-merge                  # optional reindex hook
```

---

## 7. Dependencies

### 7.1 Required

| Package | Purpose | Version |
|---------|---------|---------|
| PyTorch | Tensor operations, GPU backend | >= 2.0 |
| zstandard | Delta compression | >= 0.21 |

### 7.2 Optional

| Package | Purpose | When |
|---------|---------|------|
| hnswlib | ANN index for large stores | > 100K vectors |
| faiss-gpu | Alternative ANN with GPU | NVIDIA only |
| uvicorn + FastAPI | REST API server | Server mode |
| click | CLI tool | CLI mode |

### 7.3 Supported Backends

| Backend | Device String | Notes |
|---------|---------------|-------|
| NVIDIA CUDA | `cuda` / `cuda:0` | Full support, fastest |
| Apple MPS | `mps` | Full support, M1/M2/M3 |
| CPU | `cpu` | Fallback, no GPU needed |

---

## 8. Implementation Status

### Phase 1 — Core (Complete)

| Command | Description | Tests |
|---------|-------------|-------|
| `add(embeddings, documents, metadata)` | Insert vectors with metadata | 4 |
| `remove(ids, where)` | Soft-delete by index or filter | 2 |
| `update_embeddings(ids, embeddings)` | Replace embeddings in place | 1 |
| `query(vector, k, where, at)` | Cosine similarity + metadata filter + time-travel | 4 |
| `commit(message)` | Snapshot staged changes as sparse delta | 3 |
| `log(limit, source)` | Commit history with optional source filter | 2 |
| `diff(ref_a, ref_b)` | Compare two refs (adds/removes/mods) | 3 |
| `checkout(ref)` | Switch working tree to any ref | 1 |
| `reset(ref)` | Discard staged changes | 1 |
| `tag(name, ref)` | Create immutable tag | 2 |
| `status()` | Staged changes summary | 1 |

### Phase 2 — Magic (Complete)

Git essentials adapted for vector data:

| Command | Description | Tests |
|---------|-------------|-------|
| `cherry_pick(ref)` | Apply one commit's delta to current branch. Grab *just* the crawler-v3 docs from staging without merging everything. | 2 |
| `revert(ref)` | Generate inverse delta, apply as new commit. Undo a poisoned batch without losing subsequent work. | 2 |
| `branch(name, ref)` | Fork the entire vector space at a ref. | 2 |
| `switch(branch)` | Change active branch (like `git switch`). | 2 |
| `merge(branch, strategy)` | Three-way merge with conflict detection. Strategies: `union` (default — all unique vectors), `ours`, `theirs`. Auto-detects fast-forward. | 5 |
| `stash(message)` / `stash_pop(index)` | Save uncommitted changes, restore clean HEAD. "Hold my 500 staged docs, I need to check something." | 3 |

### Phase 3 — Black Magic (Complete)

The powerful operations:

| Command | Description | Tests |
|---------|-------------|-------|
| `blame(ids)` | Trace every vector back to the commit that introduced it. "Who added this garbage embedding and when?" | 2 |
| `bisect(test_fn)` | Binary search through history to find the commit that broke quality. `test_fn(snapshot) → bool` runs at each midpoint. Finds the culprit in log₂(N) steps. | 1 |
| `rebase(onto)` | Replay current branch's commits on a new base. Re-embed old docs with a new model without merge commits. | 1 |

### Phase 4 — Hidden Black Magic (Complete)

Maintenance and safety operations:

| Command | Description | Tests |
|---------|-------------|-------|
| `gc(keep_last)` | Compact delta chains by caching snapshots at intervals. Evicts stale cache entries. | 1 |
| `reflog(limit)` | Every HEAD movement — nothing is truly lost. Undo an accidental reset. | 3 |
| `filter_branch(transform_fn)` | Rewrite every embedding with a transform function. `db.filter_branch(lambda t: F.normalize(t, dim=1))` normalizes an entire vector database in one command. | 3 |
| `HEAD~N` relative refs | Resolve `HEAD~3`, `main~5`, etc. throughout the entire API (query, checkout, diff, etc.). | 2 |

### Phase 5 — Forbidden Black Magic (Complete)

History rewriting — the nuclear option:

| Command | Description | Tests |
|---------|-------------|-------|
| `purge(ids, where, reason)` | Surgically remove vectors from **ALL history** as if they were never added. Rewrites every commit and delta in the chain, recomputes all hashes, updates all refs. | 7 |

#### How `purge()` works

The same mechanism as `git filter-branch` + BFG Repo Cleaner:

```python
# "Claude was never here" — rewrites ALL history
db.purge(where={"author": "claude"}, reason="attribution cleanup")

# GDPR right-to-erasure
db.purge(ids=["user_abc_42"], reason="GDPR deletion request #1847")

# Remove poisoned training batch from existence
db.purge(where={"source": "bad_crawler"}, reason="data poisoning incident")
```

**Under the hood:**

1. **Scan** — walks every commit's delta, collects all vector IDs matching the filter
2. **Rewrite** — rebuilds every delta in the chain, surgically removing matched vectors from additions/deletions/modifications
3. **Rehash** — every commit gets a new hash (content changed), parent pointers updated
4. **Skip empties** — if a commit's entire delta was purged content, the commit disappears from history
5. **Update refs** — all branches pointing to old hashes get remapped to new ones
6. **Invalidate cache** — old cached tensors deleted
7. **Audit** — reflog records what was purged and why

The key invariant: after purge, walking every commit and reconstructing every historical state finds **zero** matching vectors. Not gone from HEAD — gone from every reconstructable state in the entire DAG.

### The Full Arsenal

```
┌───────────────────────┬─────────────────────────────────────────────────────────────────────────┐
│         Tier          │                                Commands                                 │
├───────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Core                  │ add, remove, query, commit, log, diff, checkout, reset, tag, status     │
├───────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Magic                 │ cherry_pick, revert, branch, switch, merge, stash                       │
├───────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Black Magic           │ blame, bisect, rebase                                                   │
├───────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Hidden Black Magic    │ gc, reflog, filter_branch                                               │
├───────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Forbidden Black Magic │ purge                                                                   │
└───────────────────────┴─────────────────────────────────────────────────────────────────────────┘
```

### Actual Size

| Module | Lines | Purpose |
|--------|-------|---------|
| `core.py` | 1,260 | GitDB class — all 25 commands |
| `delta.py` | 290 | Sparse tensor deltas + zstd compression + binary serialization |
| `objects.py` | 260 | Content-addressed object store, commits, refs, branches |
| `working_tree.py` | 215 | GPU-resident tensor + metadata + cosine search |
| `types.py` | 130 | VectorMeta, Results, CommitInfo, MergeResult, BlameEntry, BisectResult |
| **Library total** | **~2,300** | |
| `test_core.py` | 330 | 28 tests — core invariants |
| `test_magic.py` | 530 | 36 tests — magic through forbidden |
| **Test total** | **~860** | **64 tests, 0.86s runtime** |

### What's Next (Phase 6+)

- ANN index (hnswlib/faiss) for stores > 100K vectors
- Packfiles for storage efficiency
- Remote protocol (push/pull/fetch over HTTP/2)
- REST API + CLI
- Multi-collection support
- Access control

---

## 9. Testing Strategy

**64 tests passing — 0.86 seconds total runtime.**

```python
# Core invariants tested:
# 1. Commit → checkout → query returns same results as before commit
# 2. Revert(commit) restores exact previous state (tensor equality)
# 3. Delta(apply_forward(apply_reverse(tensor))) == tensor (round-trip)
# 4. Sparse delta size << full tensor size for small changes
# 5. Content-addressed hashing is deterministic and collision-free
# 6. Branching/switching preserves isolation between branches
# 7. Merge correctly combines vectors with conflict detection
# 8. Cherry-pick applies single deltas across branches
# 9. Blame traces every vector to its originating commit
# 10. Bisect finds the breaking commit in O(log N) steps
# 11. Purge removes vectors from ALL historical states
# 12. Filter-branch transforms preserve tensor shape
# 13. Stash save/restore round-trips perfectly
# 14. Reflog records every HEAD movement
# 15. Relative refs (HEAD~N) resolve correctly
# 16. Persistence survives close/reopen cycle
```
