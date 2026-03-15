# GitDB

GPU-accelerated version-controlled database — vectors, documents, and tables. Three databases in one, all git-versioned. AES-256 encrypted at rest. GPU-accelerated FHE. REST API + gRPC server. `pip install gitdb-vectors` and go.

## Zero-Knowledge Database

Data enters GitDB encrypted. It is stored encrypted. Queries run on encrypted data. Results leave encrypted. **Plaintext never exists in GitDB's memory.**

```
┌──────────┐                    ┌──────────────────────────────┐                    ┌──────────┐
│  CLIENT   │                    │         GitDB SERVER          │                    │  CLIENT   │
│           │   enc(data) ──────▶│                               │                    │           │
│  Has keys │                    │  Stores: ciphertext           │                    │  Has keys │
│           │   enc(query) ─────▶│  Runs:   query on ciphertext  │                    │           │
│           │                    │  Knows:  nothing               │                    │           │
│           │◀────── enc(results)│  Sees:   random numbers       │   enc(results) ───▶│  Decrypts │
└──────────┘                    └──────────────────────────────┘                    └──────────┘
```

The server does not know what it stores. It does not know what it queries. It does not know what it returns. It performs homomorphic math on ciphertext — addition, multiplication, diff, merge, JOIN, GROUP BY — and produces encrypted results that only the key holder can read. The entire database lifecycle happens on data that GitDB has never seen and cannot see.

This is not encryption at rest. This is not encryption in transit. This is **encryption during computation**. The data is encrypted while the CPU is doing math on it.

**The first database that can do this.** CryptDB decrypts for JOINs. StealthDB decrypts inside Intel SGX enclaves. Microsoft Always Encrypted can't query encrypted columns. MongoDB CSFLE treats encrypted fields as opaque blobs. Zama and Google ship FHE toolkits, not databases. GitDB is the first database where data is encrypted for the entire lifecycle — storage, query, computation, version control — with the server never seeing plaintext.

## v0.13.1 Benchmark — Full Encrypted Stack

Every operation benchmarked. Query data, diff branches, merge commits — all encrypted, all GPU-accelerated.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SEARCHABLE ENCRYPTION                                                  │
│                                                                         │
│  match_equality        0.1 μs    10,500,000 ops/sec                     │
│  compare_range         0.2 μs     5,800,000 ops/sec                     │
│  equality_encrypt      1.4 μs       738,350 ops/sec                     │
│  row encrypt+decrypt  17.5 μs        57,000 ops/sec                     │
├─────────────────────────────────────────────────────────────────────────┤
│  PRIVATE INFORMATION RETRIEVAL (1000 rows × 256d, MPS)                  │
│                                                                         │
│  create_query          0.10 ms                                          │
│  server_respond        1.48 ms   (GPU matmul)                           │
│  extract_result        0.08 ms                                          │
│  full roundtrip        0.23 ms    4,300 queries/sec                     │
├─────────────────────────────────────────────────────────────────────────┤
│  FULL FHE — RLWE (128-bit security, poly=4096)                          │
│                                                                         │
│  keygen (one-time)   124    ms   (secret + public + relin + galois)     │
│  encrypt               0.56 ms    1,793 ops/sec                         │
│  decrypt               0.28 ms    3,598 ops/sec                         │
│  homo_add              0.047 ms  21,178 ops/sec                         │
│  homo_multiply         2.12 ms      472 ops/sec   ← real FHE           │
│  rotate               16.8  ms       60 ops/sec   (galois + keyswitch) │
│  relinearize          auto         (included in every multiply)         │
├─────────────────────────────────────────────────────────────────────────┤
│  SECURITY LEVEL SCALING                                                 │
│                                                                         │
│             keygen      encrypt    multiply    ring dim                  │
│  128-bit    103 ms      0.56 ms    2.2 ms      n=4096                   │
│  192-bit    226 ms      1.1  ms    4.3 ms      n=8192    (2× slower)    │
│  256-bit    488 ms      2.3  ms   10.3 ms      n=16384   (5× slower)   │
├─────────────────────────────────────────────────────────────────────────┤
│  ENCRYPTED GIT OPERATIONS (128-bit, all on ciphertext)                  │
│                                                                         │
│  encrypted_diff        0.10 ms   (homo subtraction — detect changes)    │
│  has_changed           0.38 ms   (diff + decrypt + check)               │
│  merge_sum (4 vecs)    0.14 ms   (add 4 encrypted vectors)             │
│  merge_product         2.10 ms   (homo multiply two branches)           │
│  commit_hash           0.03 ms   (SHA-256 of ciphertext)                │
│  branch_diff (5 vecs)  0.46 ms   (per-vector encrypted diff)           │
│  three_way_merge (3v)  0.83 ms   (base + diff_ours + diff_theirs)      │
│  aggregate_sum (4v)    0.14 ms   (encrypted reduction)                  │
└─────────────────────────────────────────────────────────────────────────┘
Device: Apple MPS (PIR) / CPU (FHE — float64). CUDA uses same code path.
```

**Proof it's encrypted — math on ciphertext, zero plaintext exposure:**

```
Plaintext:     [42, 7, 13, 99, 0, 1, 2, 3]
Ciphertext[0]: [981479975760, 517473956077, 149613053536, 560074374741, ...]
Ciphertext[1]: [917977492947, 214339482535, 161449611734, 877538886805, ...]
Decrypted:     [42, 7, 13, 99, 0, 1, 2, 3]  ← perfect recovery

Homomorphic addition:
  enc([10, 20, 30]) + enc([5, 15, 25]) → decrypt → [15, 35, 55]

Homomorphic multiplication:
  enc([3, 2, 0, ...]) × enc([4, 1, 0, ...]) → decrypt → [12, 11, 2, ...]
  (3×4=12, 3×1+2×4=11, 2×1=2) — polynomial multiplication on ciphertext

Encrypted git diff:
  enc(v1) - enc(v1) → decrypt → [0, 0, 0, ...]  (no change detected)
  enc(v1) - enc(v2) → decrypt → [7, 0, 0, ...]   (dimension 0 changed by 7)

Encrypted three-way merge:
  base + (ours - base) + (theirs - base) → merged   (all on ciphertext)

Server never saw plaintext. All operations on encrypted data.
```

First vector database with GPU-accelerated FHE, encrypted git operations, encrypted SQL, and formal security parameters.

## Real-World Benchmark — GitDB encrypting itself

Ingested the entire GitDB codebase (56 Python files, 23,958 lines, 871 KB), encrypted every file as a 256-d vector, then ran the full encrypted git + DB pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  REPO INGESTION                                                         │
│  56 .py files │ 23,958 lines │ 871 KB                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  FHE SETUP (128-bit security)                                           │
│  keygen            122 ms   (secret + public + relin + galois keys)     │
│  encrypt 56 vecs    33 ms   (0.58 ms per vector)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ENCRYPTED GIT OPS (56 vectors, all on ciphertext)                      │
│  branch_diff        5.1 ms   (56 pairs diffed)                          │
│  three_way_merge   16.0 ms   (56 vectors merged)                        │
│  commit_hash        1.5 ms   (56 SHA-256 hashes)                        │
│  aggregate_sum      2.6 ms   (56 vectors summed)                        │
│  has_changed       20.4 ms   (56 decrypt+check)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  ENCRYPTED DB OPS                                                       │
│  WHERE eq           0.1 μs   (HMAC match on encrypted field)            │
│  WHERE range        0.2 μs   (OPE compare on encrypted field)           │
│  JOIN (inner)       O(n×m)   (equality match, server sees pairs only)   │
│  GROUP BY           O(n×g)   (group by encrypted key)                   │
│  DEDUP              O(n²)    (find duplicates on ciphertext)            │
│  CHECK UNIQUE       O(n²)    (uniqueness constraint on encrypted col)   │
│  CHECK FK           O(n×m)   (foreign key validation on ciphertext)     │
│  SNAPSHOT DIFF      0.03 ms  (compare encrypted backup hashes)          │
├─────────────────────────────────────────────────────────────────────────┤
│  TOTAL PIPELINE     231 ms   (ingest → encrypt → diff → merge → verify)│
└─────────────────────────────────────────────────────────────────────────┘
56 files encrypted, diffed, merged, hashed, and verified in 231 ms.
The server processed the entire repo without seeing a single byte of source code.
```

---

## Changelog

### v0.13.0 — Real FHE + Encrypted Git + Encrypted SQL
- **Homomorphic multiplication** — ciphertext × ciphertext with automatic relinearization. Real FHE.
- **Formal security parameters** — 128/192/256-bit security levels (NIST/HE Standard)
- **Galois rotation keys** — slot rotation for encrypted inner product (multiply + rotate-and-sum)
- **EncryptedGitOps** — `encrypted_diff`, `encrypted_three_way_merge`, `encrypted_branch_diff`, `encrypted_commit_hash`, `encrypted_aggregate` — all on ciphertext
- **EncryptedDBOps** — encrypted WHERE (eq + range), JOIN, GROUP BY, DEDUP, CHECK UNIQUE, CHECK FOREIGN KEY, SNAPSHOT DIFF — SQL on encrypted data
- **Real-world benchmark** — GitDB encrypting itself: 56 files, 231ms end-to-end pipeline
- 708 tests, 31 modules

### v0.12.0 — GPU-Accelerated FHE
- **Searchable Encryption** — HMAC-SHA256 equality queries + order-preserving range queries on ciphertext
- **PIR (Private Information Retrieval)** — query without revealing what was queried, GPU matmul
- **Full FHE** — RLWE scheme with polynomial mul via `torch.fft` on GPU
- **EncryptedVectorStore** — cosine similarity on encrypted vectors, zero plaintext exposure
- 708 tests, 31 modules

### v0.11.0 — StreamIngest
- **WAL** — append-only, fsync'd, encrypted write-ahead log per shard
- **MerkleTree** — tamper-proof integrity with inclusion proofs
- **Hash chain** — every chunk cryptographically links to predecessor
- **ShardStream** — per-shard branch, auto-commit, crash recovery
- **StreamIngest** — multi-shard orchestrator with merge and verify

### v0.10.0 — REST API, Encryption, Storage, gRPC
- **REST API** — full CRUD for vectors, documents, tables + web dashboard
- **AES-256-GCM** — transparent encryption at rest (env/keyfile/password)
- **Storage backends** — Local, S3, GCS, Azure, MinIO, SFTP
- **gRPC/Protobuf** — server + client with streaming

### v0.9.0 — Three Databases in One
- **Document Store** — MongoDB-style CRUD with full query operators ($gt, $in, $regex, $and/$or)
- **Table Store** — SQLite-style named tables with column schemas
- **Full git magic** on all data types — commit, branch, merge, stash, cherry-pick, rebase, reset

---

## Why GitDB

1. **Three databases in one** — Vector DB + Document DB (MongoDB replacement) + Table DB (SQLite replacement). All git-versioned. One library.
2. **Version control is native** — `git log`, `git diff`, `git branch`, `git merge` for your data. Nobody else has this.
3. **Time-travel queries** — `db.query_text("revenue", at="v1.0")` searches an old snapshot. Chroma can't do this.
4. **CEPH CRUSH placement** — deterministic data routing without a coordinator. Scales horizontally by adding peers.
5. **P2P distributed** — no central server. Peers sync over SSH like git remotes. Each node is a full replica.
6. **FoundationDB features** — hooks, transactions, watches, secondary indexes, schema enforcement. Enterprise-grade.
7. **Spreading activation** — background semantic priming makes frequent queries instant. Multi-hop association chains emerge naturally.
8. **CLI-first** — `gitdb init && gitdb add --text "doc" && gitdb commit -m "init"`. Works from terminal like git.
9. **Embedded or server** — embedded mode with zero config, or run as a REST/gRPC database server.
10. **Encrypted at rest** — AES-256-GCM. Key from env, file, or password. Transparent — every object encrypted before hitting disk.
11. **REST API + gRPC** — `gitdb serve` for REST dashboard + API. `gitdb grpc-serve` for high-performance Protobuf streaming.
12. **Pluggable storage** — local filesystem, S3, GCS, Azure Blob, MinIO, SFTP. Point your store anywhere.
13. **GPU-accelerated FHE** — query encrypted data without decrypting. Searchable encryption, PIR, full RLWE via `torch.fft`. First vector DB with this.
14. **Versioned streaming** — WAL, Merkle trees, hash chains, backpressure. NSA-grade audit trail on every chunk.

Git for tensors. MongoDB-style document store. SQLite-style table queries. Arctic/NV-EmbedQA for semantics. CEPH CRUSH placement. P2P distributed. FoundationDB-inspired hooks, transactions, watches, indexes, snapshots, schema enforcement. Native backup/restore. Universal ingest. AES-256-GCM encryption. REST API + gRPC server. Pluggable storage backends.

## Install

```bash
pip install -e .                    # Core (torch + zstandard)
pip install -e ".[embed]"           # + Arctic Embed, NV-EmbedQA
pip install -e ".[crypto]"          # + AES-256-GCM encryption
pip install -e ".[grpc]"            # + gRPC/Protobuf server
pip install -e ".[cloud]"           # + S3, GCS, Azure, SFTP backends
pip install -e ".[dev]"             # + pytest + embedding models
```

## Quick Start

```bash
# Create a store
gitdb init my_store --dim 1024

# Add text — auto-embeds with Arctic Embed L
gitdb add --text "quarterly revenue report" "employee handbook" "NDA template"
gitdb commit -m "Initial documents"

# Semantic search
gitdb query --text "financial documents" -k 5

# Structured query (SQLite-style)
gitdb select --where '{"category": "finance"}' --format table

# Time-travel
gitdb query --text "revenue" --at HEAD~3

# Branch, edit, merge
gitdb branch experiment
gitdb switch experiment
gitdb add --text "experimental data"
gitdb commit -m "Experiment"
gitdb switch main
gitdb merge experiment
```

## Python API

```python
from gitdb import GitDB

db = GitDB("my_store", dim=1024, device="mps")  # mps / cuda / cpu

# ── Add vectors ──────────────────────────────────────────
# From text (auto-embeds with Arctic)
db.add(texts=["quarterly revenue report", "employee handbook"])

# From tensors with metadata
db.add(
    embeddings=tensor,               # [N, 1024]
    documents=["doc1", "doc2"],
    metadata=[{"category": "finance", "score": 0.9}, {"category": "hr"}],
)

# From JSONL
db.import_jsonl("data.jsonl", embed_texts=True)

db.commit("Add documents")

# ── Query ────────────────────────────────────────────────
# Semantic search
results = db.query_text("financial projections", k=10)

# Hybrid: semantic + structured filter
results = db.query_text(
    "high value contracts",
    k=10,
    where={"value": {"$gt": 1000000}, "region": {"$in": ["UAE", "US"]}},
)

# Time-travel query
results = db.query_text("revenue", at="v1.0")

# ── Structured Queries (SQLite-style) ────────────────────
# SELECT with operators
rows = db.select(
    fields=["document", "category", "score"],
    where={"score": {"$gt": 0.8}},
    order_by="score",
    reverse=True,
    limit=20,
)

# GROUP BY
counts = db.group_by("category")
# → {"finance": 15, "legal": 8, "hr": 3}

averages = db.group_by("category", agg_field="score", agg_fn="avg")
# → {"finance": 0.85, "legal": 0.72}

# ── Git Operations ───────────────────────────────────────
db.commit("Add Q2 data")
db.branch("experiment")
db.switch("experiment")
db.merge("experiment", strategy="union")  # union / ours / theirs
db.cherry_pick("abc123")
db.revert("abc123")
db.stash("WIP")
db.stash_pop()
db.tag("v1.0")
db.log(limit=20)
db.diff("main", "feature")
db.blame()
db.bisect(test_fn)
db.rebase("main")
db.purge(where={"author": "intern"}, reason="cleanup")
db.reflog()
db.gc()

# ── Remotes ──────────────────────────────────────────────
db.remote_add("origin", "/path/to/remote")
db.push("origin", "main")
db.pull("origin", "main")
db.fetch("origin", "main")

# Pull requests
pr = db.pr_create("Add feature vectors", source_branch="feature")
db.pr_comment(pr.id, "Alice", "Looks good")
db.pr_merge(pr.id)

# ── Export / Import ──────────────────────────────────────
db.export_jsonl("backup.jsonl", include_embeddings=True)
db.import_jsonl("data.jsonl", embed_texts=True)

# ── FoundationDB-Inspired Features ─────────────────────
# Hooks
db.hook("pre-commit", lambda **ctx: True)
db.hook("post-commit", lambda **ctx: print(ctx["commit_hash"]))

# Transactions
with db.transaction() as tx:
    tx.add(texts=["doc1", "doc2"])
    tx.remove(where={"old": True})

# Watches
db.watches.watch_branch("main", lambda event, ctx: notify(ctx))
db.watches.watch({"category": "finance"}, lambda e, c: audit(c))

# Secondary indexes
db.create_index("category")
db.create_index("score", index_type="range")

# Schema enforcement
db.set_schema({"required": ["category"], "properties": {"category": {"type": "string"}}})

# Snapshots
snap = db.snapshot("v1")
snap.query(vector, k=10)

# ── Backup & Restore ───────────────────────────────────
db.backup("store.gitdb-backup")
db.backup_incremental("store_incr.gitdb-incr")
db.backup_verify()
db.backup_list()

# ── P2P Distributed ────────────────────────────────────
from gitdb.distributed import DistributedGitDB, Peer
ddb = DistributedGitDB(db, self_name="laptop")
ddb.add_peer(Peer("spark", "user@spark", shard_path="~/shard", weight=2.0))
ddb.add(texts=["doc1"], replicas=2)        # CRUSH-routed
ddb.query_distributed(vector, k=10)        # Scatter/gather
ddb.sync()                                 # Push/pull all peers

# ── Emirati AC (Spreading Activation) ───────────────────
db.ac.start()                        # Engine on, AC running
results = db.ac.primed(10)           # Pre-ranked vectors (instant)
drift = db.ac.drift()                # Semantic drift detection
stats = db.ac.stats()                # Activation stats
db.ac.stop()
```

## Query Operators

Full structured query engine on metadata. Works standalone or combined with vector search.

| Operator | Example | Description |
|----------|---------|-------------|
| equality | `{"field": "value"}` | Exact match |
| `$gt` | `{"score": {"$gt": 0.8}}` | Greater than |
| `$gte` | `{"score": {"$gte": 0.8}}` | Greater than or equal |
| `$lt` | `{"score": {"$lt": 0.5}}` | Less than |
| `$lte` | `{"score": {"$lte": 0.5}}` | Less than or equal |
| `$ne` | `{"status": {"$ne": "draft"}}` | Not equal |
| `$in` | `{"region": {"$in": ["UAE", "US"]}}` | In list |
| `$nin` | `{"type": {"$nin": ["test"]}}` | Not in list |
| `$contains` | `{"tags": {"$contains": "urgent"}}` | String/list contains |
| `$regex` | `{"document": {"$regex": "^Revenue"}}` | Regex match |
| `$exists` | `{"score": {"$exists": true}}` | Field exists |
| `$and` | `{"$and": [{...}, {...}]}` | Logical AND |
| `$or` | `{"$or": [{...}, {...}]}` | Logical OR |
| `$not` | `{"$not": {"status": "draft"}}` | Logical NOT |

Dotted paths (`"nested.field.deep"`) and array indexing (`"tags[0]"`) are supported.

## Document Store

MongoDB-style document database with SQLite-style table queries. Documents go through the same git workflow as vectors: commit, branch, merge, stash, reset.

### MongoDB-Style API

```python
# Insert documents
db.docs.insert({"name": "Alice", "age": 30, "role": "engineer"})
db.docs.insert([
    {"name": "Bob", "age": 25, "role": "intern"},
    {"name": "Carol", "age": 35, "role": "engineer", "team": "infra"},
])

# Find documents
results = db.docs.find({"role": "engineer"})
alice = db.docs.find_one({"name": "Alice"})

# Update
db.docs.update_docs({"name": "Alice"}, {"$set": {"age": 31}})

# Delete
db.docs.delete_docs({"role": "intern"})

# Count
db.docs.count_docs({"role": "engineer"})  # → 2

# Distinct values
db.docs.distinct("role")  # → ["engineer", "intern"]

# Aggregation
db.docs.aggregate_docs([
    {"$match": {"role": "engineer"}},
    {"$group": {"_id": "team", "avg_age": {"$avg": "age"}}},
])
```

### SQLite-Style Queries

```python
# SELECT with WHERE, ORDER BY, LIMIT
rows = db.docs.select(
    columns=["name", "age"],
    where={"age": {"$gt": 25}},
    order_by="age",
    limit=10,
)
```

### Full Query Operators

All operators work on documents just like on vector metadata:

`$gt`, `$gte`, `$lt`, `$lte`, `$ne`, `$eq`, `$in`, `$nin`, `$contains`, `$regex`, `$exists`, `$and`, `$or`, `$not`, nested dot paths (`"address.city"`).

### CLI

```bash
# Insert a document
gitdb insert '{"name": "Alice", "age": 30}'

# Find with query operators
gitdb find --where '{"age": {"$gt": 25}}' --columns name,age

# Update matching documents
gitdb update --where '{"name": "Alice"}' --set '{"age": 31}'

# Delete matching documents
gitdb delete --where '{"role": "intern"}'

# Count matching documents
gitdb count --where '{"role": "engineer"}'
```

Documents are staged and committed like everything else in GitDB. Branch your documents, merge them, time-travel through them.

## Embedding Models

| Model | Params | Dimensions | Matryoshka | Use Case |
|-------|--------|------------|------------|----------|
| **Arctic Embed L** (default) | 335M | 1024 | No | Fast, lightweight, strong retrieval |
| **NV-EmbedQA-1B-v2** | 1.2B | 384-2048 | Yes | Max quality, configurable dimensions |

```bash
# Use Arctic (default)
gitdb add --text "document"

# Use NV-EmbedQA with Matryoshka
gitdb add --text "document" --embed-model nv-embed-qa

# Re-embed entire store with different model
gitdb re-embed --model nv-embed-qa -c

# List available models
gitdb embed list
```

### Matryoshka Dimensions (NV-EmbedQA)

Same model, different vector sizes. Trade storage for quality:

| Dim | Size per vector | 1M vectors | Quality loss |
|-----|----------------|------------|-------------|
| 384 | 1.5 KB | 1.5 GB | ~5% |
| 512 | 2 KB | 2 GB | ~3% |
| 768 | 3 KB | 3 GB | ~1.5% |
| 1024 | 4 KB | 4 GB | ~0.5% |
| 2048 | 8 KB | 8 GB | Baseline |

## Emirati AC (Spreading Activation)

Named after how Emiratis leave their car AC running so it's pre-chilled when they walk out. The GPU continuously pattern-matches your operations against the vector store. Vectors activate neighboring vectors through multi-hop association chains.

```python
db.ac.start()  # Engine on

# As you browse, AC silently activates related vectors:
db.select(where={"type": "nda"})           # AC notices: NDA interest
# → activates confidentiality vectors      # 1 hop
# → activates IP protection vectors        # 2 hops
# → activates trade secret policies        # 3 hops

# Query — results already warm
results = db.ac.primed(10)  # Instant. Zero search time.

# Or boost normal queries with activation levels
results = db.query_text("data protection")  # AC-boosted ranking

# Drift detection
db.add(texts=["pizza menu", "catering prices"])
db.ac.drift()
# → {"magnitude": 0.42, "severity": "medium"}
# Recent additions diverging from branch centroid
```

**Tuning knobs**: `POLL_INTERVAL`, `ACTIVATION_DECAY`, `SPREAD_FACTOR`, `NEIGHBOR_K`, `HOT_CACHE_SIZE` — all configurable on `db.ac`.

## Hooks (FoundationDB-inspired)

Pre/post event hooks for commit, merge, push, and drift. Pre-hooks can reject operations.

```python
# Reject commits without metadata
def require_metadata(**ctx):
    db = ctx["db"]
    for meta in db.tree.metadata:
        if not meta.metadata:
            return False
    return True

db.hook("pre-commit", require_metadata)
db.hook("post-commit", lambda **ctx: print(f"Committed: {ctx['commit_hash'][:8]}"))
db.hook("pre-merge", lambda **ctx: ctx["branch"] != "protected")
db.hook("on-drift", lambda **ctx: alert(ctx["magnitude"]))

# List / clear
db.hooks.list_hooks()
db.hooks.clear("pre-commit")
```

**Events**: `pre-commit`, `post-commit`, `pre-merge`, `post-merge`, `pre-push`, `post-push`, `on-drift`

## Transactions

Atomic multi-operation batches. All succeed or all roll back.

```python
with db.transaction() as tx:
    tx.add(texts=["new document"])
    tx.remove(where={"status": "deprecated"})
    tx.update_embeddings([0, 1], new_embeddings)
    # If anything raises, everything rolls back
```

## Watches (Subscriptions)

Subscribe to changes on branches or metadata patterns. Callbacks fire on commit, push, or pull.

```python
# Watch a branch
wid = db.watches.watch_branch("production", lambda event, ctx: notify(ctx))

# Watch metadata patterns
wid = db.watches.watch(
    {"category": "finance"},
    lambda event, ctx: audit_log(ctx)
)

# List / unwatch
db.watches.list_watches()
db.watches.unwatch(wid)
```

## Secondary Indexes

Hash indexes for exact lookups, range indexes for min/max queries. Rebuilt on load, updated incrementally.

```python
# Create indexes
db.create_index("category")                  # Hash index (default)
db.create_index("score", index_type="range") # Range index

# Fast lookups (no scan)
results = db.indexes.lookup("category", "finance")    # O(1) exact match
results = db.indexes.range_lookup("score", 0.8, 1.0)  # O(log n) range

# List / drop
db.list_indexes()
db.drop_index("category")
```

## Schema Enforcement

JSON Schema validation on metadata. Rejects bad data at `add()` time.

```python
db.set_schema({
    "required": ["category", "source"],
    "properties": {
        "category": {"type": "string", "enum": ["finance", "legal", "hr"]},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "source": {"type": "string", "minLength": 1},
        "tags": {"type": "array"},
    },
    "additionalProperties": False,
})

# This works:
db.add(texts=["doc"], metadata=[{"category": "finance", "source": "bloomberg", "score": 0.9}])

# This raises SchemaError:
db.add(texts=["doc"], metadata=[{"category": "invalid"}])

# Validate existing data
db.get_schema()
db.set_schema(None)  # Clear
```

**Supports**: `type` (string/number/integer/boolean/array/object/null), `required`, `enum`, `minimum`/`maximum`, `minLength`/`maxLength`, `pattern`, `additionalProperties`.

## Snapshots

Cheap read-only frozen views. Cloned tensor, referenced metadata. In-memory only.

```python
snap = db.snapshot("before_experiment")

# Mutate the database freely...
db.add(texts=["experimental data"])
db.remove(where={"old": True})

# Query the snapshot — unchanged
results = snap.query(vector, k=10)
results = snap.query_text("revenue", k=5)
rows = snap.select(where={"category": "finance"})

# List snapshots
db.snapshots()
```

## Native Backup & Restore

Full and incremental backups. tar.zst format. No external tools needed.

```python
# Full backup
manifest = db.backup("my_store.gitdb-backup")

# Incremental backup (only new objects since last backup)
manifest = db.backup_incremental("my_store_incr.gitdb-incr")

# List backups
db.backup_list()

# Verify store integrity
result = db.backup_verify()
# → {"valid": True, "commits_verified": 42, "total_objects_on_disk": 156}

# Restore
from gitdb.backup import restore
restore("my_store.gitdb-backup", "/path/to/dest")
```

```bash
gitdb backup full -o my_store.gitdb-backup
gitdb backup incremental -o my_store_incr.gitdb-incr
gitdb backup list
gitdb backup verify
gitdb restore my_store.gitdb-backup /path/to/dest
```

**Backup format**: `.gitdb-backup` (full) / `.gitdb-incr` (incremental) — tar archives compressed with zstd. Each backup includes a JSON manifest sidecar with checksums, object counts, branch refs, and timestamps.

## CRUSH Algorithm (CEPH-style Placement)

Deterministic data placement across shards. No central lookup table — any node computes where any vector lives.

```python
from gitdb.crush import CRUSHMap, CRUSHRouter

# Define cluster topology
crush = CRUSHMap()
crush.add_device("ssd_1", weight=2.0, location={"host": "node1", "rack": "rack1"})
crush.add_device("ssd_2", weight=2.0, location={"host": "node1", "rack": "rack1"})
crush.add_device("ssd_3", weight=1.0, location={"host": "node2", "rack": "rack2"})
crush.add_device("ssd_4", weight=1.0, location={"host": "node2", "rack": "rack2"})

router = CRUSHRouter(crush)

# Where should this vector go? Deterministic — same answer every time.
router.place("vector_abc123", replicas=2)
# → ["ssd_1", "ssd_3"]  (different racks for fault tolerance)

# Add a node — only ~1/N data moves
crush.add_device("ssd_5", weight=1.0, location={"host": "node3", "rack": "rack3"})
plan = router.rebalance_plan(old_map)
# → {"moves": 200, "total": 1000, "percent": 20.0}

# Check distribution uniformity
router.distribution(sample_keys, replicas=2)
# → {"ssd_1": 510, "ssd_2": 498, "ssd_3": 251, "ssd_4": 241}  (weight-proportional)
```

**Straw2 bucket selection** — the same algorithm CEPH uses. Deterministic, weight-proportional, minimal disruption on topology changes.

## P2P Distributed Mode

Every node is equal. No coordinator. CRUSH routes vectors, SSH transports them, git-style merge handles conflicts.

```python
from gitdb import GitDB
from gitdb.distributed import DistributedGitDB, Peer

db = GitDB("my_shard", dim=1024, device="mps")
ddb = DistributedGitDB(db, self_name="laptop")

# ── Add peers (SSH transport, like git remotes) ────────
ddb.add_peer(Peer("spark", "xentureon@100.89.91.91",
                   shard_path="~/stores/vectors",
                   weight=2.0,  # DGX gets 2x the data
                   location={"host": "spark", "rack": "home"}))

ddb.add_peer(Peer("cloud", "ubuntu@ec2.example.com",
                   shard_path="/data/vectors",
                   weight=1.0,
                   location={"host": "cloud", "rack": "aws"}))

# ── Add vectors — CRUSH routes to correct shards ──────
ddb.add(texts=["quarterly report", "NDA draft"], replicas=2)
ddb.commit("Add documents")

# ── Query — scatter to all shards, merge top-k ────────
results = ddb.query_distributed(vector, k=10)

# ── Sync — push outbox, pull updates ──────────────────
ddb.sync()                  # All peers
ddb.sync("spark")           # Just Spark

# ── Rebalance — when topology changes ─────────────────
ddb.add_peer(Peer("gpu_box", "user@gpu.local", weight=3.0))
plan = ddb.rebalance(dry_run=True)
# → {"foreign_count": 150, "should_migrate": {"gpu_box": [...]}}
ddb.rebalance(dry_run=False)  # Actually queue migrations

# ── Gossip — peers discover each other ────────────────
other_topology = other_node.gossip_map()
ddb.apply_gossip(other_topology)  # Merges peer lists

# ── Status ────────────────────────────────────────────
ddb.status()
# → {"self": "laptop", "peers_total": 3, "peers_active": 2,
#    "local_vectors": 5000, "pending_outbox": 42}
```

```
Node A (laptop)          Node B (Spark)         Node C (cloud)
┌─────────────┐         ┌─────────────┐        ┌─────────────┐
│ GitDB shard │◄──SSH──►│ GitDB shard │◄──SSH──►│ GitDB shard │
│ CRUSH map   │  push/  │ CRUSH map   │  push/  │ CRUSH map   │
│ Peer list   │  pull   │ Peer list   │  pull   │ Peer list   │
└─────────────┘         └─────────────┘        └─────────────┘
     All nodes have the same CRUSH map.
     Any node can route any vector. No coordinator.
```

**How it works:**
- **Add**: CRUSH computes placement → vectors stored locally + queued in outbox for remote peers
- **Query**: scatter query to all active peers via SSH → merge + dedup results → return top-k
- **Sync**: push outbox via existing git remotes → pull updates from peers
- **Join**: new node gets CRUSH map → rebalance moves ~1/N data
- **Leave**: mark node down → replicas re-placed to surviving nodes
- **Conflict**: git-style branch/merge semantics handle divergence

## Universal Ingest

One command to swallow entire databases. SQLite, MongoDB, CSV, Parquet, PDF, text — auto-detected, auto-chunked, auto-committed.

```python
from gitdb import GitDB

db = GitDB("my_store", dim=1024, device="mps")

# Swallow an entire SQLite database
db.ingest("legacy.db")
# → Reads all tables, auto-detects text columns, embeds them, preserves all columns as metadata

# Import MongoDB dump (JSON, JSONL, or BSON)
db.ingest("users.jsonl")
# → Handles $oid, $date, $numberLong; flattens nested objects with dot notation

# CSV / TSV
db.ingest("data.csv")                          # Auto-detect delimiter and text column
db.ingest("data.tsv", text_column="content")   # Or specify it

# Parquet
db.ingest("warehouse.parquet")

# PDF — sentence-boundary-aware chunking
db.ingest("report.pdf", chunk_size=500, chunk_overlap=50)

# Plain text / Markdown
db.ingest("notes.md")

# Entire directory — batch ingest everything
db.ingest("./documents/", pattern="*.pdf", recursive=True)

# Cloud storage — S3, MinIO, GCS, Azure, SFTP
db.ingest("s3://my-bucket/data/")
db.ingest("gs://my-bucket/embeddings/")
db.ingest("az://my-container/exports/")
db.ingest("minio://localhost:9000/bucket/prefix/")
db.ingest("sftp://user@host/path/to/files/")
```

```bash
# CLI — local files
gitdb ingest legacy.db                       # SQLite
gitdb ingest dump.jsonl                      # MongoDB
gitdb ingest data.csv --text-column content  # CSV with explicit text column
gitdb ingest report.pdf --chunk-size 1000    # PDF with custom chunking
gitdb ingest ./docs/ --pattern "*.md"        # Directory batch
gitdb ingest data.tsv --no-commit            # Ingest without auto-commit

# CLI — cloud storage
gitdb ingest s3://my-bucket/data/            # S3
gitdb ingest gs://my-bucket/prefix/          # Google Cloud Storage
gitdb ingest az://container/path/            # Azure Blob
gitdb ingest minio://host:9000/bucket/       # MinIO
gitdb ingest sftp://user@host/path/          # SFTP
```

| Format | Extensions / Schemes | Features |
|--------|---------------------|----------|
| SQLite | `.db`, `.sqlite`, `.sqlite3` | All tables, auto-detect text columns, preserve all metadata |
| MongoDB | `.json`, `.jsonl`, `.bson` | Extended JSON (`$oid`, `$date`), nested object flattening |
| CSV/TSV | `.csv`, `.tsv`, `.tab` | Auto-detect delimiter + text column, type inference |
| Parquet | `.parquet`, `.pq` | Via pyarrow, auto-detect text column |
| PDF | `.pdf` | Via pymupdf/PyPDF2, sentence-boundary chunking with overlap |
| Text | `.txt`, `.md`, `.rst`, `.log` | Chunk with configurable size and overlap |
| S3 | `s3://` | boto3, recursive prefix listing, auto-detect file types |
| GCS | `gs://` | google-cloud-storage, recursive blob listing |
| Azure Blob | `az://` | azure-storage-blob, connection string auth |
| MinIO | `minio://` | S3-compatible with custom endpoint |
| SFTP | `sftp://` | paramiko, recursive directory listing, key or password auth |

## Web Dashboard

```bash
gitdb serve
```

Opens a browser at `http://localhost:7474`. Dark theme. No dependencies.

Semantic search bar. Commit timeline with branch labels. Branch sidebar. AC heatmap with live auto-refresh. Structured query builder. Stats footer.

Keyboard shortcut: `/` focuses the search bar.

```bash
gitdb serve --port 8080           # Custom port
gitdb serve --no-browser          # Don't auto-open browser
```

## Semantic Git Operations

Git operations that understand meaning, not just hashes.

```bash
# Cherry-pick by concept (not by commit hash)
gitdb semantic-cherry-pick feature "authentication"
# → Finds and picks commits whose vectors match "authentication"

# Blame by concept
gitdb semantic-blame "toxic content" --threshold 0.5
# → Shows which commits introduced semantically similar vectors

# Re-embed on rebase (upgrade vectors when rebasing)
db.rebase("main")
db.re_embed(embed_model="nv-embed-qa")
db.commit("Rebased + re-embedded")
```

## CLI Reference

```
gitdb init <name> [--dim N] [--device D]     Create store
gitdb status                                  Working tree status
gitdb add <file> | --text T | --text-file F   Add vectors
gitdb commit -m "message"                     Commit changes
gitdb log [-n N]                              Show history
gitdb diff <ref_a> <ref_b>                    Compare refs
gitdb query --text T | --vector F [-k N]      Search
gitdb select [--where JSON] [--fields F]      Structured query
gitdb group-by <field> [--agg-fn F]           Aggregation
gitdb branch [name]                           List/create branch
gitdb switch <branch>                         Switch branch
gitdb merge <branch> [--strategy S]           Merge
gitdb cherry-pick <ref>                       Cherry-pick commit
gitdb revert <ref>                            Revert commit
gitdb stash [save|pop|list]                   Stash changes
gitdb tag <name>                              Tag current HEAD
gitdb reset [ref]                             Reset to ref
gitdb blame                                   Vector provenance
gitdb reflog                                  HEAD movement log
gitdb rebase <onto>                           Rebase branch
gitdb gc [--keep N]                           Garbage collect
gitdb purge --where JSON --reason R           Purge from history
gitdb filter-branch <transform>               Transform all vectors
gitdb show [ref]                              Show commit details
gitdb amend [-m "msg"]                        Amend last commit
gitdb squash <N> [-m "msg"]                   Squash N commits
gitdb fork <dest>                             Fork/clone store
gitdb clone <dest>                            Clone store
gitdb note [-m "msg"]                         Notes on commits
gitdb comment [-m "msg"]                      Comments on commits
gitdb head                                    Show HEAD
gitdb remote [add|remove|list]                Manage remotes
gitdb push <remote> [branch]                  Push to remote
gitdb pull <remote> [branch]                  Pull from remote
gitdb fetch <remote> [branch]                 Fetch from remote
gitdb pr [create|list|show|merge|close]       Pull requests
gitdb export <file> [--embeddings]            Export JSONL
gitdb import <file> [--embed]                 Import JSONL
gitdb embed [list|text|info]                  Embedding tools
gitdb re-embed [--model M]                    Re-embed all vectors
gitdb semantic-blame <query>                  Blame by concept
gitdb semantic-cherry-pick <branch> <query>   Cherry-pick by meaning
gitdb ac [status|primed|drift|start|stop]     Emirati AC engine
gitdb backup [full|incremental|list|verify]   Backup operations
gitdb restore <backup> <dest> [--force]       Restore from backup
gitdb snapshot [create|list|query]            In-memory snapshots
gitdb hook [list|clear]                       Event hooks
gitdb watch [list|clear]                      Change watches
gitdb index [create|drop|list|lookup]         Secondary indexes
gitdb schema [show|set|clear|validate]        Schema enforcement
gitdb insert <json>                           Insert document
gitdb find [--where JSON] [--columns C]       Find documents
gitdb update --where JSON --set JSON          Update documents
gitdb delete --where JSON                     Delete documents
gitdb count [--where JSON]                    Count documents
gitdb ingest <file|dir> [--text-column C]     Universal ingest
gitdb serve [--port N] [--no-browser]        Web dashboard
```

## Command Reference

Every command with examples. Python API and CLI side by side.

---

### init — Create a new store

```python
db = GitDB("my_store", dim=1024, device="mps")
```

```bash
gitdb init my_store --dim 1024 --device mps
```

Creates a `.gitdb/` directory with config, HEAD, refs. Default dimension is 1024, default device is cpu.

---

### add — Add vectors

```python
# From text (auto-embeds with Arctic Embed)
db.add(texts=["quarterly revenue report", "employee handbook"])

# From embeddings directly
db.add(embeddings=tensor, documents=["doc1", "doc2"], metadata=[{"dept": "finance"}])

# From file
db.add(embeddings=torch.load("vectors.pt"))
```

```bash
gitdb add --text "quarterly revenue report" "employee handbook"
gitdb add vectors.pt --documents docs.json
gitdb add --text-file corpus.txt
```

Vectors are staged, not committed. Like `git add`.

---

### remove — Remove vectors

```python
db.remove(ids=["a3f1b2c4"])
db.remove(where={"dept": "finance"})
```

Stages removals. Takes effect on commit.

---

### commit — Commit staged changes

```python
hash = db.commit("Add Q3 data")
# → "a3f1b2c4d5e6..."
```

```bash
gitdb commit -m "Add Q3 data"
```

Creates a delta (only what changed), hashes it, stores it as a content-addressed object. Returns the commit hash.

---

### log — Show commit history

```python
history = db.log(limit=10)
for entry in history:
    print(entry.hash[:8], entry.message, entry.stats)
```

```bash
gitdb log -n 10
# a3f1b2c4 Add Q3 data (+50 -0 ~3)
# b7e2d3f5 Initial import (+500 -0 ~0)
```

---

### status — Working tree status

```python
s = db.status()
# → {"branch": "main", "staged_additions": 5, "staged_deletions": 0, "total_vectors": 500}
```

```bash
gitdb status
# On branch main
# Staged: 5 additions, 0 deletions
# Total vectors: 500
```

---

### query — Vector similarity search

```python
results = db.query(vector, k=10)
results = db.query_text("financial documents", k=5)
results = db.query_text("revenue", k=10, where={"dept": "finance"})

# Time-travel: search an old snapshot
results = db.query_text("revenue", k=5, at="v1.0")
```

```bash
gitdb query --text "financial documents" -k 5
gitdb query --vector query.pt -k 10
```

Returns ids, scores, documents, metadata. Scores are cosine similarity (1.0 = identical).

---

### select — Structured query (like SQL)

```python
rows = db.select(where={"dept": "finance", "value": {"$gt": 100000}}, fields=["document", "value"])
```

```bash
gitdb select --where '{"dept": "finance"}' --fields document,value --format table
```

No vectors involved. Pure metadata query with 13 operators.

---

### group-by — Aggregation

```python
groups = db.group_by("dept", agg_fn="count")
# → {"finance": 50, "engineering": 120, "legal": 30}
```

```bash
gitdb group-by dept --agg-fn count
gitdb group-by category --agg-fn avg --agg-field score
```

Supports count, sum, avg, min, max.

---

### diff — Compare two refs

Real `git diff` output. Shows every added/removed/modified vector with document, metadata, and cosine similarity.

```bash
gitdb diff main experiment
# diff --gitdb main/a3f1b2c4 experiment/a3f1b2c4
# new vector a3f1b2c4...
# --- /dev/null
# +++ experiment/a3f1b2c4
# @@ -0,0 +1 @@
# +document: quarterly revenue report
# +metadata: {"dept":"finance"}
#
# diff --gitdb main/c8d3e4f6 experiment/c8d3e4f6
# modified vector c8d3e4f6... (cosine similarity: 0.7234)
# --- main/c8d3e4f6
# +++ experiment/c8d3e4f6
# @@ -1,2 +1,2 @@
# -document: old projections model
# +document: updated projections
# -metadata: {"version":1}
# +metadata: {"version":2}
```

```python
diff = db.diff("main", "experiment")
# → Diff(+2 -0 ~1)

# Git-style unified format
print(diff.unified("main", "experiment"))

# Programmatic access
for entry in diff.entries:
    print(entry.change, entry.id[:8], entry.document, entry.similarity)

# Summary view
print(diff.show())
```

Same format as `git diff`. Green for added, red for removed, yellow/cyan for modified. Cosine similarity on modified vectors tells you how much the embedding actually moved.

---

### branch — Create or list branches

```python
db.branch("experiment")
branches = db.branches()
# → {"main": "a3f1...", "experiment": "a3f1..."}
```

```bash
gitdb branch experiment
gitdb branch              # list all
```

---

### switch — Switch branches

```python
db.switch("experiment")
```

```bash
gitdb switch experiment
```

Reconstructs the working tree from the target branch's HEAD.

---

### merge — Merge branches

```python
result = db.merge("experiment", strategy="union")
# → Merge(a3f1b2c4 | union +2 -0)

# Full merge summary with diff
print(result.show())
# Merge commit a3f1b2c4... (union)
#   +2 vectors added
#
# diff --gitdb ours/a3f1b2c4 experiment/a3f1b2c4
# new vector a3f1b2c4...
# --- /dev/null
# +++ experiment/a3f1b2c4
# @@ -0,0 +1 @@
# +document: quarterly revenue report
# +metadata: {"dept":"finance"}

# Access the diff programmatically
for entry in result.diff.entries:
    print(entry.change, entry.document)

# Check conflicts
if result.has_conflicts:
    print(f"Conflicts: {result.conflicts}")
```

```bash
gitdb merge experiment --strategy union
# Merge commit a3f1b2c4 (union)
#   +2 vectors added
#
# diff --gitdb ours/a3f1b2c4 experiment/a3f1b2c4
# new vector a3f1b2c4...
# --- /dev/null
# +++ experiment/a3f1b2c4
# @@ -0,0 +1 @@
# +document: quarterly revenue report
# +metadata: {"dept":"finance"}
```

Strategies: `union` (combine all), `ours` (keep our modifications), `theirs` (keep their modifications). Three-way merge finds common ancestor automatically. Shows full git-style diff of what the merge brought in.

---

### cherry-pick — Apply a specific commit

```python
new_hash = db.cherry_pick("b7e2d3f5")
```

```bash
gitdb cherry-pick b7e2d3f5
```

Applies just the delta from that commit onto your current branch.

---

### semantic-cherry-pick — Cherry-pick by meaning

```python
db.semantic_cherry_pick("feature", "authentication")
# → Finds commits on "feature" whose vectors match "authentication" and picks them
```

```bash
gitdb semantic-cherry-pick feature "authentication"
```

---

### revert — Undo a commit

```python
new_hash = db.revert("b7e2d3f5")
```

```bash
gitdb revert b7e2d3f5
```

Creates a new commit that reverses the delta of the target commit.

---

### stash — Stash uncommitted changes

```python
db.stash("work in progress")
db.stash_pop()
stashes = db.stash_list()
```

```bash
gitdb stash save "work in progress"
gitdb stash pop
gitdb stash list
```

---

### tag — Tag a commit

```python
db.tag("v1.0")
db.tag("release-candidate", ref="abc123")
```

```bash
gitdb tag v1.0
```

---

### reset — Reset to a ref

```python
db.reset("v1.0")
db.reset("HEAD~3")
```

```bash
gitdb reset v1.0
```

Reconstructs working tree from the target ref. Discards staged changes.

---

### blame — Vector provenance

```python
entries = db.blame()
for e in entries:
    print(e.id, e.commit_hash[:8], e.message)
# → "vec_001 a3f1b2c4 Add Q3 data"
```

```bash
gitdb blame
# vec_001  a3f1b2c4  Add Q3 data        2026-03-15
# vec_002  b7e2d3f5  Initial import     2026-03-14
```

Shows which commit introduced each vector. Like `git blame` for embeddings.

---

### semantic-blame — Blame by concept

```python
entries = db.semantic_blame("toxic content", threshold=0.5)
```

```bash
gitdb semantic-blame "toxic content" --threshold 0.5
```

Finds which commits introduced vectors semantically similar to your query.

---

### reflog — HEAD movement log

```python
entries = db.reflog(limit=20)
```

```bash
gitdb reflog
# a3f1b2c4 → b7e2d3f5  switch: main → experiment
# b7e2d3f5 → c8d3e4f6  commit: Add Q3 data
```

Every HEAD movement, including switches, resets, merges.

---

### rebase — Rebase branch

```python
new_hashes = db.rebase("main")
```

```bash
gitdb rebase main
```

Replays your branch's commits on top of the target.

---

### gc — Garbage collect

```python
db.gc(keep_last=10)
```

```bash
gitdb gc --keep 10
```

Creates cached snapshots for faster reconstruction. Removes old cache entries.

---

### purge — Remove from all history

```python
db.purge(where={"author": "claude"}, reason="cleanup")
```

```bash
gitdb purge --where '{"author": "claude"}' --reason "cleanup"
```

Rewrites every commit in history. After purge, the target vectors never existed in any historical state. Irreversible.

---

### filter-branch — Transform all vectors

```python
db.filter_branch(lambda embeddings, metadata: (embeddings * 0.5, metadata))
```

```bash
gitdb filter-branch normalize
```

Applies a transformation function to every commit's vectors. Rewrites history.

---

### show — Show commit details

```python
info = db.show("abc123")
# → {"hash": "abc123...", "message": "...", "parent": "...", "stats": {...}}
```

```bash
gitdb show abc123
gitdb show HEAD
```

---

### amend — Amend last commit

```python
new_hash = db.amend(message="Better message")
```

```bash
gitdb amend -m "Better message"
```

---

### squash — Squash N commits

```python
new_hash = db.squash(3, message="Combined work")
```

```bash
gitdb squash 3 -m "Combined work"
```

---

### fork / clone — Copy a store

```python
new_db = db.fork("/path/to/copy")
new_db = db.fork("/path/to/copy", branch="experiment")
```

```bash
gitdb fork /path/to/copy
gitdb clone /path/to/copy
```

---

### note — Add notes to commits

```python
db.note("abc123", "This commit has a known issue")
notes = db.notes("abc123")
```

```bash
gitdb note -m "This commit has a known issue"
```

---

### remote — Manage remotes

```python
db.remote_add("origin", "/path/to/remote")
db.remote_add("spark", "ssh://xentureon@100.89.91.91/~/stores/vectors")
db.remote_remove("origin")
remotes = db.remotes()
```

```bash
gitdb remote add origin /path/to/remote
gitdb remote add spark ssh://xentureon@100.89.91.91/~/stores/vectors
gitdb remote list
gitdb remote remove origin
```

Supports local paths and SSH remotes.

---

### push — Push to remote

```python
result = db.push("origin")
result = db.push("spark", branch="experiment")
```

```bash
gitdb push origin
gitdb push spark experiment
```

Transfers objects + refs over SSH or local copy.

---

### pull — Pull from remote

```python
result = db.pull("origin")
result = db.pull("spark", branch="main")
```

```bash
gitdb pull origin
gitdb pull spark main
```

Fetches + merges in one step.

---

### fetch — Fetch without merging

```python
result = db.fetch("origin")
```

```bash
gitdb fetch origin
gitdb fetch spark experiment
```

---

### pr — Pull requests on vectors

```python
# Create
db.pr_create(title="Q3 forecast", source="experiment", target="main",
             description="New revenue model vectors")

# Review
prs = db.pr_list()
pr = db.pr_show(1)

# Comment
db.pr_comment(1, "reviewer", "Looks good, projections vectors improved")

# Merge or close
result = db.pr_merge(1)
db.pr_close(2)
```

```bash
gitdb pr create --title "Q3 forecast" --source experiment --target main
gitdb pr list
gitdb pr show 1
gitdb comment -m "LGTM"
gitdb pr merge 1
gitdb pr close 2
```

Full PR workflow for vectors. Create, review diffs, comment, merge or close. Same flow as GitHub PRs but for tensors.

---

### export / import — JSONL

```python
db.export_jsonl("data.jsonl", include_embeddings=True)
db.import_jsonl("data.jsonl", embed_texts=True)
```

```bash
gitdb export data.jsonl --embeddings
gitdb import data.jsonl --embed
```

---

### embed — Embedding tools

```bash
gitdb embed list                          # List available models
gitdb embed text "hello world"            # Embed text, print vector
gitdb embed info                          # Current model info
```

---

### re-embed — Re-embed all vectors

```python
db.re_embed(embed_model="nv-embed-qa")
```

```bash
gitdb re-embed --model nv-embed-qa
```

Re-embeds every vector with a different model. Useful after model upgrades.

---

### ac — Spreading activation engine

```python
db.ac.start()
db.ac.feed_vectors(query_vector)
primed = db.ac.primed(k=10)       # Hot vectors ready for instant recall
drift = db.ac.drift()             # Detect embedding drift
stats = db.ac.stats()
db.ac.stop()
```

```bash
gitdb ac start
gitdb ac status
gitdb ac primed -k 10
gitdb ac drift
gitdb ac stop
```

Background thread that spreads activation through your vectors. Feed it a query, it pre-activates related vectors across multi-hop chains. Makes frequent queries instant.

---

### backup — Backup operations

```python
manifest = db.backup("backup.gitdb-backup")
manifest = db.backup_incremental("backup.gitdb-incr")
backups = db.backup_list()
result = db.backup_verify()
```

```bash
gitdb backup full backup.gitdb-backup
gitdb backup incremental backup.gitdb-incr
gitdb backup list
gitdb backup verify
```

Full or incremental. tar.zst compressed. Includes manifest with checksums.

---

### restore — Restore from backup

```python
manifest = db.backup_restore("backup.gitdb-backup", overwrite=True)
```

```bash
gitdb restore backup.gitdb-backup /path/to/dest --force
```

---

### snapshot — Read-only frozen views

```python
snap = db.snapshot("before_experiment")
results = snap.query(vector, k=5)
rows = snap.select(where={"dept": "finance"})
```

```bash
gitdb snapshot create before_experiment
gitdb snapshot list
gitdb snapshot query before_experiment --text "revenue" -k 5
```

Zero-copy frozen view of the current state. Query it while the main store keeps changing.

---

### hook — Event hooks

```python
db.hook("pre-commit", lambda **ctx: validate(ctx))
db.hook("post-commit", lambda **ctx: notify(ctx["commit_hash"]))
db.hook("pre-merge", lambda **ctx: check_approval(ctx))
db.unhook("pre-commit", my_hook)
```

```bash
gitdb hook list
gitdb hook clear pre-commit
```

Events: pre-commit, post-commit, pre-merge, post-merge, on-drift. Pre-hooks return False to reject the operation.

---

### watch — Change subscriptions

```python
watch_id = db.watch(where={"dept": "finance"}, on_change=lambda delta: alert(delta))
watch_id = db.watch(branch="experiment", on_change=callback)
db.watches_list()
```

```bash
gitdb watch list
gitdb watch clear
```

Fires callback when matching vectors change.

---

### index — Secondary indexes

```python
db.create_index("dept", index_type="hash")
db.create_index("score", index_type="range")
results = db.index_lookup("dept", "finance")
```

```bash
gitdb index create dept --type hash
gitdb index create score --type range
gitdb index lookup dept finance
gitdb index list
gitdb index drop dept
```

Hash indexes for O(1) exact lookup. Range indexes for O(log n) range queries.

---

### schema — Schema enforcement

```python
db.set_schema({
    "required": ["dept", "author"],
    "properties": {
        "dept": {"type": "string", "enum": ["finance", "legal", "eng"]},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
    }
})
# Now db.add() rejects metadata that doesn't match
```

```bash
gitdb schema set schema.json
gitdb schema show
gitdb schema validate           # Check all existing rows
gitdb schema clear
```

JSON Schema subset. Validates on every add().

---

### ingest — Universal ingest

```python
db.ingest("legacy.db")                    # SQLite
db.ingest("dump.jsonl")                   # MongoDB
db.ingest("data.csv")                     # CSV
db.ingest("report.pdf")                   # PDF
db.ingest("./documents/")                 # Directory
db.ingest("s3://my-bucket/data/")         # S3
db.ingest("gs://my-bucket/prefix/")       # GCS
db.ingest("sftp://user@host/path/")       # SFTP
```

```bash
gitdb ingest legacy.db
gitdb ingest data.csv --text-column content
gitdb ingest report.pdf --chunk-size 1000
gitdb ingest ./docs/ --pattern "*.md"
gitdb ingest s3://my-bucket/data/
```

Auto-detect file type, auto-chunk text, auto-embed, auto-commit.

---

### serve — Web dashboard

```python
# From Python
from gitdb.server import start_server
start_server(db, port=7474)
```

```bash
gitdb serve
gitdb serve --port 8080 --no-browser
```

Dark theme browser UI at localhost:7474. Semantic search, commit graph, branch sidebar, AC heatmap, structured queries. No dependencies.

---

### transaction — Atomic operations

```python
with db.transaction() as tx:
    tx.add(texts=["new doc"])
    tx.remove(where={"old": True})
    # Either both happen or neither — rolls back on exception
```

All-or-nothing. Snapshots state on enter, restores on exception.

---

## Encryption

```python
# Option 1: Environment variable
os.environ["GITDB_KEY"] = EncryptionManager.generate_key().hex()
db = GitDB("my_store", dim=1024)  # Auto-detects key from env

# Option 2: Password-derived key
from gitdb import EncryptionManager
enc = EncryptionManager.from_password("my-secret-password")
db = GitDB("my_store", dim=1024, encryption=enc)

# Option 3: CLI
gitdb encrypt init          # Generate key to .gitdb/keyfile
gitdb encrypt status        # Check if encryption is enabled
```

AES-256-GCM. Every object, document, and table encrypted before hitting disk. Key from `GITDB_KEY` env, `GITDB_KEY_FILE`, `.gitdb/keyfile`, or password (PBKDF2, 600k iterations).

---

## REST API Server

```bash
gitdb serve --port 8080     # Launch REST API + web dashboard
```

```python
# Full CRUD API
POST   /api/vectors/add          # Add vectors
POST   /api/vectors/query        # Semantic search
POST   /api/docs/insert          # Insert documents
POST   /api/docs/find            # MongoDB-style query
POST   /api/tables/{name}/insert # Insert table rows
POST   /api/tables/{name}/select # SQL-style SELECT

# Git operations
POST   /api/commit               # Commit changes
POST   /api/branch               # Create branch
POST   /api/merge                # Merge branches
GET    /api/log                  # Commit history
GET    /api/status               # Store status
GET    /api/branches             # List branches
```

Threaded HTTP server with web dashboard. All three data types (vectors, docs, tables) and all git operations exposed via REST.

---

## gRPC / Protobuf

```bash
gitdb grpc-serve --port 50051   # Launch gRPC server
```

```python
from gitdb.grpc_service import GitDBClient

client = GitDBClient("localhost:50051")
client.insert({"name": "Alice", "age": 30})
client.find(where={"age": {"$gt": 25}})
client.commit("added users")
```

Full Protobuf service definition with streaming support. High-performance alternative to REST for production deployments.

---

## Storage Backends

```python
from gitdb.storage import parse_storage_uri, copy_between

# Local (default)
store = parse_storage_uri("/path/to/store")

# S3
store = parse_storage_uri("s3://my-bucket/gitdb-data")

# MinIO
store = parse_storage_uri("minio://localhost:9000/mybucket/data")

# Google Cloud Storage
store = parse_storage_uri("gs://gcs-bucket/prefix")

# Azure Blob
store = parse_storage_uri("az://mycontainer/prefix")

# SFTP
store = parse_storage_uri("sftp://user@host.com/data/store")

# Migrate between backends
copy_between(local_store, s3_store, prefix="objects/")
```

Pluggable storage with `copy_between()` for zero-downtime migration.

---

## Fully Homomorphic Encryption (FHE)

Three tiers of encrypted computation. Query encrypted data without ever decrypting it. All GPU-accelerated via `torch.fft`.

### Tier 1: Searchable Encryption (fast, production-ready)

```python
from gitdb import SearchableEncryption

se = SearchableEncryption(key=os.urandom(32))

# Encrypt metadata — server can still query it
encrypted = se.encrypt_row(
    {"clearance": "TS/SCI", "priority": 9, "content": "classified"},
    field_types={"clearance": "equality", "priority": "range", "content": "exact"}
)

# Equality match on ciphertext — no decryption needed
se.match_equality(se.encrypt_field("TS/SCI"), encrypted["clearance"])  # True

# Range query on ciphertext — order preserved
se.compare_range(encrypted_a["priority"], encrypted_b["priority"])  # -1, 0, 1
```

### Tier 2: Private Information Retrieval (PIR)

```python
from gitdb import PIRClient, PIRServer

# Server holds the database — never learns what client queries
server = PIRServer(database_tensor)  # GPU-resident

# Client queries row 42 — server sees random noise
client = PIRClient(n_items=1000, item_dim=1024, device="mps")
query, noise = client.create_query(target_index=42)
response = server.respond(query)  # GPU matmul
row_42 = client.extract_result(response, noise, database_tensor)
```

### Tier 3: Full FHE — Encrypted Vector Search + Encrypted Git

```python
from gitdb import FHEScheme, EncryptedVectorStore, EncryptedGitOps

# 128-bit security, GPU-accelerated RLWE
fhe = FHEScheme(security_level=128, device="mps")  # or "cuda"
fhe.keygen()  # generates secret key, public key, relin key, galois keys

# Encrypted vector store — similarity via multiply + rotate-and-sum
store = EncryptedVectorStore(fhe)
store.add_encrypted(vector_a)
store.add_encrypted(vector_b)

# Query without decrypting — real FHE inner product
encrypted_scores = store.encrypted_query(query_vector, k=10)

# Homomorphic math — addition AND multiplication
ct_sum = fhe.add(enc_a, enc_b)       # enc(a + b)
ct_prod = fhe.multiply(enc_a, enc_b) # enc(a × b) with relinearization

# Encrypted git operations — version control on ciphertext
ops = EncryptedGitOps(fhe)
ct_diff = ops.encrypted_diff(enc_v1, enc_v2)           # what changed?
merged = ops.encrypted_three_way_merge(base, ours, theirs) # merge branches
commit = ops.encrypted_commit_hash(enc_v1)             # content-address
```

The GPU advantage: FHE's bottleneck is polynomial multiplication (NTT ≈ FFT). `torch.fft.fft` runs on CUDA/MPS. First vector DB with GPU-accelerated FHE + encrypted git + encrypted SQL.

### Tier 4: Encrypted Database Operations

```python
from gitdb import FHEScheme, EncryptedDBOps, SearchableEncryption
import os

key = os.urandom(32)
se = SearchableEncryption(key)
fhe = FHEScheme(security_level=128)
fhe.keygen()
db = EncryptedDBOps(fhe, se)

# Encrypt rows with per-column encryption types
rows = [
    se.encrypt_row({"name": "Alice", "dept": "eng", "salary": 150000},
                   {"name": "equality", "dept": "equality", "salary": "range"}),
    se.encrypt_row({"name": "Bob", "dept": "sales", "salary": 120000},
                   {"name": "equality", "dept": "equality", "salary": "range"}),
    se.encrypt_row({"name": "Carol", "dept": "eng", "salary": 180000},
                   {"name": "equality", "dept": "equality", "salary": "range"}),
]

# WHERE — filter on encrypted fields
eng = db.encrypted_where_eq(rows, "dept", "eng")          # [0, 2]
high_pay = db.encrypted_where_range(rows, "salary", "gt", 140000)  # [0, 2]

# JOIN — combine encrypted tables
orders = [se.encrypt_row({"name": "Alice", "item": "GPU"}, {"name": "equality", "item": "exact"})]
pairs = db.encrypted_join(rows, orders, "name")            # [(0, 0)]

# GROUP BY — group on encrypted keys
groups = db.encrypted_group_by(rows, "dept")               # {enc("eng"): [0,2], enc("sales"): [1]}

# DEDUP — find unique rows on encrypted field
unique = db.encrypted_dedup(rows, "name")                  # [0, 1, 2]

# Constraint checks — all on ciphertext
db.encrypted_check_unique(rows, "name")                    # True
db.encrypted_check_foreign_key(orders, "name", rows, "name")  # [] (no violations)

# Backup verification — compare encrypted snapshots
diff = db.encrypted_snapshot_diff(snapshot_v1, snapshot_v2)
# {"same_count": 50, "changed_indices": [3, 17], "added": 2, "removed": 0}
```

The server executes every query — WHERE, JOIN, GROUP BY, DEDUP, constraint checks — without ever seeing a plaintext value.

### What can you actually DO with FHE?

In plain English: your data is encrypted. It stays encrypted. The server does math on the encrypted data and gives you encrypted results. You decrypt the results. At no point did the server see your actual data.

**What works today in GitDB:**

| Operation | How | Speed |
|-----------|-----|-------|
| **Search encrypted vectors** | `store.encrypted_query(query, k=10)` — cosine similarity via multiply + rotate-and-sum | 252 q/s (100 vecs) |
| **Equality queries on encrypted fields** | "Find all rows where clearance = TS/SCI" — without decrypting | 738,350 ops/s |
| **Range queries on encrypted fields** | "Find all rows where priority > 5" — order preserved in ciphertext | 5,795,982 ops/s |
| **Private retrieval** | "Give me row 42" — server returns it without knowing you asked for 42 | 19,128 q/s (1000 rows) |
| **Homomorphic addition** | `enc(a) + enc(b) → enc(a + b)` — sum encrypted values | ~20,000 ops/s |
| **Homomorphic multiplication** | `enc(a) × enc(b) → enc(a × b)` — multiply encrypted values | ~470 ops/s |
| **Encrypted diff** | Detect which vectors changed between commits — without decrypting | Homomorphic subtraction |
| **Encrypted merge** | Three-way merge on encrypted branches — conflict resolution via addition | Full merge pipeline |
| **Encrypted branch diff** | Per-vector diff between two encrypted branches | All on ciphertext |
| **Encrypted commit hash** | Content-address encrypted objects — SHA-256 of ciphertext | Instant |
| **Encrypted aggregation** | Sum/mean over encrypted vectors across branches | ~20,000 ops/s |
| **Encrypted WHERE (eq)** | `db.encrypted_where_eq(rows, "dept", "eng")` — filter on encrypted field | 10,500,000 ops/s |
| **Encrypted WHERE (range)** | `db.encrypted_where_range(rows, "salary", "gt", 100000)` | 5,800,000 ops/s |
| **Encrypted JOIN** | `db.encrypted_join(users, orders, "user_id")` — inner join on ciphertext | O(n×m) |
| **Encrypted GROUP BY** | `db.encrypted_group_by(rows, "dept")` — group by encrypted key | O(n×g) |
| **Encrypted DEDUP** | `db.encrypted_dedup(rows, "email")` — deduplicate on encrypted field | O(n²) |
| **Encrypted UNIQUE check** | `db.encrypted_check_unique(rows, "id")` — constraint on ciphertext | O(n²) |
| **Encrypted FK check** | `db.encrypted_check_foreign_key(children, "pid", parents, "id")` | O(n×m) |
| **Encrypted backup diff** | `db.encrypted_snapshot_diff(snap_a, snap_b)` — compare encrypted backups | 0.03 ms |

**Encrypted git operations — version control on data you cannot read:**

```python
from gitdb.fhe import FHEScheme, EncryptedGitOps

fhe = FHEScheme(security_level=128, device="cuda")
fhe.keygen()
ops = EncryptedGitOps(fhe)

# Encrypt your vectors
ct_v1 = fhe.encrypt(vector_1)
ct_v2 = fhe.encrypt(vector_2)

# Diff — detect changes without decrypting
ct_diff = ops.encrypted_diff(ct_v1, ct_v2)
changed = ops.decrypt_has_changed(ct_diff)  # only key holder knows

# Three-way merge — combine encrypted branches
merged = ops.encrypted_three_way_merge(base_branch, our_branch, their_branch)

# Branch diff — compare entire encrypted branches
diffs = ops.encrypted_branch_diff(branch_a, branch_b)

# Content-address encrypted objects
commit_hash = ops.encrypted_commit_hash(ct_v1)  # SHA-256 of ciphertext

# Aggregate across branches
total = ops.encrypted_aggregate([ct_v1, ct_v2, ct_v3], op="sum")
```

The server manages version history, merges branches, diffs commits — all on data it has never seen and cannot read. Every operation is homomorphic. Only the key holder can decrypt the results.

### Plaintext vs Encrypted — Benchmark Comparison

Same queries, same data. 128-bit security. How much does encryption cost?

```
Queries Per Second (log scale):

  Small (10 vectors × 64d):
    Plaintext ███████████████████████████████████   52,999 q/s
    PIR       █████████████████████████████████░░   31,361 q/s   (1.7× slower)
    FHE       ████████████████████████░░░░░░░░░░░    1,788 q/s   (30× slower)

  Medium (100 vectors × 256d):
    Plaintext ██████████████████████████████████░   46,666 q/s
    PIR       ███████████████████████████████████   62,289 q/s   (0.7× — FASTER)
    FHE       ████████████████░░░░░░░░░░░░░░░░░░░      199 q/s   (235× slower)

  Large (1000 vectors × 1024d):
    Plaintext ████████████████████████████░░░░░░░    3,342 q/s
    PIR       ███████████████████████████████████   19,128 q/s   (0.2× — 6× FASTER)
    FHE       █████████░░░░░░░░░░░░░░░░░░░░░░░░░░       14 q/s   (242× slower)

FHE Operation Latency (128-bit security, poly=4096):

  Operation          Latency      Throughput
  ─────────          ───────      ──────────
  Homo. add          0.05 ms      ~20,000 ops/s
  Encrypt+decrypt    0.90 ms       ~1,100 ops/s
  Homo. multiply     2.10 ms         ~470 ops/s   ← real FHE
  Relinearize        automatic (included in multiply)

Overhead Factor (Plaintext = 1×):

                    PIR         FHE
  Small (10×64d)    1.7×         30×
  Medium (100×256d) 0.7×        235×
  Large (1000×1024d)0.2×        242×

Device: MPS/CUDA/CPU — torch.fft, same code, any device
```

**Three things that stand out:**

1. **PIR is FASTER than plaintext at scale.** At 1000 vectors, PIR does 19,128 queries/sec vs plaintext's 3,342. The matmul-based retrieval is just an efficient GPU operation. You get privacy AND speed.

2. **FHE is 30-242× slower, not 1000-10000×.** Traditional FHE libraries (Microsoft SEAL, HElib) are thousands of times slower than plaintext. We use `torch.fft` for polynomial multiplication — the same FFT that already runs on your GPU. Our FHE runs wherever PyTorch runs.

3. **Homomorphic multiply at 470 ops/sec.** This is real FHE — addition AND multiplication on ciphertext. Production FHE libraries do multiply in 10-100ms. We do it in 2.1ms. That's what `torch.fft` buys you.

**When to use which tier:**

- **Searchable Encryption** — query encrypted metadata fast. Best for: compliance (HIPAA, GDPR), field-level access control. Speed: nearly free.
- **PIR** — the server never knows WHAT you searched for. Best for: classified queries, private analytics, whistleblower protection. Speed: faster than plaintext at scale.
- **Full FHE** — math on encrypted data. The server computes similarity, diffs branches, merges commits — without ever seeing a single plaintext value. Best for: multi-party computation, encrypted ML inference, zero-trust git. Speed: 30-242× overhead.
- **Encrypted Git Ops** — version control on encrypted data. Diff, merge, branch, commit — all on ciphertext. The server manages your repo's history without reading your data.

---

## Streaming Ingest

NSA-grade versioned streaming pipeline. Every chunk content-addressed, hash-chained, Merkle-verified, encrypted at rest.

```python
from gitdb import StreamIngest

with StreamIngest("my_store", dim=1024) as stream:
    # Multiple concurrent feeds — each shard gets its own branch + WAL
    sensor_a = stream.shard("sensor-a")
    sensor_b = stream.shard("sensor-b")

    # Ingest with full provenance
    sensor_a.ingest(data, source="building-7")
    sensor_b.ingest(data, source="building-12")

    # Auto-commits every N chunks or T seconds
    # Merkle root stored in every commit message

    # Merge all shards into main
    stream.merge_all()

    # Verify nothing was tampered with
    assert stream.verify()
```

**Components:**
- **WAL** — append-only, fsync'd, encrypted write-ahead log per shard. Survives crashes
- **Hash chain** — every chunk cryptographically links to its predecessor. Break the chain, we know
- **Merkle tree** — tamper-proof integrity with inclusion proofs. One bit changes, the root changes
- **Backpressure** — bounded buffers, slow consumers wait
- **Dedup** — same content = same hash = stored once
- **Crash recovery** — replay WAL from last committed sequence

**Use cases:** IoT sensor networks, financial trading audit trails (SEC/FINRA), medical device data streams (FDA chain of custody), evidence management, intelligence collection pipelines, training data ingestion with poisoning detection.

---

## ML / AI Integration

Training data management is the killer use case nobody's solved well.

**The problem today:** Every ML team has a "data mess" — training data lives in S3 buckets, gets modified by 5 people, nobody knows which version of the dataset produced which model. When a model starts hallucinating, you can't trace it back to the training data that caused it.

### 1. Training Data Versioning

```python
db = GitDB("training_data", dim=1024)
db.ingest("datasets/v1/")
db.commit("baseline training set")

# Experiment with filtered data
db.branch("no-toxic")
db.purge(where={"toxicity": {"$gt": 0.8}})
db.commit("removed toxic samples")

# Which dataset produced the better model? Diff them.
db.diff("main", "no-toxic")
```

Nobody else can do this. DVC comes close but it's file-level, not record-level. We diff individual training examples.

### 2. RAG Infrastructure

Every RAG system needs a vector DB. GitDB gives you:
- **Branch per user/tenant** — multi-tenant RAG without separate databases
- **Time-travel retrieval** — "answer this question using only knowledge from before March 2026"
- **Audit trail** — which documents were retrieved for which answer (compliance, regulated industries)

### 3. Embedding Management

Models get re-embedded constantly — new embedding model drops, you re-embed everything. GitDB tracks this:

```python
db.commit("arctic-embed-l embeddings")
# New model comes out
db.branch("nv-embedqa")
db.re_embed(model="nvidia/NV-EmbedQA")
db.commit("re-embedded with NV-EmbedQA")
# Compare retrieval quality between embedding versions
```

### 4. Dataset Curation / RLHF

- Human annotators work on branches (each annotator = branch)
- Merge reviewed annotations into main
- Cherry-pick high-quality examples from one annotator's work
- Full blame — who labeled what, when, and which labels got merged into the training set
- This is exactly what Scale AI, Surge AI, and Labelbox charge millions for, minus the versioning

### 5. Model Evaluation Store

Store eval results as documents/tables with git versioning:

```python
db.create_table("evals", {"model": "text", "benchmark": "text", "score": "float"})
db.insert_into("evals", {"model": "v3.5-mini", "benchmark": "MMLU", "score": 0.74})
db.commit("v3.5-mini eval run")
# Compare across model versions
db.diff("v3.5-mini-evals", "v4.0-evals")
```

### 6. Streaming Training Data

- Live data feeds (user interactions, feedback) streaming into shards
- Auto-versioned, auto-committed
- Merkle-verified — prove your training data wasn't poisoned
- This is a real concern: training data poisoning attacks. Merkle trees catch them

### 7. Feature Store

Tables with git versioning = versioned feature store:
- Branch for experimental features
- Merge when validated
- Time-travel to reproduce any historical model's exact feature set
- Every ML platform (Tecton, Feast) charges for this

### Where GitDB fits in the AI stack

| Need | Current Solution | GitDB |
|------|-----------------|-------|
| Vector search | Pinecone/Weaviate/Chroma | Yes + version control |
| Training data versioning | DVC (file-level only) | Record-level diffs |
| Dataset curation | Labelbox/Scale ($$$) | Branch per annotator, merge |
| Feature store | Tecton/Feast ($$$) | Tables + git versioning |
| RAG | Any vector DB | + time-travel + audit trail |
| Eval tracking | W&B/MLflow | Tables + branching |
| Data provenance | Nothing good | Merkle trees + hash chains |

**The pitch in one line:** GitDB is the version control layer that sits under your entire ML pipeline — training data, embeddings, features, evals, RAG — and gives you the same branch/merge/diff/blame workflow that git gave source code.

Nobody's done this. DVC versions files. MLflow tracks experiments. W&B tracks metrics. None of them version the actual data at the record level with branch/merge semantics.

---

## Architecture

```
24,000+ lines of Python across 31 modules. 708 tests.

┌──────────────────────────────────────────────────────────────┐
│                    GitDB v0.13.1                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  P2P Layer ──→ Distributed Cluster   ← P2P Network          │
│       │              │                  gossip, sync         │
│       │         distributed.py          scatter/gather query │
│       ▼              │                                       │
│  CRUSH ────→ Vector Placement        ← CEPH Algorithm       │
│       │              │                  straw2 selection     │
│       │         crush.py                deterministic routing│
│       ▼              │                                       │
│  JSONL/JSON ──→ Metadata Store       ← Structured DB        │
│       │              │                  select, group-by     │
│       │         structured.py           $gt $in $regex       │
│       ▼              │                                       │
│  Schema ────→ Validation             ← Schema Enforcement   │
│       │              │                  JSON Schema subset   │
│       │         schema.py               required, types,    │
│       ▼              │                  enum, bounds, regex  │
│  Embedding ──→ GPU Tensor Store      ← Vector DB            │
│       │              │                  cosine sim           │
│       │         working_tree.py         top-k search         │
│       ▼              │                                       │
│  Indexes ───→ Fast Lookups           ← Secondary Indexes    │
│       │              │                  hash + range indexes │
│       │         indexes.py              O(1) exact, O(logn) │
│       ▼              │                                       │
│  Delta Chain ──→ Version History     ← Git Engine            │
│       │              │                  branch, merge        │
│       │         objects.py              blame, bisect        │
│       ▼              │                  purge, push/pull     │
│  Hooks ────→ Event Callbacks         ← FoundationDB-style   │
│       │              │                  pre/post commit/merge│
│       │         hooks.py                reject operations    │
│       ▼              │                                       │
│  Watches ──→ Subscriptions           ← Change Notifications │
│       │              │                  branch + metadata    │
│       │         watches.py              pattern matching     │
│       ▼              │                                       │
│  Emirati AC ──→ Hot Cache            ← Spreading Activation │
│       │              │                  multi-hop chains     │
│       │         ambient.py              drift detection      │
│       ▼              │                                       │
│  Snapshots ──→ Frozen Views          ← Read-only Snapshots  │
│       │              │                  zero-copy metadata   │
│       │         snapshots.py            query, select        │
│       ▼              │                                       │
│  Backup ───→ tar.zst Archives        ← Native Backup        │
│       │              │                  full + incremental   │
│       │         backup.py               verify, restore     │
│       ▼              │                                       │
│  Ingest ───→ Universal Import         ← Auto-detect         │
│       │              │                  SQLite, MongoDB      │
│       │         ingest.py               CSV, Parquet, PDF   │
│       │              │                                       │
│  Cloud ────→ S3/GCS/Azure/MinIO/SFTP  ← Cloud Transport     │
│       │              │                  stream + ingest      │
│       │         cloud_ingest.py         temp + cleanup       │
│       ▼              │                                       │
│  Encryption ──→ AES-256-GCM           ← Transparent          │
│       │              │                  encrypt on write      │
│       │         encryption.py           decrypt on read       │
│       ▼              │                                       │
│  Storage ───→ Backend Abstraction      ← Pluggable            │
│       │              │                  Local, S3, GCS        │
│       │         storage.py              Azure, SFTP, MinIO   │
│       ▼              │                                       │
│  Dashboard ──→ REST API + Web UI       ← gitdb serve          │
│       │              │                  full CRUD API         │
│       │         server.py               commit graph, search │
│       ▼              │                                       │
│  gRPC ──────→ Protobuf Service         ← gitdb grpc-serve     │
│       │              │                  streaming, high-perf  │
│       │         grpc_service.py         client + server       │
│       ▼              │                                       │
│  StreamIngest ──→ Versioned Pipeline   ← NSA-grade            │
│       │              │                  WAL, Merkle, hash     │
│       │         streaming.py            chain, backpressure  │
│       ▼              │                                       │
│  FHE ────────→ Encrypted Compute       ← GPU-accelerated     │
│       │              │                  searchable, PIR       │
│       │         fhe.py                  RLWE via torch.fft   │
│       ▼              │                                       │
│  QUERY ──→ Boosted Results                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Modules (30):
  core.py          2,600+ lines  Main interface (70+ methods)
  cli.py           1,900+ lines  CLI (75+ commands)
  server.py        1,200+ lines  REST API + web dashboard
  grpc_service.py    700+ lines  gRPC/Protobuf server + client
  ingest.py          909 lines   Universal ingest (6 formats)
  distributed.py     765 lines   P2P distributed layer
  crush.py           600 lines   CEPH CRUSH algorithm
  documents.py       500+ lines  DocumentStore + TableStore
  ambient.py         459 lines   Emirati AC engine
  backup.py          457 lines   Native backup & restore
  remote.py          387 lines   Push/pull/fetch (local + SSH)
  cloud_ingest.py    349 lines   Cloud storage transport
  storage.py         300+ lines  Pluggable storage backends
  structured.py      322 lines   Query operators, aggregation
  delta.py           288 lines   Sparse tensor deltas + zstd
  objects.py         270 lines   Content-addressed object store
  fhe.py             310+ lines  GPU-accelerated FHE (searchable, PIR, RLWE)
  streaming.py       400+ lines  StreamIngest pipeline (WAL, Merkle, shards)
  encryption.py      200+ lines  AES-256-GCM encryption
  working_tree.py    208 lines   GPU-resident tensor + search
  embed.py           200 lines   Arctic + NV-EmbedQA
  pullrequest.py     197 lines   PR store with comments
  indexes.py         138 lines   Secondary indexes (hash + range)
  types.py           127 lines   Data types
  schema.py          125 lines   JSON Schema validation
  watches.py         123 lines   Change subscriptions
  snapshots.py       118 lines   Read-only frozen views
  hooks.py            78 lines   Event hook system
```

## What Nothing Else Can Do

| Capability | GitDB | Pinecone | Weaviate | Qdrant | ChromaDB | SQLite |
|-----------|-------|----------|----------|--------|----------|--------|
| Vector similarity search | Yes | Yes | Yes | Yes | Yes | No |
| Document store (MongoDB-style) | Yes | No | No | No | No | No |
| Table queries (SQLite-style) | Yes | No | No | No | No | Yes |
| Structured metadata queries | Yes | Limited | Yes | Yes | Yes | Yes |
| Version control (branch/merge) | Yes | No | No | No | No | No |
| Time-travel queries | Yes | No | No | No | No | No |
| Semantic blame/bisect | Yes | No | No | No | No | No |
| Cherry-pick by meaning | Yes | No | No | No | No | No |
| Purge from all history | Yes | No | No | No | No | No |
| Spreading activation | Yes | No | No | No | No | No |
| Drift detection | Yes | No | No | No | No | No |
| Pull requests on vectors | Yes | No | No | No | No | No |
| Auto-embed from text | Yes | No | Yes | No | Yes | No |
| Matryoshka dimensions | Yes | No | No | No | No | No |
| GPU-accelerated | Yes | Cloud | Cloud | Yes | No | No |
| Export/Import JSONL | Yes | No | Yes | Yes | Yes | No |
| Pre/post event hooks | Yes | No | No | No | No | No |
| Atomic transactions | Yes | No | No | No | No | Yes |
| Change watches/subscriptions | Yes | No | No | No | No | No |
| Secondary indexes | Yes | Cloud | Auto | Yes | No | Yes |
| Schema enforcement | Yes | No | Yes | No | No | Yes |
| Read-only snapshots | Yes | No | No | No | No | No |
| Native backup/restore | Yes | Cloud | Cloud | Yes | No | No |
| CRUSH placement algorithm | Yes | No | No | No | No | No |
| P2P distributed (no coordinator) | Yes | Cloud | Cloud | Cloud | No | No |
| Scatter/gather distributed query | Yes | Cloud | Cloud | Cloud | No | No |
| Gossip topology discovery | Yes | No | No | No | No | No |
| Universal ingest (6 formats) | Yes | No | No | No | No | No |
| Cloud ingest (S3/GCS/Azure/MinIO/SFTP) | Yes | No | No | No | No | No |
| Built-in web dashboard | Yes | Cloud | Cloud | Yes | No | No |
| Encryption at rest | Yes | Cloud | Cloud | No | No | Extension |
| REST API + gRPC | Both | REST | Both | REST | No | No |
| Pluggable storage backends | Yes | Cloud | Cloud | Cloud | No | No |
| Versioned streaming ingest | Yes | No | No | No | No | No |
| Merkle integrity proofs | Yes | No | No | No | No | No |
| FHE (encrypted computation) | Yes | No | No | No | No | No |
| Private information retrieval | Yes | No | No | No | No | No |

## LLM Integration

GitDB connects to LLMs via three pathways, depending on your stack.

### 1. MCP Server (Claude, Cursor, AI IDEs)

MCP (Model Context Protocol) is the standard for tool integration with Claude and AI-native IDEs. GitDB exposes its full API as MCP tools:

```python
# gitdb_mcp_server.py — minimal MCP server
from gitdb import GitDB
import json

# MCP tools map directly to GitDB methods:
# - gitdb_query(text, k, where) → db.query_text()
# - gitdb_add(texts) → db.add(texts=)
# - gitdb_select(where, fields) → db.select()
# - gitdb_log(limit) → db.log()
# - gitdb_diff(ref_a, ref_b) → db.diff()
# - gitdb_commit(message) → db.commit()
```

### 2. REST API (Ollama, LMStudio, any HTTP client)

Run GitDB as an HTTP server with OpenAI-compatible endpoints:

```python
# gitdb_server.py — REST API
from flask import Flask, request, jsonify
from gitdb import GitDB

app = Flask(__name__)
db = GitDB("my_store", dim=1024, device="mps")
db.ac.start()

@app.route("/v1/query", methods=["POST"])
def query():
    data = request.json
    results = db.query_text(data["text"], k=data.get("k", 10))
    return jsonify({
        "results": [{"document": d, "score": s, "metadata": m}
                     for d, s, m in zip(results.documents, results.scores, results.metadata)]
    })

@app.route("/v1/add", methods=["POST"])
def add():
    data = request.json
    indices = db.add(texts=data["texts"], metadata=data.get("metadata"))
    return jsonify({"added": len(indices)})

@app.route("/v1/select", methods=["POST"])
def select():
    data = request.json
    rows = db.select(where=data.get("where"), fields=data.get("fields"))
    return jsonify({"rows": rows})

# OpenAI-compatible embeddings endpoint
@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    from gitdb.embed import embed
    data = request.json
    vectors = embed(data["input"])
    return jsonify({
        "data": [{"embedding": v.tolist(), "index": i} for i, v in enumerate(vectors)]
    })
```

### 3. Python SDK (LangChain, LlamaIndex, direct)

```python
# LangChain retriever
from gitdb import GitDB

class GitDBRetriever:
    def __init__(self, path, dim=1024):
        self.db = GitDB(path, dim=dim, device="mps")

    def get_relevant_documents(self, query, k=5):
        results = self.db.query_text(query, k=k)
        return [{"page_content": d, "metadata": m}
                for d, m in zip(results.documents, results.metadata)]
```

## JSONL Format

GitDB uses JSONL (JSON Lines) for import/export. One JSON object per line.

### Full format (with embeddings)

```json
{"id": "a3f1b2c4", "document": "NDA with Acme Corp", "metadata": {"type": "nda", "value": 500000}, "embedding": [0.123, -0.456, ...]}
{"id": "b7e2d3f5", "document": "MSA with Globex", "metadata": {"type": "msa", "value": 2000000}, "embedding": [0.789, 0.012, ...]}
```

### Text-only format (auto-embed on import)

```json
{"document": "NDA with Acme Corp", "metadata": {"type": "nda", "value": 500000}}
{"document": "MSA with Globex", "metadata": {"type": "msa", "value": 2000000}}
```

Import text-only JSONL with auto-embedding:

```bash
gitdb import data.jsonl --embed -c
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | No | Vector ID (auto-generated if omitted) |
| `document` | string | No | Raw text (used for embedding and display) |
| `metadata` | object | No | Arbitrary JSON metadata |
| `embedding` | float[] | No | Pre-computed embedding vector |

## On-Disk Structure

```
my_store/
  .gitdb/
    config                    # Store config (dim, device, version)
    HEAD                      # Current commit hash
    refs/
      heads/
        main                  # Branch refs
        feature
      tags/
        v1.0                  # Tag refs
    objects/
      a3/f1b2c4...           # Content-addressed objects (commits + deltas)
      b7/e2d3f5...           # Two-char prefix directories (git-style)
    cache/
      tensors/
        <hash>.pt             # Cached reconstructed tensors
        <hash>.meta.json      # Cached metadata
    remotes.json              # Remote configurations
    prs/
      index.json              # Pull request store
    stash/
      0.pt, 0.meta.json      # Stashed states
    backups/
      history.json            # Backup history
    distributed/
      peers.json              # P2P peer registry
      crush_map.json          # CRUSH topology + rules
      outbox/                 # Queued vectors for remote peers
        spark.jsonl
        cloud.jsonl
    reflog                    # Append-only HEAD movement log
    schema.json               # Schema definition (if set)
    notes/                    # Commit notes
    comments/                 # Commit comments
```

### Object Format

Commits and deltas are stored as content-addressed objects, hashed with SHA-256.

**Commit object** (JSON):
```json
{
  "parent": "abc123...",
  "parent2": null,
  "delta_hash": "def456...",
  "message": "Add Q2 data",
  "timestamp": 1710460800.0,
  "stats": {"added": 50, "removed": 0, "modified": 3},
  "tensor_rows": 500
}
```

**Delta object** (binary):
```
GDBD (magic bytes) + version byte + packed indices + tensor bytes + JSON metadata
Compressed with zstd level 3
```

Deltas are sparse: they only store what changed (additions, deletions, modifications), not the full tensor. A commit adding 50 vectors to a 10,000-vector store only stores 50 new vectors, not 10,050.

## Internals

### Delta Chain Reconstruction

To reconstruct state at any commit, GitDB walks back to the root and replays deltas forward:

```
root → delta1 → delta2 → delta3 → ... → HEAD
```

Cached snapshots short-circuit this walk. `gitdb gc` creates periodic checkpoints.

### Three-Way Merge

When merging branches that have diverged:

1. Find common ancestor (merge base)
2. Compute delta from ancestor → ours
3. Compute delta from ancestor → theirs
4. Apply merge strategy:
   - **union**: Combine all additions from both sides
   - **ours**: Keep our modifications, take their additions
   - **theirs**: Keep their modifications, take our additions
5. Detect conflicts: vectors modified in both branches

### Purge (History Rewriting)

`gitdb purge` rewrites every commit in history:

1. Walk every commit from root to HEAD
2. Filter every delta: remove target vectors from additions/modifications
3. Skip commits that become empty after filtering
4. Rehash everything (new content = new hash)
5. Update all branch refs and tags to new hashes
6. Invalidate all caches

After purge, the target vectors never existed in any historical state.

### Spreading Activation (Emirati AC)

Background thread architecture:

```
Every 3 seconds:
  1. Embed recent context (last 10 operations) → single vector
  2. GPU matmul against all vectors → direct activation scores
  3. Decay existing activations by 10%
  4. Activate top-20 matches (direct)
  5. Spread: each active vector activates its top-5 neighbors
     (weighted by cosine similarity × spread factor)
  6. Update hot cache (top 50 by activation level)
  7. Check drift (recent additions vs branch centroid)
```

Multi-hop chains emerge naturally:
```
"auth tokens" (query)
  → session_management (0.65)    ← direct hit
    → security_audit (0.42)      ← 1 hop
      → vulnerability_scan (0.31) ← 2 hops
```

## Configuration

### Store Config

```python
db = GitDB(
    "my_store",
    dim=1024,      # Embedding dimension
    device="mps",  # mps (Apple Silicon) / cuda (NVIDIA) / cpu
)
```

### Emirati AC Tuning

```python
db.ac.POLL_INTERVAL = 3.0       # Seconds between cycles
db.ac.ACTIVATION_DECAY = 0.90   # 10% decay per cycle
db.ac.SPREAD_FACTOR = 0.25      # Spread intensity
db.ac.SPREAD_MIN_SIM = 0.40     # Min similarity to spread
db.ac.NEIGHBOR_K = 5            # Neighbors per vector
db.ac.HOT_CACHE_SIZE = 50       # Hot cache capacity
db.ac.MIN_ACTIVATION = 0.05     # Deactivation threshold
db.ac.DIRECT_WEIGHT = 0.5       # Direct match weight
db.ac.REINFORCE_BONUS = 0.1     # Re-activation bonus
```

## Testing

```bash
# Run all tests (fast, no model downloads)
python -m pytest tests/ -m "not slow" -q

# Run specific test suites
python -m pytest tests/test_core.py        # 28 tests — basic operations
python -m pytest tests/test_magic.py       # 55 tests — git commands
python -m pytest tests/test_remote.py      # 18 tests — push/pull/fetch/PRs
python -m pytest tests/test_embed.py       # 11 unit + 18 integration
python -m pytest tests/test_structured.py  # 45 tests — query engine
python -m pytest tests/test_ambient.py     # 20 tests — Emirati AC
python -m pytest tests/test_hooks.py       # 31 tests — hooks + integration
python -m pytest tests/test_watches.py     # 14 tests — change subscriptions
python -m pytest tests/test_schema.py      # 18 tests — schema enforcement
python -m pytest tests/test_indexes.py     # 23 tests — secondary indexes
python -m pytest tests/test_snapshots.py   # 17 tests — snapshots
python -m pytest tests/test_transactions.py # 12 tests — atomic transactions
python -m pytest tests/test_backup.py      # 9 tests — backup & restore
python -m pytest tests/test_crush.py       # 32 tests — CRUSH algorithm
python -m pytest tests/test_distributed.py # 33 tests — P2P distributed
python -m pytest tests/test_ingest.py      # 33 tests — universal ingest
python -m pytest tests/test_cloud_ingest.py # 43 tests — cloud storage ingest
python -m pytest tests/test_server.py      # 22 tests — web dashboard

# Run integration tests (requires model download)
python -m pytest tests/test_embed.py -m slow
```

### Test Coverage

| Suite | Tests | What's Covered |
|-------|-------|---------------|
| `test_core.py` | 28 | add, remove, query, commit, log, diff, checkout, reset, tags, delta serialization, persistence, time-travel |
| `test_magic.py` | 55 | cherry-pick, revert, branching, merge (5 strategies), stash, blame, bisect, rebase, GC, reflog, filter-branch, purge (7 tests), HEAD, show, amend, squash, fork, notes |
| `test_remote.py` | 18 | remote add/remove, push, pull, fetch, branch push, PRs (create/list/comment/merge/close/filter), commit comments |
| `test_embed.py` | 29 | model registry, similarity, embedding, normalization, semantic filter, Matryoshka, GitDB text add/query, re-embed, semantic blame |
| `test_structured.py` | 45 | field resolution, all 13 operators, $and/$or/$not, select, group-by (5 agg functions), JSONL export/import, hybrid vector+structured queries |
| `test_ambient.py` | 20 | lifecycle, direct activation, spreading, decay, drift detection, AC-boosted queries, hot cache |
| `test_hooks.py` | 31 | register/unregister/fire, pre-hook rejection, post-hook fire, idempotent register, clear, integration with commit/merge |
| `test_hooks_integration.py` | 5 | pre-commit reject/accept, post-commit fires, pre-merge reject, post-merge fires |
| `test_watches.py` | 14 | watch/unwatch, branch watches, where watches, fire on commit, list, ID increment |
| `test_schema.py` | 18 | required fields, all type checks, enum, bounds, length, pattern, additionalProperties, GitDB integration, persistence |
| `test_indexes.py` | 23 | hash index create/rebuild/lookup/update, range index lookup, drop, list, duplicate create, GitDB integration |
| `test_snapshots.py` | 17 | create, query, select, isolation, duplicate name, get/list, repr, tombstone handling |
| `test_transactions.py` | 12 | commit, rollback on error, remove, nested operations, edge cases |
| `test_backup.py` | 9 | full backup, manifest sidecar, history, incremental, restore, restore+query, overwrite, verify valid/corrupt |
| `test_crush.py` | 32 | CRUSH map CRUD, straw2 determinism/weight/uniformity, single/multi-replica placement, host separation, down devices, router place/batch/distribution/rebalance, 144-device cluster |
| `test_distributed.py` | 33 | peer registry CRUD/persistence, CRUSH routing (single/multi/deterministic/batch/uniform), distributed add with outbox, local + distributed query with dedup, rebalance planning, gossip discovery, sync |
| `test_ingest.py` | 33 | SQLite (single/multi-table, auto-detect text, all metadata), MongoDB (JSON/JSONL, extended JSON, nested flattening), CSV (auto-detect, explicit column, TSV), text chunking (overlap, sentence boundaries), universal auto-detect, directory batch, helpers |
| `test_cloud_ingest.py` | 43 | URI parsing (all 5 schemes), is_cloud_uri detection, mocked S3/GCS/Azure/SFTP transports, error handling (missing deps, bad URIs), temp file cleanup, aggregate result counting |
| `test_server.py` | 22 | API endpoints (status, log, branches, tags, query, select, diff, show, AC status/heatmap), HTML dashboard serve, JSON response validation, server lifecycle |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- zstandard >= 0.21
- sentence-transformers >= 2.2 (optional, for embedding)

## License

MIT
