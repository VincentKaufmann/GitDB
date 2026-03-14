# GitDB

GPU-accelerated version-controlled vector database. No server. No Docker. No config. `pip install gitdb` and go.

## Why GitDB

1. **Version control is native** — `git log`, `git diff`, `git branch`, `git merge` for your vectors. Nobody else has this.
2. **Time-travel queries** — `db.query_text("revenue", at="v1.0")` searches an old snapshot. Chroma can't do this.
3. **CEPH CRUSH placement** — deterministic data routing without a coordinator. Scales horizontally by adding peers.
4. **P2P distributed** — no central server. Peers sync over SSH like git remotes. Each node is a full replica.
5. **FoundationDB features** — hooks, transactions, watches, secondary indexes, schema enforcement. Enterprise-grade.
6. **Spreading activation** — background semantic priming makes frequent queries instant. Multi-hop association chains emerge naturally.
7. **CLI-first** — `gitdb init && gitdb add --text "doc" && gitdb commit -m "init"`. Works from terminal like git.
8. **Embedded** — no server, no Docker, no config. `pip install gitdb` and go.

Git for tensors. SQLite for metadata. Arctic/NV-EmbedQA for semantics. CEPH CRUSH placement. P2P distributed. FoundationDB-inspired hooks, transactions, watches, indexes, snapshots, schema enforcement. Native backup/restore. Universal ingest (SQLite, MongoDB, CSV, Parquet, PDF, text, S3, GCS, Azure, MinIO, SFTP).

## Install

```bash
pip install -e .                    # Core (torch + zstandard)
pip install -e ".[embed]"           # + Arctic Embed, NV-EmbedQA
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
gitdb ingest <file|dir> [--text-column C]     Universal ingest
gitdb serve [--port N] [--no-browser]        Web dashboard
```

## Architecture

```
14,400 lines of Python across 23 modules. 459 tests.

┌──────────────────────────────────────────────────────────────┐
│                    GitDB v0.8.0                               │
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
│  Dashboard ──→ Web UI                 ← gitdb serve          │
│       │              │                  commit graph, search │
│       │         server.py               AC heatmap, queries  │
│       ▼              │                                       │
│  QUERY ──→ Boosted Results                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Modules:
  core.py          2,383 lines   Main interface (60+ methods)
  cli.py           1,455 lines   CLI (67+ commands)
  distributed.py     765 lines   P2P distributed layer
  crush.py           600 lines   CEPH CRUSH algorithm
  ambient.py         459 lines   Emirati AC engine
  backup.py          457 lines   Native backup & restore
  remote.py          387 lines   Push/pull/fetch (local + SSH)
  structured.py      322 lines   Query operators, aggregation
  delta.py           288 lines   Sparse tensor deltas + zstd
  objects.py         260 lines   Content-addressed object store
  working_tree.py    208 lines   GPU-resident tensor + search
  embed.py           200 lines   Arctic + NV-EmbedQA
  pullrequest.py     197 lines   PR store with comments
  indexes.py         138 lines   Secondary indexes (hash + range)
  types.py           127 lines   Data types
  schema.py          125 lines   JSON Schema validation
  watches.py         123 lines   Change subscriptions
  snapshots.py       118 lines   Read-only frozen views
  ingest.py          909 lines   Universal ingest (6 formats)
  cloud_ingest.py    349 lines   Cloud storage transport (S3/GCS/Azure/MinIO/SFTP)
  server.py          673 lines   Web dashboard (gitdb serve)
  hooks.py            78 lines   Event hook system
```

## What Nothing Else Can Do

| Capability | GitDB | Pinecone | Weaviate | Qdrant | ChromaDB | SQLite |
|-----------|-------|----------|----------|--------|----------|--------|
| Vector similarity search | Yes | Yes | Yes | Yes | Yes | No |
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
