# GitDB: A GPU-Accelerated Version-Controlled Vector Database

**Version**: 0.1 Draft
**Date**: March 14, 2026
**Authors**: Vincent Kaufmann, Slick

---

## Abstract

Modern AI applications rely on vector databases for retrieval-augmented generation (RAG), semantic search, and embedding storage. These databases are mutable, multi-source, and mission-critical — yet they lack version control. Meanwhile, existing version-controlled databases (Dolt, lakeFS) operate on structured tabular data with no support for high-dimensional vectors or GPU-accelerated similarity search.

GitDB bridges this gap: a GPU-accelerated vector database where every mutation is a git-style delta. It provides full version control semantics — branches, commits, reverts, cherry-picks, merges — over tensor-backed embedding stores. The working tree lives on GPU memory as a contiguous tensor; the history is stored as sparse deltas using COO/CSR format, achieving compression ratios exceeding 99% for typical incremental updates.

This paper presents the architecture, storage format, query engine, and seven key use cases for LLM applications including RAG version control, embedding model upgrades, training data provenance, multi-tenant isolation, collaborative knowledge bases, poisoning recovery, and time-travel queries.

---

## 1. Introduction

### 1.1 The Problem

Every company deploying LLM-powered applications faces the same set of unsolved problems:

**Mutable vector stores are unauditable.** Documents are ingested, re-embedded, and deleted from vector databases with no record of what changed, when, or why. When retrieval quality degrades, there is no `git bisect` equivalent to find the breaking change.

**Embedding model upgrades are one-way operations.** Upgrading from one embedding model to another requires re-embedding the entire corpus. If the new model performs worse on certain query types, rollback means re-embedding again with the old model — hours or days of compute wasted.

**Bad data is catastrophic.** A single bad ingest — corrupted documents, adversarial injections, or miscategorized data — poisons the entire vector store. Recovery requires rebuilding from scratch because there is no way to identify and surgically remove only the affected vectors.

**Regulatory compliance is manual.** The EU AI Act and similar regulations require organizations to demonstrate exactly what data their AI systems had access to at any point in time. Without version control, this requires maintaining separate audit logs that inevitably drift from reality.

**Multi-tenant data isolation is fragile.** Serving multiple customers from a shared knowledge base requires careful partitioning. Adding shared knowledge to all tenants or removing a specific tenant's data is error-prone with no branching or merging semantics.

### 1.2 The Insight

Git solved these problems for source code thirty years ago. The key insight is that **the same version control semantics that work for code can work for vector data** — with one critical adaptation: the working tree is not a filesystem of text files, but a GPU-resident tensor of embeddings.

```
Traditional Git:
  Working Tree = files on disk
  Index = staged changes
  Objects = compressed file snapshots
  Refs = branches, tags

GitDB:
  Working Tree = [N, dim] tensor on GPU
  Index = pending vector mutations
  Objects = sparse tensor deltas (COO/CSR)
  Refs = branches, tags (same semantics)
```

### 1.3 What Exists vs. What Doesn't

| System | Vectors | GPU | Version Control |
|--------|---------|-----|-----------------|
| ChromaDB | Yes | No | No |
| Qdrant | Yes | No | No |
| Pinecone | Yes | No | No (namespaces only) |
| Milvus | Yes | Partial | No |
| Weaviate | Yes | No | No |
| Dolt | No (SQL) | No | Yes (full git) |
| lakeFS | No (object store) | No | Yes (git-like) |
| DVC | No (ML artifacts) | No | Yes (git-based) |
| RAPIDS cuDF | No (DataFrames) | Yes | No |
| HeavyDB | No (analytics) | Yes | No |
| **GitDB** | **Yes** | **Yes** | **Yes** |

GitDB occupies an empty cell in the matrix.

---

## 2. Architecture

### 2.1 System Overview

```
                    ┌─────────────────────────────────┐
                    │           GitDB Instance         │
                    │                                  │
  Ingest ────────>  │  ┌──────────────────────────┐   │
                    │  │  Staging Area (CPU)        │   │
  Query ─────────>  │  │  - Pending additions       │   │
                    │  │  - Pending deletions        │   │
  Commit ────────>  │  │  - Metadata mutations       │   │
                    │  └──────────┬───────────────┘   │
                    │             │ commit             │
                    │             v                    │
                    │  ┌──────────────────────────┐   │
                    │  │  Working Tree (GPU)        │   │
                    │  │                            │   │
                    │  │  embeddings: [N, dim]      │   │  <── Query target
                    │  │  metadata:   [N] structs   │   │
                    │  │  index:      HNSW/IVF      │   │
                    │  └──────────┬───────────────┘   │
                    │             │                    │
                    │             v                    │
                    │  ┌──────────────────────────┐   │
                    │  │  Object Store (Disk)       │   │
                    │  │                            │   │
                    │  │  Commits: sparse deltas    │   │
                    │  │  Packfiles: compressed     │   │
                    │  │  Refs: branches, tags      │   │
                    │  │  Remotes: push/pull/fetch  │   │
                    │  └──────────────────────────┘   │
                    │                                  │
                    └──────────────────────────────────┘
```

### 2.2 The Three Layers

**Layer 1: Working Tree (GPU)**

The active state of the database. A contiguous tensor of shape `[N, dim]` resident in GPU memory (CUDA or MPS). All queries execute against this tensor via matrix multiplication:

```python
scores = torch.mm(query_vec.unsqueeze(0), working_tree.T)  # [1, N]
top_k = torch.topk(scores, k=10)
```

For larger stores (>1M vectors), an ANN index (HNSW or IVF) is maintained alongside the tensor for sub-linear search. The index is rebuilt on checkout operations.

Metadata (document text, source, tags, timestamps) is stored in a companion structure indexed by row position in the tensor.

**Layer 2: Staging Area (CPU)**

Mutations are staged before committing, exactly like `git add`:

```python
db.add(documents=["new doc"], embeddings=[vec], source="api_v2")  # staged
db.remove(where={"source": "bad_crawler"})                        # staged
db.commit("Remove bad crawler data, add API v2 docs")             # applied
```

Staging allows atomic multi-operation commits. A commit either fully applies or doesn't — no partial states.

**Layer 3: Object Store (Disk)**

The git-compatible history. Each commit stores:
- A **sparse tensor delta** (what changed in the embedding matrix)
- A **metadata diff** (what changed in document metadata)
- Parent commit hash(es)
- Author, timestamp, message

The object store uses content-addressed hashing (SHA-256 of the delta content). Packfiles compress sequential deltas for storage efficiency.

### 2.3 Sparse Tensor Deltas

The key storage innovation. When you add 100 documents to a 100,000-row tensor, the delta is:

```
Delta = {
    type: "sparse_coo",
    additions: {
        indices: [100000, 100001, ..., 100099],  # new row positions
        values: tensor([100, dim]),                # the new embeddings
    },
    deletions: [],                                 # row indices removed
    dim: 1024,
    prev_rows: 100000,
    new_rows: 100100,
}
```

Storage cost: `100 * 1024 * 4 bytes = 400 KB` instead of `100100 * 1024 * 4 = 400 MB`. That's a **99.9% compression ratio** for a 100-document ingest.

For deletions: store only the removed indices. For modifications (re-embedding): store the row indices and new values. The full tensor is never duplicated.

**Reconstruction**: To check out a historical state, apply deltas in reverse from HEAD to the target commit. For frequently accessed historical states, GitDB caches reconstructed tensors (like git's loose objects vs packfiles).

### 2.4 Branching and Merging

Branches are lightweight refs pointing to commit hashes — identical to git.

```
main ────── c1 ── c2 ── c3 ── c4 (HEAD)
                    \
staging              c5 ── c6 (staging HEAD)
                            \
tenant/acme                  c7 (tenant/acme HEAD)
```

**Merge strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Union | Add all vectors from both branches, dedup by content hash | Combining knowledge bases |
| Ours/Theirs | Keep one branch's version on conflict | Tenant isolation |
| Rebase | Replay one branch's deltas on top of another | Sequential updates |
| Cherry-pick | Apply specific commits from another branch | Selective data sharing |

**Conflict detection**: Two commits conflict when they modify the same row index. This is detected by comparing the sparse delta indices. Resolution is explicit — the user chooses which embedding to keep.

---

## 3. Query Engine

### 3.1 Similarity Search

The primary query operation. Executed entirely on GPU:

```python
def query(self, vector: torch.Tensor, k: int = 10,
          where: dict = None, branch: str = "main") -> Results:
    """
    Similarity search against the working tree.

    If branch != current branch, reconstructs the target branch's
    tensor (cached for repeated queries).
    """
    if branch != self.current_branch:
        tensor = self._reconstruct(branch)
    else:
        tensor = self.working_tree

    # Exact search (< 100K vectors): single matmul
    scores = F.cosine_similarity(
        vector.unsqueeze(0), tensor, dim=1
    )

    # Apply metadata filters (WHERE clause)
    if where:
        mask = self._eval_where(where)
        scores[~mask] = -float('inf')

    top_k = torch.topk(scores, k)
    return self._resolve_metadata(top_k)
```

### 3.2 Cross-Branch Queries

Query multiple branches simultaneously to compare retrieval quality:

```python
# A/B test: which branch returns better results for this query?
results_main = db.query(vec, k=10, branch="main")
results_new_embed = db.query(vec, k=10, branch="experiment/new-embedder")

# Diff: what documents are in staging but not main?
diff = db.diff("main", "staging")
# Returns: {added: [...], removed: [...], modified: [...]}
```

### 3.3 Time-Travel Queries

```python
# What would this query have returned last Tuesday?
results = db.query(vec, k=10, at="2026-03-10T00:00:00")

# Equivalent to:
db.checkout("main", at="2026-03-10")  # reconstruct historical state
results = db.query(vec, k=10)
db.checkout("main")                    # return to HEAD
```

### 3.4 Source-Filtered Operations

Every vector has a `source` metadata field tracking its origin:

```python
# Find all vectors from a specific source
db.log(source="web_crawler_v2")

# Remove all vectors from a poisoned source
db.revert(source="web_crawler_v2", since="2026-03-01")

# Cherry-pick only vetted sources into production
db.cherry_pick(commit="abc123", where={"source": "manual_review"})
```

---

## 4. Use Cases for LLM Applications

### 4.1 RAG Version Control

**Problem**: A RAG pipeline ingests documents from multiple sources. A new crawler introduces malformed documents that degrade retrieval quality. There is no way to identify which ingest caused the regression or surgically undo it.

**Solution**:
```python
db = GitDB("rag_store")

# Routine ingest — each source is a tagged commit
db.add(documents, embeddings, source="support_docs_v3")
db.commit("Ingest support docs v3 — 500 articles")

# Quality degrades. Find the cause:
for commit in db.log():
    db.checkout(commit)
    score = evaluate_retrieval(test_queries)
    print(f"{commit.hash[:8]} ({commit.message}): {score:.3f}")
    # Bisect to find the breaking commit

# Found it. Revert just that commit:
db.revert("abc123")  # Removes only that commit's vectors
db.commit("Revert bad crawler ingest")
```

### 4.2 Embedding Model Upgrades

**Problem**: Upgrading embedding models requires re-embedding the entire corpus. If the new model is worse, you've lost the old embeddings.

**Solution**:
```python
# Branch for the experiment
db.branch("experiment/arctic-embed-l")
db.checkout("experiment/arctic-embed-l")

# Re-embed everything with the new model
new_embeddings = arctic_embed.encode(db.all_documents())
db.update_embeddings(new_embeddings)
db.commit("Re-embed with Arctic Embed L (1024d)")

# A/B test
for query in test_suite:
    old = db.query(query, branch="main")
    new = db.query(query, branch="experiment/arctic-embed-l")
    compare(old, new)

# New model wins? Merge. Loses? Delete branch. Zero risk.
db.checkout("main")
db.merge("experiment/arctic-embed-l")
```

### 4.3 Training Data Provenance

**Problem**: Regulators ask "what data was used to fine-tune model V2.3?" and you can't answer precisely.

**Solution**:
```python
# Tag each training dataset version
db.tag("training-v2.3")

# Months later, auditor asks:
snapshot = db.checkout("training-v2.3")
manifest = db.export_manifest()
# Returns: every document, its source, ingest date, hash
# Cryptographic proof of exactly what was in the training set
```

### 4.4 Multi-Tenant Branching

**Problem**: Multiple customers share a knowledge base. Each needs private data isolated. Adding shared knowledge to all tenants is manual and error-prone.

**Solution**:
```python
# Shared knowledge on main
db.checkout("main")
db.add(public_docs, embeddings, source="shared_kb")
db.commit("Add shared knowledge base")

# Each tenant branches from main
for tenant in ["acme", "globex", "initech"]:
    db.branch(f"tenant/{tenant}", from_branch="main")
    db.checkout(f"tenant/{tenant}")
    db.add(tenant_docs[tenant], tenant_embeddings[tenant],
           source=f"tenant_{tenant}")
    db.commit(f"Add {tenant} private documents")

# New shared knowledge? Update main, tenants pull:
db.checkout("main")
db.add(new_shared_docs, embeddings, source="shared_kb_v2")
db.commit("Shared KB update — Q1 2026 policies")

for tenant in ["acme", "globex", "initech"]:
    db.checkout(f"tenant/{tenant}")
    db.merge("main")  # Gets new shared docs, keeps private data
```

### 4.5 Collaborative Knowledge Bases

**Problem**: Multiple teams contribute documents to a shared RAG store. No review process exists for data quality.

**Solution**:
```python
# Team A proposes new documents
db.branch("proposal/team-a-product-docs")
db.checkout("proposal/team-a-product-docs")
db.add(team_a_docs, embeddings, source="team_a")
db.commit("Add Q1 product documentation")

# Review: compare retrieval quality
diff = db.diff("main", "proposal/team-a-product-docs")
print(f"Adding {diff.added_count} vectors, removing {diff.removed_count}")

# Run quality checks
for query in critical_queries:
    old = db.query(query, branch="main")
    new = db.query(query, branch="proposal/team-a-product-docs")
    assert new.precision >= old.precision, f"Regression on: {query}"

# Approved → merge
db.checkout("main")
db.merge("proposal/team-a-product-docs")
```

### 4.6 Poisoning Recovery

**Problem**: Adversarial documents injected into the vector store via RAG cause the LLM to output harmful or incorrect information. Identifying and removing only the poisoned vectors is impossible.

**Solution**:
```python
# Incident detected. Find the poisoned ingest:
suspicious = db.log(source="user_uploads", since="2026-03-01")

# Isolate: create a clean branch without the suspect commits
db.branch("clean", from_branch="main")
db.checkout("clean")
for commit in suspicious:
    db.revert(commit)
db.commit("Remove all user uploads since March 1")

# Verify: the attack vector is gone
result = db.query(attack_query, branch="clean")
assert attack_document not in result

# Deploy the clean branch
db.checkout("main")
db.reset("clean")  # main now points to the clean state
```

### 4.7 Time-Travel Queries

**Problem**: "Our chatbot gave wrong answers last Tuesday but works fine now. What changed?"

**Solution**:
```python
# Reconstruct Tuesday's state
tuesday_results = db.query(
    problematic_query,
    at="2026-03-10T14:00:00"
)

# Compare with current
current_results = db.query(problematic_query)

# Find what changed between then and now
changes = db.log(since="2026-03-10", until="2026-03-14")
for c in changes:
    print(f"{c.date} {c.author}: {c.message} (+{c.added} -{c.removed})")
```

---

## 5. Storage Format

### 5.1 On-Disk Layout

```
.gitdb/
  HEAD                    # ref to current branch
  refs/
    heads/
      main               # commit hash
      staging             # commit hash
      tenant/acme         # commit hash
    tags/
      v1.0               # commit hash
      training-v2.3      # commit hash
  objects/
    ab/
      c123...            # commit object
    de/
      f456...            # delta object (sparse tensor)
    gh/
      i789...            # metadata diff object
  pack/
    pack-001.gitdb       # compressed sequential deltas
    pack-001.idx         # pack index
  config                 # repository configuration
  cache/
    tensors/             # cached reconstructed tensors for branches
```

### 5.2 Commit Object Format

```json
{
  "hash": "abc123...",
  "parent": "def456...",
  "author": "ingest_pipeline",
  "timestamp": 1773492000.0,
  "message": "Ingest support docs v3 — 500 articles",
  "delta_hash": "ghi789...",
  "metadata_hash": "jkl012...",
  "stats": {
    "added": 500,
    "removed": 0,
    "modified": 0,
    "prev_rows": 50000,
    "new_rows": 50500
  }
}
```

### 5.3 Delta Object Format

```python
{
    "format": "sparse_coo",          # COO sparse tensor format
    "dtype": "float32",
    "dim": 1024,                     # embedding dimension
    "additions": {
        "indices": [50000, 50001, ...],  # row positions
        "values": bytes(...),            # raw tensor data
        "count": 500,
    },
    "deletions": {
        "indices": [1234, 5678],     # removed row positions
        "count": 2,
    },
    "modifications": {
        "indices": [42, 99],         # modified row positions
        "old_values": bytes(...),    # for reverse application
        "new_values": bytes(...),
        "count": 2,
    },
    "compressed": True,              # zstd compression
    "compressed_size": 204800,       # 200 KB
    "uncompressed_size": 2048000,    # 2 MB (10x compression)
}
```

### 5.4 Compression Analysis

For typical RAG workloads (incremental document ingestion):

| Operation | Tensor Size | Delta Size | Ratio |
|-----------|-------------|------------|-------|
| Add 100 docs to 100K store | 400 MB | 400 KB | 99.9% |
| Add 1K docs to 100K store | 400 MB | 4 MB | 99.0% |
| Add 10K docs to 100K store | 400 MB | 40 MB | 90.0% |
| Delete 50 docs from 100K store | 400 MB | 200 B (indices only) | 99.99% |
| Re-embed 100K docs (model upgrade) | 400 MB | 400 MB | 0% (full rewrite) |

Model upgrades are the worst case — every vector changes. But even then, the old state is one delta away from reconstruction.

With zstd compression on the delta values, typical compression adds another 2-4x reduction (embedding vectors compress well due to clustered distributions).

---

## 6. Implementation Plan

### 6.1 Phase 1: Core (MVP)

- In-memory tensor working tree (PyTorch)
- Basic operations: add, remove, query (cosine similarity)
- Commit/checkout with sparse COO deltas
- Linear history (no branches yet)
- File-based object store
- Python API

### 6.2 Phase 2: Branching

- Branch creation/deletion
- Checkout (reconstruct tensor from deltas)
- Merge (union strategy)
- Diff between branches
- Tag support

### 6.3 Phase 3: Performance

- ANN index (HNSW via hnswlib or FAISS)
- Packfile compression
- Tensor reconstruction caching
- Batch ingest optimization
- Memory-mapped object store

### 6.4 Phase 4: Collaboration

- Remote protocol (push/pull between instances)
- Conflict detection and resolution
- Access control per branch
- Audit logging

### 6.5 Phase 5: Production

- REST API server
- Multi-GPU support
- Streaming ingest
- Backup/restore
- Monitoring and metrics

---

## 7. Performance Characteristics

### 7.1 Query Performance

| Store Size | Exact Search (matmul) | ANN Search (HNSW) |
|------------|----------------------|---------------------|
| 10K vectors | 0.1 ms | 0.05 ms |
| 100K vectors | 0.8 ms | 0.1 ms |
| 1M vectors | 8 ms | 0.2 ms |
| 10M vectors | 80 ms | 0.5 ms |

*Estimated on NVIDIA A100 80GB, 1024d embeddings, k=10.*

For stores under 100K vectors, exact matmul search is fast enough and avoids the complexity of maintaining an ANN index. GitDB automatically switches to HNSW above a configurable threshold.

### 7.2 Commit Performance

| Operation | 100K Store | 1M Store |
|-----------|-----------|----------|
| Add 100 docs | 2 ms (delta) + 0.5 ms (write) | 2 ms + 0.5 ms |
| Delete 100 docs | 0.1 ms (delta) + 0.1 ms (write) | 0.1 ms + 0.1 ms |
| Checkout (1 commit back) | 3 ms (apply delta) | 30 ms |
| Checkout (100 commits back) | 50 ms (100 deltas) | 500 ms |
| Full reconstruction | 200 ms | 2 s |

Commit performance is independent of store size for additions/deletions — only the delta size matters. Checkout performance scales with the number of deltas to apply.

### 7.3 Storage Overhead

For a 100K vector store (400 MB working tree) with 1000 commits averaging 100 additions each:

```
Working tree:  400 MB  (GPU)
Object store:  ~50 MB  (1000 deltas, compressed)
Total:         ~450 MB
Overhead:      12.5%
```

Compare with naive snapshots: 1000 * 400 MB = 400 GB. GitDB achieves **8000x storage reduction** through delta encoding.

---

## 8. Related Work

**Dolt** (DoltHub, 2019) is the closest system — a SQL database with full git semantics. Dolt proves that version-controlled databases are viable and useful. However, Dolt operates on structured tabular data with B-tree indexes, not high-dimensional vectors with GPU-accelerated similarity search. GitDB extends Dolt's insight to the vector domain.

**lakeFS** (Treeverse, 2020) provides git-like version control for data lakes (S3-compatible object stores). It operates at the file/object level, not the vector level, and has no query engine.

**Milvus** (Zilliz, 2019) is a GPU-capable vector database but lacks version control entirely. Data mutations are irreversible without external backup systems.

**FAISS** (Meta, 2017) provides GPU-accelerated vector search but is a library, not a database. It has no persistence, no versioning, and no multi-tenancy.

**DVC** (Iterative, 2017) provides git-based version control for ML data and models. It tracks files, not individual vectors, and has no query capabilities.

GitDB uniquely combines: (1) vector-native storage, (2) GPU-accelerated queries, and (3) full git version control semantics.

---

## 9. Conclusion

GitDB addresses a fundamental gap in the AI infrastructure stack: vector databases have no version control, and version-controlled databases don't support vectors. By storing the working state as a GPU tensor and history as sparse deltas, GitDB provides the performance characteristics of a modern vector database with the safety and auditability of git.

The seven use cases presented — RAG version control, embedding model upgrades, training data provenance, multi-tenant branching, collaborative knowledge bases, poisoning recovery, and time-travel queries — represent real, unsolved problems faced by every organization deploying LLM applications at scale.

The implementation is tractable: the core MVP (Phase 1) requires approximately 1,500 lines of Python, building on PyTorch for tensor operations and borrowing storage format concepts from git's packfile specification.

---

## References

1. Torvalds, L. (2005). Git: A distributed version control system.
2. Johnson, J., Douze, M., & Jegou, H. (2017). Billion-scale similarity search with GPUs. arXiv:1702.08734.
3. DoltHub. (2019). Dolt: Git for data. https://www.dolthub.com/
4. Treeverse. (2020). lakeFS: Git-like version control for data lakes. https://lakefs.io/
5. Zilliz. (2019). Milvus: A purpose-built vector database. https://milvus.io/
6. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using HNSW graphs. arXiv:1603.09320.
