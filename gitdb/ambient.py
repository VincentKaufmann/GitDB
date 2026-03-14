"""Emirati AC — Spreading Activation for GitDB.

Named after how Emiratis leave their car AC running so it's pre-chilled
when they walk out. The GPU continuously pattern-matches recent operations
against the entire vector store. Vectors activate neighboring vectors
through multi-hop chains. By the time you query, the answer is already hot.

Architecture (from SoulKeeper, adapted for version-controlled vectors):
  1. Context feed: every gitdb operation (log, diff, add, query) feeds context
  2. Direct activation: GPU matmul finds top matches (microseconds)
  3. Spreading activation: active vectors activate their neighbors
  4. Decay: activations fade over time (vectors cool down)
  5. Hot cache: top N activated vectors, sorted by activation level
  6. Drift detection: monitors semantic centroid drift on current branch

Multi-hop chains emerge naturally:
    "auth tokens" → session_management (0.65) → security_audit (0.58)
        → vulnerability_scan (0.52) → penetration_test (0.48)
"""

import hashlib
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from gitdb.types import Results, VectorMeta


class EmiratiAC:
    """Background spreading activation engine for GitDB.

    Usage:
        db = GitDB("store", dim=1024, device="mps")
        db.ac.start()           # Engine on, AC running
        db.add(texts=["..."])   # Context auto-fed
        hot = db.ac.primed(10)  # Already-ranked vectors, zero compute
        db.ac.stop()            # Engine off
    """

    # ── Tuning knobs ──────────────────────────────────────────
    POLL_INTERVAL = 3.0         # Seconds between ambient cycles
    ACTIVATION_DECAY = 0.90     # 10% decay per cycle
    SPREAD_FACTOR = 0.25        # 25% of activation spreads to neighbors
    SPREAD_MIN_SIM = 0.40       # Only spread to vectors with sim > 0.4
    SPREAD_MIN_LEVEL = 0.20     # Only spread from vectors activated > 0.2
    MAX_SPREAD_SOURCES = 15     # Max vectors to spread from per cycle
    NEIGHBOR_K = 5              # Neighbors per vector for spreading
    HOT_CACHE_SIZE = 50         # Keep top 50 activated vectors
    MIN_ACTIVATION = 0.05       # Below this, vector is deactivated
    DIRECT_WEIGHT = 0.5         # Weight for direct matmul matches
    REINFORCE_BONUS = 0.1       # Bonus when same vector re-activates
    DRIFT_WINDOW = 20           # Track last N additions for drift detection

    def __init__(self, gitdb: Any):
        self.db = gitdb

        # Activation state
        self.activations: Dict[int, float] = {}       # row_idx → activation level
        self.hot_cache: List[Tuple[int, float]] = []   # sorted (idx, level) pairs
        self.context_buffer: List[str] = []             # recent operation descriptions
        self._last_context_hash = ""

        # Drift tracking
        self._branch_centroid: Optional[torch.Tensor] = None
        self._recent_additions: List[torch.Tensor] = []
        self._drift_alerts: List[Dict] = []

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._new_context = threading.Event()

        # Callbacks
        self._on_drift: Optional[Callable] = None

        # Stats
        self.cycles = 0
        self.last_cycle_ms = 0.0
        self.total_spread_hops = 0
        self.peak_active = 0

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self):
        """Start the ambient activation loop."""
        if self._running:
            return
        self._running = True
        self._compute_centroid()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="EmiratiAC")
        self._thread.start()

    def stop(self):
        """Stop the ambient loop."""
        self._running = False
        self._new_context.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._running

    # ── Context Feeding ────────────────────────────────────────

    def feed(self, operation: str, detail: str = ""):
        """Feed context from a gitdb operation.

        Called automatically by GitDB methods when AC is running.
        Examples:
            feed("query", "search for authentication vectors")
            feed("commit", "Add 50 finance embeddings")
            feed("diff", "main vs feature-branch")
            feed("log", "viewing last 20 commits")
        """
        with self._lock:
            ctx = f"{operation}: {detail}" if detail else operation
            self.context_buffer.append(ctx)
            self.context_buffer = self.context_buffer[-20:]  # sliding window
        self._new_context.set()

    def feed_text(self, text: str):
        """Feed raw text context (e.g., query text, document text)."""
        self.feed("text", text)

    def feed_vectors(self, vectors: torch.Tensor):
        """Feed vectors directly (e.g., from add or query operations).

        Runs direct activation without needing an embedding model.
        """
        if self.db.tree.embeddings is None or self.db.tree.embeddings.shape[0] == 0:
            return

        with self._lock:
            try:
                emb = self.db.tree.embeddings
                if vectors.dim() == 1:
                    vectors = vectors.unsqueeze(0)
                vectors = vectors.to(emb.device)

                # Direct matmul for activation
                norms_a = F.normalize(vectors, p=2, dim=1)
                norms_b = F.normalize(emb, p=2, dim=1)
                sims = torch.mm(norms_a, norms_b.t())  # (Q, N)

                # Activate top matches from each query vector
                for q in range(sims.shape[0]):
                    top_scores, top_idx = torch.topk(
                        sims[q], k=min(20, sims.shape[1]))
                    for idx_t, score_t in zip(top_idx, top_scores):
                        idx = int(idx_t.item())
                        score = float(score_t.item())
                        if score > 0.20:
                            current = self.activations.get(idx, 0)
                            if current > 0:
                                new_level = current + score * self.DIRECT_WEIGHT + self.REINFORCE_BONUS
                            else:
                                new_level = score * self.DIRECT_WEIGHT
                            self.activations[idx] = min(1.0, new_level)
            except Exception:
                pass

        self._new_context.set()

    def track_addition(self, vectors: torch.Tensor):
        """Track new additions for drift detection."""
        with self._lock:
            if vectors.dim() == 1:
                vectors = vectors.unsqueeze(0)
            for i in range(vectors.shape[0]):
                self._recent_additions.append(vectors[i].cpu().clone())
            self._recent_additions = self._recent_additions[-self.DRIFT_WINDOW:]

    # ── Query Integration ──────────────────────────────────────

    def primed(self, top_k: int = 10) -> Results:
        """Return pre-activated vectors — instant, zero computation.

        This is the payoff: results are already ranked before you ask.
        """
        with self._lock:
            ids, scores, docs, metas = [], [], [], []
            for idx, level in self.hot_cache[:top_k]:
                if idx < len(self.db.tree.metadata):
                    meta = self.db.tree.metadata[idx]
                    ids.append(meta.id)
                    scores.append(round(level, 4))
                    docs.append(meta.document)
                    metas.append(meta.metadata)
            return Results(ids=ids, scores=scores, documents=docs, metadata=metas)

    def boost_results(self, results: Results, boost_weight: float = 0.3) -> Results:
        """Boost query results using activation levels.

        Re-ranks results by combining cosine similarity with activation level.
        Activated vectors get a score boost — the AC "priming" effect.
        """
        if not results.ids:
            return results

        activation_map = self.get_activation_map()
        boosted = []
        for i in range(len(results.ids)):
            vid = results.ids[i]
            base_score = results.scores[i]
            # Find activation for this vector
            act_level = 0.0
            for idx, level in self.hot_cache:
                if idx < len(self.db.tree.metadata) and self.db.tree.metadata[idx].id == vid:
                    act_level = level
                    break
            boosted_score = base_score * (1 - boost_weight) + act_level * boost_weight
            boosted.append((i, boosted_score))

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        return Results(
            ids=[results.ids[i] for i, _ in boosted],
            scores=[s for _, s in boosted],
            documents=[results.documents[i] for i, _ in boosted],
            metadata=[results.metadata[i] for i, _ in boosted],
        )

    def get_activation_map(self) -> Dict[int, float]:
        """Return current activation levels."""
        with self._lock:
            return dict(self.activations)

    # ── Drift Detection ────────────────────────────────────────

    def drift(self) -> Optional[Dict]:
        """Check semantic drift of recent additions vs branch centroid.

        Returns None if no drift, or dict with drift magnitude and details.
        """
        with self._lock:
            if self._branch_centroid is None or not self._recent_additions:
                return None

            recent = torch.stack(self._recent_additions)
            recent_centroid = F.normalize(recent.mean(dim=0, keepdim=True), p=2, dim=1)
            branch_norm = F.normalize(self._branch_centroid.unsqueeze(0), p=2, dim=1)

            drift_sim = F.cosine_similarity(recent_centroid, branch_norm).item()
            drift_magnitude = 1.0 - drift_sim

            if drift_magnitude > 0.3:  # Significant drift
                return {
                    "magnitude": round(drift_magnitude, 4),
                    "similarity_to_centroid": round(drift_sim, 4),
                    "recent_count": len(self._recent_additions),
                    "severity": "high" if drift_magnitude > 0.5 else "medium",
                }
            return None

    def on_drift(self, callback: Callable):
        """Register a callback for drift alerts."""
        self._on_drift = callback

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Current AC stats."""
        with self._lock:
            return {
                "running": self._running,
                "active_vectors": len(self.activations),
                "hot_cache_size": len(self.hot_cache),
                "cycles": self.cycles,
                "last_cycle_ms": round(self.last_cycle_ms, 1),
                "peak_active": self.peak_active,
                "total_spread_hops": self.total_spread_hops,
                "top_score": round(self.hot_cache[0][1], 3) if self.hot_cache else 0,
                "context_depth": len(self.context_buffer),
                "drift": self.drift(),
            }

    # ── Internal ───────────────────────────────────────────────

    def _compute_centroid(self):
        """Compute semantic centroid of current branch state."""
        if self.db.tree.embeddings is not None and self.db.tree.embeddings.shape[0] > 0:
            self._branch_centroid = self.db.tree.embeddings.mean(dim=0).cpu()

    def _run_loop(self):
        """Main loop — runs until stopped."""
        while self._running:
            self._new_context.wait(timeout=self.POLL_INTERVAL)
            self._new_context.clear()
            if not self._running:
                break
            try:
                self._cycle()
            except Exception:
                pass

    def _cycle(self):
        """One activation cycle: context → matmul → spread → cache → drift check."""
        if self.db.tree.embeddings is None or self.db.tree.embeddings.shape[0] == 0:
            return

        t0 = time.time()

        # Check if context changed
        with self._lock:
            if not self.context_buffer:
                self._decay_only()
                return
            context_text = " | ".join(self.context_buffer[-10:])

        context_hash = hashlib.md5(context_text.encode()).hexdigest()
        if context_hash == self._last_context_hash:
            self._decay_only()
            return
        self._last_context_hash = context_hash

        # Step 1: Try to embed context text (requires embed module)
        context_vec = self._embed_context(context_text)

        if context_vec is not None:
            # Step 2: GPU matmul — direct activation
            emb = self.db.tree.embeddings
            device = emb.device
            q = F.normalize(context_vec.unsqueeze(0).to(device), p=2, dim=1)
            norms = F.normalize(emb, p=2, dim=1)
            sims = torch.mm(q, norms.t()).squeeze(0)

            top_scores, top_idx = torch.topk(sims, k=min(20, len(sims)))

            # Step 3: Decay existing activations
            with self._lock:
                for idx in list(self.activations.keys()):
                    self.activations[idx] *= self.ACTIVATION_DECAY
                    if self.activations[idx] < self.MIN_ACTIVATION:
                        del self.activations[idx]

            # Step 4: Activate direct matches
            for idx_t, score_t in zip(top_idx, top_scores):
                idx = int(idx_t.item())
                score = float(score_t.item())
                if score > 0.20:
                    with self._lock:
                        current = self.activations.get(idx, 0)
                        if current > 0:
                            new_level = current + score * self.DIRECT_WEIGHT + self.REINFORCE_BONUS
                        else:
                            new_level = score * self.DIRECT_WEIGHT
                        self.activations[idx] = min(1.0, new_level)
        else:
            # No embedder — just decay
            with self._lock:
                for idx in list(self.activations.keys()):
                    self.activations[idx] *= self.ACTIVATION_DECAY
                    if self.activations[idx] < self.MIN_ACTIVATION:
                        del self.activations[idx]

        # Step 5: Spreading activation (the magic)
        hops = self._spread()

        # Step 6: Update hot cache
        with self._lock:
            sorted_acts = sorted(
                self.activations.items(), key=lambda x: x[1], reverse=True)
            self.hot_cache = sorted_acts[:self.HOT_CACHE_SIZE]
            active_count = len(self.activations)
            if active_count > self.peak_active:
                self.peak_active = active_count

        # Step 7: Drift detection
        drift = self.drift()
        if drift and self._on_drift:
            self._on_drift(drift)
            self._drift_alerts.append({**drift, "timestamp": time.time()})

        self.cycles += 1
        self.total_spread_hops += hops
        self.last_cycle_ms = (time.time() - t0) * 1000

    def _decay_only(self):
        """Decay activations when context hasn't changed."""
        with self._lock:
            for idx in list(self.activations.keys()):
                self.activations[idx] *= self.ACTIVATION_DECAY
                if self.activations[idx] < self.MIN_ACTIVATION:
                    del self.activations[idx]
            sorted_acts = sorted(
                self.activations.items(), key=lambda x: x[1], reverse=True)
            self.hot_cache = sorted_acts[:self.HOT_CACHE_SIZE]

    def _spread(self) -> int:
        """Spreading activation — active vectors activate their neighbors.

        This is how human memory works: searching for "auth tokens" doesn't just
        find auth vectors — it activates session management, security audit,
        vulnerability scanning through multi-hop cosine similarity chains.
        """
        emb = self.db.tree.embeddings
        if emb is None:
            return 0

        with self._lock:
            to_spread = [
                (idx, level) for idx, level in self.activations.items()
                if level > self.SPREAD_MIN_LEVEL
            ][:self.MAX_SPREAD_SOURCES]

        if not to_spread:
            return 0

        hops = 0
        n = emb.shape[0]
        device = emb.device

        for idx, level in to_spread:
            if idx >= n:
                continue
            try:
                # Cosine similarity of this vector vs all others
                vec = F.normalize(emb[idx].unsqueeze(0), p=2, dim=1)
                all_norm = F.normalize(emb, p=2, dim=1)
                sims = torch.mm(vec, all_norm.t()).squeeze(0)

                # Top-k neighbors (excluding self)
                top_scores, top_idx = torch.topk(
                    sims, k=min(self.NEIGHBOR_K + 1, n))

                for n_idx_t, n_score_t in zip(top_idx, top_scores):
                    n_idx = int(n_idx_t.item())
                    n_score = float(n_score_t.item())
                    if n_idx == idx:
                        continue
                    if n_score < self.SPREAD_MIN_SIM:
                        continue

                    spread = level * self.SPREAD_FACTOR * n_score
                    with self._lock:
                        current = self.activations.get(n_idx, 0)
                        if current < spread:
                            self.activations[n_idx] = min(1.0, current + spread)
                            if current == 0:
                                hops += 1
            except Exception:
                continue

        return hops

    def _embed_context(self, text: str) -> Optional[torch.Tensor]:
        """Try to embed context text. Returns None if no embedder available."""
        try:
            from gitdb.embed import embed_query
            return embed_query(text, dim=self.db.dim)
        except (ImportError, Exception):
            return None
