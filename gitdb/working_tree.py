"""WorkingTree — the GPU-resident active state of the database."""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from gitdb.types import Results, VectorMeta


class WorkingTree:
    """The GPU-resident active state of the database.

    Holds the current embedding tensor, per-row metadata, and a
    tombstone set for soft-deleted rows (compacted on commit).
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.embeddings: Optional[torch.Tensor] = None  # [N, dim] on device
        self.metadata: List[VectorMeta] = []
        self.tombstones: Set[int] = set()

    @property
    def size(self) -> int:
        """Number of active (non-tombstoned) rows."""
        total = len(self.metadata)
        return total - len(self.tombstones)

    @property
    def total_rows(self) -> int:
        """Total rows including tombstoned."""
        return len(self.metadata)

    def add(
        self,
        embeddings: torch.Tensor,
        documents: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[int]:
        """Add vectors to the working tree. Returns row indices."""
        n = embeddings.shape[0]
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {embeddings.shape[1]}")

        emb = embeddings.to(self.device)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = torch.cat([self.embeddings, emb], dim=0)

        start_idx = len(self.metadata)
        indices = list(range(start_idx, start_idx + n))

        for i in range(n):
            doc = documents[i] if documents and i < len(documents) else None
            meta = metadata[i] if metadata and i < len(metadata) else {}
            vid = ids[i] if ids and i < len(ids) else _content_hash(emb[i], doc)
            self.metadata.append(VectorMeta(id=vid, document=doc, metadata=meta))

        return indices

    def remove(self, indices: List[int]):
        """Soft-delete rows by marking them as tombstoned."""
        for idx in indices:
            if idx < 0 or idx >= self.total_rows:
                raise IndexError(f"Row index out of range: {idx}")
            self.tombstones.add(idx)

    def remove_where(self, where: Dict[str, Any]) -> List[int]:
        """Soft-delete rows matching a metadata filter. Returns removed indices."""
        removed = []
        for i, meta in enumerate(self.metadata):
            if i in self.tombstones:
                continue
            if _matches_where(meta, where):
                self.tombstones.add(i)
                removed.append(i)
        return removed

    def update_embeddings(self, indices: List[int], embeddings: torch.Tensor):
        """Update embeddings for specific rows."""
        if len(indices) != embeddings.shape[0]:
            raise ValueError("Indices and embeddings count mismatch")
        emb = embeddings.to(self.device)
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= self.total_rows:
                raise IndexError(f"Row index out of range: {idx}")
            self.embeddings[idx] = emb[i]

    def query(
        self,
        vector: torch.Tensor,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Results:
        """Cosine similarity search with optional metadata filter."""
        if self.embeddings is None or self.size == 0:
            return Results(ids=[], scores=[], documents=[], metadata=[])

        q = vector.to(self.device)
        if q.dim() == 1:
            q = q.unsqueeze(0)

        # Build active mask
        n = self.total_rows
        active_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        if self.tombstones:
            tomb_indices = torch.tensor(list(self.tombstones), dtype=torch.long, device=self.device)
            active_mask[tomb_indices] = False

        if where:
            for i in range(n):
                if active_mask[i] and not _matches_where(self.metadata[i], where):
                    active_mask[i] = False

        active_indices = torch.where(active_mask)[0]
        if len(active_indices) == 0:
            return Results(ids=[], scores=[], documents=[], metadata=[])

        active_emb = self.embeddings[active_indices]

        # Cosine similarity
        scores = F.cosine_similarity(q, active_emb, dim=1)
        actual_k = min(k, len(scores))
        top_scores, top_local = torch.topk(scores, actual_k)

        # Map back to global indices
        top_global = active_indices[top_local]

        ids = []
        docs = []
        metas = []
        for idx in top_global.cpu().tolist():
            m = self.metadata[idx]
            ids.append(m.id)
            docs.append(m.document)
            metas.append(m.metadata)

        return Results(
            ids=ids,
            scores=top_scores.cpu().tolist(),
            documents=docs,
            metadata=metas,
        )

    def compact(self) -> Tuple[List[int], List[VectorMeta]]:
        """Remove tombstoned rows, returning removed (indices, metadata).

        Call this during commit to reclaim space.
        """
        if not self.tombstones:
            return [], []

        removed_indices = sorted(self.tombstones)
        removed_meta = [self.metadata[i] for i in removed_indices]

        # Keep non-tombstoned rows
        keep_mask = torch.ones(self.total_rows, dtype=torch.bool)
        for idx in removed_indices:
            keep_mask[idx] = False

        self.embeddings = self.embeddings[keep_mask]
        self.metadata = [m for i, m in enumerate(self.metadata) if i not in self.tombstones]
        self.tombstones.clear()

        return removed_indices, removed_meta

    def snapshot_tensor(self) -> Optional[torch.Tensor]:
        """Get a CPU copy of the current embeddings (excluding tombstones)."""
        if self.embeddings is None:
            return None
        if not self.tombstones:
            return self.embeddings.cpu().clone()
        keep_mask = torch.ones(self.total_rows, dtype=torch.bool)
        for idx in self.tombstones:
            keep_mask[idx] = False
        return self.embeddings[keep_mask].cpu().clone()

    def load_state(self, embeddings: torch.Tensor, metadata: List[VectorMeta]):
        """Load a complete state (used for checkout)."""
        self.embeddings = embeddings.to(self.device)
        self.metadata = list(metadata)
        self.tombstones.clear()


def _content_hash(embedding: torch.Tensor, document: Optional[str] = None) -> str:
    """Generate a content-based ID from embedding + document."""
    import hashlib
    h = hashlib.sha256()
    h.update(embedding.cpu().to(torch.float32).numpy().tobytes())
    if document:
        h.update(document.encode("utf-8"))
    return h.hexdigest()[:16]


def _matches_where(meta: VectorMeta, where: Dict[str, Any]) -> bool:
    """Check if metadata matches a where filter.

    Supports full structured queries: comparisons, $and/$or/$not,
    $in/$nin, $contains, $regex, $exists, dotted field paths.
    See gitdb.structured for full documentation.
    """
    from gitdb.structured import matches
    return matches(meta, where)
