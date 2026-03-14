"""Snapshot — cheap read-only frozen views of the current state."""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from gitdb.types import Results, VectorMeta


class Snapshot:
    """A read-only frozen view of the database at a point in time.

    Zero-copy for metadata (reference), cloned tensor for embeddings.
    Supports query, query_text, and select operations.
    """

    def __init__(
        self,
        embeddings: Optional[torch.Tensor],
        metadata: List[VectorMeta],
        name: str,
        timestamp: float,
    ):
        self.embeddings = embeddings  # cloned tensor
        self.metadata = metadata      # reference (read-only contract)
        self.name = name
        self.timestamp = timestamp
        self._dim = embeddings.shape[1] if embeddings is not None and embeddings.shape[0] > 0 else 0

    @property
    def size(self) -> int:
        """Number of vectors in this snapshot."""
        return len(self.metadata)

    def query(
        self,
        vector: torch.Tensor,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Results:
        """Cosine similarity search with optional metadata filter."""
        if self.embeddings is None or self.size == 0:
            return Results(ids=[], scores=[], documents=[], metadata=[])

        device = self.embeddings.device
        q = vector.to(device)
        if q.dim() == 1:
            q = q.unsqueeze(0)

        n = len(self.metadata)
        active_mask = torch.ones(n, dtype=torch.bool, device=device)

        if where:
            from gitdb.structured import matches
            for i in range(n):
                if active_mask[i] and not matches(self.metadata[i], where):
                    active_mask[i] = False

        active_indices = torch.where(active_mask)[0]
        if len(active_indices) == 0:
            return Results(ids=[], scores=[], documents=[], metadata=[])

        active_emb = self.embeddings[active_indices]
        scores = F.cosine_similarity(q, active_emb, dim=1)
        actual_k = min(k, len(scores))
        top_scores, top_local = torch.topk(scores, actual_k)
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

    def query_text(
        self,
        text: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        embed_model: Optional[str] = None,
    ) -> Results:
        """Semantic query — embed text then search the snapshot."""
        from gitdb.embed import embed_query, DEFAULT_MODEL
        model = embed_model or DEFAULT_MODEL
        vector = embed_query(text, model_name=model, dim=self._dim)
        return self.query(vector, k=k, where=where)

    def select(
        self,
        where: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
    ) -> List[dict]:
        """Select metadata from snapshot with optional filter and projection."""
        from gitdb.structured import matches, project, to_table

        active = list(self.metadata)
        if where:
            active = [m for m in active if matches(m, where)]

        if fields:
            return project(active, fields)
        return to_table(active, fields)

    def __repr__(self):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return f"Snapshot({self.name!r}, {self.size} vectors, {ts})"
