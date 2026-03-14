"""Sparse tensor deltas — the heart of GitDB's version control."""

import hashlib
import io
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import zstandard as zstd

from gitdb.types import VectorMeta


# Delta file magic number and version
DELTA_MAGIC = b"GDBD"  # GitDB Delta
DELTA_VERSION = 1

# Compression level (3 = good balance of speed/ratio)
ZSTD_LEVEL = 3


@dataclass
class Delta:
    """Sparse representation of changes between two states.

    Stores only what changed — additions, deletions, modifications,
    and metadata-only changes. Serializes to a compact binary format
    with zstd compression.
    """

    # Additions: new rows appended
    add_indices: List[int] = field(default_factory=list)
    add_values: Optional[torch.Tensor] = None   # [num_additions, dim]
    add_metadata: List[VectorMeta] = field(default_factory=list)

    # Deletions: rows removed (store old values for reverse)
    del_indices: List[int] = field(default_factory=list)
    del_values: Optional[torch.Tensor] = None    # [num_deletions, dim] old embeddings
    del_metadata: List[VectorMeta] = field(default_factory=list)

    # Modifications: rows with changed embeddings
    mod_indices: List[int] = field(default_factory=list)
    mod_old_values: Optional[torch.Tensor] = None  # for reverse application
    mod_new_values: Optional[torch.Tensor] = None

    # Metadata-only changes: (index, old_meta, new_meta)
    meta_changes: List[Tuple[int, VectorMeta, VectorMeta]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            not self.add_indices
            and not self.del_indices
            and not self.mod_indices
            and not self.meta_changes
        )

    def apply_forward(
        self,
        tensor: torch.Tensor,
        metadata: List[VectorMeta],
    ) -> Tuple[torch.Tensor, List[VectorMeta]]:
        """Apply this delta to move forward in history."""
        t = tensor.clone()
        meta = list(metadata)

        # Apply modifications in place
        for i, idx in enumerate(self.mod_indices):
            t[idx] = self.mod_new_values[i]

        # Apply metadata-only changes
        for idx, _old, new in self.meta_changes:
            meta[idx] = new

        # Apply deletions (must track index shifts)
        # Sort descending so removals don't shift subsequent indices
        if self.del_indices:
            for idx in sorted(self.del_indices, reverse=True):
                # Remove row from tensor
                t = torch.cat([t[:idx], t[idx + 1:]], dim=0)
                meta.pop(idx)

        # Apply additions
        if self.add_values is not None and len(self.add_indices) > 0:
            t = torch.cat([t, self.add_values.to(t.device)], dim=0)
            meta.extend(self.add_metadata)

        return t, meta

    def apply_reverse(
        self,
        tensor: torch.Tensor,
        metadata: List[VectorMeta],
    ) -> Tuple[torch.Tensor, List[VectorMeta]]:
        """Apply this delta in reverse to move backward in history."""
        t = tensor.clone()
        meta = list(metadata)

        # Reverse additions (remove the rows that were added)
        if self.add_values is not None and len(self.add_indices) > 0:
            n_added = len(self.add_indices)
            t = t[:-n_added]
            meta = meta[:-n_added]

        # Reverse deletions (re-insert the deleted rows)
        if self.del_indices and self.del_values is not None:
            # Sort by original index so we insert in order
            sorted_pairs = sorted(zip(self.del_indices, range(len(self.del_indices))))
            for orig_idx, del_i in sorted_pairs:
                row_emb = self.del_values[del_i].to(t.device)
                row_meta = self.del_metadata[del_i]
                # Re-insert at original position
                t = torch.cat([t[:orig_idx], row_emb.unsqueeze(0), t[orig_idx:]], dim=0)
                meta.insert(orig_idx, row_meta)

        # Reverse modifications
        for i, idx in enumerate(self.mod_indices):
            if idx < len(t):
                t[idx] = self.mod_old_values[i]

        # Reverse metadata-only changes
        for idx, old, _new in self.meta_changes:
            if idx < len(meta):
                meta[idx] = old

        return t, meta

    def serialize(self) -> bytes:
        """Serialize to binary format with zstd compression."""
        buf = io.BytesIO()

        # Header
        buf.write(DELTA_MAGIC)
        buf.write(struct.pack("<B", DELTA_VERSION))

        # Additions
        n_add = len(self.add_indices)
        buf.write(struct.pack("<I", n_add))
        if n_add > 0:
            buf.write(struct.pack(f"<{n_add}I", *self.add_indices))
            # Tensor as raw bytes (float32)
            t_bytes = self.add_values.cpu().to(torch.float32).numpy().tobytes()
            buf.write(struct.pack("<I", len(t_bytes)))
            buf.write(t_bytes)
            # Metadata as compact repr
            meta_bytes = _serialize_meta_list(self.add_metadata)
            buf.write(struct.pack("<I", len(meta_bytes)))
            buf.write(meta_bytes)

        # Deletions
        n_del = len(self.del_indices)
        buf.write(struct.pack("<I", n_del))
        if n_del > 0:
            buf.write(struct.pack(f"<{n_del}I", *self.del_indices))
            # Store old embedding values for reverse
            if self.del_values is not None:
                del_bytes = self.del_values.cpu().to(torch.float32).numpy().tobytes()
                buf.write(struct.pack("<I", len(del_bytes)))
                buf.write(del_bytes)
            else:
                buf.write(struct.pack("<I", 0))
            meta_bytes = _serialize_meta_list(self.del_metadata)
            buf.write(struct.pack("<I", len(meta_bytes)))
            buf.write(meta_bytes)

        # Modifications
        n_mod = len(self.mod_indices)
        buf.write(struct.pack("<I", n_mod))
        if n_mod > 0:
            buf.write(struct.pack(f"<{n_mod}I", *self.mod_indices))
            old_bytes = self.mod_old_values.cpu().to(torch.float32).numpy().tobytes()
            buf.write(struct.pack("<I", len(old_bytes)))
            buf.write(old_bytes)
            new_bytes = self.mod_new_values.cpu().to(torch.float32).numpy().tobytes()
            buf.write(struct.pack("<I", len(new_bytes)))
            buf.write(new_bytes)

        # Meta-only changes
        n_meta = len(self.meta_changes)
        buf.write(struct.pack("<I", n_meta))
        if n_meta > 0:
            indices = [mc[0] for mc in self.meta_changes]
            buf.write(struct.pack(f"<{n_meta}I", *indices))
            old_meta = _serialize_meta_list([mc[1] for mc in self.meta_changes])
            buf.write(struct.pack("<I", len(old_meta)))
            buf.write(old_meta)
            new_meta = _serialize_meta_list([mc[2] for mc in self.meta_changes])
            buf.write(struct.pack("<I", len(new_meta)))
            buf.write(new_meta)

        # Compress
        raw = buf.getvalue()
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        return cctx.compress(raw)

    @staticmethod
    def deserialize(data: bytes, dim: int) -> "Delta":
        """Deserialize from compressed binary format."""
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(data)
        buf = io.BytesIO(raw)

        magic = buf.read(4)
        if magic != DELTA_MAGIC:
            raise ValueError(f"Invalid delta magic: {magic!r}")
        version = struct.unpack("<B", buf.read(1))[0]
        if version != DELTA_VERSION:
            raise ValueError(f"Unsupported delta version: {version}")

        delta = Delta()

        # Additions
        n_add = struct.unpack("<I", buf.read(4))[0]
        if n_add > 0:
            delta.add_indices = list(struct.unpack(f"<{n_add}I", buf.read(4 * n_add)))
            t_len = struct.unpack("<I", buf.read(4))[0]
            import numpy as np
            arr = np.frombuffer(buf.read(t_len), dtype=np.float32).reshape(n_add, dim)
            delta.add_values = torch.from_numpy(arr.copy())
            meta_len = struct.unpack("<I", buf.read(4))[0]
            delta.add_metadata = _deserialize_meta_list(buf.read(meta_len))

        # Deletions
        n_del = struct.unpack("<I", buf.read(4))[0]
        if n_del > 0:
            delta.del_indices = list(struct.unpack(f"<{n_del}I", buf.read(4 * n_del)))
            del_val_len = struct.unpack("<I", buf.read(4))[0]
            if del_val_len > 0:
                import numpy as np
                del_arr = np.frombuffer(buf.read(del_val_len), dtype=np.float32).reshape(n_del, dim)
                delta.del_values = torch.from_numpy(del_arr.copy())
            meta_len = struct.unpack("<I", buf.read(4))[0]
            delta.del_metadata = _deserialize_meta_list(buf.read(meta_len))

        # Modifications
        n_mod = struct.unpack("<I", buf.read(4))[0]
        if n_mod > 0:
            delta.mod_indices = list(struct.unpack(f"<{n_mod}I", buf.read(4 * n_mod)))
            old_len = struct.unpack("<I", buf.read(4))[0]
            import numpy as np
            old_arr = np.frombuffer(buf.read(old_len), dtype=np.float32).reshape(n_mod, dim)
            delta.mod_old_values = torch.from_numpy(old_arr.copy())
            new_len = struct.unpack("<I", buf.read(4))[0]
            new_arr = np.frombuffer(buf.read(new_len), dtype=np.float32).reshape(n_mod, dim)
            delta.mod_new_values = torch.from_numpy(new_arr.copy())

        # Meta-only changes
        n_meta = struct.unpack("<I", buf.read(4))[0]
        if n_meta > 0:
            indices = list(struct.unpack(f"<{n_meta}I", buf.read(4 * n_meta)))
            old_len = struct.unpack("<I", buf.read(4))[0]
            old_metas = _deserialize_meta_list(buf.read(old_len))
            new_len = struct.unpack("<I", buf.read(4))[0]
            new_metas = _deserialize_meta_list(buf.read(new_len))
            delta.meta_changes = list(zip(indices, old_metas, new_metas))

        return delta

    def content_hash(self) -> str:
        """SHA-256 hash of the serialized delta."""
        return hashlib.sha256(self.serialize()).hexdigest()


# ─── Metadata serialization helpers ─────────────────────────────

import json


def _serialize_meta_list(metas: List[VectorMeta]) -> bytes:
    """Serialize a list of VectorMeta to JSON bytes."""
    data = []
    for m in metas:
        data.append({
            "id": m.id,
            "document": m.document,
            "metadata": m.metadata,
        })
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def _deserialize_meta_list(raw: bytes) -> List[VectorMeta]:
    """Deserialize a list of VectorMeta from JSON bytes."""
    data = json.loads(raw.decode("utf-8"))
    return [
        VectorMeta(id=d["id"], document=d.get("document"), metadata=d.get("metadata", {}))
        for d in data
    ]
