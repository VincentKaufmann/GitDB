"""IndexManager — secondary indexes on metadata fields for fast lookups."""

import bisect
from collections import defaultdict
from typing import Any, Dict, List, Optional


class IndexManager:
    """In-memory secondary indexes on metadata fields.

    Supports hash indexes (exact lookup) and range indexes (min/max queries).
    Rebuilt on load, updated incrementally on add.
    """

    def __init__(self):
        self._indexes: Dict[str, dict] = {}  # field -> {type, data}

    def create_index(self, field: str, index_type: str = "hash") -> None:
        """Create an index on a metadata field.

        Args:
            field: Metadata field name (e.g. "category", "score").
            index_type: "hash" for exact lookups, "range" for range queries.
        """
        if index_type not in ("hash", "range"):
            raise ValueError(f"Unknown index type: {index_type}. Use 'hash' or 'range'.")
        if field in self._indexes:
            raise ValueError(f"Index already exists on field: {field}")
        if index_type == "hash":
            self._indexes[field] = {"type": "hash", "data": defaultdict(list)}
        else:
            # Range index: sorted list of (value, idx) tuples
            self._indexes[field] = {"type": "range", "keys": [], "entries": []}

    def drop_index(self, field: str) -> None:
        """Remove an index."""
        if field not in self._indexes:
            raise ValueError(f"No index on field: {field}")
        del self._indexes[field]

    def list_indexes(self) -> List[dict]:
        """List all indexes."""
        return [
            {"field": field, "type": info["type"], "entries": self._index_size(field)}
            for field, info in self._indexes.items()
        ]

    def _index_size(self, field: str) -> int:
        info = self._indexes[field]
        if info["type"] == "hash":
            return sum(len(v) for v in info["data"].values())
        else:
            return len(info["keys"])

    def lookup(self, field: str, value: Any) -> List[int]:
        """Exact lookup. Returns row indices matching field==value."""
        if field not in self._indexes:
            raise ValueError(f"No index on field: {field}")
        info = self._indexes[field]
        if info["type"] == "hash":
            return list(info["data"].get(value, []))
        else:
            # Range index: binary search for exact match
            keys = info["keys"]
            entries = info["entries"]
            lo = bisect.bisect_left(keys, value)
            result = []
            while lo < len(keys) and keys[lo] == value:
                result.append(entries[lo])
                lo += 1
            return result

    def range_lookup(self, field: str, min_val=None, max_val=None) -> List[int]:
        """Range lookup. Returns row indices where min_val <= field <= max_val."""
        if field not in self._indexes:
            raise ValueError(f"No index on field: {field}")
        info = self._indexes[field]
        if info["type"] != "range":
            raise ValueError(f"Range lookup requires a 'range' index, got '{info['type']}'")

        keys = info["keys"]
        entries = info["entries"]

        if min_val is not None:
            lo = bisect.bisect_left(keys, min_val)
        else:
            lo = 0

        if max_val is not None:
            hi = bisect.bisect_right(keys, max_val)
        else:
            hi = len(keys)

        return [entries[i] for i in range(lo, hi)]

    def rebuild(self, metadata: list) -> None:
        """Rebuild all indexes from metadata list.

        Args:
            metadata: List of VectorMeta objects.
        """
        for field, info in self._indexes.items():
            if info["type"] == "hash":
                info["data"] = defaultdict(list)
                for idx, meta in enumerate(metadata):
                    val = meta.metadata.get(field)
                    if val is not None:
                        info["data"][val].append(idx)
            else:
                pairs = []
                for idx, meta in enumerate(metadata):
                    val = meta.metadata.get(field)
                    if val is not None:
                        pairs.append((val, idx))
                pairs.sort(key=lambda x: x[0])
                info["keys"] = [p[0] for p in pairs]
                info["entries"] = [p[1] for p in pairs]

    def update(self, idx: int, metadata: dict) -> None:
        """Update indexes for one row.

        Args:
            idx: Row index.
            metadata: The metadata dict for this row.
        """
        for field, info in self._indexes.items():
            val = metadata.get(field)
            if val is not None:
                if info["type"] == "hash":
                    info["data"][val].append(idx)
                else:
                    pos = bisect.bisect_left(info["keys"], val)
                    info["keys"].insert(pos, val)
                    info["entries"].insert(pos, idx)

    def has_index(self, field: str) -> bool:
        """Check if an index exists for a field."""
        return field in self._indexes
