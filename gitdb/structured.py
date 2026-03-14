"""Structured data layer — relational queries, JSON path, JSONL import/export.

GitDB stores arbitrary metadata per vector. This module makes that metadata
queryable like a real database: SQL-style filters, comparisons, aggregations,
JSON path queries, and full JSONL round-trip.

Combined with vector search, this enables hybrid queries:
  "Find vectors similar to X WHERE metadata.category = 'finance' AND metadata.score > 0.8"
"""

import json
import operator
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from gitdb.types import VectorMeta

# ═══════════════════════════════════════════════════════════════
#  Query Operators
# ═══════════════════════════════════════════════════════════════

# Supported operators in where clauses:
#   {"field": value}                → equality
#   {"field": {"$gt": 5}}          → greater than
#   {"field": {"$gte": 5}}         → greater than or equal
#   {"field": {"$lt": 5}}          → less than
#   {"field": {"$lte": 5}}         → less than or equal
#   {"field": {"$ne": 5}}          → not equal
#   {"field": {"$in": [1,2,3]}}    → in list
#   {"field": {"$nin": [1,2,3]}}   → not in list
#   {"field": {"$contains": "x"}}  → string/list contains
#   {"field": {"$regex": "pat"}}   → regex match
#   {"field": {"$exists": true}}   → field exists
#   {"$and": [{...}, {...}]}       → logical AND
#   {"$or": [{...}, {...}]}        → logical OR
#   {"$not": {...}}                → logical NOT

OPS = {
    "$gt": operator.gt,
    "$gte": operator.ge,
    "$lt": operator.lt,
    "$lte": operator.le,
    "$ne": operator.ne,
    "$eq": operator.eq,
}


def _resolve_field(meta: VectorMeta, field: str) -> Any:
    """Resolve a dotted field path against metadata.

    Supports:
      "id"               → meta.id
      "document"         → meta.document
      "category"         → meta.metadata["category"]
      "nested.field"     → meta.metadata["nested"]["field"]
      "tags[0]"          → meta.metadata["tags"][0]
    """
    if field == "id":
        return meta.id
    if field == "document":
        return meta.document

    # Navigate into meta.metadata
    obj = meta.metadata
    for part in field.split("."):
        # Handle array indexing: "tags[0]"
        m = re.match(r"(\w+)\[(\d+)\]$", part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if isinstance(obj, dict):
                obj = obj.get(key)
            if isinstance(obj, (list, tuple)) and idx < len(obj):
                obj = obj[idx]
            else:
                return None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
        if obj is None:
            return None
    return obj


def _eval_condition(meta: VectorMeta, field: str, condition: Any) -> bool:
    """Evaluate a single field condition."""
    value = _resolve_field(meta, field)

    # Simple equality: {"field": "value"}
    if not isinstance(condition, dict):
        return value == condition

    # Operator conditions: {"field": {"$gt": 5}}
    for op_key, op_val in condition.items():
        if op_key in OPS:
            if value is None:
                return False
            try:
                if not OPS[op_key](value, op_val):
                    return False
            except TypeError:
                return False
        elif op_key == "$in":
            if value not in op_val:
                return False
        elif op_key == "$nin":
            if value in op_val:
                return False
        elif op_key == "$contains":
            if value is None:
                return False
            if isinstance(value, str):
                if op_val not in value:
                    return False
            elif isinstance(value, (list, tuple)):
                if op_val not in value:
                    return False
            else:
                return False
        elif op_key == "$regex":
            if value is None or not isinstance(value, str):
                return False
            if not re.search(op_val, value):
                return False
        elif op_key == "$exists":
            exists = value is not None
            if exists != op_val:
                return False
        else:
            return False  # Unknown operator

    return True


def matches(meta: VectorMeta, where: Dict[str, Any]) -> bool:
    """Check if a VectorMeta matches a structured where clause.

    Supports nested $and/$or/$not and all comparison operators.
    """
    for key, condition in where.items():
        if key == "$and":
            if not all(matches(meta, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(matches(meta, sub) for sub in condition):
                return False
        elif key == "$not":
            if matches(meta, condition):
                return False
        else:
            if not _eval_condition(meta, key, condition):
                return False
    return True


# ═══════════════════════════════════════════════════════════════
#  Aggregation
# ═══════════════════════════════════════════════════════════════

def aggregate(
    metadata_list: List[VectorMeta],
    group_by: str,
    agg_field: Optional[str] = None,
    agg_fn: str = "count",
) -> Dict[Any, Any]:
    """Group metadata by a field and aggregate.

    Args:
        metadata_list: List of VectorMeta to aggregate.
        group_by: Field to group by (dotted path supported).
        agg_field: Field to aggregate (for sum/avg/min/max).
        agg_fn: One of count, sum, avg, min, max, collect.

    Returns:
        Dict mapping group key → aggregated value.
    """
    groups: Dict[Any, List] = {}
    for meta in metadata_list:
        key = _resolve_field(meta, group_by)
        if key is None:
            key = "__null__"
        groups.setdefault(key, []).append(meta)

    result = {}
    for key, metas in groups.items():
        if agg_fn == "count":
            result[key] = len(metas)
        elif agg_fn == "collect":
            result[key] = [_resolve_field(m, agg_field) for m in metas] if agg_field else metas
        elif agg_field:
            values = [_resolve_field(m, agg_field) for m in metas]
            values = [v for v in values if v is not None and isinstance(v, (int, float))]
            if not values:
                result[key] = None
            elif agg_fn == "sum":
                result[key] = sum(values)
            elif agg_fn == "avg":
                result[key] = sum(values) / len(values)
            elif agg_fn == "min":
                result[key] = min(values)
            elif agg_fn == "max":
                result[key] = max(values)
        else:
            result[key] = len(metas)

    return result


# ═══════════════════════════════════════════════════════════════
#  Projection (SELECT fields)
# ═══════════════════════════════════════════════════════════════

def project(
    metadata_list: List[VectorMeta],
    fields: List[str],
) -> List[Dict[str, Any]]:
    """Project specific fields from metadata (like SQL SELECT).

    Args:
        metadata_list: Source metadata.
        fields: List of dotted field paths to extract.

    Returns:
        List of dicts with only the requested fields.
    """
    results = []
    for meta in metadata_list:
        row = {}
        for f in fields:
            row[f] = _resolve_field(meta, f)
        results.append(row)
    return results


# ═══════════════════════════════════════════════════════════════
#  Sort
# ═══════════════════════════════════════════════════════════════

def sort_by(
    metadata_list: List[VectorMeta],
    field: str,
    reverse: bool = False,
) -> List[VectorMeta]:
    """Sort metadata by a field value."""
    def key_fn(meta):
        v = _resolve_field(meta, field)
        if v is None:
            return (1, "")  # Nulls last
        return (0, v)
    return sorted(metadata_list, key=key_fn, reverse=reverse)


# ═══════════════════════════════════════════════════════════════
#  JSONL Import / Export
# ═══════════════════════════════════════════════════════════════

def export_jsonl(
    metadata_list: List[VectorMeta],
    output_path: Union[str, Path],
    include_embeddings: bool = False,
    embeddings=None,
):
    """Export metadata (and optionally embeddings) to JSONL.

    Each line: {"id": ..., "document": ..., "metadata": {...}, "embedding": [...]}
    """
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        for i, meta in enumerate(metadata_list):
            row = {
                "id": meta.id,
                "document": meta.document,
                "metadata": meta.metadata,
            }
            if include_embeddings and embeddings is not None and i < len(embeddings):
                row["embedding"] = embeddings[i].cpu().tolist()
            f.write(json.dumps(row) + "\n")


def import_jsonl(
    input_path: Union[str, Path],
) -> Tuple[List[str], List[Optional[str]], List[Dict], List[Optional[List[float]]]]:
    """Import from JSONL. Returns (ids, documents, metadata_list, embeddings_or_none).

    Each line: {"id": ..., "document": ..., "metadata": {...}, "embedding": [...]}
    """
    input_path = Path(input_path)
    ids, docs, metas, embs = [], [], [], []
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids.append(obj.get("id", ""))
            docs.append(obj.get("document"))
            metas.append(obj.get("metadata", {}))
            embs.append(obj.get("embedding"))
    return ids, docs, metas, embs


# ═══════════════════════════════════════════════════════════════
#  JSON Export (for relational-style tables)
# ═══════════════════════════════════════════════════════════════

def to_table(
    metadata_list: List[VectorMeta],
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert metadata to a flat table (list of dicts).

    If fields is None, auto-discovers all fields across all metadata.
    """
    if fields:
        return project(metadata_list, fields)

    # Auto-discover fields
    all_fields = {"id", "document"}
    for meta in metadata_list:
        all_fields.update(meta.metadata.keys())

    return project(metadata_list, sorted(all_fields))
