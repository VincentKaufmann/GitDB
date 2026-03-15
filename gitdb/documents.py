"""Document & Table store — versioned JSON + SQL storage without vectors.

MongoDB replacement:
    db.collection("users").insert({"name": "Alice", "age": 30})
    db.collection("users").find(where={"age": {"$gt": 25}})

SQLite replacement:
    db.create_table("users", {"name": "text", "age": "integer", "email": "text"})
    db.table("users").insert({"name": "Alice", "age": 30, "email": "a@b.com"})
    db.table("users").select(columns=["name"], where={"age": {"$gt": 25}})

Both versioned through git: commit, branch, merge, stash, cherry-pick, diff.
"""

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════════
#  Table — SQLite-style named table with column schema
# ═══════════════════════════════════════════════════════════════

COLUMN_TYPES = {
    "text": str, "string": str, "str": str,
    "integer": int, "int": int,
    "float": float, "real": float, "number": float,
    "boolean": bool, "bool": bool,
    "json": (dict, list),
    "any": object,
}


class Table:
    """A named table with optional column schema enforcement.

    Like an SQLite table: named, typed columns, INSERT/SELECT/UPDATE/DELETE.
    Schema is optional — omit it for schemaless (MongoDB-like) behavior.
    """

    def __init__(self, name: str, columns: Optional[Dict[str, str]] = None):
        self.name = name
        self.columns = columns  # {"name": "text", "age": "integer"} or None
        self._rows: List[Dict[str, Any]] = []
        self._tombstones: set = set()
        self._auto_id = 0

    @property
    def size(self) -> int:
        return len(self._rows) - len(self._tombstones)

    def _validate(self, row: Dict[str, Any]):
        """Validate row against column schema."""
        if not self.columns:
            return
        for col, col_type in self.columns.items():
            if col in row and row[col] is not None:
                expected = COLUMN_TYPES.get(col_type, object)
                if expected is not object and not isinstance(row[col], expected):
                    try:
                        # Try coercion
                        if expected is int:
                            row[col] = int(row[col])
                        elif expected is float:
                            row[col] = float(row[col])
                        elif expected is str:
                            row[col] = str(row[col])
                        elif expected is bool:
                            row[col] = bool(row[col])
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Column '{col}' expects {col_type}, got {type(row[col]).__name__}"
                        )

    def insert(self, row: Dict[str, Any]) -> int:
        """Insert a row. Returns row ID."""
        row = dict(row)
        if "_rowid" not in row:
            self._auto_id += 1
            row["_rowid"] = self._auto_id
        self._validate(row)
        self._rows.append(row)
        return row["_rowid"]

    def insert_many(self, rows: List[Dict[str, Any]]) -> List[int]:
        return [self.insert(r) for r in rows]

    def select(
        self,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """SQL-style SELECT."""
        rows = []
        for i, row in enumerate(self._rows):
            if i in self._tombstones:
                continue
            if where and not _match_doc(row, where):
                continue
            rows.append(row)

        if order_by:
            rows.sort(
                key=lambda r: (r.get(order_by) is None, r.get(order_by, "")),
                reverse=desc,
            )
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        if columns:
            rows = [{c: r.get(c) for c in columns} for r in rows]
        return rows

    def update(self, where: Dict[str, Any], set_fields: Dict[str, Any]) -> int:
        """UPDATE rows matching where."""
        count = 0
        for i, row in enumerate(self._rows):
            if i in self._tombstones:
                continue
            if _match_doc(row, where):
                for k, v in set_fields.items():
                    row[k] = v
                self._validate(row)
                count += 1
        return count

    def delete(self, where: Dict[str, Any]) -> int:
        """DELETE rows matching where."""
        count = 0
        for i, row in enumerate(self._rows):
            if i in self._tombstones:
                continue
            if _match_doc(row, where):
                self._tombstones.add(i)
                count += 1
        return count

    def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        if where is None:
            return self.size
        return len(self.select(where=where))

    def distinct(self, field: str, where: Optional[Dict[str, Any]] = None) -> List[Any]:
        rows = self.select(where=where)
        seen = set()
        result = []
        for r in rows:
            v = r.get(field)
            key = v if not isinstance(v, dict) else json.dumps(v, sort_keys=True)
            if v is not None and key not in seen:
                seen.add(key)
                result.append(v)
        return result

    def aggregate(
        self, group_by: str, agg_field: Optional[str] = None,
        agg_fn: str = "count", where: Optional[Dict[str, Any]] = None,
    ) -> Dict[Any, Any]:
        rows = self.select(where=where)
        groups: Dict[Any, List] = {}
        for r in rows:
            key = r.get(group_by, "__null__")
            groups.setdefault(key, []).append(r)
        result = {}
        for key, group in groups.items():
            if agg_fn == "count":
                result[key] = len(group)
            elif agg_fn == "collect":
                result[key] = [r.get(agg_field) for r in group] if agg_field else group
            elif agg_field:
                vals = [r.get(agg_field) for r in group if isinstance(r.get(agg_field), (int, float))]
                if not vals:
                    result[key] = None
                elif agg_fn == "sum":
                    result[key] = sum(vals)
                elif agg_fn == "avg":
                    result[key] = sum(vals) / len(vals)
                elif agg_fn == "min":
                    result[key] = min(vals)
                elif agg_fn == "max":
                    result[key] = max(vals)
            else:
                result[key] = len(group)
        return result

    def compact(self):
        self._rows = [r for i, r in enumerate(self._rows) if i not in self._tombstones]
        self._tombstones.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize table to dict for persistence."""
        self.compact()
        return {
            "name": self.name,
            "columns": self.columns,
            "auto_id": self._auto_id,
            "rows": list(self._rows),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Table":
        """Restore table from dict."""
        t = cls(data["name"], data.get("columns"))
        t._auto_id = data.get("auto_id", 0)
        t._rows = data.get("rows", [])
        return t

    def __repr__(self):
        cols = f", columns={list(self.columns.keys())}" if self.columns else ""
        return f"Table('{self.name}', {self.size} rows{cols})"


# ═══════════════════════════════════════════════════════════════
#  TableStore — manages multiple named tables
# ═══════════════════════════════════════════════════════════════

class TableStore:
    """Collection of named tables, serializable as one JSON blob.

    Persisted per-commit alongside vector data and documents.
    """

    def __init__(self):
        self._tables: Dict[str, Table] = {}

    def create(self, name: str, columns: Optional[Dict[str, str]] = None) -> Table:
        """Create a new table. Returns it."""
        if name in self._tables:
            raise ValueError(f"Table '{name}' already exists")
        t = Table(name, columns)
        self._tables[name] = t
        return t

    def get(self, name: str) -> Table:
        """Get table by name."""
        if name not in self._tables:
            raise ValueError(f"Table '{name}' does not exist")
        return self._tables[name]

    def get_or_create(self, name: str, columns: Optional[Dict[str, str]] = None) -> Table:
        """Get existing table or create it."""
        if name not in self._tables:
            return self.create(name, columns)
        return self._tables[name]

    def drop(self, name: str):
        """Drop a table."""
        if name not in self._tables:
            raise ValueError(f"Table '{name}' does not exist")
        del self._tables[name]

    def list_tables(self) -> List[str]:
        return list(self._tables.keys())

    @property
    def size(self) -> int:
        return sum(t.size for t in self._tables.values())

    def snapshot(self) -> bytes:
        """Serialize all tables to JSON bytes."""
        data = {name: t.to_dict() for name, t in self._tables.items()}
        return json.dumps(data, default=str).encode()

    def restore(self, data: bytes):
        """Restore all tables from JSON bytes."""
        self._tables = {}
        raw = json.loads(data.decode())
        for name, tdata in raw.items():
            self._tables[name] = Table.from_dict(tdata)

    def __repr__(self):
        names = ", ".join(self._tables.keys()) if self._tables else "empty"
        return f"TableStore({names})"


class DocumentStore:
    """In-memory document collection with JSONL persistence.

    Each document is a dict with an auto-generated _id field.
    Supports MongoDB-style queries and SQL-style selects.
    """

    def __init__(self):
        self._docs: List[Dict[str, Any]] = []
        self._tombstones: set = set()  # indices of deleted docs

    @property
    def size(self) -> int:
        return len(self._docs) - len(self._tombstones)

    @property
    def total_rows(self) -> int:
        return len(self._docs)

    # ─── CRUD ─────────────────────────────────────────────────

    def insert(self, doc: Dict[str, Any]) -> str:
        """Insert a document. Returns the _id."""
        doc = dict(doc)  # copy
        if "_id" not in doc:
            doc["_id"] = hashlib.sha256(
                json.dumps(doc, sort_keys=True, default=str).encode()
                + str(time.time()).encode()
                + uuid.uuid4().bytes
            ).hexdigest()[:16]
        self._docs.append(doc)
        return doc["_id"]

    def insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents. Returns list of _ids."""
        return [self.insert(d) for d in docs]

    def find(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Find documents matching a query (MongoDB-style).

        Args:
            where: Query filter using MongoDB operators ($gt, $lt, $in, etc.)
            limit: Max documents to return.
            skip: Number to skip (for pagination).
        """
        results = []
        for i, doc in enumerate(self._docs):
            if i in self._tombstones:
                continue
            if where and not _match_doc(doc, where):
                continue
            results.append(doc)

        if skip:
            results = results[skip:]
        if limit is not None:
            results = results[:limit]
        return results

    def find_one(self, where: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Find a single matching document."""
        results = self.find(where=where, limit=1)
        return results[0] if results else None

    def select(
        self,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """SQL-style select with projection, filtering, sorting.

        Args:
            columns: Fields to return (None = all).
            where: Filter conditions.
            order_by: Field to sort by.
            desc: Sort descending.
            limit: Max rows.
            offset: Skip rows.
        """
        rows = self.find(where=where)

        if order_by:
            rows.sort(
                key=lambda d: (d.get(order_by) is None, d.get(order_by, "")),
                reverse=desc,
            )

        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]

        if columns:
            rows = [{c: d.get(c) for c in columns} for d in rows]

        return rows

    def update(
        self,
        where: Dict[str, Any],
        set_fields: Dict[str, Any],
    ) -> int:
        """Update documents matching where clause. Returns count updated."""
        count = 0
        for i, doc in enumerate(self._docs):
            if i in self._tombstones:
                continue
            if _match_doc(doc, where):
                for k, v in set_fields.items():
                    doc[k] = v
                count += 1
        return count

    def delete(self, where: Dict[str, Any]) -> int:
        """Delete documents matching where clause. Returns count deleted."""
        count = 0
        for i, doc in enumerate(self._docs):
            if i in self._tombstones:
                continue
            if _match_doc(doc, where):
                self._tombstones.add(i)
                count += 1
        return count

    def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching query."""
        if where is None:
            return self.size
        return len(self.find(where=where))

    def distinct(self, field: str, where: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Get distinct values for a field."""
        docs = self.find(where=where)
        seen = set()
        result = []
        for d in docs:
            v = d.get(field)
            if v is not None and v not in seen:
                seen.add(v if not isinstance(v, dict) else json.dumps(v, sort_keys=True))
                result.append(v)
        return result

    def aggregate(
        self,
        group_by: str,
        agg_field: Optional[str] = None,
        agg_fn: str = "count",
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[Any, Any]:
        """Group by a field and aggregate.

        agg_fn: count, sum, avg, min, max, collect
        """
        docs = self.find(where=where)
        groups: Dict[Any, List] = {}
        for d in docs:
            key = d.get(group_by, "__null__")
            groups.setdefault(key, []).append(d)

        result = {}
        for key, group in groups.items():
            if agg_fn == "count":
                result[key] = len(group)
            elif agg_fn == "collect":
                result[key] = [d.get(agg_field) for d in group] if agg_field else group
            elif agg_field:
                vals = [d.get(agg_field) for d in group if isinstance(d.get(agg_field), (int, float))]
                if not vals:
                    result[key] = None
                elif agg_fn == "sum":
                    result[key] = sum(vals)
                elif agg_fn == "avg":
                    result[key] = sum(vals) / len(vals)
                elif agg_fn == "min":
                    result[key] = min(vals)
                elif agg_fn == "max":
                    result[key] = max(vals)
            else:
                result[key] = len(group)
        return result

    # ─── Snapshot / Restore ───────────────────────────────────

    def compact(self):
        """Remove tombstoned docs."""
        self._docs = [d for i, d in enumerate(self._docs) if i not in self._tombstones]
        self._tombstones.clear()

    def snapshot(self) -> bytes:
        """Serialize current state to JSONL bytes."""
        live = [d for i, d in enumerate(self._docs) if i not in self._tombstones]
        lines = [json.dumps(d, default=str) for d in live]
        return ("\n".join(lines) + "\n" if lines else "").encode()

    def restore(self, data: bytes):
        """Restore from JSONL bytes."""
        self._docs = []
        self._tombstones = set()
        text = data.decode()
        for line in text.strip().split("\n"):
            if line.strip():
                self._docs.append(json.loads(line))

    def get_docs(self) -> List[Dict[str, Any]]:
        """Get all live documents."""
        return [d for i, d in enumerate(self._docs) if i not in self._tombstones]

    def load_docs(self, docs: List[Dict[str, Any]]):
        """Replace all documents."""
        self._docs = list(docs)
        self._tombstones = set()

    def diff_docs(self, other_docs: List[Dict[str, Any]]) -> Dict[str, List]:
        """Diff current state against another doc list.

        Returns {"added": [...], "removed": [...], "modified": [...]}.
        """
        current = {d["_id"]: d for d in self.get_docs() if "_id" in d}
        other = {d["_id"]: d for d in other_docs if "_id" in d}

        added = [other[k] for k in other if k not in current]
        removed = [current[k] for k in current if k not in other]
        modified = []
        for k in current:
            if k in other and current[k] != other[k]:
                modified.append({"_id": k, "before": current[k], "after": other[k]})

        return {"added": added, "removed": removed, "modified": modified}


# ═══════════════════════════════════════════════════════════════
#  Query matching (MongoDB-style operators)
# ═══════════════════════════════════════════════════════════════

import operator
import re

_OPS = {
    "$gt": operator.gt,
    "$gte": operator.ge,
    "$lt": operator.lt,
    "$lte": operator.le,
    "$ne": operator.ne,
    "$eq": operator.eq,
}


def _resolve(doc: Dict[str, Any], field: str) -> Any:
    """Resolve dotted field path: 'a.b.c' → doc['a']['b']['c']."""
    obj = doc
    for part in field.split("."):
        # Array index: "items[0]"
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


def _eval_cond(doc: Dict[str, Any], field: str, condition: Any) -> bool:
    """Evaluate one field condition."""
    value = _resolve(doc, field)

    if not isinstance(condition, dict):
        return value == condition

    for op, op_val in condition.items():
        if op in _OPS:
            if value is None:
                return False
            try:
                if not _OPS[op](value, op_val):
                    return False
            except TypeError:
                return False
        elif op == "$in":
            if value not in op_val:
                return False
        elif op == "$nin":
            if value in op_val:
                return False
        elif op == "$contains":
            if value is None or (
                isinstance(value, (str, list, tuple)) and op_val not in value
            ):
                return False
        elif op == "$regex":
            if not isinstance(value, str) or not re.search(op_val, value):
                return False
        elif op == "$exists":
            if (value is not None) != op_val:
                return False
        else:
            return False
    return True


def _match_doc(doc: Dict[str, Any], where: Dict[str, Any]) -> bool:
    """Check if a document matches a query."""
    for key, cond in where.items():
        if key == "$and":
            if not all(_match_doc(doc, sub) for sub in cond):
                return False
        elif key == "$or":
            if not any(_match_doc(doc, sub) for sub in cond):
                return False
        elif key == "$not":
            if _match_doc(doc, cond):
                return False
        else:
            if not _eval_cond(doc, key, cond):
                return False
    return True
