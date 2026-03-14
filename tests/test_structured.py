"""Tests for structured data layer — relational queries, JSON path, JSONL."""

import json
import pytest
import torch

from gitdb import GitDB
from gitdb.types import VectorMeta
from gitdb.structured import (
    matches, aggregate, project, sort_by, to_table,
    export_jsonl, import_jsonl, _resolve_field,
)


def rvec(n=1, dim=8):
    return torch.randn(n, dim)


@pytest.fixture
def db_with_data(tmp_path):
    """GitDB with structured metadata."""
    db = GitDB(str(tmp_path / "store"), dim=8, device="cpu")
    db.add(
        rvec(5),
        documents=["Revenue Q1", "Revenue Q2", "Legal memo", "HR policy", "Legal contract"],
        metadata=[
            {"category": "finance", "score": 0.9, "quarter": 1, "tags": ["important", "reviewed"]},
            {"category": "finance", "score": 0.7, "quarter": 2, "tags": ["draft"]},
            {"category": "legal", "score": 0.8, "author": "Alice", "tags": ["reviewed"]},
            {"category": "hr", "score": 0.5, "author": "Bob"},
            {"category": "legal", "score": 0.95, "author": "Alice", "tags": ["important", "signed"]},
        ],
    )
    db.commit("Initial structured data")
    return db


# ═══════════════════════════════════════════════════════════════
#  Field Resolution
# ═══════════════════════════════════════════════════════════════

class TestFieldResolution:
    def test_resolve_id(self):
        meta = VectorMeta(id="abc123", document="test", metadata={})
        assert _resolve_field(meta, "id") == "abc123"

    def test_resolve_document(self):
        meta = VectorMeta(id="x", document="hello", metadata={})
        assert _resolve_field(meta, "document") == "hello"

    def test_resolve_simple_field(self):
        meta = VectorMeta(id="x", metadata={"category": "finance"})
        assert _resolve_field(meta, "category") == "finance"

    def test_resolve_nested_field(self):
        meta = VectorMeta(id="x", metadata={"nested": {"deep": "value"}})
        assert _resolve_field(meta, "nested.deep") == "value"

    def test_resolve_array_index(self):
        meta = VectorMeta(id="x", metadata={"tags": ["a", "b", "c"]})
        assert _resolve_field(meta, "tags[1]") == "b"

    def test_resolve_missing_returns_none(self):
        meta = VectorMeta(id="x", metadata={})
        assert _resolve_field(meta, "nonexistent") is None


# ═══════════════════════════════════════════════════════════════
#  Query Operators
# ═══════════════════════════════════════════════════════════════

class TestMatches:
    def test_simple_equality(self):
        meta = VectorMeta(id="x", metadata={"category": "finance"})
        assert matches(meta, {"category": "finance"})
        assert not matches(meta, {"category": "legal"})

    def test_gt(self):
        meta = VectorMeta(id="x", metadata={"score": 0.8})
        assert matches(meta, {"score": {"$gt": 0.5}})
        assert not matches(meta, {"score": {"$gt": 0.9}})

    def test_gte(self):
        meta = VectorMeta(id="x", metadata={"score": 0.8})
        assert matches(meta, {"score": {"$gte": 0.8}})
        assert not matches(meta, {"score": {"$gte": 0.9}})

    def test_lt(self):
        meta = VectorMeta(id="x", metadata={"score": 0.3})
        assert matches(meta, {"score": {"$lt": 0.5}})

    def test_lte(self):
        meta = VectorMeta(id="x", metadata={"score": 0.5})
        assert matches(meta, {"score": {"$lte": 0.5}})

    def test_ne(self):
        meta = VectorMeta(id="x", metadata={"category": "finance"})
        assert matches(meta, {"category": {"$ne": "legal"}})
        assert not matches(meta, {"category": {"$ne": "finance"}})

    def test_in(self):
        meta = VectorMeta(id="x", metadata={"category": "finance"})
        assert matches(meta, {"category": {"$in": ["finance", "legal"]}})
        assert not matches(meta, {"category": {"$in": ["hr"]}})

    def test_nin(self):
        meta = VectorMeta(id="x", metadata={"category": "finance"})
        assert matches(meta, {"category": {"$nin": ["hr", "legal"]}})
        assert not matches(meta, {"category": {"$nin": ["finance"]}})

    def test_contains_string(self):
        meta = VectorMeta(id="x", document="Revenue Q1 report")
        assert matches(meta, {"document": {"$contains": "Revenue"}})
        assert not matches(meta, {"document": {"$contains": "Legal"}})

    def test_contains_list(self):
        meta = VectorMeta(id="x", metadata={"tags": ["important", "reviewed"]})
        assert matches(meta, {"tags": {"$contains": "important"}})
        assert not matches(meta, {"tags": {"$contains": "draft"}})

    def test_regex(self):
        meta = VectorMeta(id="x", document="Revenue Q1 2025")
        assert matches(meta, {"document": {"$regex": r"Q\d+"}})
        assert not matches(meta, {"document": {"$regex": r"^Legal"}})

    def test_exists(self):
        meta = VectorMeta(id="x", metadata={"score": 0.5})
        assert matches(meta, {"score": {"$exists": True}})
        assert matches(meta, {"missing": {"$exists": False}})
        assert not matches(meta, {"score": {"$exists": False}})

    def test_and(self):
        meta = VectorMeta(id="x", metadata={"category": "finance", "score": 0.9})
        assert matches(meta, {"$and": [{"category": "finance"}, {"score": {"$gt": 0.5}}]})
        assert not matches(meta, {"$and": [{"category": "finance"}, {"score": {"$gt": 0.95}}]})

    def test_or(self):
        meta = VectorMeta(id="x", metadata={"category": "hr"})
        assert matches(meta, {"$or": [{"category": "finance"}, {"category": "hr"}]})
        assert not matches(meta, {"$or": [{"category": "finance"}, {"category": "legal"}]})

    def test_not(self):
        meta = VectorMeta(id="x", metadata={"category": "finance"})
        assert matches(meta, {"$not": {"category": "legal"}})
        assert not matches(meta, {"$not": {"category": "finance"}})

    def test_combined_implicit_and(self):
        meta = VectorMeta(id="x", metadata={"category": "finance", "score": 0.9})
        assert matches(meta, {"category": "finance", "score": {"$gt": 0.5}})

    def test_nested_field_query(self):
        meta = VectorMeta(id="x", metadata={"nested": {"value": 42}})
        assert matches(meta, {"nested.value": 42})
        assert matches(meta, {"nested.value": {"$gt": 40}})


# ═══════════════════════════════════════════════════════════════
#  GitDB Integration — select()
# ═══════════════════════════════════════════════════════════════

class TestSelect:
    def test_select_all(self, db_with_data):
        rows = db_with_data.select()
        assert len(rows) == 5

    def test_select_with_where(self, db_with_data):
        rows = db_with_data.select(where={"category": "finance"})
        assert len(rows) == 2

    def test_select_with_fields(self, db_with_data):
        rows = db_with_data.select(fields=["document", "category"])
        assert all(set(r.keys()) == {"document", "category"} for r in rows)

    def test_select_with_gt(self, db_with_data):
        rows = db_with_data.select(where={"score": {"$gt": 0.8}})
        assert len(rows) == 2  # 0.9 and 0.95

    def test_select_with_order(self, db_with_data):
        rows = db_with_data.select(
            fields=["document", "score"],
            where={"score": {"$exists": True}},
            order_by="score",
            reverse=True,
        )
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_select_with_limit(self, db_with_data):
        rows = db_with_data.select(limit=2)
        assert len(rows) == 2

    def test_select_or_filter(self, db_with_data):
        rows = db_with_data.select(
            where={"$or": [{"category": "finance"}, {"category": "legal"}]}
        )
        assert len(rows) == 4  # 2 finance + 2 legal

    def test_select_contains(self, db_with_data):
        rows = db_with_data.select(where={"tags": {"$contains": "important"}})
        assert len(rows) == 2  # Q1 finance + legal contract

    def test_select_regex(self, db_with_data):
        rows = db_with_data.select(where={"document": {"$regex": r"^Revenue"}})
        assert len(rows) == 2


# ═══════════════════════════════════════════════════════════════
#  GROUP BY
# ═══════════════════════════════════════════════════════════════

class TestGroupBy:
    def test_count(self, db_with_data):
        result = db_with_data.group_by("category")
        assert result["finance"] == 2
        assert result["legal"] == 2
        assert result["hr"] == 1

    def test_avg(self, db_with_data):
        result = db_with_data.group_by("category", agg_field="score", agg_fn="avg")
        assert result["finance"] == pytest.approx(0.8, abs=0.01)  # (0.9+0.7)/2

    def test_max(self, db_with_data):
        result = db_with_data.group_by("category", agg_field="score", agg_fn="max")
        assert result["legal"] == 0.95

    def test_with_where(self, db_with_data):
        result = db_with_data.group_by(
            "category",
            where={"score": {"$gt": 0.7}},
        )
        assert result.get("hr") is None  # HR score is 0.5, filtered out
        assert result["finance"] == 1     # Only Q1 (0.9) passes

    def test_collect(self, db_with_data):
        result = db_with_data.group_by("category", agg_field="document", agg_fn="collect")
        assert len(result["finance"]) == 2
        assert "Revenue Q1" in result["finance"]


# ═══════════════════════════════════════════════════════════════
#  JSONL Export / Import
# ═══════════════════════════════════════════════════════════════

class TestJsonl:
    def test_export_import_roundtrip(self, db_with_data, tmp_path):
        out_path = str(tmp_path / "export.jsonl")
        db_with_data.export_jsonl(out_path, include_embeddings=True)

        # Read back
        lines = Path(out_path).read_text().strip().split("\n")
        assert len(lines) == 5
        first = json.loads(lines[0])
        assert "id" in first
        assert "document" in first
        assert "metadata" in first
        assert "embedding" in first
        assert len(first["embedding"]) == 8  # dim=8

    def test_export_without_embeddings(self, db_with_data, tmp_path):
        out_path = str(tmp_path / "export.jsonl")
        db_with_data.export_jsonl(out_path, include_embeddings=False)
        first = json.loads(Path(out_path).read_text().strip().split("\n")[0])
        assert "embedding" not in first

    def test_import_with_embeddings(self, tmp_path):
        # Create JSONL with embeddings
        jsonl_path = tmp_path / "data.jsonl"
        lines = []
        for i in range(3):
            lines.append(json.dumps({
                "id": f"id_{i}",
                "document": f"doc_{i}",
                "metadata": {"idx": i},
                "embedding": torch.randn(8).tolist(),
            }))
        jsonl_path.write_text("\n".join(lines))

        db = GitDB(str(tmp_path / "store"), dim=8, device="cpu")
        indices = db.import_jsonl(str(jsonl_path))
        assert len(indices) == 3
        assert db.tree.size == 3

    def test_import_roundtrip(self, db_with_data, tmp_path):
        """Export then import into fresh store — data survives."""
        export_path = str(tmp_path / "rt.jsonl")
        db_with_data.export_jsonl(export_path, include_embeddings=True)

        db2 = GitDB(str(tmp_path / "store2"), dim=8, device="cpu")
        indices = db2.import_jsonl(export_path)
        assert len(indices) == 5
        assert db2.tree.size == 5

        # Metadata survived
        rows = db2.select(where={"category": "finance"})
        assert len(rows) == 2


# ═══════════════════════════════════════════════════════════════
#  Hybrid: Vector + Structured Query
# ═══════════════════════════════════════════════════════════════

class TestHybridQuery:
    def test_vector_query_with_structured_filter(self, db_with_data):
        """Combine cosine similarity with structured metadata filter."""
        q = torch.randn(8)
        results = db_with_data.query(q, k=10, where={"category": "finance"})
        assert len(results.ids) == 2
        for m in results.metadata:
            assert m["category"] == "finance"

    def test_vector_query_with_gt_filter(self, db_with_data):
        q = torch.randn(8)
        results = db_with_data.query(q, k=10, where={"score": {"$gt": 0.8}})
        assert len(results.ids) == 2  # 0.9 and 0.95

    def test_vector_query_with_or_filter(self, db_with_data):
        q = torch.randn(8)
        results = db_with_data.query(
            q, k=10,
            where={"$or": [{"category": "finance"}, {"category": "hr"}]}
        )
        assert len(results.ids) == 3  # 2 finance + 1 hr

    def test_vector_query_with_contains_filter(self, db_with_data):
        q = torch.randn(8)
        results = db_with_data.query(q, k=10, where={"tags": {"$contains": "reviewed"}})
        assert len(results.ids) == 2  # Q1 finance + legal memo


from pathlib import Path
