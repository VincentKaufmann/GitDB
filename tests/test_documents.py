"""Tests for the document store — MongoDB/SQLite replacement."""

import json
import pytest
import torch
from pathlib import Path

from gitdb import GitDB
from gitdb.documents import DocumentStore, _match_doc


# ═══════════════════════════════════════════════════════════════
#  DocumentStore unit tests
# ═══════════════════════════════════════════════════════════════

class TestDocumentStore:
    def test_insert_returns_id(self):
        ds = DocumentStore()
        _id = ds.insert({"name": "Alice"})
        assert isinstance(_id, str)
        assert len(_id) == 16

    def test_insert_preserves_custom_id(self):
        ds = DocumentStore()
        _id = ds.insert({"_id": "custom123", "name": "Bob"})
        assert _id == "custom123"

    def test_insert_many(self):
        ds = DocumentStore()
        ids = ds.insert_many([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(ids) == 3
        assert ds.size == 3

    def test_find_all(self):
        ds = DocumentStore()
        ds.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        results = ds.find()
        assert len(results) == 3

    def test_find_with_where(self):
        ds = DocumentStore()
        ds.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        results = ds.find(where={"x": {"$gt": 1}})
        assert len(results) == 2

    def test_find_equality(self):
        ds = DocumentStore()
        ds.insert_many([{"name": "Alice"}, {"name": "Bob"}])
        results = ds.find(where={"name": "Alice"})
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_find_with_limit(self):
        ds = DocumentStore()
        ds.insert_many([{"x": i} for i in range(10)])
        results = ds.find(limit=3)
        assert len(results) == 3

    def test_find_with_skip(self):
        ds = DocumentStore()
        ds.insert_many([{"x": i} for i in range(5)])
        results = ds.find(skip=3)
        assert len(results) == 2

    def test_find_one(self):
        ds = DocumentStore()
        ds.insert_many([{"name": "Alice"}, {"name": "Bob"}])
        result = ds.find_one(where={"name": "Bob"})
        assert result["name"] == "Bob"

    def test_find_one_none(self):
        ds = DocumentStore()
        result = ds.find_one(where={"name": "Nobody"})
        assert result is None

    def test_select_with_columns(self):
        ds = DocumentStore()
        ds.insert_many([
            {"name": "Alice", "age": 30, "role": "eng"},
            {"name": "Bob", "age": 25, "role": "pm"},
        ])
        results = ds.select(columns=["name", "age"])
        assert all(set(r.keys()) == {"name", "age"} for r in results)

    def test_select_with_order_by(self):
        ds = DocumentStore()
        ds.insert_many([{"name": "Charlie", "age": 35}, {"name": "Alice", "age": 25}])
        results = ds.select(order_by="age")
        assert results[0]["age"] == 25

    def test_select_desc(self):
        ds = DocumentStore()
        ds.insert_many([{"x": 1}, {"x": 3}, {"x": 2}])
        results = ds.select(order_by="x", desc=True)
        assert results[0]["x"] == 3

    def test_update(self):
        ds = DocumentStore()
        ds.insert({"name": "Alice", "age": 30})
        count = ds.update({"name": "Alice"}, {"age": 31})
        assert count == 1
        assert ds.find_one({"name": "Alice"})["age"] == 31

    def test_update_multiple(self):
        ds = DocumentStore()
        ds.insert_many([{"role": "eng", "level": 1}, {"role": "eng", "level": 2}])
        count = ds.update({"role": "eng"}, {"dept": "engineering"})
        assert count == 2

    def test_delete(self):
        ds = DocumentStore()
        ds.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        count = ds.delete({"x": 2})
        assert count == 1
        assert ds.size == 2

    def test_count(self):
        ds = DocumentStore()
        ds.insert_many([{"a": 1}, {"a": 2}, {"b": 1}])
        assert ds.count() == 3
        assert ds.count(where={"a": {"$exists": True}}) == 2

    def test_distinct(self):
        ds = DocumentStore()
        ds.insert_many([{"color": "red"}, {"color": "blue"}, {"color": "red"}])
        assert set(ds.distinct("color")) == {"red", "blue"}

    def test_aggregate_count(self):
        ds = DocumentStore()
        ds.insert_many([
            {"dept": "eng", "name": "A"},
            {"dept": "eng", "name": "B"},
            {"dept": "pm", "name": "C"},
        ])
        result = ds.aggregate("dept")
        assert result["eng"] == 2
        assert result["pm"] == 1

    def test_aggregate_avg(self):
        ds = DocumentStore()
        ds.insert_many([
            {"dept": "eng", "salary": 100},
            {"dept": "eng", "salary": 200},
            {"dept": "pm", "salary": 150},
        ])
        result = ds.aggregate("dept", agg_field="salary", agg_fn="avg")
        assert result["eng"] == 150

    def test_snapshot_restore(self):
        ds = DocumentStore()
        ds.insert_many([{"x": 1}, {"x": 2}])
        snap = ds.snapshot()
        ds2 = DocumentStore()
        ds2.restore(snap)
        assert ds2.size == 2

    def test_compact(self):
        ds = DocumentStore()
        ds.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        ds.delete({"x": 2})
        assert ds.size == 2
        ds.compact()
        assert ds.size == 2
        assert len(ds._docs) == 2  # actually removed

    def test_diff_docs(self):
        ds = DocumentStore()
        ds.insert({"_id": "a1", "val": 1})
        ds.insert({"_id": "a2", "val": 2})
        other = [{"_id": "a1", "val": 10}, {"_id": "a3", "val": 3}]
        diff = ds.diff_docs(other)
        assert len(diff["added"]) == 1
        assert len(diff["removed"]) == 1
        assert len(diff["modified"]) == 1


# ═══════════════════════════════════════════════════════════════
#  Query operator tests
# ═══════════════════════════════════════════════════════════════

class TestQueryOperators:
    def test_gt(self):
        assert _match_doc({"age": 30}, {"age": {"$gt": 25}})
        assert not _match_doc({"age": 20}, {"age": {"$gt": 25}})

    def test_lt(self):
        assert _match_doc({"age": 20}, {"age": {"$lt": 25}})

    def test_gte_lte(self):
        assert _match_doc({"x": 5}, {"x": {"$gte": 5}})
        assert _match_doc({"x": 5}, {"x": {"$lte": 5}})

    def test_ne(self):
        assert _match_doc({"x": 1}, {"x": {"$ne": 2}})
        assert not _match_doc({"x": 1}, {"x": {"$ne": 1}})

    def test_in(self):
        assert _match_doc({"role": "eng"}, {"role": {"$in": ["eng", "pm"]}})
        assert not _match_doc({"role": "qa"}, {"role": {"$in": ["eng", "pm"]}})

    def test_nin(self):
        assert _match_doc({"role": "qa"}, {"role": {"$nin": ["eng", "pm"]}})

    def test_contains(self):
        assert _match_doc({"tags": ["a", "b"]}, {"tags": {"$contains": "a"}})
        assert _match_doc({"name": "Alice"}, {"name": {"$contains": "lic"}})

    def test_regex(self):
        assert _match_doc({"email": "alice@example.com"}, {"email": {"$regex": r"@example\.com$"}})

    def test_exists(self):
        assert _match_doc({"x": 1}, {"x": {"$exists": True}})
        assert _match_doc({}, {"x": {"$exists": False}})

    def test_and(self):
        assert _match_doc({"x": 5}, {"$and": [{"x": {"$gt": 3}}, {"x": {"$lt": 10}}]})

    def test_or(self):
        assert _match_doc({"x": 1}, {"$or": [{"x": 1}, {"x": 2}]})

    def test_not(self):
        assert _match_doc({"x": 5}, {"$not": {"x": {"$lt": 3}}})

    def test_nested_field(self):
        doc = {"user": {"address": {"city": "NYC"}}}
        assert _match_doc(doc, {"user.address.city": "NYC"})


# ═══════════════════════════════════════════════════════════════
#  GitDB integration tests
# ═══════════════════════════════════════════════════════════════

class TestGitDBDocuments:
    def setup_method(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.db = GitDB(f"{self.tmpdir}/test_store", dim=4, device="cpu")

    def test_insert_and_find(self):
        self.db.insert({"name": "Alice", "age": 30})
        self.db.insert({"name": "Bob", "age": 25})
        results = self.db.find(where={"age": {"$gt": 27}})
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_insert_list(self):
        ids = self.db.insert([{"a": 1}, {"a": 2}])
        assert len(ids) == 2
        assert self.db.count_docs() == 2

    def test_commit_persists_documents(self):
        self.db.insert({"name": "Alice"})
        h = self.db.commit("add alice")
        assert h is not None
        # Reset and verify
        self.db.reset("HEAD")
        results = self.db.find()
        assert len(results) == 1

    def test_documents_and_vectors_together(self):
        # Add vectors
        self.db.add(torch.randn(3, 4), documents=["a", "b", "c"])
        # Add documents
        self.db.insert({"type": "config", "key": "version", "value": "1.0"})
        h = self.db.commit("mixed data")
        assert self.db.tree.size == 3
        assert self.db.docs.size == 1

    def test_stash_preserves_documents(self):
        self.db.insert({"name": "WIP doc"})
        self.db.stash("work in progress")
        # After stash, documents should be restored to HEAD (empty)
        assert self.db.docs.size == 0
        # Pop should bring them back
        self.db.stash_pop()
        assert self.db.docs.size == 1
        assert self.db.find_one()["name"] == "WIP doc"

    def test_update_docs(self):
        self.db.insert({"name": "Alice", "age": 30})
        count = self.db.update_docs({"name": "Alice"}, {"age": 31})
        assert count == 1
        assert self.db.find_one({"name": "Alice"})["age"] == 31

    def test_delete_docs(self):
        self.db.insert({"name": "Alice"})
        self.db.insert({"name": "Bob"})
        count = self.db.delete_docs({"name": "Bob"})
        assert count == 1
        assert self.db.count_docs() == 1

    def test_count_docs(self):
        self.db.insert([{"x": 1}, {"x": 2}, {"x": 3}])
        assert self.db.count_docs() == 3
        assert self.db.count_docs(where={"x": {"$gt": 1}}) == 2

    def test_distinct(self):
        self.db.insert([{"color": "red"}, {"color": "blue"}, {"color": "red"}])
        assert set(self.db.distinct("color")) == {"red", "blue"}

    def test_aggregate_docs(self):
        self.db.insert([
            {"dept": "eng", "salary": 100},
            {"dept": "eng", "salary": 200},
            {"dept": "pm", "salary": 150},
        ])
        result = self.db.aggregate_docs("dept", agg_field="salary", agg_fn="sum")
        assert result["eng"] == 300

    def test_document_only_commit(self):
        """Can commit with only document changes, no vectors."""
        self.db.insert({"key": "value"})
        h = self.db.commit("docs only")
        assert h is not None

    def test_status_shows_documents(self):
        self.db.insert({"x": 1})
        s = self.db.status()
        assert s["staged_doc_inserts"] == 1
        assert s["documents"] == 1

    def test_multiple_commits_with_documents(self):
        self.db.insert({"version": 1})
        self.db.commit("v1")
        self.db.insert({"version": 2})
        self.db.commit("v2")
        assert self.db.count_docs() == 2

    def test_reset_restores_documents(self):
        self.db.insert({"x": 1})
        h1 = self.db.commit("first")
        self.db.insert({"x": 2})
        self.db.commit("second")
        self.db.reset(h1)
        assert self.db.count_docs() == 1

    def test_branch_and_switch_documents(self):
        self.db.insert({"branch": "main", "val": 1})
        self.db.commit("main doc")
        self.db.branch("feature")
        self.db.switch("feature")
        self.db.insert({"branch": "feature", "val": 2})
        self.db.commit("feature doc")
        assert self.db.count_docs() == 2
        self.db.switch("main")
        assert self.db.count_docs() == 1
        self.db.switch("feature")
        assert self.db.count_docs() == 2

    def test_merge_documents(self):
        self.db.insert({"_id": "shared", "val": 1})
        self.db.commit("base")
        self.db.branch("feature")
        self.db.switch("feature")
        self.db.insert({"_id": "feat_doc", "val": 2})
        self.db.commit("feature doc")
        self.db.switch("main")
        result = self.db.merge("feature")
        # Should have both documents after merge
        assert self.db.count_docs() == 2
        assert self.db.find_one({"_id": "feat_doc"}) is not None

    def test_cherry_pick_documents(self):
        self.db.insert({"_id": "base", "val": 0})
        self.db.commit("base")
        self.db.branch("feature")
        self.db.switch("feature")
        self.db.insert({"_id": "cherry", "val": 42})
        feat_hash = self.db.commit("add cherry doc")
        self.db.switch("main")
        assert self.db.count_docs() == 1
        self.db.cherry_pick(feat_hash)
        assert self.db.count_docs() == 2
        assert self.db.find_one({"_id": "cherry"})["val"] == 42

    def test_document_only_mode(self):
        """Pure document store — no vectors at all."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        db = GitDB(f"{tmpdir}/docstore", dim=4, device="cpu")
        db.insert({"name": "Alice", "role": "engineer"})
        db.insert({"name": "Bob", "role": "pm"})
        h1 = db.commit("initial")
        db.insert({"name": "Charlie", "role": "engineer"})
        db.commit("add charlie")
        # SQL-style query
        engineers = db.docs.select(
            columns=["name"], where={"role": "engineer"}, order_by="name"
        )
        assert len(engineers) == 2
        assert engineers[0]["name"] == "Alice"
        # Reset to first commit
        db.reset(h1)
        assert db.count_docs() == 2

    def test_merge_documents_with_vectors(self):
        """Merge works when both branches have docs AND vectors."""
        self.db.add(torch.randn(2, 4), documents=["v1", "v2"])
        self.db.insert({"_id": "doc_main", "source": "main"})
        self.db.commit("main data")
        self.db.branch("experiment")
        self.db.switch("experiment")
        self.db.add(torch.randn(1, 4), documents=["v3"])
        self.db.insert({"_id": "doc_exp", "source": "experiment"})
        self.db.commit("experiment data")
        self.db.switch("main")
        result = self.db.merge("experiment")
        assert self.db.tree.size == 3  # 2 + 1 vectors
        assert self.db.count_docs() == 2  # doc_main + doc_exp
