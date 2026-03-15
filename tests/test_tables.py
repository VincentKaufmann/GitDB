"""Tests for SQL-style tables — the SQLite replacement.

Every git operation should work on tables: commit, branch, merge,
cherry-pick, stash, rebase, reset, checkout, amend, squash.
"""

import pytest
import tempfile
import torch

from gitdb import GitDB
from gitdb.documents import Table, TableStore


# ═══════════════════════════════════════════════════════════════
#  Table unit tests
# ═══════════════════════════════════════════════════════════════

class TestTable:
    def test_create_table(self):
        t = Table("users", {"name": "text", "age": "integer"})
        assert t.name == "users"
        assert t.size == 0

    def test_insert(self):
        t = Table("t", {"x": "integer"})
        rid = t.insert({"x": 42})
        assert isinstance(rid, int)
        assert t.size == 1

    def test_insert_many(self):
        t = Table("t")
        ids = t.insert_many([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(ids) == 3
        assert t.size == 3

    def test_select_all(self):
        t = Table("t")
        t.insert_many([{"x": 1}, {"x": 2}])
        rows = t.select()
        assert len(rows) == 2

    def test_select_where(self):
        t = Table("t")
        t.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        rows = t.select(where={"x": {"$gt": 1}})
        assert len(rows) == 2

    def test_select_columns(self):
        t = Table("t", {"name": "text", "age": "integer", "email": "text"})
        t.insert({"name": "Alice", "age": 30, "email": "a@b.com"})
        rows = t.select(columns=["name", "age"])
        assert rows == [{"name": "Alice", "age": 30}]

    def test_select_order_by(self):
        t = Table("t")
        t.insert_many([{"x": 3}, {"x": 1}, {"x": 2}])
        rows = t.select(order_by="x")
        assert [r["x"] for r in rows] == [1, 2, 3]

    def test_select_desc(self):
        t = Table("t")
        t.insert_many([{"x": 1}, {"x": 3}, {"x": 2}])
        rows = t.select(order_by="x", desc=True)
        assert rows[0]["x"] == 3

    def test_select_limit_offset(self):
        t = Table("t")
        t.insert_many([{"x": i} for i in range(10)])
        rows = t.select(order_by="x", limit=3, offset=2)
        assert len(rows) == 3
        assert rows[0]["x"] == 2

    def test_update(self):
        t = Table("t")
        t.insert({"name": "Alice", "age": 30})
        count = t.update({"name": "Alice"}, {"age": 31})
        assert count == 1
        assert t.select(where={"name": "Alice"})[0]["age"] == 31

    def test_delete(self):
        t = Table("t")
        t.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        count = t.delete({"x": 2})
        assert count == 1
        assert t.size == 2

    def test_count(self):
        t = Table("t")
        t.insert_many([{"x": 1}, {"x": 2}, {"x": 3}])
        assert t.count() == 3
        assert t.count(where={"x": {"$gt": 1}}) == 2

    def test_distinct(self):
        t = Table("t")
        t.insert_many([{"c": "red"}, {"c": "blue"}, {"c": "red"}])
        assert set(t.distinct("c")) == {"red", "blue"}

    def test_aggregate(self):
        t = Table("t")
        t.insert_many([
            {"dept": "eng", "sal": 100},
            {"dept": "eng", "sal": 200},
            {"dept": "pm", "sal": 150},
        ])
        result = t.aggregate("dept", agg_field="sal", agg_fn="avg")
        assert result["eng"] == 150

    def test_schema_validation(self):
        t = Table("t", {"name": "text", "age": "integer"})
        t.insert({"name": "Alice", "age": 30})  # OK
        t.insert({"name": "Bob", "age": "25"})   # coerced to int
        assert t.select(where={"name": "Bob"})[0]["age"] == 25

    def test_schema_validation_error(self):
        t = Table("t", {"age": "integer"})
        with pytest.raises(ValueError, match="expects integer"):
            t.insert({"age": "not_a_number"})

    def test_serialize_restore(self):
        t = Table("t", {"name": "text"})
        t.insert_many([{"name": "Alice"}, {"name": "Bob"}])
        d = t.to_dict()
        t2 = Table.from_dict(d)
        assert t2.size == 2
        assert t2.columns == {"name": "text"}


# ═══════════════════════════════════════════════════════════════
#  TableStore unit tests
# ═══════════════════════════════════════════════════════════════

class TestTableStore:
    def test_create_and_get(self):
        ts = TableStore()
        ts.create("users", {"name": "text"})
        t = ts.get("users")
        assert t.name == "users"

    def test_list_tables(self):
        ts = TableStore()
        ts.create("a")
        ts.create("b")
        assert set(ts.list_tables()) == {"a", "b"}

    def test_drop(self):
        ts = TableStore()
        ts.create("x")
        ts.drop("x")
        assert ts.list_tables() == []

    def test_snapshot_restore(self):
        ts = TableStore()
        t = ts.create("users", {"name": "text"})
        t.insert({"name": "Alice"})
        snap = ts.snapshot()
        ts2 = TableStore()
        ts2.restore(snap)
        assert ts2.list_tables() == ["users"]
        assert ts2.get("users").size == 1


# ═══════════════════════════════════════════════════════════════
#  GitDB table integration — full git magic
# ═══════════════════════════════════════════════════════════════

class TestGitDBTables:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = GitDB(f"{self.tmpdir}/test_store", dim=4, device="cpu")

    def test_create_table_and_insert(self):
        self.db.create_table("users", {"name": "text", "age": "integer"})
        self.db.insert_into("users", {"name": "Alice", "age": 30})
        rows = self.db.select_from("users")
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"

    def test_commit_persists_tables(self):
        self.db.create_table("config", {"key": "text", "value": "text"})
        self.db.insert_into("config", {"key": "version", "value": "1.0"})
        self.db.commit("add config")
        self.db.reset("HEAD")
        assert "config" in self.db.list_tables()
        assert self.db.select_from("config")[0]["value"] == "1.0"

    def test_branch_and_switch_tables(self):
        self.db.create_table("t", {"x": "integer"})
        self.db.insert_into("t", {"x": 1})
        self.db.commit("main table")
        self.db.branch("feature")
        self.db.switch("feature")
        self.db.insert_into("t", {"x": 2})
        self.db.commit("feature row")
        assert self.db.table("t").size == 2
        self.db.switch("main")
        assert self.db.table("t").size == 1
        self.db.switch("feature")
        assert self.db.table("t").size == 2

    def test_merge_tables(self):
        self.db.create_table("t")
        self.db.insert_into("t", {"_rowid": 1, "val": "base"})
        self.db.commit("base")
        self.db.branch("feature")
        self.db.switch("feature")
        self.db.insert_into("t", {"_rowid": 2, "val": "feature"})
        self.db.commit("feature row")
        self.db.switch("main")
        self.db.merge("feature")
        assert self.db.table("t").size == 2

    def test_merge_new_table_from_branch(self):
        self.db.create_table("base_t")
        self.db.insert_into("base_t", {"x": 1})
        self.db.commit("base")
        self.db.branch("feature")
        self.db.switch("feature")
        self.db.create_table("feature_t", {"y": "integer"})
        self.db.insert_into("feature_t", {"y": 42})
        self.db.commit("new table on feature")
        self.db.switch("main")
        assert "feature_t" not in self.db.list_tables()
        self.db.merge("feature")
        assert "feature_t" in self.db.list_tables()
        assert self.db.select_from("feature_t")[0]["y"] == 42

    def test_stash_and_pop_tables(self):
        self.db.create_table("wip")
        self.db.insert_into("wip", {"status": "in progress"})
        self.db.stash("WIP")
        assert "wip" not in self.db.list_tables()
        self.db.stash_pop()
        assert "wip" in self.db.list_tables()
        assert self.db.table("wip").size == 1

    def test_cherry_pick_tables(self):
        self.db.create_table("t")
        self.db.insert_into("t", {"_rowid": 1, "v": "base"})
        self.db.commit("base")
        self.db.branch("feature")
        self.db.switch("feature")
        self.db.insert_into("t", {"_rowid": 2, "v": "cherry"})
        feat_hash = self.db.commit("cherry row")
        self.db.switch("main")
        self.db.cherry_pick(feat_hash)
        assert self.db.table("t").size == 2

    def test_reset_restores_tables(self):
        self.db.create_table("t")
        self.db.insert_into("t", {"x": 1})
        h1 = self.db.commit("v1")
        self.db.insert_into("t", {"x": 2})
        self.db.commit("v2")
        assert self.db.table("t").size == 2
        self.db.reset(h1)
        assert self.db.table("t").size == 1

    def test_checkout_loads_tables(self):
        self.db.create_table("t")
        self.db.insert_into("t", {"x": 1})
        h1 = self.db.commit("v1")
        self.db.insert_into("t", {"x": 2})
        h2 = self.db.commit("v2")
        self.db.checkout(h1)
        assert self.db.table("t").size == 1

    def test_tables_with_vectors_and_docs(self):
        """All three data types in one store."""
        self.db.add(torch.randn(2, 4), documents=["vec1", "vec2"])
        self.db.insert({"type": "doc", "val": 1})
        self.db.create_table("metrics", {"name": "text", "score": "float"})
        self.db.insert_into("metrics", {"name": "accuracy", "score": 0.95})
        h = self.db.commit("mixed")
        assert self.db.tree.size == 2
        assert self.db.docs.size == 1
        assert self.db.table("metrics").size == 1

    def test_collection_alias(self):
        """collection() is MongoDB-style alias for get_or_create table."""
        coll = self.db.collection("events")
        coll.insert({"type": "click", "ts": 1234})
        coll.insert({"type": "view", "ts": 1235})
        assert coll.size == 2
        assert self.db.collection("events").size == 2

    def test_update_and_delete_table(self):
        self.db.create_table("t")
        self.db.insert_into("t", [
            {"name": "Alice", "score": 90},
            {"name": "Bob", "score": 80},
            {"name": "Charlie", "score": 70},
        ])
        count = self.db.update_table("t", {"name": "Bob"}, {"score": 85})
        assert count == 1
        count = self.db.delete_from("t", {"score": {"$lt": 75}})
        assert count == 1
        assert self.db.table("t").size == 2

    def test_status_shows_tables(self):
        self.db.create_table("t")
        self.db.insert_into("t", {"x": 1})
        s = self.db.status()
        assert "t" in s["tables"]
        assert s["table_rows"] == 1
        assert s["staged_table_changes"] > 0

    def test_table_only_commit(self):
        """Can commit with only table changes, no vectors or docs."""
        self.db.create_table("t")
        self.db.insert_into("t", {"x": 1})
        h = self.db.commit("tables only")
        assert h is not None

    def test_drop_table(self):
        self.db.create_table("temp")
        self.db.insert_into("temp", {"x": 1})
        self.db.drop_table("temp")
        assert "temp" not in self.db.list_tables()

    def test_amend_with_tables(self):
        self.db.create_table("t")
        self.db.insert_into("t", {"x": 1})
        h1 = self.db.commit("initial")
        self.db.insert_into("t", {"x": 2})
        h2 = self.db.amend(message="amended with more rows")
        assert h2 != h1
        # Table state should be saved for amended commit
        self.db.reset("HEAD")
        assert self.db.table("t").size == 2

    def test_squash_with_tables(self):
        self.db.create_table("t")
        self.db.insert_into("t", {"x": 1})
        self.db.commit("c1")
        self.db.insert_into("t", {"x": 2})
        self.db.commit("c2")
        self.db.insert_into("t", {"x": 3})
        self.db.commit("c3")
        h = self.db.squash(3, message="squashed")
        self.db.reset("HEAD")
        assert self.db.table("t").size == 3
