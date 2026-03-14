"""Tests for IndexManager — secondary indexes on metadata fields."""

import pytest
import torch

from gitdb import GitDB
from gitdb.indexes import IndexManager
from gitdb.types import VectorMeta


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_store")


@pytest.fixture
def db(db_path):
    return GitDB(db_path, dim=8, device="cpu")


def random_vectors(n, dim=8):
    return torch.randn(n, dim)


class TestIndexManagerUnit:
    """Unit tests on IndexManager directly."""

    def test_create_hash_index(self):
        im = IndexManager()
        im.create_index("category", "hash")
        indexes = im.list_indexes()
        assert len(indexes) == 1
        assert indexes[0]["field"] == "category"
        assert indexes[0]["type"] == "hash"

    def test_create_range_index(self):
        im = IndexManager()
        im.create_index("score", "range")
        indexes = im.list_indexes()
        assert indexes[0]["type"] == "range"

    def test_create_duplicate_raises(self):
        im = IndexManager()
        im.create_index("x")
        with pytest.raises(ValueError):
            im.create_index("x")

    def test_create_bad_type_raises(self):
        im = IndexManager()
        with pytest.raises(ValueError):
            im.create_index("x", "btree")

    def test_drop_index(self):
        im = IndexManager()
        im.create_index("x")
        im.drop_index("x")
        assert len(im.list_indexes()) == 0

    def test_drop_nonexistent_raises(self):
        im = IndexManager()
        with pytest.raises(ValueError):
            im.drop_index("nope")

    def test_hash_lookup(self):
        im = IndexManager()
        im.create_index("category")
        metadata = [
            VectorMeta(id="a", metadata={"category": "finance"}),
            VectorMeta(id="b", metadata={"category": "legal"}),
            VectorMeta(id="c", metadata={"category": "finance"}),
        ]
        im.rebuild(metadata)
        result = im.lookup("category", "finance")
        assert sorted(result) == [0, 2]

    def test_hash_lookup_missing_value(self):
        im = IndexManager()
        im.create_index("category")
        im.rebuild([VectorMeta(id="a", metadata={"category": "x"})])
        result = im.lookup("category", "nonexistent")
        assert result == []

    def test_range_lookup(self):
        im = IndexManager()
        im.create_index("score", "range")
        metadata = [
            VectorMeta(id="a", metadata={"score": 0.1}),
            VectorMeta(id="b", metadata={"score": 0.5}),
            VectorMeta(id="c", metadata={"score": 0.9}),
            VectorMeta(id="d", metadata={"score": 0.3}),
        ]
        im.rebuild(metadata)
        result = im.range_lookup("score", min_val=0.3, max_val=0.6)
        assert sorted(result) == [1, 3]  # indices with score 0.5 and 0.3

    def test_range_lookup_open_ended(self):
        im = IndexManager()
        im.create_index("score", "range")
        metadata = [
            VectorMeta(id="a", metadata={"score": 1}),
            VectorMeta(id="b", metadata={"score": 5}),
            VectorMeta(id="c", metadata={"score": 10}),
        ]
        im.rebuild(metadata)
        result = im.range_lookup("score", min_val=5)
        assert sorted(result) == [1, 2]

    def test_range_lookup_on_hash_raises(self):
        im = IndexManager()
        im.create_index("x", "hash")
        im.rebuild([])
        with pytest.raises(ValueError):
            im.range_lookup("x", min_val=0)

    def test_update_incremental(self):
        im = IndexManager()
        im.create_index("tag")
        im.update(0, {"tag": "a"})
        im.update(1, {"tag": "b"})
        im.update(2, {"tag": "a"})
        assert sorted(im.lookup("tag", "a")) == [0, 2]

    def test_has_index(self):
        im = IndexManager()
        im.create_index("x")
        assert im.has_index("x")
        assert not im.has_index("y")

    def test_rebuild_clears_old_data(self):
        im = IndexManager()
        im.create_index("tag")
        im.update(0, {"tag": "old"})
        im.rebuild([VectorMeta(id="a", metadata={"tag": "new"})])
        assert im.lookup("tag", "old") == []
        assert im.lookup("tag", "new") == [0]

    def test_lookup_no_index_raises(self):
        im = IndexManager()
        with pytest.raises(ValueError):
            im.lookup("missing", "val")


class TestIndexIntegration:
    """Integration tests with GitDB."""

    def test_create_and_list(self, db):
        db.create_index("category")
        indexes = db.list_indexes()
        assert len(indexes) == 1
        assert indexes[0]["field"] == "category"

    def test_index_populated_on_create(self, db):
        db.add(
            random_vectors(3),
            metadata=[{"category": "a"}, {"category": "b"}, {"category": "a"}],
        )
        db.create_index("category")
        result = db.indexes.lookup("category", "a")
        assert sorted(result) == [0, 2]

    def test_index_updated_on_add(self, db):
        db.create_index("tag")
        db.add(random_vectors(2), metadata=[{"tag": "x"}, {"tag": "y"}])
        assert db.indexes.lookup("tag", "x") == [0]
        assert db.indexes.lookup("tag", "y") == [1]

    def test_drop_index(self, db):
        db.create_index("tag")
        db.drop_index("tag")
        assert db.list_indexes() == []

    def test_index_rebuilt_on_load(self, db_path):
        # Create, add, commit
        db1 = GitDB(db_path, dim=8, device="cpu")
        db1.create_index("category")
        db1.add(
            random_vectors(3),
            metadata=[{"category": "a"}, {"category": "b"}, {"category": "a"}],
        )
        db1.commit("indexed data")

        # Reopen — indexes must be recreated manually (they are in-memory)
        db2 = GitDB(db_path, dim=8, device="cpu")
        db2.create_index("category")
        # rebuild happens during create_index since it calls rebuild
        result = db2.indexes.lookup("category", "a")
        assert sorted(result) == [0, 2]

    def test_range_index_with_db(self, db):
        db.create_index("score", index_type="range")
        db.add(
            random_vectors(4),
            metadata=[{"score": 1}, {"score": 5}, {"score": 10}, {"score": 3}],
        )
        result = db.indexes.range_lookup("score", min_val=3, max_val=6)
        assert sorted(result) == [1, 3]
