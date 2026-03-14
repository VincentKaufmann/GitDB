"""Tests for schema enforcement."""

import pytest
from gitdb.schema import Schema, SchemaError


class TestSchema:
    def test_required_fields(self):
        s = Schema({"required": ["name", "category"]})
        assert s.validate({"name": "x", "category": "y"}) == []
        errors = s.validate({"name": "x"})
        assert len(errors) == 1
        assert "category" in errors[0]

    def test_type_string(self):
        s = Schema({"properties": {"name": {"type": "string"}}})
        assert s.validate({"name": "hello"}) == []
        errors = s.validate({"name": 42})
        assert len(errors) == 1

    def test_type_number(self):
        s = Schema({"properties": {"score": {"type": "number"}}})
        assert s.validate({"score": 3.14}) == []
        assert s.validate({"score": 42}) == []
        errors = s.validate({"score": "high"})
        assert len(errors) == 1

    def test_type_integer(self):
        s = Schema({"properties": {"count": {"type": "integer"}}})
        assert s.validate({"count": 5}) == []
        errors = s.validate({"count": 5.5})
        assert len(errors) == 1

    def test_type_boolean(self):
        s = Schema({"properties": {"active": {"type": "boolean"}}})
        assert s.validate({"active": True}) == []
        errors = s.validate({"active": 1})
        assert len(errors) == 1

    def test_type_array(self):
        s = Schema({"properties": {"tags": {"type": "array"}}})
        assert s.validate({"tags": [1, 2]}) == []
        errors = s.validate({"tags": "not_array"})
        assert len(errors) == 1

    def test_enum(self):
        s = Schema({"properties": {"status": {"enum": ["active", "inactive"]}}})
        assert s.validate({"status": "active"}) == []
        errors = s.validate({"status": "unknown"})
        assert len(errors) == 1

    def test_minimum_maximum(self):
        s = Schema({"properties": {"score": {"type": "number", "minimum": 0, "maximum": 100}}})
        assert s.validate({"score": 50}) == []
        assert len(s.validate({"score": -1})) == 1
        assert len(s.validate({"score": 101})) == 1

    def test_min_max_length(self):
        s = Schema({"properties": {"name": {"type": "string", "minLength": 2, "maxLength": 10}}})
        assert s.validate({"name": "ok"}) == []
        assert len(s.validate({"name": "x"})) == 1
        assert len(s.validate({"name": "a" * 11})) == 1

    def test_pattern(self):
        s = Schema({"properties": {"email": {"type": "string", "pattern": r"@.*\."}}})
        assert s.validate({"email": "a@b.c"}) == []
        assert len(s.validate({"email": "nope"})) == 1

    def test_additional_properties_false(self):
        s = Schema({"properties": {"name": {}}, "additionalProperties": False})
        assert s.validate({"name": "x"}) == []
        errors = s.validate({"name": "x", "extra": 1})
        assert len(errors) == 1

    def test_additional_properties_true(self):
        s = Schema({"properties": {"name": {}}, "additionalProperties": True})
        assert s.validate({"name": "x", "extra": 1}) == []

    def test_save_and_load(self, tmp_path):
        definition = {"required": ["name"], "properties": {"name": {"type": "string"}}}
        s = Schema(definition)
        path = str(tmp_path / "schema.json")
        s.save(path)
        s2 = Schema.from_file(path)
        assert s2.validate({"name": "x"}) == []


class TestSchemaInGitDB:
    def test_schema_rejects_bad_metadata(self, tmp_path):
        import torch
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "s"), dim=8, device="cpu")
        db.set_schema({"required": ["category"], "properties": {"category": {"type": "string"}}})
        with pytest.raises(SchemaError):
            db.add(torch.randn(1, 8), metadata=[{"wrong_field": 1}])

    def test_schema_accepts_valid_metadata(self, tmp_path):
        import torch
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "s"), dim=8, device="cpu")
        db.set_schema({"required": ["category"], "properties": {"category": {"type": "string"}}})
        db.add(torch.randn(1, 8), metadata=[{"category": "test"}])
        assert db.tree.size == 1

    def test_schema_persists(self, tmp_path):
        import torch
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "s"), dim=8, device="cpu")
        db.set_schema({"required": ["x"]})
        db2 = GitDB(str(tmp_path / "s"), dim=8, device="cpu")
        assert db2.get_schema() is not None

    def test_clear_schema(self, tmp_path):
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "s"), dim=8, device="cpu")
        db.set_schema({"required": ["x"]})
        db.set_schema(None)
        assert db.get_schema() is None

    def test_add_without_metadata_skips_validation(self, tmp_path):
        import torch
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "s"), dim=8, device="cpu")
        db.set_schema({"required": ["category"]})
        indices = db.add(torch.randn(2, 8), documents=["a", "b"])
        assert len(indices) == 2

    def test_numeric_bounds_reject(self, tmp_path):
        import torch
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "s"), dim=8, device="cpu")
        db.set_schema({"properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}}})
        with pytest.raises(SchemaError):
            db.add(torch.randn(1, 8), metadata=[{"score": 2.0}])

    def test_get_schema_returns_definition(self, tmp_path):
        from gitdb import GitDB
        db = GitDB(str(tmp_path / "s"), dim=8, device="cpu")
        defn = {"required": ["x"], "properties": {"x": {"type": "string"}}}
        db.set_schema(defn)
        assert db.get_schema() == defn
