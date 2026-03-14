"""Tests for universal ingest."""

import csv
import json
import sqlite3
import pytest
import torch

from gitdb import GitDB
from gitdb.ingest import (
    ingest_sqlite, ingest_mongodb, ingest_csv, ingest_text,
    ingest_file, ingest_directory, detect_type,
    _chunk_text, _try_parse_value, _clean_mongo_doc, _detect_csv_text_column,
)


@pytest.fixture
def db(tmp_path):
    return GitDB(str(tmp_path / "store"), dim=8, device="cpu")


# ═══ SQLite ══════════════════════════════════════════════════

class TestSQLiteIngest:
    def _make_db(self, tmp_path, table="documents", rows=10):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, title TEXT, content TEXT, score REAL)")
        for i in range(rows):
            conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?)",
                        (i, f"Title {i}", f"This is document number {i} with some longer content for embedding.", i * 0.1))
        conn.commit()
        conn.close()
        return str(db_path)

    def test_ingest_sqlite_basic(self, db, tmp_path):
        sqlite_path = self._make_db(tmp_path)
        result = ingest_sqlite(db, sqlite_path, embed=False)
        assert result["total_rows"] == 10
        assert "documents" in result["tables"]
        assert db.tree.size == 10

    def test_ingest_sqlite_metadata(self, db, tmp_path):
        sqlite_path = self._make_db(tmp_path)
        ingest_sqlite(db, sqlite_path, embed=False)
        # Check metadata
        meta = db.tree.metadata[0].metadata
        assert "title" in meta
        assert "_source_table" in meta
        assert meta["_source_db"] == "test.db"

    def test_ingest_sqlite_specific_tables(self, db, tmp_path):
        db_path = tmp_path / "multi.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE a (id INTEGER, val TEXT)")
        conn.execute("CREATE TABLE b (id INTEGER, val TEXT)")
        conn.execute("INSERT INTO a VALUES (1, 'hello')")
        conn.execute("INSERT INTO b VALUES (1, 'world')")
        conn.commit()
        conn.close()
        result = ingest_sqlite(db, str(db_path), tables=["a"], embed=False)
        assert result["total_rows"] == 1

    def test_ingest_sqlite_text_column(self, db, tmp_path):
        sqlite_path = self._make_db(tmp_path)
        result = ingest_sqlite(db, sqlite_path, text_columns={"documents": "content"}, embed=False)
        assert result["tables"]["documents"]["text_column"] == "content"

    def test_ingest_sqlite_not_found(self, db):
        with pytest.raises(FileNotFoundError):
            ingest_sqlite(db, "/nonexistent/path.db")

    def test_ingest_sqlite_multiple_tables(self, db, tmp_path):
        db_path = tmp_path / "multi.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("CREATE TABLE posts (id INTEGER, body TEXT)")
        for i in range(5):
            conn.execute("INSERT INTO users VALUES (?, ?)", (i, f"User {i}"))
            conn.execute("INSERT INTO posts VALUES (?, ?)", (i, f"Post content {i}"))
        conn.commit()
        conn.close()
        result = ingest_sqlite(db, str(db_path), embed=False)
        assert result["total_rows"] == 10
        assert len(result["tables"]) == 2


# ═══ MongoDB/JSON ════════════════════════════════════════════

class TestMongoDBIngest:
    def test_ingest_jsonl(self, db, tmp_path):
        path = tmp_path / "export.jsonl"
        docs = [{"_id": {"$oid": f"abc{i}"}, "title": f"Doc {i}", "body": f"Content of document {i}"} for i in range(5)]
        path.write_text("\n".join(json.dumps(d) for d in docs))
        result = ingest_mongodb(db, str(path), embed=False)
        assert result["documents"] == 5
        assert db.tree.size == 5

    def test_ingest_json_array(self, db, tmp_path):
        path = tmp_path / "export.json"
        docs = [{"name": f"Item {i}", "description": f"A longer description for item number {i}"} for i in range(3)]
        path.write_text(json.dumps(docs))
        result = ingest_mongodb(db, str(path), embed=False)
        assert result["documents"] == 3

    def test_ingest_mongo_objectid(self, db, tmp_path):
        path = tmp_path / "export.jsonl"
        path.write_text(json.dumps({"_id": {"$oid": "507f1f77bcf86cd799439011"}, "text": "hello"}))
        ingest_mongodb(db, str(path), embed=False)
        meta = db.tree.metadata[0].metadata
        assert meta["_id"] == "507f1f77bcf86cd799439011"

    def test_ingest_mongo_extended_json(self, db, tmp_path):
        path = tmp_path / "export.jsonl"
        doc = {
            "count": {"$numberLong": "42"},
            "value": {"$numberDouble": "3.14"},
            "created": {"$date": "2024-01-01T00:00:00Z"},
            "text": "hello world",
        }
        path.write_text(json.dumps(doc))
        ingest_mongodb(db, str(path), embed=False)
        meta = db.tree.metadata[0].metadata
        assert meta["count"] == 42
        assert meta["value"] == 3.14

    def test_ingest_mongo_not_found(self, db):
        with pytest.raises(FileNotFoundError):
            ingest_mongodb(db, "/nonexistent.jsonl")

    def test_clean_mongo_doc(self):
        doc = {
            "_id": {"$oid": "abc123"},
            "name": "test",
            "nested": {"key": "val", "num": 42},
            "tags": ["a", "b"],
            "score": 0.95,
        }
        clean = _clean_mongo_doc(doc)
        assert clean["_id"] == "abc123"
        assert clean["name"] == "test"
        assert clean["nested.key"] == "val"
        assert clean["tags"] == ["a", "b"]


# ═══ CSV ═════════════════════════════════════════════════════

class TestCSVIngest:
    def test_ingest_csv_basic(self, db, tmp_path):
        path = tmp_path / "data.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "description", "score"])
            for i in range(10):
                writer.writerow([i, f"Item {i}", f"Description of item {i} for testing", i * 0.1])
        result = ingest_csv(db, str(path), embed=False)
        assert result["rows"] == 10
        assert "description" in result["columns"]

    def test_ingest_tsv(self, db, tmp_path):
        path = tmp_path / "data.tsv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id", "text"])
            writer.writerow([1, "hello world"])
            writer.writerow([2, "goodbye world"])
        result = ingest_csv(db, str(path), embed=False)
        assert result["rows"] == 2

    def test_csv_auto_detect_text_column(self, db, tmp_path):
        """Auto-detect only runs when embed=True; verify via _detect_csv_text_column directly."""
        rows = [
            {"id": "1", "short": "x", "long_content": "This is a much longer piece of text for testing auto-detection"},
            {"id": "2", "short": "y", "long_content": "Another long text value that should be detected as the text column"},
        ]
        detected = _detect_csv_text_column(rows, ["id", "short", "long_content"])
        assert detected == "long_content"

        # When embed=False, text_column stays None (no embedding needed)
        path = tmp_path / "data.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "short", "long_content"])
            writer.writerow([1, "x", "This is a much longer piece of text for testing auto-detection"])
            writer.writerow([2, "y", "Another long text value that should be detected as the text column"])
        result = ingest_csv(db, str(path), embed=False)
        assert result["text_column"] is None

    def test_csv_explicit_text_column(self, db, tmp_path):
        path = tmp_path / "data.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b"])
            writer.writerow(["hello", "world"])
        result = ingest_csv(db, str(path), text_column="a", embed=False)
        assert result["text_column"] == "a"

    def test_csv_numeric_parsing(self, db, tmp_path):
        path = tmp_path / "data.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "count", "score", "active"])
            writer.writerow(["test", "42", "3.14", "true"])
        ingest_csv(db, str(path), embed=False)
        meta = db.tree.metadata[0].metadata
        assert meta["count"] == 42
        assert meta["score"] == 3.14
        assert meta["active"] is True

    def test_csv_not_found(self, db):
        with pytest.raises(FileNotFoundError):
            ingest_csv(db, "/nonexistent.csv")


# ═══ Text ════════════════════════════════════════════════════

class TestTextIngest:
    def test_ingest_text_basic(self, db, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("Hello world. " * 100)
        result = ingest_text(db, str(path), embed=False, chunk_size=200)
        assert result["chunks"] > 1
        assert result["total_chars"] > 0

    def test_ingest_text_small(self, db, tmp_path):
        path = tmp_path / "small.txt"
        path.write_text("Short text.")
        result = ingest_text(db, str(path), embed=False)
        assert result["chunks"] == 1

    def test_ingest_markdown(self, db, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text("# Header\n\nSome content. " * 50)
        result = ingest_text(db, str(path), embed=False, chunk_size=200)
        assert result["chunks"] > 0

    def test_text_metadata(self, db, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("Hello world content here.")
        ingest_text(db, str(path), embed=False)
        meta = db.tree.metadata[0].metadata
        assert meta["_source_file"] == "test.txt"
        assert meta["_source_type"] == "text"


# ═══ Chunking ════════════════════════════════════════════════

class TestChunking:
    def test_chunk_small(self):
        chunks = _chunk_text("Hello world.", 500, 50)
        assert len(chunks) == 1

    def test_chunk_large(self):
        text = "Sentence one. " * 100
        chunks = _chunk_text(text, 100, 20)
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        text = "A" * 300
        chunks = _chunk_text(text, 100, 20)
        # Chunks should overlap
        assert len(chunks) >= 3

    def test_chunk_empty(self):
        chunks = _chunk_text("", 100, 20)
        assert len(chunks) == 0


# ═══ Universal Ingest ════════════════════════════════════════

class TestUniversalIngest:
    def test_detect_type(self):
        assert detect_type("data.db") == "sqlite"
        assert detect_type("data.sqlite") == "sqlite"
        assert detect_type("export.json") == "mongodb"
        assert detect_type("export.jsonl") == "mongodb"
        assert detect_type("data.csv") == "csv"
        assert detect_type("data.tsv") == "csv"
        assert detect_type("data.parquet") == "parquet"
        assert detect_type("doc.pdf") == "pdf"
        assert detect_type("doc.txt") == "text"
        assert detect_type("doc.md") == "text"
        assert detect_type("unknown.xyz") is None

    def test_ingest_file_csv(self, db, tmp_path):
        path = tmp_path / "data.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            writer.writerow(["hello"])
        result = ingest_file(db, str(path), embed=False)
        assert result["rows"] == 1

    def test_ingest_file_sqlite(self, db, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello')")
        conn.commit()
        conn.close()
        result = ingest_file(db, str(db_path), embed=False)
        assert result["total_rows"] == 1

    def test_ingest_file_text(self, db, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("Hello world.")
        result = ingest_file(db, str(path), embed=False)
        assert result["chunks"] == 1


class TestDirectoryIngest:
    def test_ingest_directory(self, db, tmp_path):
        # Create mixed files
        (tmp_path / "a.txt").write_text("Hello from file A.")
        (tmp_path / "b.txt").write_text("Hello from file B.")
        with open(tmp_path / "c.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            writer.writerow(["csv row"])
        result = ingest_directory(db, str(tmp_path), embed=False)
        assert result["total_files"] >= 3
        assert result["total_rows"] >= 3


# ═══ Helpers ═════════════════════════════════════════════════

class TestHelpers:
    def test_try_parse_value(self):
        assert _try_parse_value("42") == 42
        assert _try_parse_value("3.14") == 3.14
        assert _try_parse_value("true") is True
        assert _try_parse_value("false") is False
        assert _try_parse_value("hello") == "hello"

    def test_detect_csv_text_column(self):
        rows = [
            {"id": "1", "name": "short", "desc": "This is a much longer description text"},
            {"id": "2", "name": "tiny", "desc": "Another long description for detection testing"},
        ]
        assert _detect_csv_text_column(rows, ["id", "name", "desc"]) == "desc"
