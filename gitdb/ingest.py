"""Universal ingest — swallow SQLite, MongoDB, CSV, Parquet, PDF, and text into GitDB.

One command to replace your entire stack:
    gitdb ingest my_database.db
    gitdb ingest dump.bson
    gitdb ingest data.csv --text-column content
    gitdb ingest report.pdf
    gitdb ingest *.txt
"""

import csv
import io
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


# ═══════════════════════════════════════════════════════════════
#  SQLite Ingest
# ═══════════════════════════════════════════════════════════════

def ingest_sqlite(
    db,  # GitDB instance
    sqlite_path: str,
    tables: Optional[List[str]] = None,
    text_columns: Optional[Dict[str, str]] = None,
    embed: bool = True,
    embed_model: Optional[str] = None,
    batch_size: int = 500,
    autocommit: bool = True,
) -> Dict[str, Any]:
    """Ingest an entire SQLite database into GitDB.

    Each row becomes a vector. Text columns are embedded. All columns become metadata.

    Args:
        db: GitDB instance.
        sqlite_path: Path to .db or .sqlite file.
        tables: List of tables to ingest (default: all).
        text_columns: {table_name: column_name} — which column to embed per table.
                      If not specified, auto-detects the longest TEXT column.
        embed: If True, auto-embed text columns.
        embed_model: Embedding model name (default: arctic).
        batch_size: Rows per batch.
        autocommit: Commit after each table.

    Returns:
        {"tables": {name: {"rows": N, "columns": [...]}}, "total_rows": N}
    """
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get table list
    if tables is None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]

    if not text_columns:
        text_columns = {}

    result = {"tables": {}, "total_rows": 0}

    for table in tables:
        # Get column info
        cursor.execute(f"PRAGMA table_info({_quote_ident(table)})")
        columns = cursor.fetchall()
        col_names = [c[1] for c in columns]
        col_types = {c[1]: c[2].upper() for c in columns}

        # Auto-detect text column if not specified
        text_col = text_columns.get(table)
        if text_col is None and embed:
            text_col = _detect_text_column(cursor, table, col_names, col_types)

        # Read rows
        cursor.execute(f"SELECT * FROM {_quote_ident(table)}")
        rows = cursor.fetchall()

        table_result = {"rows": 0, "columns": col_names, "text_column": text_col}

        # Batch ingest
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = []
            metadata_list = []
            documents = []

            for row in batch:
                row_dict = dict(row)
                # Add source table info
                row_dict["_source_table"] = table
                row_dict["_source_db"] = str(sqlite_path.name)

                # Extract text for embedding
                doc_text = None
                if text_col and text_col in row_dict:
                    doc_text = str(row_dict[text_col]) if row_dict[text_col] is not None else None

                # Clean metadata (remove None values, convert types)
                meta = {}
                for k, v in row_dict.items():
                    if v is not None:
                        if isinstance(v, (int, float, str, bool)):
                            meta[k] = v
                        elif isinstance(v, bytes):
                            meta[k] = f"<blob:{len(v)}bytes>"
                        else:
                            meta[k] = str(v)

                if doc_text and embed:
                    texts.append(doc_text)
                documents.append(doc_text)
                metadata_list.append(meta)

            # Add to GitDB
            if texts and embed:
                db.add(texts=texts, metadata=metadata_list, embed_model=embed_model)
            elif documents:
                # No embedding — use random vectors as placeholders (metadata-only ingest)
                n = len(documents)
                placeholder = torch.randn(n, db.dim)
                db.add(embeddings=placeholder, documents=documents, metadata=metadata_list)

            table_result["rows"] += len(batch)

        if autocommit and table_result["rows"] > 0:
            db.commit(f"Ingest SQLite table: {table} ({table_result['rows']} rows)")

        result["tables"][table] = table_result
        result["total_rows"] += table_result["rows"]

    conn.close()
    return result


def _detect_text_column(cursor, table, col_names, col_types):
    """Auto-detect the best text column for embedding."""
    # Prefer TEXT/VARCHAR columns
    text_cols = [c for c in col_names if col_types.get(c, "") in ("TEXT", "VARCHAR", "CLOB", "")]
    if not text_cols:
        return None
    if len(text_cols) == 1:
        return text_cols[0]

    # Pick the one with the longest average content
    best_col = None
    best_avg = 0
    for col in text_cols:
        try:
            cursor.execute(f"SELECT AVG(LENGTH({_quote_ident(col)})) FROM {_quote_ident(table)} WHERE {_quote_ident(col)} IS NOT NULL")
            avg_len = cursor.fetchone()[0]
            if avg_len and avg_len > best_avg:
                best_avg = avg_len
                best_col = col
        except Exception:
            continue
    return best_col


def _quote_ident(name):
    """Quote a SQL identifier to prevent injection."""
    return '"' + name.replace('"', '""') + '"'


# ═══════════════════════════════════════════════════════════════
#  MongoDB / BSON / JSON Ingest
# ═══════════════════════════════════════════════════════════════

def ingest_mongodb(
    db,
    path: str,
    text_field: Optional[str] = None,
    embed: bool = True,
    embed_model: Optional[str] = None,
    batch_size: int = 500,
    collection_name: Optional[str] = None,
    autocommit: bool = True,
) -> Dict[str, Any]:
    """Ingest a MongoDB JSON/BSON export into GitDB.

    Supports:
      - mongoexport JSON (one doc per line, or JSON array)
      - mongodump BSON (requires bson package)
      - Plain JSON array files

    Args:
        db: GitDB instance.
        path: Path to .json, .jsonl, or .bson file.
        text_field: Field name to embed. Auto-detects if not specified.
        embed: If True, auto-embed text fields.
        embed_model: Embedding model name.
        batch_size: Documents per batch.
        collection_name: Name for commit message.
        autocommit: Commit after ingest.

    Returns:
        {"documents": N, "text_field": str, "fields": [...]}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Parse documents
    docs = _parse_mongo_file(path)

    if not docs:
        return {"documents": 0, "text_field": None, "fields": []}

    # Auto-detect text field
    if text_field is None and embed:
        text_field = _detect_mongo_text_field(docs[:100])  # sample first 100

    # Collect all field names
    all_fields = set()
    for doc in docs:
        all_fields.update(doc.keys())

    result = {"documents": 0, "text_field": text_field, "fields": sorted(all_fields)}

    # Batch ingest
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        texts = []
        metadata_list = []
        documents = []

        for doc in batch:
            # Clean MongoDB special types
            meta = _clean_mongo_doc(doc)
            meta["_source_file"] = str(path.name)
            if collection_name:
                meta["_collection"] = collection_name

            # Extract text
            doc_text = None
            if text_field and text_field in meta:
                doc_text = str(meta[text_field]) if meta[text_field] is not None else None

            if doc_text and embed:
                texts.append(doc_text)
            documents.append(doc_text)
            metadata_list.append(meta)

        if texts and embed:
            db.add(texts=texts, metadata=metadata_list, embed_model=embed_model)
        elif documents:
            n = len(documents)
            placeholder = torch.randn(n, db.dim)
            db.add(embeddings=placeholder, documents=documents, metadata=metadata_list)

        result["documents"] += len(batch)

    if autocommit and result["documents"] > 0:
        name = collection_name or path.stem
        db.commit(f"Ingest MongoDB: {name} ({result['documents']} docs)")

    return result


def _parse_mongo_file(path: Path) -> List[dict]:
    """Parse MongoDB export file (JSON, JSONL, or BSON)."""
    if path.suffix == ".bson":
        try:
            import bson
            docs = []
            with open(path, "rb") as f:
                data = f.read()
            # BSON decode
            offset = 0
            while offset < len(data):
                size = int.from_bytes(data[offset:offset+4], "little")
                if size <= 0 or offset + size > len(data):
                    break
                doc_bytes = data[offset:offset+size]
                docs.append(bson.decode(doc_bytes))
                offset += size
            return docs
        except ImportError:
            raise ImportError("Install 'pymongo' or 'bson' package to read BSON files: pip install pymongo")

    # JSON or JSONL
    content = path.read_text(encoding="utf-8", errors="replace")
    content = content.strip()

    # Try JSON array first
    if content.startswith("["):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

    # JSONL (one JSON object per line)
    docs = []
    for line in content.split("\n"):
        line = line.strip()
        if line:
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return docs


def _detect_mongo_text_field(docs: List[dict]) -> Optional[str]:
    """Auto-detect the best text field from sample documents."""
    field_lengths = {}
    for doc in docs:
        for key, value in doc.items():
            if isinstance(value, str) and len(value) > 10:
                field_lengths.setdefault(key, []).append(len(value))

    if not field_lengths:
        return None

    # Pick field with longest average string length
    best_field = max(field_lengths, key=lambda k: sum(field_lengths[k]) / len(field_lengths[k]))
    return best_field


def _clean_mongo_doc(doc: dict) -> dict:
    """Clean MongoDB document for GitDB metadata."""
    clean = {}
    for key, value in doc.items():
        # Handle MongoDB ObjectId
        if key == "_id":
            if isinstance(value, dict) and "$oid" in value:
                clean["_id"] = value["$oid"]
            elif isinstance(value, str):
                clean["_id"] = value
            else:
                clean["_id"] = str(value)
        elif isinstance(value, dict):
            # Handle MongoDB extended JSON types
            if "$date" in value:
                clean[key] = str(value["$date"])
            elif "$numberLong" in value:
                clean[key] = int(value["$numberLong"])
            elif "$numberDouble" in value:
                clean[key] = float(value["$numberDouble"])
            else:
                # Nested object — flatten with dot notation
                for subkey, subval in value.items():
                    flat_key = f"{key}.{subkey}"
                    if isinstance(subval, (str, int, float, bool)):
                        clean[flat_key] = subval
        elif isinstance(value, list):
            # Store lists as JSON strings for metadata compatibility
            if all(isinstance(v, (str, int, float)) for v in value):
                clean[key] = value  # Keep simple lists
            else:
                clean[key] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            pass  # Skip nulls
        else:
            clean[key] = str(value)
    return clean


# ═══════════════════════════════════════════════════════════════
#  CSV / TSV Ingest
# ═══════════════════════════════════════════════════════════════

def ingest_csv(
    db,
    path: str,
    text_column: Optional[str] = None,
    delimiter: Optional[str] = None,
    embed: bool = True,
    embed_model: Optional[str] = None,
    batch_size: int = 500,
    autocommit: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Ingest a CSV or TSV file into GitDB.

    Each row becomes a vector. One column is embedded, rest become metadata.

    Args:
        db: GitDB instance.
        path: Path to .csv or .tsv file.
        text_column: Column name to embed. Auto-detects if not specified.
        delimiter: Column delimiter (auto-detects from extension).
        embed: If True, auto-embed text column.
        embed_model: Embedding model name.
        batch_size: Rows per batch.
        autocommit: Commit after ingest.
        encoding: File encoding.

    Returns:
        {"rows": N, "columns": [...], "text_column": str}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Auto-detect delimiter
    if delimiter is None:
        if path.suffix in (".tsv", ".tab"):
            delimiter = "\t"
        else:
            delimiter = ","

    # Read CSV
    with open(path, "r", encoding=encoding, errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        columns = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return {"rows": 0, "columns": columns, "text_column": None}

    # Auto-detect text column
    if text_column is None and embed:
        text_column = _detect_csv_text_column(rows[:100], columns)

    result = {"rows": 0, "columns": columns, "text_column": text_column}

    # Batch ingest
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = []
        metadata_list = []
        documents = []

        for row in batch:
            meta = {"_source_file": str(path.name)}
            doc_text = None

            for col in columns:
                val = row.get(col, "")
                if val is None or val == "":
                    continue
                # Try to parse numeric values
                parsed = _try_parse_value(val)
                meta[col] = parsed

                if col == text_column:
                    doc_text = str(val)

            if doc_text and embed:
                texts.append(doc_text)
            documents.append(doc_text)
            metadata_list.append(meta)

        if texts and embed:
            db.add(texts=texts, metadata=metadata_list, embed_model=embed_model)
        elif documents:
            n = len(documents)
            placeholder = torch.randn(n, db.dim)
            db.add(embeddings=placeholder, documents=documents, metadata=metadata_list)

        result["rows"] += len(batch)

    if autocommit and result["rows"] > 0:
        db.commit(f"Ingest CSV: {path.name} ({result['rows']} rows)")

    return result


def _detect_csv_text_column(rows: List[dict], columns: List[str]) -> Optional[str]:
    """Auto-detect the best text column in a CSV."""
    col_lengths = {}
    for row in rows:
        for col in columns:
            val = row.get(col, "")
            if val and isinstance(val, str) and not _is_numeric(val):
                col_lengths.setdefault(col, []).append(len(val))

    if not col_lengths:
        return columns[0] if columns else None

    return max(col_lengths, key=lambda k: sum(col_lengths[k]) / len(col_lengths[k]))


def _try_parse_value(val: str):
    """Try to parse a string value as int, float, or bool."""
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _is_numeric(val: str) -> bool:
    """Check if a string is numeric."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


# ═══════════════════════════════════════════════════════════════
#  Parquet Ingest
# ═══════════════════════════════════════════════════════════════

def ingest_parquet(
    db,
    path: str,
    text_column: Optional[str] = None,
    embed: bool = True,
    embed_model: Optional[str] = None,
    batch_size: int = 500,
    autocommit: bool = True,
) -> Dict[str, Any]:
    """Ingest a Parquet file into GitDB.

    Requires: pip install pyarrow or pip install fastparquet
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Install pyarrow to read Parquet files: pip install pyarrow")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    table = pq.read_table(str(path))
    columns = table.column_names

    # Convert to Python dicts
    rows = []
    for i in range(len(table)):
        row = {}
        for col in columns:
            val = table.column(col)[i].as_py()
            row[col] = val
        rows.append(row)

    if not rows:
        return {"rows": 0, "columns": columns, "text_column": None}

    # Auto-detect text column
    if text_column is None and embed:
        text_column = _detect_parquet_text_column(rows[:100], columns)

    result = {"rows": 0, "columns": columns, "text_column": text_column}

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = []
        metadata_list = []
        documents = []

        for row in batch:
            meta = {"_source_file": str(path.name)}
            doc_text = None

            for col in columns:
                val = row.get(col)
                if val is not None:
                    if isinstance(val, (str, int, float, bool)):
                        meta[col] = val
                    elif isinstance(val, list):
                        meta[col] = val
                    else:
                        meta[col] = str(val)

                if col == text_column and val is not None:
                    doc_text = str(val)

            if doc_text and embed:
                texts.append(doc_text)
            documents.append(doc_text)
            metadata_list.append(meta)

        if texts and embed:
            db.add(texts=texts, metadata=metadata_list, embed_model=embed_model)
        elif documents:
            n = len(documents)
            placeholder = torch.randn(n, db.dim)
            db.add(embeddings=placeholder, documents=documents, metadata=metadata_list)

        result["rows"] += len(batch)

    if autocommit and result["rows"] > 0:
        db.commit(f"Ingest Parquet: {path.name} ({result['rows']} rows)")

    return result


def _detect_parquet_text_column(rows, columns):
    """Auto-detect text column in Parquet data."""
    col_lengths = {}
    for row in rows:
        for col in columns:
            val = row.get(col)
            if isinstance(val, str) and len(val) > 10:
                col_lengths.setdefault(col, []).append(len(val))
    if not col_lengths:
        return None
    return max(col_lengths, key=lambda k: sum(col_lengths[k]) / len(col_lengths[k]))


# ═══════════════════════════════════════════════════════════════
#  PDF Ingest
# ═══════════════════════════════════════════════════════════════

def ingest_pdf(
    db,
    path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embed: bool = True,
    embed_model: Optional[str] = None,
    autocommit: bool = True,
) -> Dict[str, Any]:
    """Ingest a PDF file into GitDB.

    Extracts text, chunks it, embeds each chunk as a vector.

    Requires: pip install pymupdf (or pip install PyPDF2 as fallback)

    Args:
        db: GitDB instance.
        path: Path to .pdf file.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        embed: If True, auto-embed chunks.
        embed_model: Embedding model name.
        autocommit: Commit after ingest.

    Returns:
        {"pages": N, "chunks": N, "total_chars": N}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Extract text from PDF
    pages_text = _extract_pdf_text(path)

    if not pages_text:
        return {"pages": 0, "chunks": 0, "total_chars": 0}

    # Chunk the text
    chunks = []
    for page_num, text in enumerate(pages_text):
        page_chunks = _chunk_text(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(page_chunks):
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_num + 1,
                    "chunk_index": i,
                    "source": str(path.name),
                })

    result = {
        "pages": len(pages_text),
        "chunks": len(chunks),
        "total_chars": sum(len(p) for p in pages_text),
    }

    if not chunks:
        return result

    # Ingest chunks
    texts = [c["text"] for c in chunks]
    metadata = [{
        "page": c["page"],
        "chunk_index": c["chunk_index"],
        "_source_file": c["source"],
        "_source_type": "pdf",
    } for c in chunks]

    if embed:
        db.add(texts=texts, metadata=metadata, embed_model=embed_model)
    else:
        placeholder = torch.randn(len(texts), db.dim)
        db.add(embeddings=placeholder, documents=texts, metadata=metadata)

    if autocommit:
        db.commit(f"Ingest PDF: {path.name} ({result['pages']} pages, {result['chunks']} chunks)")

    return result


def _extract_pdf_text(path: Path) -> List[str]:
    """Extract text from PDF, one string per page."""
    # Try pymupdf first (faster, better quality)
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return pages
    except ImportError:
        pass

    # Fallback to PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
    except ImportError:
        raise ImportError(
            "Install a PDF library to read PDFs:\n"
            "  pip install pymupdf    (recommended)\n"
            "  pip install PyPDF2     (fallback)"
        )


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near the boundary
            for sep in [". ", ".\n", "! ", "? ", "\n\n", "\n"]:
                boundary = text.rfind(sep, start + chunk_size // 2, end + 50)
                if boundary > start:
                    end = boundary + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ═══════════════════════════════════════════════════════════════
#  Plain Text / Markdown Ingest
# ═══════════════════════════════════════════════════════════════

def ingest_text(
    db,
    path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embed: bool = True,
    embed_model: Optional[str] = None,
    autocommit: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Ingest a plain text or markdown file into GitDB.

    Chunks the text and embeds each chunk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding=encoding, errors="replace")

    if not text.strip():
        return {"chunks": 0, "total_chars": 0}

    chunks = _chunk_text(text, chunk_size, chunk_overlap)

    result = {"chunks": len(chunks), "total_chars": len(text)}

    if not chunks:
        return result

    metadata = [{
        "chunk_index": i,
        "_source_file": str(path.name),
        "_source_type": "text",
    } for i in range(len(chunks))]

    if embed:
        db.add(texts=chunks, metadata=metadata, embed_model=embed_model)
    else:
        placeholder = torch.randn(len(chunks), db.dim)
        db.add(embeddings=placeholder, documents=chunks, metadata=metadata)

    if autocommit:
        db.commit(f"Ingest text: {path.name} ({len(chunks)} chunks)")

    return result


# ═══════════════════════════════════════════════════════════════
#  Directory Ingest (batch)
# ═══════════════════════════════════════════════════════════════

def ingest_directory(
    db,
    dir_path: str,
    pattern: str = "*",
    recursive: bool = True,
    embed: bool = True,
    embed_model: Optional[str] = None,
    autocommit: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Ingest all supported files from a directory.

    Auto-detects file types and routes to the appropriate ingest function.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    glob_fn = dir_path.rglob if recursive else dir_path.glob
    files = sorted(glob_fn(pattern))

    results = {"files": {}, "total_rows": 0, "total_files": 0}

    for f in files:
        if f.is_dir():
            continue
        try:
            r = ingest_file(db, str(f), embed=embed, embed_model=embed_model,
                          autocommit=autocommit, **kwargs)
            results["files"][str(f.relative_to(dir_path))] = r
            results["total_rows"] += r.get("rows", r.get("chunks", r.get("documents", 0)))
            results["total_files"] += 1
        except (ImportError, Exception) as e:
            results["files"][str(f.relative_to(dir_path))] = {"error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════════
#  Universal Ingest (auto-detect)
# ═══════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = {
    ".db": "sqlite", ".sqlite": "sqlite", ".sqlite3": "sqlite",
    ".json": "mongodb", ".jsonl": "mongodb", ".bson": "mongodb",
    ".csv": "csv", ".tsv": "csv", ".tab": "csv",
    ".parquet": "parquet", ".pq": "parquet",
    ".pdf": "pdf",
    ".txt": "text", ".md": "text", ".markdown": "text",
    ".rst": "text", ".log": "text",
}

def ingest_file(
    db,
    path: str,
    embed: bool = True,
    embed_model: Optional[str] = None,
    autocommit: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Universal ingest — auto-detect file type and ingest.

    Supports: SQLite (.db, .sqlite), MongoDB (.json, .jsonl, .bson),
    CSV/TSV (.csv, .tsv), Parquet (.parquet), PDF (.pdf),
    Text (.txt, .md, .rst, .log)

    Returns ingest result dict.
    """
    path = Path(path)
    ext = path.suffix.lower()

    file_type = SUPPORTED_EXTENSIONS.get(ext)
    if file_type is None:
        # Default to text for unknown extensions
        file_type = "text"

    if file_type == "sqlite":
        return ingest_sqlite(db, str(path), embed=embed, embed_model=embed_model,
                           autocommit=autocommit, **kwargs)
    elif file_type == "mongodb":
        return ingest_mongodb(db, str(path), embed=embed, embed_model=embed_model,
                            autocommit=autocommit, **kwargs)
    elif file_type == "csv":
        return ingest_csv(db, str(path), embed=embed, embed_model=embed_model,
                        autocommit=autocommit, **kwargs)
    elif file_type == "parquet":
        return ingest_parquet(db, str(path), embed=embed, embed_model=embed_model,
                            autocommit=autocommit, **kwargs)
    elif file_type == "pdf":
        return ingest_pdf(db, str(path), embed=embed, embed_model=embed_model,
                        autocommit=autocommit, **kwargs)
    elif file_type == "text":
        return ingest_text(db, str(path), embed=embed, embed_model=embed_model,
                         autocommit=autocommit, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def detect_type(path: str) -> Optional[str]:
    """Detect file type from extension. Returns type name or None."""
    return SUPPORTED_EXTENSIONS.get(Path(path).suffix.lower())
