"""GitDB gRPC service layer — server and client.

Provides a high-performance gRPC interface to GitDB. Requires grpcio and
grpcio-tools. Proto stubs are auto-generated on first import if missing.

Usage (server):
    from gitdb.grpc_service import serve_grpc
    serve_grpc("/path/to/store", port=50051)

Usage (client):
    from gitdb.grpc_service import GitDBClient
    client = GitDBClient("localhost:50051")
    client.insert({"name": "Alice"})
    client.find(where={"name": "Alice"})
    client.commit("added users")

Usage (CLI):
    gitdb grpc-serve --port 50051

Stub generation:
    python -m grpc_tools.protoc -I gitdb --python_out=gitdb --grpc_python_out=gitdb gitdb/gitdb.proto
"""

import io
import json
import logging
import threading
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ─── Lazy imports for grpc (don't crash if not installed) ────────────

_grpc = None
_grpc_pb2 = None
_grpc_pb2_grpc = None


def _ensure_grpc():
    """Lazy-import grpc and generated stubs. Auto-generate stubs if missing."""
    global _grpc, _grpc_pb2, _grpc_pb2_grpc

    if _grpc is not None:
        return

    try:
        import grpc
    except ImportError:
        raise ImportError(
            "grpcio is required for gRPC support. Install with:\n"
            "  pip install grpcio grpcio-tools"
        )

    _grpc = grpc

    # Try importing generated stubs
    try:
        from gitdb import gitdb_pb2, gitdb_pb2_grpc
        _grpc_pb2 = gitdb_pb2
        _grpc_pb2_grpc = gitdb_pb2_grpc
    except ImportError:
        # Auto-generate from proto
        _compile_proto()
        from gitdb import gitdb_pb2, gitdb_pb2_grpc
        _grpc_pb2 = gitdb_pb2
        _grpc_pb2_grpc = gitdb_pb2_grpc


def _compile_proto():
    """Compile gitdb.proto into Python stubs."""
    try:
        from grpc_tools import protoc
    except ImportError:
        raise ImportError(
            "grpcio-tools is required to compile proto files. Install with:\n"
            "  pip install grpcio-tools\n"
            "Or manually generate stubs:\n"
            "  python -m grpc_tools.protoc -I gitdb --python_out=gitdb "
            "--grpc_python_out=gitdb gitdb/gitdb.proto"
        )

    proto_dir = str(Path(__file__).parent)
    proto_file = str(Path(__file__).parent / "gitdb.proto")
    result = protoc.main([
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        proto_file,
    ])
    if result != 0:
        raise RuntimeError(f"protoc failed with exit code {result}")
    logger.info("Auto-generated gRPC stubs from gitdb.proto")


# ─── Serialization helpers ───────────────────────────────────────────

def _tensor_to_bytes(tensor) -> bytes:
    """Serialize a torch tensor to bytes."""
    import torch
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _bytes_to_tensor(data: bytes):
    """Deserialize bytes back to a torch tensor."""
    import torch
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True)


def _json_or_none(s: str):
    """Parse JSON string, return None if empty."""
    if not s:
        return None
    return json.loads(s)


def _to_json(obj) -> str:
    """Serialize to JSON string."""
    return json.dumps(obj, separators=(",", ":"))


# ─── Service Implementation ─────────────────────────────────────────

class GitDBServicer:
    """gRPC service implementation wrapping a GitDB instance.

    All DB operations are guarded by a threading lock for safety.
    """

    def __init__(self, db):
        self.db = db
        self._lock = threading.Lock()

    def _wrap(self, context, fn, *args, **kwargs):
        """Execute fn under lock, catching errors as gRPC status codes."""
        try:
            with self._lock:
                return fn(*args, **kwargs)
        except ValueError as e:
            context.set_code(_grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return None
        except KeyError as e:
            context.set_code(_grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return None
        except Exception as e:
            context.set_code(_grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            logger.exception("gRPC handler error")
            return None

    # ─── Git operations ──────────────────────────────────────

    def Status(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            s = self.db.status()
            return pb2.StatusResponse(
                branch=s["branch"],
                head=s.get("head") or "",
                vectors=s.get("active_rows", 0),
                documents=s.get("documents", 0),
                tables=s.get("tables", []),
                table_rows=s.get("table_rows", 0),
            )
        result = self._wrap(context, _do)
        return result or _grpc_pb2.StatusResponse()

    def Commit(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            h = self.db.commit(request.message)
            return pb2.CommitResponse(hash=h)
        result = self._wrap(context, _do)
        return result or pb2.CommitResponse()

    def Log(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            entries = self.db.log()
            log_entries = []
            for c in entries:
                log_entries.append(pb2.LogEntry(
                    hash=c.hash,
                    parent=c.parent or "",
                    message=c.message,
                    timestamp=c.timestamp,
                    added=c.stats.added,
                    removed=c.stats.removed,
                    modified=c.stats.modified,
                ))
            return pb2.LogResponse(entries=log_entries)
        result = self._wrap(context, _do)
        return result or pb2.LogResponse()

    def Branch(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            self.db.branch(request.name)
            return pb2.Empty()
        result = self._wrap(context, _do)
        return result or pb2.Empty()

    def Switch(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            self.db.switch(request.branch)
            return pb2.Empty()
        result = self._wrap(context, _do)
        return result or pb2.Empty()

    def Merge(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            strategy = request.strategy or "union"
            r = self.db.merge(request.branch, strategy=strategy)
            return pb2.MergeResponse(
                commit_hash=r.commit_hash,
                strategy=r.strategy,
                added=r.added,
                removed=r.removed,
                conflicts=r.conflicts,
            )
        result = self._wrap(context, _do)
        return result or pb2.MergeResponse()

    def Stash(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            msg = request.message or "WIP"
            self.db.stash(msg)
            return pb2.Empty()
        result = self._wrap(context, _do)
        return result or pb2.Empty()

    def StashPop(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            entry = self.db.stash_pop()
            return pb2.StashEntry(
                index=entry.index,
                message=entry.message,
                timestamp=entry.timestamp,
            )
        result = self._wrap(context, _do)
        return result or pb2.StashEntry()

    def StashList(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            entries = self.db.stash_list()
            return pb2.StashListResponse(entries=[
                pb2.StashEntry(
                    index=e.index,
                    message=e.message,
                    timestamp=e.timestamp,
                ) for e in entries
            ])
        result = self._wrap(context, _do)
        return result or pb2.StashListResponse()

    def Reset(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            self.db.reset(request.ref or "HEAD")
            return pb2.Empty()
        result = self._wrap(context, _do)
        return result or pb2.Empty()

    def CherryPick(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            h = self.db.cherry_pick(request.ref)
            return pb2.CommitResponse(hash=h)
        result = self._wrap(context, _do)
        return result or pb2.CommitResponse()

    def Revert(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            h = self.db.revert(request.ref)
            return pb2.CommitResponse(hash=h)
        result = self._wrap(context, _do)
        return result or pb2.CommitResponse()

    def Branches(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            b = self.db.branches()
            return pb2.BranchListResponse(
                branches=b,
                current=self.db.current_branch,
            )
        result = self._wrap(context, _do)
        return result or pb2.BranchListResponse()

    # ─── Vector operations ───────────────────────────────────

    def AddVectors(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            import torch
            kwargs = {}

            # Texts for auto-embedding
            if request.texts:
                kwargs["texts"] = list(request.texts)

            # Pre-computed embeddings
            if request.embeddings:
                tensors = [_bytes_to_tensor(b) for b in request.embeddings]
                kwargs["embeddings"] = torch.stack(tensors) if len(tensors) > 1 else tensors[0]

            # Metadata
            if request.metadata_json:
                kwargs["metadata"] = json.loads(request.metadata_json)

            # Documents default to texts
            if "texts" in kwargs and "embeddings" not in kwargs:
                kwargs["documents"] = kwargs.get("texts")

            indices = self.db.add(**kwargs)
            return pb2.AddVectorsResponse(indices=indices)
        result = self._wrap(context, _do)
        return result or pb2.AddVectorsResponse()

    def QueryVectors(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            k = request.k or 10
            where = _json_or_none(request.where_json)

            if request.text:
                results = self.db.query_text(request.text, k=k, where=where)
            elif request.vector:
                vec = _bytes_to_tensor(request.vector)
                results = self.db.query(vec, k=k, where=where)
            else:
                raise ValueError("Must provide text or vector")

            return pb2.QueryResponse(
                ids=results.ids,
                scores=results.scores,
                documents=[d or "" for d in results.documents],
                metadata_json=[_to_json(m) for m in results.metadata],
            )
        result = self._wrap(context, _do)
        return result or pb2.QueryResponse()

    # ─── Document operations ─────────────────────────────────

    def InsertDoc(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            doc = json.loads(request.doc_json)
            result = self.db.insert(doc)
            if isinstance(result, list):
                return pb2.InsertDocResponse(ids=result)
            return pb2.InsertDocResponse(ids=[result])
        result = self._wrap(context, _do)
        return result or pb2.InsertDocResponse()

    def FindDocs(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            where = _json_or_none(request.where_json)
            limit = request.limit or None
            skip = request.skip or 0
            docs = self.db.find(where=where, limit=limit, skip=skip)
            return pb2.FindResponse(docs_json=[_to_json(d) for d in docs])
        result = self._wrap(context, _do)
        return result or pb2.FindResponse()

    def UpdateDocs(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            where = json.loads(request.where_json)
            set_fields = json.loads(request.set_json)
            count = self.db.update_docs(where, set_fields)
            return pb2.CountResponse(count=count)
        result = self._wrap(context, _do)
        return result or pb2.CountResponse()

    def DeleteDocs(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            where = json.loads(request.where_json)
            count = self.db.delete_docs(where)
            return pb2.CountResponse(count=count)
        result = self._wrap(context, _do)
        return result or pb2.CountResponse()

    def CountDocs(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            where = _json_or_none(request.where_json)
            count = self.db.count_docs(where=where)
            return pb2.CountResponse(count=count)
        result = self._wrap(context, _do)
        return result or pb2.CountResponse()

    # ─── Table operations ────────────────────────────────────

    def CreateTable(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            columns = _json_or_none(request.columns_json)
            self.db.create_table(request.name, columns)
            return pb2.Empty()
        result = self._wrap(context, _do)
        return result or pb2.Empty()

    def ListTables(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            names = self.db.list_tables()
            return pb2.TableListResponse(names=names)
        result = self._wrap(context, _do)
        return result or pb2.TableListResponse()

    def TableInsert(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            rows = json.loads(request.rows_json)
            result = self.db.insert_into(request.table, rows)
            if isinstance(result, list):
                return pb2.CountResponse(count=len(result))
            return pb2.CountResponse(count=1)
        result = self._wrap(context, _do)
        return result or pb2.CountResponse()

    def TableSelect(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            columns = _json_or_none(request.columns_json)
            where = _json_or_none(request.where_json)
            rows = self.db.select_from(
                request.table,
                columns=columns,
                where=where,
                order_by=request.order_by or None,
                desc=request.desc,
                limit=request.limit or None,
                offset=request.offset or 0,
            )
            return pb2.TableSelectResponse(rows_json=[_to_json(r) for r in rows])
        result = self._wrap(context, _do)
        return result or pb2.TableSelectResponse()

    def TableUpdate(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            where = json.loads(request.where_json)
            set_fields = json.loads(request.set_json)
            count = self.db.update_table(request.table, where, set_fields)
            return pb2.CountResponse(count=count)
        result = self._wrap(context, _do)
        return result or pb2.CountResponse()

    def TableDelete(self, request, context):
        pb2 = _grpc_pb2
        def _do():
            where = json.loads(request.where_json)
            count = self.db.delete_from(request.table, where)
            return pb2.CountResponse(count=count)
        result = self._wrap(context, _do)
        return result or pb2.CountResponse()


# ─── Server ──────────────────────────────────────────────────────────

def serve_grpc(
    db_path: str,
    port: int = 50051,
    dim: int = 1024,
    device: str = "cpu",
    max_workers: int = 10,
    block: bool = True,
):
    """Start the gRPC server wrapping a GitDB instance.

    Args:
        db_path: Path to the GitDB store.
        port: Port to listen on (default 50051).
        dim: Vector dimension for new stores.
        device: Torch device (cpu, cuda, mps).
        max_workers: Thread pool size.
        block: If True, block until interrupted. If False, return server.

    Returns:
        The grpc.Server instance (useful when block=False).
    """
    _ensure_grpc()
    from gitdb import GitDB

    db = GitDB(db_path, dim=dim, device=device)
    servicer = GitDBServicer(db)

    server = _grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    _grpc_pb2_grpc.add_GitDBServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"GitDB gRPC server listening on port {port}")

    if block:
        print(f"GitDB gRPC server running on port {port} — Ctrl+C to stop")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            server.stop(grace=2)
            print("\nServer stopped.")
    return server


# ─── Client ──────────────────────────────────────────────────────────

class GitDBClient:
    """Pythonic gRPC client for GitDB.

    Wraps the raw gRPC stubs with clean method signatures.

    Usage:
        client = GitDBClient("localhost:50051")
        client.insert({"name": "Alice", "age": 30})
        docs = client.find(where={"name": "Alice"})
        client.commit("added users")
        client.close()
    """

    def __init__(self, target: str = "localhost:50051"):
        _ensure_grpc()
        self._channel = _grpc.insecure_channel(target)
        self._stub = _grpc_pb2_grpc.GitDBServiceStub(self._channel)
        self._pb2 = _grpc_pb2

    def close(self):
        """Close the gRPC channel."""
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ─── Git operations ──────────────────────────────────────

    def status(self) -> dict:
        """Get database status."""
        r = self._stub.Status(self._pb2.Empty())
        return {
            "branch": r.branch,
            "head": r.head,
            "vectors": r.vectors,
            "documents": r.documents,
            "tables": list(r.tables),
            "table_rows": r.table_rows,
        }

    def commit(self, message: str) -> str:
        """Commit staged changes. Returns commit hash."""
        r = self._stub.Commit(self._pb2.CommitRequest(message=message))
        return r.hash

    def log(self) -> List[dict]:
        """Get commit history."""
        r = self._stub.Log(self._pb2.Empty())
        return [
            {
                "hash": e.hash,
                "parent": e.parent,
                "message": e.message,
                "timestamp": e.timestamp,
                "added": e.added,
                "removed": e.removed,
                "modified": e.modified,
            }
            for e in r.entries
        ]

    def branch(self, name: str):
        """Create a new branch."""
        self._stub.Branch(self._pb2.BranchRequest(name=name))

    def switch(self, branch: str):
        """Switch to a branch."""
        self._stub.Switch(self._pb2.SwitchRequest(branch=branch))

    def merge(self, branch: str, strategy: str = "union") -> dict:
        """Merge a branch."""
        r = self._stub.Merge(self._pb2.MergeRequest(
            branch=branch, strategy=strategy,
        ))
        return {
            "commit_hash": r.commit_hash,
            "strategy": r.strategy,
            "added": r.added,
            "removed": r.removed,
            "conflicts": list(r.conflicts),
        }

    def stash(self, message: str = "WIP"):
        """Stash current changes."""
        self._stub.Stash(self._pb2.StashRequest(message=message))

    def stash_pop(self) -> dict:
        """Pop the last stash entry."""
        r = self._stub.StashPop(self._pb2.Empty())
        return {"index": r.index, "message": r.message, "timestamp": r.timestamp}

    def stash_list(self) -> List[dict]:
        """List stash entries."""
        r = self._stub.StashList(self._pb2.Empty())
        return [
            {"index": e.index, "message": e.message, "timestamp": e.timestamp}
            for e in r.entries
        ]

    def reset(self, ref: str = "HEAD"):
        """Reset to a ref."""
        self._stub.Reset(self._pb2.RefRequest(ref=ref))

    def cherry_pick(self, ref: str) -> str:
        """Cherry-pick a commit. Returns new commit hash."""
        r = self._stub.CherryPick(self._pb2.RefRequest(ref=ref))
        return r.hash

    def revert(self, ref: str) -> str:
        """Revert a commit. Returns new commit hash."""
        r = self._stub.Revert(self._pb2.RefRequest(ref=ref))
        return r.hash

    def branches(self) -> dict:
        """List branches. Returns {name: hash, ...} and 'current'."""
        r = self._stub.Branches(self._pb2.Empty())
        return {"branches": dict(r.branches), "current": r.current}

    # ─── Vector operations ───────────────────────────────────

    def add_vectors(
        self,
        texts: Optional[List[str]] = None,
        embeddings=None,
        metadata: Optional[List[dict]] = None,
    ) -> List[int]:
        """Add vectors by text (auto-embed) or pre-computed embeddings.

        Args:
            texts: Texts to embed and add.
            embeddings: Pre-computed torch tensors (list or stacked tensor).
            metadata: Per-vector metadata dicts.

        Returns:
            List of assigned indices.
        """
        req = self._pb2.AddVectorsRequest()
        if texts:
            req.texts.extend(texts)
        if embeddings is not None:
            import torch
            if isinstance(embeddings, torch.Tensor):
                if embeddings.dim() == 1:
                    embeddings = [embeddings]
                else:
                    embeddings = [embeddings[i] for i in range(embeddings.size(0))]
            for t in embeddings:
                req.embeddings.append(_tensor_to_bytes(t))
        if metadata:
            req.metadata_json = _to_json(metadata)

        r = self._stub.AddVectors(req)
        return list(r.indices)

    def query(
        self,
        text: Optional[str] = None,
        vector=None,
        k: int = 10,
        where: Optional[dict] = None,
    ) -> dict:
        """Query vectors by text or vector.

        Returns:
            Dict with ids, scores, documents, metadata.
        """
        req = self._pb2.QueryRequest(k=k)
        if text:
            req.text = text
        if vector is not None:
            req.vector = _tensor_to_bytes(vector)
        if where:
            req.where_json = _to_json(where)

        r = self._stub.QueryVectors(req)
        return {
            "ids": list(r.ids),
            "scores": list(r.scores),
            "documents": list(r.documents),
            "metadata": [json.loads(m) for m in r.metadata_json] if r.metadata_json else [],
        }

    # ─── Document operations ─────────────────────────────────

    def insert(self, doc: Union[dict, List[dict]]) -> List[str]:
        """Insert document(s). Returns list of _ids."""
        r = self._stub.InsertDoc(self._pb2.InsertDocRequest(
            doc_json=_to_json(doc),
        ))
        return list(r.ids)

    def find(
        self,
        where: Optional[dict] = None,
        limit: int = 0,
        skip: int = 0,
    ) -> List[dict]:
        """Find documents matching a query."""
        r = self._stub.FindDocs(self._pb2.FindRequest(
            where_json=_to_json(where) if where else "",
            limit=limit,
            skip=skip,
        ))
        return [json.loads(d) for d in r.docs_json]

    def update_docs(self, where: dict, set_fields: dict) -> int:
        """Update matching documents. Returns count."""
        r = self._stub.UpdateDocs(self._pb2.UpdateDocRequest(
            where_json=_to_json(where),
            set_json=_to_json(set_fields),
        ))
        return r.count

    def delete_docs(self, where: dict) -> int:
        """Delete matching documents. Returns count."""
        r = self._stub.DeleteDocs(self._pb2.DeleteDocRequest(
            where_json=_to_json(where),
        ))
        return r.count

    def count_docs(self, where: Optional[dict] = None) -> int:
        """Count matching documents."""
        r = self._stub.CountDocs(self._pb2.FindRequest(
            where_json=_to_json(where) if where else "",
        ))
        return r.count

    # ─── Table operations ────────────────────────────────────

    def create_table(self, name: str, columns: Optional[dict] = None):
        """Create a named table with optional column schema."""
        self._stub.CreateTable(self._pb2.CreateTableRequest(
            name=name,
            columns_json=_to_json(columns) if columns else "",
        ))

    def list_tables(self) -> List[str]:
        """List all table names."""
        r = self._stub.ListTables(self._pb2.Empty())
        return list(r.names)

    def table_insert(self, table: str, rows: Union[dict, List[dict]]) -> int:
        """Insert row(s) into a table. Returns count inserted."""
        r = self._stub.TableInsert(self._pb2.TableInsertRequest(
            table=table,
            rows_json=_to_json(rows),
        ))
        return r.count

    def table_select(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        where: Optional[dict] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        limit: int = 0,
        offset: int = 0,
    ) -> List[dict]:
        """SELECT from a table."""
        r = self._stub.TableSelect(self._pb2.TableSelectRequest(
            table=table,
            columns_json=_to_json(columns) if columns else "",
            where_json=_to_json(where) if where else "",
            order_by=order_by or "",
            desc=desc,
            limit=limit,
            offset=offset,
        ))
        return [json.loads(row) for row in r.rows_json]

    def table_update(self, table: str, where: dict, set_fields: dict) -> int:
        """UPDATE rows in a table. Returns count."""
        r = self._stub.TableUpdate(self._pb2.TableUpdateRequest(
            table=table,
            where_json=_to_json(where),
            set_json=_to_json(set_fields),
        ))
        return r.count

    def table_delete(self, table: str, where: dict) -> int:
        """DELETE from a table. Returns count."""
        r = self._stub.TableDelete(self._pb2.TableDeleteRequest(
            table=table,
            where_json=_to_json(where),
        ))
        return r.count


# ─── CLI entry point ─────────────────────────────────────────────────

def cmd_grpc_serve(args):
    """CLI handler for `gitdb grpc-serve`."""
    import sys
    from pathlib import Path

    path = getattr(args, "path", None)
    if path is None:
        # Try to find store in current directory
        from gitdb.cli import find_store
        path = find_store()
        if path is None:
            print("fatal: not a gitdb repository (or any parent)", file=sys.stderr)
            sys.exit(1)

    port = getattr(args, "port", 50051)
    device = getattr(args, "device", "cpu")
    dim = getattr(args, "dim", 1024)
    workers = getattr(args, "workers", 10)

    serve_grpc(
        db_path=path,
        port=port,
        dim=dim,
        device=device,
        max_workers=workers,
        block=True,
    )
