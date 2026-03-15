"""Tests for GitDB gRPC service layer.

Tests the servicer logic directly (no network), serialization helpers,
client method signatures, and error handling.
"""

import io
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from gitdb import GitDB
from gitdb.types import CommitInfo, CommitStats, MergeResult, Results, StashEntry


# ─── Helpers ─────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    return GitDB(str(tmp_path / "grpc_test"), dim=8, device="cpu")


class FakeContext:
    """Mock gRPC context for testing servicer methods directly."""

    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def make_pb2_mocks():
    """Create mock pb2 message classes that behave like simple containers."""
    class Msg:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    return Msg


# ─── Serialization tests ────────────────────────────────────────────

class TestSerialization:
    def test_tensor_roundtrip(self):
        from gitdb.grpc_service import _tensor_to_bytes, _bytes_to_tensor
        t = torch.randn(4, 8)
        data = _tensor_to_bytes(t)
        assert isinstance(data, bytes)
        assert len(data) > 0
        t2 = _bytes_to_tensor(data)
        assert torch.allclose(t, t2)

    def test_tensor_1d_roundtrip(self):
        from gitdb.grpc_service import _tensor_to_bytes, _bytes_to_tensor
        t = torch.randn(8)
        data = _tensor_to_bytes(t)
        t2 = _bytes_to_tensor(data)
        assert torch.allclose(t, t2)

    def test_json_or_none_empty(self):
        from gitdb.grpc_service import _json_or_none
        assert _json_or_none("") is None
        assert _json_or_none(None) is None

    def test_json_or_none_valid(self):
        from gitdb.grpc_service import _json_or_none
        result = _json_or_none('{"name": "Alice"}')
        assert result == {"name": "Alice"}

    def test_to_json(self):
        from gitdb.grpc_service import _to_json
        result = _to_json({"a": 1, "b": [2, 3]})
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": [2, 3]}
        # Compact format (no spaces)
        assert " " not in result


# ─── Servicer direct tests (no gRPC transport) ──────────────────────

class TestServicerDirect:
    """Test GitDBServicer methods by calling them directly with a real DB."""

    def _make_servicer(self, db):
        """Build a servicer with real DB but mock pb2 references."""
        from gitdb.grpc_service import GitDBServicer
        return GitDBServicer(db)

    def test_servicer_lock_exists(self, db):
        servicer = self._make_servicer(db)
        assert hasattr(servicer, "_lock")
        assert isinstance(servicer._lock, type(threading.Lock()))

    def test_wrap_catches_value_error(self, db):
        """_wrap should catch ValueError and set INVALID_ARGUMENT."""
        servicer = self._make_servicer(db)
        ctx = FakeContext()

        # Manually patch grpc status codes
        import gitdb.grpc_service as gs
        mock_grpc = MagicMock()
        mock_grpc.StatusCode.INVALID_ARGUMENT = "INVALID_ARGUMENT"
        old_grpc = gs._grpc
        gs._grpc = mock_grpc
        try:
            def bad():
                raise ValueError("test error")
            result = servicer._wrap(ctx, bad)
            assert result is None
            assert ctx.code == "INVALID_ARGUMENT"
            assert ctx.details == "test error"
        finally:
            gs._grpc = old_grpc

    def test_wrap_catches_key_error(self, db):
        """_wrap should catch KeyError and set NOT_FOUND."""
        servicer = self._make_servicer(db)
        ctx = FakeContext()

        import gitdb.grpc_service as gs
        mock_grpc = MagicMock()
        mock_grpc.StatusCode.NOT_FOUND = "NOT_FOUND"
        old_grpc = gs._grpc
        gs._grpc = mock_grpc
        try:
            def bad():
                raise KeyError("missing_table")
            result = servicer._wrap(ctx, bad)
            assert result is None
            assert ctx.code == "NOT_FOUND"
        finally:
            gs._grpc = old_grpc

    def test_wrap_catches_generic_error(self, db):
        """_wrap should catch Exception and set INTERNAL."""
        servicer = self._make_servicer(db)
        ctx = FakeContext()

        import gitdb.grpc_service as gs
        mock_grpc = MagicMock()
        mock_grpc.StatusCode.INTERNAL = "INTERNAL"
        old_grpc = gs._grpc
        gs._grpc = mock_grpc
        try:
            def bad():
                raise RuntimeError("boom")
            result = servicer._wrap(ctx, bad)
            assert result is None
            assert ctx.code == "INTERNAL"
        finally:
            gs._grpc = old_grpc

    def test_wrap_returns_result_on_success(self, db):
        """_wrap should return the function's result on success."""
        servicer = self._make_servicer(db)
        ctx = FakeContext()

        import gitdb.grpc_service as gs
        gs._grpc = MagicMock()
        try:
            result = servicer._wrap(ctx, lambda: 42)
            assert result == 42
            assert ctx.code is None
        finally:
            gs._grpc = None


# ─── Client class tests (no server needed) ──────────────────────────

class TestClientMethods:
    """Test GitDBClient method signatures and logic with a mocked stub."""

    def _make_client(self):
        """Create a client with mocked channel and stub."""
        import gitdb.grpc_service as gs

        # Mock the grpc module and pb2 modules
        mock_grpc = MagicMock()
        mock_channel = MagicMock()
        mock_grpc.insecure_channel.return_value = mock_channel

        # Create a minimal mock pb2 with message classes
        mock_pb2 = MagicMock()
        mock_pb2_grpc = MagicMock()

        old_grpc = gs._grpc
        old_pb2 = gs._grpc_pb2
        old_pb2_grpc = gs._grpc_pb2_grpc

        gs._grpc = mock_grpc
        gs._grpc_pb2 = mock_pb2
        gs._grpc_pb2_grpc = mock_pb2_grpc

        client = gs.GitDBClient("localhost:50051")

        # Return client and cleanup function
        def cleanup():
            gs._grpc = old_grpc
            gs._grpc_pb2 = old_pb2
            gs._grpc_pb2_grpc = old_pb2_grpc

        return client, cleanup

    def test_client_context_manager(self):
        client, cleanup = self._make_client()
        try:
            with client as c:
                assert c is client
            # Channel should be closed
            client._channel.close.assert_called_once()
        finally:
            cleanup()

    def test_client_insert_single_doc(self):
        client, cleanup = self._make_client()
        try:
            mock_response = MagicMock()
            mock_response.ids = ["abc123"]
            client._stub.InsertDoc.return_value = mock_response

            result = client.insert({"name": "Alice"})
            assert result == ["abc123"]

            # Verify the request was created with correct JSON
            call_args = client._pb2.InsertDocRequest.call_args
            doc_json = call_args[1].get("doc_json") or call_args[0][0] if call_args[0] else call_args[1]["doc_json"]
            parsed = json.loads(doc_json)
            assert parsed == {"name": "Alice"}
        finally:
            cleanup()

    def test_client_insert_batch(self):
        client, cleanup = self._make_client()
        try:
            mock_response = MagicMock()
            mock_response.ids = ["id1", "id2"]
            client._stub.InsertDoc.return_value = mock_response

            docs = [{"name": "Bob"}, {"name": "Charlie"}]
            result = client.insert(docs)
            assert result == ["id1", "id2"]
        finally:
            cleanup()

    def test_client_find(self):
        client, cleanup = self._make_client()
        try:
            mock_response = MagicMock()
            mock_response.docs_json = ['{"name":"Alice","_id":"1"}', '{"name":"Bob","_id":"2"}']
            client._stub.FindDocs.return_value = mock_response

            result = client.find(where={"name": "Alice"})
            assert len(result) == 2
            assert result[0]["name"] == "Alice"
        finally:
            cleanup()

    def test_client_commit(self):
        client, cleanup = self._make_client()
        try:
            mock_response = MagicMock()
            mock_response.hash = "abc123def456"
            client._stub.Commit.return_value = mock_response

            h = client.commit("test commit")
            assert h == "abc123def456"
        finally:
            cleanup()

    def test_client_status(self):
        client, cleanup = self._make_client()
        try:
            mock_response = MagicMock()
            mock_response.branch = "main"
            mock_response.head = "abc123"
            mock_response.vectors = 100
            mock_response.documents = 50
            mock_response.tables = ["users", "logs"]
            mock_response.table_rows = 75
            client._stub.Status.return_value = mock_response

            s = client.status()
            assert s["branch"] == "main"
            assert s["vectors"] == 100
            assert s["documents"] == 50
            assert s["tables"] == ["users", "logs"]
        finally:
            cleanup()

    def test_client_create_table(self):
        client, cleanup = self._make_client()
        try:
            client._stub.CreateTable.return_value = MagicMock()
            client.create_table("users", {"name": "text", "age": "integer"})

            call_args = client._pb2.CreateTableRequest.call_args
            kwargs = call_args[1]
            assert kwargs["name"] == "users"
            parsed = json.loads(kwargs["columns_json"])
            assert parsed == {"name": "text", "age": "integer"}
        finally:
            cleanup()

    def test_client_table_select(self):
        client, cleanup = self._make_client()
        try:
            mock_response = MagicMock()
            mock_response.rows_json = ['{"name":"Alice","age":30}']
            client._stub.TableSelect.return_value = mock_response

            rows = client.table_select("users", where={"name": "Alice"})
            assert len(rows) == 1
            assert rows[0]["age"] == 30
        finally:
            cleanup()

    def test_client_log(self):
        client, cleanup = self._make_client()
        try:
            entry = MagicMock()
            entry.hash = "abc"
            entry.parent = ""
            entry.message = "init"
            entry.timestamp = 1234.0
            entry.added = 5
            entry.removed = 0
            entry.modified = 0

            mock_response = MagicMock()
            mock_response.entries = [entry]
            client._stub.Log.return_value = mock_response

            log = client.log()
            assert len(log) == 1
            assert log[0]["hash"] == "abc"
            assert log[0]["added"] == 5
        finally:
            cleanup()

    def test_client_branches(self):
        client, cleanup = self._make_client()
        try:
            mock_response = MagicMock()
            mock_response.branches = {"main": "abc", "dev": "def"}
            mock_response.current = "main"
            client._stub.Branches.return_value = mock_response

            b = client.branches()
            assert b["current"] == "main"
            assert "main" in b["branches"]
        finally:
            cleanup()


# ─── Proto file existence test ───────────────────────────────────────

class TestProtoFile:
    def test_proto_file_exists(self):
        proto_path = Path(__file__).parent.parent / "gitdb" / "gitdb.proto"
        assert proto_path.exists(), f"Proto file missing: {proto_path}"

    def test_proto_has_service_definition(self):
        proto_path = Path(__file__).parent.parent / "gitdb" / "gitdb.proto"
        content = proto_path.read_text()
        assert "service GitDBService" in content
        assert "rpc Status" in content
        assert "rpc Commit" in content
        assert "rpc InsertDoc" in content
        assert "rpc TableSelect" in content
        assert "rpc QueryVectors" in content

    def test_proto_has_all_messages(self):
        proto_path = Path(__file__).parent.parent / "gitdb" / "gitdb.proto"
        content = proto_path.read_text()
        expected = [
            "StatusResponse", "CommitRequest", "CommitResponse",
            "LogEntry", "LogResponse", "QueryRequest", "QueryResponse",
            "InsertDocRequest", "FindRequest", "FindResponse",
            "CreateTableRequest", "TableSelectRequest", "TableSelectResponse",
            "MergeRequest", "MergeResponse", "BranchListResponse",
        ]
        for msg in expected:
            assert f"message {msg}" in content, f"Missing message: {msg}"


# ─── Compile test (validates proto syntax) ───────────────────────────

class TestProtoCompile:
    def test_proto_compiles(self):
        """Validate that the proto file has valid syntax by attempting compilation."""
        try:
            from grpc_tools import protoc
        except ImportError:
            pytest.skip("grpcio-tools not installed")

        proto_dir = str(Path(__file__).parent.parent / "gitdb")
        proto_file = str(Path(__file__).parent.parent / "gitdb" / "gitdb.proto")

        # Compile to a temp directory so we don't pollute the source
        with tempfile.TemporaryDirectory() as tmpdir:
            result = protoc.main([
                "grpc_tools.protoc",
                f"-I{proto_dir}",
                f"--python_out={tmpdir}",
                f"--grpc_python_out={tmpdir}",
                proto_file,
            ])
            assert result == 0, "Proto compilation failed"

            # Verify output files were created
            assert (Path(tmpdir) / "gitdb_pb2.py").exists()
            assert (Path(tmpdir) / "gitdb_pb2_grpc.py").exists()
