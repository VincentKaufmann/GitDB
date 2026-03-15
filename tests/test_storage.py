"""Tests for gitdb.storage — pluggable storage backends."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from gitdb.storage import (
    AzureStorage,
    GCSStorage,
    LocalStorage,
    S3Storage,
    SFTPStorage,
    StorageBackend,
    copy_between,
    parse_storage_uri,
)


# ---------------------------------------------------------------------------
# LocalStorage CRUD
# ---------------------------------------------------------------------------

@pytest.fixture
def local(tmp_path):
    return LocalStorage(tmp_path / "store")


def test_local_write_and_read(local):
    local.write("a/b", b"hello")
    assert local.read("a/b") == b"hello"


def test_local_exists(local):
    assert not local.exists("x")
    local.write("x", b"1")
    assert local.exists("x")


def test_local_delete(local):
    local.write("k", b"data")
    local.delete("k")
    assert not local.exists("k")


def test_local_delete_missing_raises(local):
    with pytest.raises(KeyError):
        local.delete("nope")


def test_local_read_missing_raises(local):
    with pytest.raises(KeyError):
        local.read("nope")


def test_local_overwrite(local):
    local.write("f", b"v1")
    local.write("f", b"v2")
    assert local.read("f") == b"v2"


# ---------------------------------------------------------------------------
# Key listing with prefix
# ---------------------------------------------------------------------------

def test_local_list_keys_all(local):
    local.write("objects/aaa", b"1")
    local.write("objects/bbb", b"2")
    local.write("cache/ccc", b"3")
    keys = local.list_keys()
    assert keys == ["cache/ccc", "objects/aaa", "objects/bbb"]


def test_local_list_keys_with_prefix(local):
    local.write("objects/aaa", b"1")
    local.write("objects/bbb", b"2")
    local.write("cache/ccc", b"3")
    keys = local.list_keys("objects/")
    assert keys == ["objects/aaa", "objects/bbb"]


def test_local_list_keys_empty(local):
    assert local.list_keys() == []


# ---------------------------------------------------------------------------
# parse_storage_uri — all schemes
# ---------------------------------------------------------------------------

def test_parse_local_path():
    with tempfile.TemporaryDirectory() as d:
        backend = parse_storage_uri(d)
        assert isinstance(backend, LocalStorage)
        assert str(backend.root) == d


def test_parse_s3_uri():
    backend = parse_storage_uri("s3://my-bucket/some/prefix")
    assert isinstance(backend, S3Storage)
    assert backend.bucket == "my-bucket"
    assert backend.prefix == "some/prefix"


def test_parse_minio_uri():
    backend = parse_storage_uri("minio://localhost:9000/mybucket/data")
    assert isinstance(backend, S3Storage)
    assert backend.bucket == "mybucket"
    assert backend.prefix == "data/"
    assert backend._endpoint_url == "http://localhost:9000"


def test_parse_gs_uri():
    backend = parse_storage_uri("gs://gcs-bucket/prefix/path")
    assert isinstance(backend, GCSStorage)
    assert backend.bucket_name == "gcs-bucket"
    assert backend.prefix == "prefix/path"


def test_parse_az_uri():
    with mock.patch.dict(os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "fake"}):
        backend = parse_storage_uri("az://mycontainer/pre")
        assert isinstance(backend, AzureStorage)
        assert backend.container_name == "mycontainer"
        assert backend.prefix == "pre"


def test_parse_sftp_uri():
    backend = parse_storage_uri("sftp://alice@myhost.com/data/store")
    assert isinstance(backend, SFTPStorage)
    assert backend.host == "myhost.com"
    assert backend.username == "alice"
    assert backend.remote_root == "/data/store"


def test_parse_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_storage_uri("ftp://something")


# ---------------------------------------------------------------------------
# copy_between
# ---------------------------------------------------------------------------

def test_copy_between_local(tmp_path):
    src = LocalStorage(tmp_path / "src")
    dst = LocalStorage(tmp_path / "dst")

    src.write("objects/a", b"alpha")
    src.write("objects/b", b"beta")
    src.write("other/c", b"gamma")

    count = copy_between(src, dst, prefix="objects/")
    assert count == 2
    assert dst.read("objects/a") == b"alpha"
    assert dst.read("objects/b") == b"beta"
    assert not dst.exists("other/c")


def test_copy_between_all(tmp_path):
    src = LocalStorage(tmp_path / "s")
    dst = LocalStorage(tmp_path / "d")
    src.write("x", b"1")
    src.write("y", b"2")
    assert copy_between(src, dst) == 2
    assert dst.list_keys() == ["x", "y"]


# ---------------------------------------------------------------------------
# Cloud backends with mocked clients
# ---------------------------------------------------------------------------

def test_s3_write_read_mocked():
    s3 = S3Storage("bucket", prefix="pfx/")
    mock_client = mock.MagicMock()
    s3._client = mock_client

    # write
    s3.write("key1", b"data")
    mock_client.put_object.assert_called_once_with(
        Bucket="bucket", Key="pfx/key1", Body=b"data",
    )

    # read
    body = mock.MagicMock()
    body.read.return_value = b"data"
    mock_client.get_object.return_value = {"Body": body}
    assert s3.read("key1") == b"data"
    mock_client.get_object.assert_called_once_with(
        Bucket="bucket", Key="pfx/key1",
    )


def test_s3_exists_mocked():
    s3 = S3Storage("b")
    mock_client = mock.MagicMock()
    s3._client = mock_client

    mock_client.head_object.return_value = {}
    assert s3.exists("k") is True

    mock_client.head_object.side_effect = Exception("nope")
    assert s3.exists("k") is False


def test_s3_list_keys_mocked():
    s3 = S3Storage("b", prefix="p/")
    mock_client = mock.MagicMock()
    s3._client = mock_client

    paginator = mock.MagicMock()
    mock_client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "p/a"}, {"Key": "p/b"}]},
    ]
    keys = s3.list_keys()
    assert keys == ["a", "b"]


def test_gcs_write_read_mocked():
    gcs = GCSStorage("bucket", prefix="g/")
    mock_bucket = mock.MagicMock()
    gcs._bucket = mock_bucket

    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob

    gcs.write("f", b"bytes")
    mock_blob.upload_from_string.assert_called_once_with(b"bytes")

    mock_blob.exists.return_value = True
    mock_blob.download_as_bytes.return_value = b"bytes"
    assert gcs.read("f") == b"bytes"


def test_azure_write_read_mocked():
    az = AzureStorage("ctr", connection_string="fake", prefix="az/")
    mock_container = mock.MagicMock()
    az._container_client = mock_container

    az.write("obj", b"stuff")
    mock_container.upload_blob.assert_called_once_with(
        "az/obj", b"stuff", overwrite=True,
    )

    mock_container.download_blob.return_value.readall.return_value = b"stuff"
    assert az.read("obj") == b"stuff"


# ---------------------------------------------------------------------------
# Thread-safety sanity (LocalStorage)
# ---------------------------------------------------------------------------

def test_local_concurrent_writes(tmp_path):
    import threading

    store = LocalStorage(tmp_path / "concurrent")
    errors = []

    def writer(i):
        try:
            store.write(f"k/{i}", f"v{i}".encode())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(store.list_keys("k/")) == 20
