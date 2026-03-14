"""Tests for cloud storage ingest transport layer."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import torch

from gitdb import GitDB
from gitdb.cloud_ingest import (
    is_cloud_uri, parse_cloud_uri, ingest_cloud,
    _ingest_s3, _ingest_minio, _ingest_gcs, _ingest_azure, _ingest_sftp,
    CLOUD_SCHEMES,
)


@pytest.fixture
def db(tmp_path):
    return GitDB(str(tmp_path / "store"), dim=8, device="cpu")


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for ingest tests."""
    p = tmp_path / "data.csv"
    p.write_text("id,name,content\n1,Alice,Hello world\n2,Bob,Goodbye world\n")
    return str(p)


# ═══ URI Detection ═════════════════════════════════════════════

class TestIsCloudUri:
    def test_s3(self):
        assert is_cloud_uri("s3://bucket/prefix/") is True

    def test_gs(self):
        assert is_cloud_uri("gs://bucket/prefix/") is True

    def test_az(self):
        assert is_cloud_uri("az://container/path/") is True

    def test_minio(self):
        assert is_cloud_uri("minio://localhost:9000/bucket/prefix/") is True

    def test_sftp(self):
        assert is_cloud_uri("sftp://host/path/") is True

    def test_local_path(self):
        assert is_cloud_uri("/home/user/data.csv") is False

    def test_relative_path(self):
        assert is_cloud_uri("data/file.csv") is False

    def test_http_not_cloud(self):
        assert is_cloud_uri("http://example.com/file") is False

    def test_empty_string(self):
        assert is_cloud_uri("") is False

    def test_none(self):
        assert is_cloud_uri(None) is False

    def test_integer(self):
        assert is_cloud_uri(42) is False


# ═══ URI Parsing ══════════════════════════════════════════════

class TestParseCloudUri:
    def test_s3_basic(self):
        scheme, bucket, prefix, extra = parse_cloud_uri("s3://my-bucket/data/files/")
        assert scheme == "s3"
        assert bucket == "my-bucket"
        assert prefix == "data/files/"
        assert extra == {}

    def test_s3_no_prefix(self):
        scheme, bucket, prefix, extra = parse_cloud_uri("s3://my-bucket")
        assert scheme == "s3"
        assert bucket == "my-bucket"
        assert prefix == ""

    def test_s3_single_prefix(self):
        scheme, bucket, prefix, extra = parse_cloud_uri("s3://bucket/prefix")
        assert bucket == "bucket"
        assert prefix == "prefix"

    def test_gs(self):
        scheme, bucket, prefix, _ = parse_cloud_uri("gs://gcs-bucket/some/path/")
        assert scheme == "gs"
        assert bucket == "gcs-bucket"
        assert prefix == "some/path/"

    def test_az(self):
        scheme, container, path, _ = parse_cloud_uri("az://my-container/blob/path/")
        assert scheme == "az"
        assert container == "my-container"
        assert path == "blob/path/"

    def test_minio(self):
        scheme, bucket, prefix, extra = parse_cloud_uri("minio://localhost:9000/mybucket/data/")
        assert scheme == "minio"
        assert bucket == "mybucket"
        assert prefix == "data/"
        assert "endpoint" in extra
        assert "localhost:9000" in extra["endpoint"]

    def test_minio_no_prefix(self):
        scheme, bucket, prefix, extra = parse_cloud_uri("minio://host:9000/bucket")
        assert bucket == "bucket"
        assert prefix == ""

    def test_sftp_basic(self):
        scheme, host, path, extra = parse_cloud_uri("sftp://myhost/data/files/")
        assert scheme == "sftp"
        assert host == "myhost"
        assert path == "data/files/"

    def test_sftp_with_user(self):
        scheme, host, path, extra = parse_cloud_uri("sftp://admin@myhost/data/")
        assert host == "myhost"
        assert extra["username"] == "admin"

    def test_sftp_with_port(self):
        scheme, host, path, extra = parse_cloud_uri("sftp://myhost:2222/data/")
        assert host == "myhost"
        assert extra["port"] == "2222"

    def test_sftp_with_user_and_port(self):
        scheme, host, path, extra = parse_cloud_uri("sftp://user@myhost:2222/path/")
        assert host == "myhost"
        assert extra["username"] == "user"
        assert extra["port"] == "2222"

    def test_invalid_no_scheme(self):
        with pytest.raises(ValueError, match="no scheme"):
            parse_cloud_uri("/local/path")

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_cloud_uri("ftp://host/path")

    def test_minio_missing_bucket(self):
        with pytest.raises(ValueError, match="bucket"):
            parse_cloud_uri("minio://host:9000")


# ═══ S3 Transport (mocked) ═══════════════════════════════════

class TestS3Ingest:
    def test_s3_ingest_flow(self, db, sample_csv, tmp_path):
        """Test full S3 ingest with mocked boto3."""
        mock_s3 = MagicMock()

        # Mock paginator to return one object
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/data.csv"}]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        # When download_file is called, copy our sample CSV
        def fake_download(bucket, key, path):
            shutil.copy(sample_csv, path)
        mock_s3.download_file.side_effect = fake_download

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            tmpdir = tempfile.mkdtemp()
            try:
                results = _ingest_s3(db, "test-bucket", "prefix/", tmpdir, embed=False, autocommit=False)
                assert len(results) == 1
                assert results[0].get("rows", 0) == 2
                assert results[0]["_cloud_key"] == "s3://test-bucket/prefix/data.csv"
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_s3_skips_directory_markers(self, db, tmp_path):
        """S3 should skip keys ending with /."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/"}, {"Key": "prefix/subdir/"}]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            tmpdir = tempfile.mkdtemp()
            try:
                results = _ingest_s3(db, "bucket", "prefix/", tmpdir, embed=False, autocommit=False)
                assert len(results) == 0
                mock_s3.download_file.assert_not_called()
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_s3_empty_bucket(self, db, tmp_path):
        """S3 with no objects returns empty results."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_s3.get_paginator.return_value = mock_paginator

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            tmpdir = tempfile.mkdtemp()
            try:
                results = _ingest_s3(db, "bucket", "prefix/", tmpdir, embed=False, autocommit=False)
                assert len(results) == 0
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_s3_missing_boto3(self, db):
        """Should raise ImportError with helpful message if boto3 not installed."""
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3"):
                _ingest_s3(db, "bucket", "prefix/", "/tmp/fake", embed=False)


# ═══ MinIO Transport (mocked) ════════════════════════════════

class TestMinIOIngest:
    def test_minio_uses_custom_endpoint(self, db, sample_csv):
        """MinIO should pass endpoint_url to boto3 client."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/data.csv"}]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def fake_download(bucket, key, path):
            shutil.copy(sample_csv, path)
        mock_s3.download_file.side_effect = fake_download

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            tmpdir = tempfile.mkdtemp()
            try:
                results = _ingest_minio(db, "bucket", "prefix/", "http://localhost:9000", tmpdir,
                                        embed=False, autocommit=False)
                mock_boto3.client.assert_called_with("s3", endpoint_url="http://localhost:9000")
                assert len(results) == 1
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)


# ═══ GCS Transport (mocked) ══════════════════════════════════

class TestGCSIngest:
    def test_gcs_ingest_flow(self, db, sample_csv):
        """Test GCS ingest with mocked google-cloud-storage."""
        mock_blob = MagicMock()
        mock_blob.name = "prefix/data.csv"

        def fake_download(path):
            shutil.copy(sample_csv, path)
        mock_blob.download_to_filename.side_effect = fake_download

        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [mock_blob]

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        mock_gcs = MagicMock()
        mock_gcs.Client.return_value = mock_client

        # Patch google.cloud.storage
        mock_google = MagicMock()
        mock_google.cloud.storage = mock_gcs

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.cloud": mock_google.cloud,
            "google.cloud.storage": mock_gcs,
        }):
            tmpdir = tempfile.mkdtemp()
            try:
                results = _ingest_gcs(db, "gcs-bucket", "prefix/", tmpdir, embed=False, autocommit=False)
                assert len(results) == 1
                assert results[0]["_cloud_key"] == "gs://gcs-bucket/prefix/data.csv"
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_gcs_missing_dependency(self, db):
        with patch.dict("sys.modules", {
            "google": None,
            "google.cloud": None,
            "google.cloud.storage": None,
        }):
            with pytest.raises(ImportError, match="google-cloud-storage"):
                _ingest_gcs(db, "bucket", "prefix/", "/tmp/fake", embed=False)


# ═══ Azure Transport (mocked) ════════════════════════════════

class TestAzureIngest:
    def test_azure_ingest_flow(self, db, sample_csv):
        """Test Azure Blob ingest with mocked azure-storage-blob."""
        mock_blob_props = MagicMock()
        mock_blob_props.name = "path/data.csv"

        mock_download = MagicMock()
        mock_download.readall.return_value = open(sample_csv, "rb").read()

        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.return_value = mock_download

        mock_container = MagicMock()
        mock_container.list_blobs.return_value = [mock_blob_props]
        mock_container.get_blob_client.return_value = mock_blob_client

        mock_service = MagicMock()
        mock_service.get_container_client.return_value = mock_container

        mock_azure = MagicMock()
        mock_azure.BlobServiceClient.from_connection_string.return_value = mock_service

        with patch.dict("sys.modules", {
            "azure": MagicMock(),
            "azure.storage": MagicMock(),
            "azure.storage.blob": mock_azure,
        }):
            with patch.dict(os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "fake-conn-str"}):
                tmpdir = tempfile.mkdtemp()
                try:
                    results = _ingest_azure(db, "my-container", "path/", tmpdir, embed=False, autocommit=False)
                    assert len(results) == 1
                    assert results[0]["_cloud_key"] == "az://my-container/path/data.csv"
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

    def test_azure_missing_connection_string(self, db):
        """Should raise ValueError if AZURE_STORAGE_CONNECTION_STRING not set."""
        mock_azure = MagicMock()
        with patch.dict("sys.modules", {
            "azure": MagicMock(),
            "azure.storage": MagicMock(),
            "azure.storage.blob": mock_azure,
        }):
            with patch.dict(os.environ, {}, clear=True):
                # Remove the env var if present
                os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
                with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING"):
                    _ingest_azure(db, "container", "path/", "/tmp/fake", embed=False)

    def test_azure_missing_dependency(self, db):
        with patch.dict("sys.modules", {
            "azure": None,
            "azure.storage": None,
            "azure.storage.blob": None,
        }):
            with pytest.raises(ImportError, match="azure-storage-blob"):
                _ingest_azure(db, "container", "path/", "/tmp/fake", embed=False)


# ═══ SFTP Transport (mocked) ═════════════════════════════════

class TestSFTPIngest:
    def test_sftp_ingest_flow(self, db, sample_csv):
        """Test SFTP ingest with mocked paramiko."""
        import stat as stat_module

        mock_entry = MagicMock()
        mock_entry.filename = "data.csv"
        mock_entry.st_mode = 0o100644  # regular file

        mock_sftp = MagicMock()
        mock_sftp.listdir_attr.return_value = [mock_entry]

        def fake_get(remote, local):
            shutil.copy(sample_csv, local)
        mock_sftp.get.side_effect = fake_get

        mock_ssh = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp

        mock_paramiko = MagicMock()
        mock_paramiko.SSHClient.return_value = mock_ssh
        mock_paramiko.AutoAddPolicy.return_value = MagicMock()

        with patch.dict("sys.modules", {"paramiko": mock_paramiko}):
            tmpdir = tempfile.mkdtemp()
            try:
                results = _ingest_sftp(db, "myhost", "data", {"username": "user"}, tmpdir,
                                       embed=False, autocommit=False)
                assert len(results) == 1
                mock_ssh.connect.assert_called_once()
                mock_sftp.close.assert_called_once()
                mock_ssh.close.assert_called_once()
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_sftp_missing_dependency(self, db):
        with patch.dict("sys.modules", {"paramiko": None}):
            with pytest.raises(ImportError, match="paramiko"):
                _ingest_sftp(db, "host", "path", {}, "/tmp/fake", embed=False)


# ═══ Main ingest_cloud Entry Point ═══════════════════════════

class TestIngestCloud:
    def test_routes_to_s3(self, db, sample_csv):
        """ingest_cloud should route s3:// URIs to S3 transport."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "data/file.csv"}]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def fake_download(bucket, key, path):
            shutil.copy(sample_csv, path)
        mock_s3.download_file.side_effect = fake_download

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = ingest_cloud(db, "s3://test-bucket/data/", embed=False)
            assert result["scheme"] == "s3"
            assert result["uri"] == "s3://test-bucket/data/"
            assert result["files_ingested"] == 1
            assert result["total_rows"] == 2

    def test_aggregate_results(self, db, sample_csv):
        """Multiple files should produce correct aggregate counts."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [
                {"Key": "data/a.csv"},
                {"Key": "data/b.csv"},
            ]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def fake_download(bucket, key, path):
            shutil.copy(sample_csv, path)
        mock_s3.download_file.side_effect = fake_download

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = ingest_cloud(db, "s3://bucket/data/", embed=False)
            assert result["files_ingested"] == 2
            assert result["total_rows"] == 4  # 2 rows per CSV * 2 files

    def test_temp_cleanup(self, db, sample_csv):
        """Temp directory should be cleaned up after ingest."""
        created_tmpdirs = []

        original_mkdtemp = tempfile.mkdtemp
        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_tmpdirs.append(d)
            return d

        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "data/file.csv"}]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def fake_download(bucket, key, path):
            shutil.copy(sample_csv, path)
        mock_s3.download_file.side_effect = fake_download

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            with patch("gitdb.cloud_ingest.tempfile.mkdtemp", side_effect=tracking_mkdtemp):
                ingest_cloud(db, "s3://bucket/data/", embed=False)

        # Temp dir should be gone
        for d in created_tmpdirs:
            assert not os.path.exists(d), f"Temp directory was not cleaned up: {d}"

    def test_temp_cleanup_on_error(self, db):
        """Temp directory should be cleaned up even if ingest fails."""
        created_tmpdirs = []

        original_mkdtemp = tempfile.mkdtemp
        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_tmpdirs.append(d)
            return d

        mock_s3 = MagicMock()
        mock_s3.get_paginator.side_effect = Exception("Connection refused")

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            with patch("gitdb.cloud_ingest.tempfile.mkdtemp", side_effect=tracking_mkdtemp):
                with pytest.raises(Exception, match="Connection refused"):
                    ingest_cloud(db, "s3://bucket/data/", embed=False)

        for d in created_tmpdirs:
            assert not os.path.exists(d), f"Temp directory was not cleaned up after error: {d}"

    def test_invalid_uri(self, db):
        with pytest.raises(ValueError):
            ingest_cloud(db, "ftp://invalid/path")

    def test_empty_results(self, db):
        """Ingest with no files returns zero counts."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_s3.get_paginator.return_value = mock_paginator

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = ingest_cloud(db, "s3://empty-bucket/", embed=False)
            assert result["files_ingested"] == 0
            assert result["total_rows"] == 0
