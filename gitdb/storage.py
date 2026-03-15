"""Pluggable storage backends for GitDB.

LocalStorage works with zero dependencies. Cloud backends (S3, GCS, Azure)
and SFTPStorage use lazy imports so the SDK is only required when actually used.
"""

import os
import shutil
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse


class StorageBackend(ABC):
    """Abstract storage backend for GitDB objects."""

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Read raw bytes for *key*. Raises KeyError if missing."""
        ...

    @abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """Write *data* under *key*, creating intermediates as needed."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if *key* exists in the store."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete *key*. Raises KeyError if missing."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """Return all keys that start with *prefix*."""
        ...


# ---------------------------------------------------------------------------
# Local filesystem
# ---------------------------------------------------------------------------

class LocalStorage(StorageBackend):
    """Default: local filesystem storage. Maps keys to files under a root dir."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        # Normalise separators so callers can use "/" on any OS
        return self.root / key.replace("/", os.sep)

    def read(self, key: str) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise KeyError(key)
        return p.read_bytes()

    def write(self, key: str, data: bytes) -> None:
        p = self._path(key)
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if not p.is_file():
            raise KeyError(key)
        p.unlink()

    def list_keys(self, prefix: str = "") -> List[str]:
        results: List[str] = []
        base = self.root
        for dirpath, _, filenames in os.walk(base):
            for fname in filenames:
                full = Path(dirpath) / fname
                rel = full.relative_to(base).as_posix()
                if rel.startswith(prefix):
                    results.append(rel)
        results.sort()
        return results


# ---------------------------------------------------------------------------
# Amazon S3 / MinIO
# ---------------------------------------------------------------------------

class S3Storage(StorageBackend):
    """Amazon S3 / MinIO storage.

    Usage::

        storage = S3Storage("my-bucket", prefix="gitdb/mystore/",
                            endpoint_url="http://localhost:9000")  # MinIO
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self._endpoint_url = endpoint_url
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            import boto3  # lazy
            kwargs: dict = {}
            if self._endpoint_url:
                kwargs["endpoint_url"] = self._endpoint_url
            if self._aws_access_key_id:
                kwargs["aws_access_key_id"] = self._aws_access_key_id
            if self._aws_secret_access_key:
                kwargs["aws_secret_access_key"] = self._aws_secret_access_key
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _full_key(self, key: str) -> str:
        return self.prefix + key

    def read(self, key: str) -> bytes:
        client = self._get_client()
        try:
            resp = client.get_object(Bucket=self.bucket, Key=self._full_key(key))
            return resp["Body"].read()
        except client.exceptions.NoSuchKey:
            raise KeyError(key)

    def write(self, key: str, data: bytes) -> None:
        client = self._get_client()
        with self._lock:
            client.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def exists(self, key: str) -> bool:
        client = self._get_client()
        try:
            client.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        if not self.exists(key):
            raise KeyError(key)
        client = self._get_client()
        client.delete_object(Bucket=self.bucket, Key=self._full_key(key))

    def list_keys(self, prefix: str = "") -> List[str]:
        client = self._get_client()
        full_prefix = self.prefix + prefix
        keys: List[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.startswith(self.prefix):
                    keys.append(k[len(self.prefix):])
        keys.sort()
        return keys


# ---------------------------------------------------------------------------
# Google Cloud Storage
# ---------------------------------------------------------------------------

class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket_name = bucket
        self.prefix = prefix
        self._bucket = None
        self._lock = threading.Lock()

    def _get_bucket(self):
        if self._bucket is None:
            from google.cloud import storage as gcs  # lazy
            client = gcs.Client()
            self._bucket = client.bucket(self.bucket_name)
        return self._bucket

    def _full_key(self, key: str) -> str:
        return self.prefix + key

    def read(self, key: str) -> bytes:
        blob = self._get_bucket().blob(self._full_key(key))
        if not blob.exists():
            raise KeyError(key)
        return blob.download_as_bytes()

    def write(self, key: str, data: bytes) -> None:
        blob = self._get_bucket().blob(self._full_key(key))
        with self._lock:
            blob.upload_from_string(data)

    def exists(self, key: str) -> bool:
        return self._get_bucket().blob(self._full_key(key)).exists()

    def delete(self, key: str) -> None:
        blob = self._get_bucket().blob(self._full_key(key))
        if not blob.exists():
            raise KeyError(key)
        blob.delete()

    def list_keys(self, prefix: str = "") -> List[str]:
        full_prefix = self.prefix + prefix
        keys: List[str] = []
        for blob in self._get_bucket().list_blobs(prefix=full_prefix):
            name = blob.name
            if name.startswith(self.prefix):
                keys.append(name[len(self.prefix):])
        keys.sort()
        return keys


# ---------------------------------------------------------------------------
# Azure Blob Storage
# ---------------------------------------------------------------------------

class AzureStorage(StorageBackend):
    """Azure Blob Storage backend."""

    def __init__(
        self,
        container: str,
        connection_string: str,
        prefix: str = "",
    ) -> None:
        self.container_name = container
        self._connection_string = connection_string
        self.prefix = prefix
        self._container_client = None
        self._lock = threading.Lock()

    def _get_container(self):
        if self._container_client is None:
            from azure.storage.blob import BlobServiceClient  # lazy
            service = BlobServiceClient.from_connection_string(self._connection_string)
            self._container_client = service.get_container_client(self.container_name)
        return self._container_client

    def _full_key(self, key: str) -> str:
        return self.prefix + key

    def read(self, key: str) -> bytes:
        container = self._get_container()
        try:
            return container.download_blob(self._full_key(key)).readall()
        except Exception:
            raise KeyError(key)

    def write(self, key: str, data: bytes) -> None:
        container = self._get_container()
        with self._lock:
            container.upload_blob(
                self._full_key(key), data, overwrite=True,
            )

    def exists(self, key: str) -> bool:
        container = self._get_container()
        try:
            container.get_blob_properties(self._full_key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        if not self.exists(key):
            raise KeyError(key)
        self._get_container().delete_blob(self._full_key(key))

    def list_keys(self, prefix: str = "") -> List[str]:
        container = self._get_container()
        full_prefix = self.prefix + prefix
        keys: List[str] = []
        for blob in container.list_blobs(name_starts_with=full_prefix):
            name = blob.name
            if name.startswith(self.prefix):
                keys.append(name[len(self.prefix):])
        keys.sort()
        return keys


# ---------------------------------------------------------------------------
# SFTP
# ---------------------------------------------------------------------------

class SFTPStorage(StorageBackend):
    """SFTP remote storage backend."""

    def __init__(
        self,
        host: str,
        path: str,
        username: str,
        password: Optional[str] = None,
        key_file: Optional[str] = None,
        port: int = 22,
    ) -> None:
        self.host = host
        self.remote_root = path.rstrip("/")
        self.username = username
        self._password = password
        self._key_file = key_file
        self.port = port
        self._sftp = None
        self._transport = None
        self._lock = threading.Lock()

    def _get_sftp(self):
        if self._sftp is None:
            import paramiko  # lazy
            self._transport = paramiko.Transport((self.host, self.port))
            if self._key_file:
                pkey = paramiko.RSAKey.from_private_key_file(self._key_file)
                self._transport.connect(username=self.username, pkey=pkey)
            else:
                self._transport.connect(
                    username=self.username, password=self._password,
                )
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        return self._sftp

    def _remote_path(self, key: str) -> str:
        return f"{self.remote_root}/{key}"

    def read(self, key: str) -> bytes:
        sftp = self._get_sftp()
        try:
            with sftp.open(self._remote_path(key), "rb") as f:
                return f.read()
        except FileNotFoundError:
            raise KeyError(key)
        except IOError:
            raise KeyError(key)

    def write(self, key: str, data: bytes) -> None:
        sftp = self._get_sftp()
        rpath = self._remote_path(key)
        # Ensure parent dirs exist
        parts = key.split("/")
        if len(parts) > 1:
            current = self.remote_root
            for part in parts[:-1]:
                current = f"{current}/{part}"
                try:
                    sftp.stat(current)
                except (FileNotFoundError, IOError):
                    sftp.mkdir(current)
        with self._lock:
            with sftp.open(rpath, "wb") as f:
                f.write(data)

    def exists(self, key: str) -> bool:
        sftp = self._get_sftp()
        try:
            sftp.stat(self._remote_path(key))
            return True
        except (FileNotFoundError, IOError):
            return False

    def delete(self, key: str) -> None:
        if not self.exists(key):
            raise KeyError(key)
        self._get_sftp().remove(self._remote_path(key))

    def list_keys(self, prefix: str = "") -> List[str]:
        sftp = self._get_sftp()
        keys: List[str] = []
        self._walk_sftp(sftp, self.remote_root, "", prefix, keys)
        keys.sort()
        return keys

    def _walk_sftp(self, sftp, base: str, rel: str, prefix: str, out: List[str]):
        current = f"{base}/{rel}" if rel else base
        try:
            entries = sftp.listdir_attr(current)
        except IOError:
            return
        import stat as stat_mod
        for entry in entries:
            child_rel = f"{rel}/{entry.filename}" if rel else entry.filename
            if stat_mod.S_ISDIR(entry.st_mode):
                self._walk_sftp(sftp, base, child_rel, prefix, out)
            else:
                if child_rel.startswith(prefix):
                    out.append(child_rel)

    def close(self):
        """Close the SFTP connection."""
        if self._sftp:
            self._sftp.close()
        if self._transport:
            self._transport.close()
        self._sftp = None
        self._transport = None


# ---------------------------------------------------------------------------
# URI parser
# ---------------------------------------------------------------------------

def parse_storage_uri(uri: str) -> StorageBackend:
    """Parse a storage URI and return the appropriate backend.

    Supported URIs::

        /path/to/dir           -> LocalStorage
        s3://bucket/prefix     -> S3Storage
        minio://host/bucket    -> S3Storage with endpoint
        gs://bucket/prefix     -> GCSStorage
        az://container/prefix  -> AzureStorage  (needs AZURE_STORAGE_CONNECTION_STRING)
        sftp://user@host/path  -> SFTPStorage
    """
    # Absolute local path (no scheme)
    if uri.startswith("/"):
        return LocalStorage(Path(uri))

    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme == "s3":
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return S3Storage(bucket, prefix=prefix)

    if scheme == "minio":
        # minio://host:port/bucket/prefix
        host = parsed.netloc
        parts = parsed.path.strip("/").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] + "/" if len(parts) > 1 else ""
        endpoint = f"http://{host}"
        return S3Storage(bucket, prefix=prefix, endpoint_url=endpoint)

    if scheme == "gs":
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return GCSStorage(bucket, prefix=prefix)

    if scheme == "az":
        container = parsed.netloc
        prefix = parsed.path.lstrip("/")
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
        return AzureStorage(container, connection_string=conn_str, prefix=prefix)

    if scheme == "sftp":
        username = parsed.username or ""
        host = parsed.hostname or ""
        path = parsed.path or "/"
        port = parsed.port or 22
        return SFTPStorage(host=host, path=path, username=username, port=port)

    raise ValueError(f"Unsupported storage URI scheme: {scheme!r}")


# ---------------------------------------------------------------------------
# Migration utility
# ---------------------------------------------------------------------------

def copy_between(
    src: StorageBackend,
    dst: StorageBackend,
    prefix: str = "",
) -> int:
    """Copy all keys matching *prefix* from *src* to *dst*.

    Returns the number of keys copied.
    """
    keys = src.list_keys(prefix)
    for key in keys:
        data = src.read(key)
        dst.write(key, data)
    return len(keys)
