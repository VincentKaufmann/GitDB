"""Cloud storage transport layer for GitDB ingest.

Ingest files directly from cloud storage into GitDB:
    gitdb ingest s3://my-bucket/data/
    gitdb ingest gs://my-bucket/prefix/
    gitdb ingest az://my-container/path/
    gitdb ingest minio://localhost:9000/bucket/prefix/
    gitdb ingest sftp://host/path/to/files/
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from gitdb.ingest import ingest_file


# ═══════════════════════════════════════════════════════════════
#  URI Parsing
# ═══════════════════════════════════════════════════════════════

CLOUD_SCHEMES = {"s3", "gs", "az", "minio", "sftp"}


def is_cloud_uri(path: str) -> bool:
    """Returns True if the path looks like a cloud storage URI."""
    if not isinstance(path, str):
        return False
    scheme = path.split("://", 1)[0].lower() if "://" in path else ""
    return scheme in CLOUD_SCHEMES


def parse_cloud_uri(uri: str) -> Tuple[str, str, str, Dict[str, str]]:
    """Parse a cloud URI into components.

    Returns:
        (scheme, bucket_or_host, prefix_or_path, extra_params)

    Examples:
        s3://bucket/prefix/       -> ("s3", "bucket", "prefix/", {})
        gs://bucket/prefix/       -> ("gs", "bucket", "prefix/", {})
        az://container/path/      -> ("az", "container", "path/", {})
        minio://host:9000/bucket/prefix/ -> ("minio", "bucket", "prefix/", {"endpoint": "http://host:9000"})
        sftp://user@host:22/path/ -> ("sftp", "host", "path/", {"username": "user", "port": "22"})
    """
    if "://" not in uri:
        raise ValueError(f"Invalid cloud URI (no scheme): {uri}")

    scheme = uri.split("://", 1)[0].lower()
    if scheme not in CLOUD_SCHEMES:
        raise ValueError(f"Unsupported cloud scheme '{scheme}'. Supported: {', '.join(sorted(CLOUD_SCHEMES))}")

    rest = uri.split("://", 1)[1]
    extra = {}

    if scheme == "minio":
        # minio://endpoint:port/bucket/prefix/
        parts = rest.split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"MinIO URI must include endpoint and bucket: {uri}")
        endpoint = parts[0]
        bucket = parts[1]
        prefix = parts[2] if len(parts) > 2 else ""
        # Build endpoint URL
        proto = "https" if "443" in endpoint else "http"
        extra["endpoint"] = f"{proto}://{endpoint}"
        return scheme, bucket, prefix, extra

    if scheme == "sftp":
        parsed = urlparse(uri)
        host = parsed.hostname or ""
        path = parsed.path.lstrip("/")
        if parsed.username:
            extra["username"] = parsed.username
        if parsed.port:
            extra["port"] = str(parsed.port)
        if parsed.password:
            extra["password"] = parsed.password
        return scheme, host, path, extra

    # S3, GCS, Azure: scheme://bucket/prefix
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return scheme, bucket, prefix, extra


# ═══════════════════════════════════════════════════════════════
#  S3 Transport
# ═══════════════════════════════════════════════════════════════

def _ingest_s3(db, bucket: str, prefix: str, tmpdir: str, **kwargs) -> List[Dict[str, Any]]:
    """Download and ingest files from S3."""
    try:
        import boto3
    except ImportError:
        raise ImportError("Install boto3 for S3 support: pip install boto3")

    s3 = boto3.client("s3")
    results = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue  # skip "directory" markers

            local_path = os.path.join(tmpdir, os.path.basename(key))
            s3.download_file(bucket, key, local_path)

            result = ingest_file(db, local_path, **kwargs)
            result["_cloud_key"] = f"s3://{bucket}/{key}"
            results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════
#  MinIO Transport (S3-compatible with custom endpoint)
# ═══════════════════════════════════════════════════════════════

def _ingest_minio(db, bucket: str, prefix: str, endpoint: str, tmpdir: str, **kwargs) -> List[Dict[str, Any]]:
    """Download and ingest files from MinIO."""
    try:
        import boto3
    except ImportError:
        raise ImportError("Install boto3 for MinIO support: pip install boto3")

    s3 = boto3.client("s3", endpoint_url=endpoint)
    results = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            local_path = os.path.join(tmpdir, os.path.basename(key))
            s3.download_file(bucket, key, local_path)

            result = ingest_file(db, local_path, **kwargs)
            result["_cloud_key"] = f"minio://{bucket}/{key}"
            results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════
#  GCS Transport
# ═══════════════════════════════════════════════════════════════

def _ingest_gcs(db, bucket: str, prefix: str, tmpdir: str, **kwargs) -> List[Dict[str, Any]]:
    """Download and ingest files from Google Cloud Storage."""
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        raise ImportError("Install google-cloud-storage for GCS support: pip install google-cloud-storage")

    client = gcs_storage.Client()
    bucket_obj = client.bucket(bucket)
    results = []

    for blob in bucket_obj.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue

        local_path = os.path.join(tmpdir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)

        result = ingest_file(db, local_path, **kwargs)
        result["_cloud_key"] = f"gs://{bucket}/{blob.name}"
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════
#  Azure Blob Transport
# ═══════════════════════════════════════════════════════════════

def _ingest_azure(db, container: str, prefix: str, tmpdir: str, **kwargs) -> List[Dict[str, Any]]:
    """Download and ingest files from Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ImportError("Install azure-storage-blob for Azure support: pip install azure-storage-blob")

    # Uses AZURE_STORAGE_CONNECTION_STRING env var by default
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("Set AZURE_STORAGE_CONNECTION_STRING environment variable for Azure Blob access")

    service = BlobServiceClient.from_connection_string(conn_str)
    container_client = service.get_container_client(container)
    results = []

    for blob in container_client.list_blobs(name_starts_with=prefix or None):
        if blob.name.endswith("/"):
            continue

        local_path = os.path.join(tmpdir, os.path.basename(blob.name))
        blob_client = container_client.get_blob_client(blob.name)
        with open(local_path, "wb") as f:
            data = blob_client.download_blob().readall()
            f.write(data)

        result = ingest_file(db, local_path, **kwargs)
        result["_cloud_key"] = f"az://{container}/{blob.name}"
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════
#  SFTP Transport
# ═══════════════════════════════════════════════════════════════

def _ingest_sftp(db, host: str, remote_path: str, extra: Dict[str, str], tmpdir: str, **kwargs) -> List[Dict[str, Any]]:
    """Download and ingest files from SFTP."""
    try:
        import paramiko
    except ImportError:
        raise ImportError("Install paramiko for SFTP support: pip install paramiko")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {"hostname": host}
    if "username" in extra:
        connect_kwargs["username"] = extra["username"]
    if "password" in extra:
        connect_kwargs["password"] = extra["password"]
    if "port" in extra:
        connect_kwargs["port"] = int(extra["port"])

    ssh.connect(**connect_kwargs)
    sftp = ssh.open_sftp()
    results = []

    try:
        files = _sftp_list_recursive(sftp, remote_path)
        for remote_file in files:
            local_path = os.path.join(tmpdir, os.path.basename(remote_file))
            sftp.get(remote_file, local_path)

            result = ingest_file(db, local_path, **kwargs)
            result["_cloud_key"] = f"sftp://{host}/{remote_file}"
            results.append(result)
    finally:
        sftp.close()
        ssh.close()

    return results


def _sftp_list_recursive(sftp, path: str) -> List[str]:
    """Recursively list files on SFTP server."""
    import stat as stat_module
    files = []
    try:
        entries = sftp.listdir_attr(path)
    except IOError:
        return []

    for entry in entries:
        full_path = f"{path.rstrip('/')}/{entry.filename}"
        if stat_module.S_ISDIR(entry.st_mode):
            files.extend(_sftp_list_recursive(sftp, full_path))
        else:
            files.append(full_path)
    return files


# ═══════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════

def ingest_cloud(
    db,
    uri: str,
    embed: bool = True,
    embed_model: Optional[str] = None,
    autocommit: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Ingest files from a cloud storage URI into GitDB.

    Supports s3://, gs://, az://, minio://, sftp:// URIs.
    Downloads each file to a temp directory, runs it through ingest_file(),
    then cleans up.

    Args:
        db: GitDB instance.
        uri: Cloud storage URI.
        embed: If True, auto-embed text content.
        embed_model: Embedding model name.
        autocommit: Commit after ingest.
        **kwargs: Passed through to ingest_file().

    Returns:
        {"uri": str, "scheme": str, "files_ingested": int,
         "files_failed": int, "total_rows": int, "results": [...]}
    """
    scheme, bucket, prefix, extra = parse_cloud_uri(uri)

    ingest_kwargs = dict(embed=embed, embed_model=embed_model, autocommit=False, **kwargs)

    tmpdir = tempfile.mkdtemp(prefix="gitdb_cloud_")
    try:
        if scheme == "s3":
            file_results = _ingest_s3(db, bucket, prefix, tmpdir, **ingest_kwargs)
        elif scheme == "minio":
            endpoint = extra.get("endpoint", "http://localhost:9000")
            file_results = _ingest_minio(db, bucket, prefix, endpoint, tmpdir, **ingest_kwargs)
        elif scheme == "gs":
            file_results = _ingest_gcs(db, bucket, prefix, tmpdir, **ingest_kwargs)
        elif scheme == "az":
            file_results = _ingest_azure(db, bucket, prefix, tmpdir, **ingest_kwargs)
        elif scheme == "sftp":
            file_results = _ingest_sftp(db, bucket, prefix, extra, tmpdir, **ingest_kwargs)
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Aggregate results
    total_rows = 0
    for r in file_results:
        total_rows += r.get("rows", r.get("chunks", r.get("documents", 0)))

    if autocommit and file_results:
        db.commit(f"Ingest cloud: {uri} ({len(file_results)} files, {total_rows} rows)")

    return {
        "uri": uri,
        "scheme": scheme,
        "files_ingested": len(file_results),
        "files_failed": 0,
        "total_rows": total_rows,
        "results": file_results,
    }
