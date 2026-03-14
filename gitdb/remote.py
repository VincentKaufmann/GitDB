"""Remote protocol — push, pull, fetch between GitDB stores.

Supports two transport modes:
  1. Local path: gitdb remote add origin /path/to/other/store
  2. SSH:        gitdb remote add origin ssh://user@host/path/to/store

Wire format: packfiles of missing objects transferred via file copy or SSH.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from gitdb.objects import Commit, ObjectStore, RefStore


class Remote:
    """A named remote pointing to another GitDB store."""

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self._is_ssh = url.startswith("ssh://")
        if self._is_ssh:
            # ssh://user@host/path/to/store
            rest = url[len("ssh://"):]
            self.ssh_target, self.remote_path = rest.split(":", 1) if ":" in rest else (rest.split("/", 1)[0], "/" + rest.split("/", 1)[1])
        else:
            self.remote_path = url

    @property
    def gitdb_dir(self) -> str:
        return os.path.join(self.remote_path, ".gitdb")


class RemoteManager:
    """Manages remotes for a GitDB store."""

    def __init__(self, gitdb_dir: Path):
        self._config_file = gitdb_dir / "remotes.json"
        self._remotes: Dict[str, str] = {}
        if self._config_file.exists():
            self._remotes = json.loads(self._config_file.read_text())

    def add(self, name: str, url: str):
        if name in self._remotes:
            raise ValueError(f"Remote already exists: {name}")
        self._remotes[name] = url
        self._save()

    def remove(self, name: str):
        if name not in self._remotes:
            raise ValueError(f"Remote not found: {name}")
        del self._remotes[name]
        self._save()

    def get(self, name: str) -> Remote:
        if name not in self._remotes:
            raise ValueError(f"Remote not found: {name}")
        return Remote(name, self._remotes[name])

    def list(self) -> Dict[str, str]:
        return dict(self._remotes)

    def _save(self):
        self._config_file.write_text(json.dumps(self._remotes, indent=2))


def _read_remote_refs(remote: Remote) -> Dict[str, str]:
    """Read all branch refs from a remote store."""
    if remote._is_ssh:
        # Read refs via SSH
        cmd = f"cat {remote.gitdb_dir}/refs/heads/* 2>/dev/null; for f in $(ls {remote.gitdb_dir}/refs/heads/ 2>/dev/null); do echo \"$f $(cat {remote.gitdb_dir}/refs/heads/$f)\"; done"
        result = subprocess.run(
            ["ssh", remote.ssh_target, f"for f in $(find {remote.gitdb_dir}/refs/heads -type f 2>/dev/null); do name=$(echo $f | sed 's|{remote.gitdb_dir}/refs/heads/||'); echo \"$name $(cat $f)\"; done"],
            capture_output=True, text=True, timeout=30,
        )
        refs = {}
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 2:
                    refs[parts[0]] = parts[1]
        return refs
    else:
        # Local — read directly
        heads_dir = Path(remote.gitdb_dir) / "refs" / "heads"
        refs = {}
        if heads_dir.exists():
            for p in heads_dir.rglob("*"):
                if p.is_file():
                    name = str(p.relative_to(heads_dir))
                    refs[name] = p.read_text().strip()
        return refs


def _collect_missing_objects(
    local_objects: ObjectStore,
    remote_refs: Dict[str, str],
    local_refs: Dict[str, str],
    branch: str,
) -> Set[str]:
    """Find commit hashes that need to be transferred."""
    remote_head = remote_refs.get(branch)
    local_head = local_refs.get(branch)

    if local_head is None:
        return set()

    # Walk local history, collecting objects remote doesn't have
    missing = set()
    current = local_head
    while current is not None:
        if current == remote_head:
            break  # Remote has everything from here back
        missing.add(current)
        # Also add the delta object
        commit = local_objects.read_commit(current)
        missing.add(commit.delta_hash)
        current = commit.parent

    return missing


def _pack_objects(objects: ObjectStore, hashes: Set[str], dest_dir: Path):
    """Pack objects into a temporary directory for transfer."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for h in hashes:
        if objects.has(h):
            data = objects.read(h)
            obj_dir = dest_dir / h[:2]
            obj_dir.mkdir(exist_ok=True)
            (obj_dir / h[2:]).write_bytes(data)


def _unpack_objects(src_dir: Path, objects: ObjectStore):
    """Unpack transferred objects into the object store."""
    if not src_dir.exists():
        return
    for prefix_dir in src_dir.iterdir():
        if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
            for obj_file in prefix_dir.iterdir():
                h = prefix_dir.name + obj_file.name
                if not objects.has(h):
                    data = obj_file.read_bytes()
                    objects.write(data)
                    # Also write by hash directly (for commits)
                    obj_path = objects._path(h)
                    if not obj_path.exists():
                        obj_path.parent.mkdir(parents=True, exist_ok=True)
                        obj_path.write_bytes(data)


def push(
    local_objects: ObjectStore,
    local_refs: RefStore,
    remote: Remote,
    branch: str,
) -> Dict[str, any]:
    """Push a branch to a remote."""
    remote_refs = _read_remote_refs(remote)
    local_branches = local_refs.list_branches()
    local_head = local_branches.get(branch)

    if local_head is None:
        raise ValueError(f"Local branch not found: {branch}")

    # Find what's missing
    missing = _collect_missing_objects(local_objects, remote_refs, local_branches, branch)

    if not missing:
        return {"status": "up-to-date", "objects_pushed": 0}

    with tempfile.TemporaryDirectory() as tmpdir:
        pack_dir = Path(tmpdir) / "pack"
        _pack_objects(local_objects, missing, pack_dir)

        if remote._is_ssh:
            # Transfer via SSH
            remote_tmp = f"/tmp/gitdb_push_{os.getpid()}"
            subprocess.run(
                ["scp", "-r", str(pack_dir), f"{remote.ssh_target}:{remote_tmp}"],
                check=True, capture_output=True, timeout=120,
            )
            # Unpack on remote side
            unpack_script = f"""
import sys, os, shutil
src = '{remote_tmp}'
obj_dir = '{remote.gitdb_dir}/objects'
for prefix in os.listdir(src):
    prefix_path = os.path.join(src, prefix)
    if os.path.isdir(prefix_path) and len(prefix) == 2:
        dest_prefix = os.path.join(obj_dir, prefix)
        os.makedirs(dest_prefix, exist_ok=True)
        for f in os.listdir(prefix_path):
            src_file = os.path.join(prefix_path, f)
            dest_file = os.path.join(dest_prefix, f)
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)
shutil.rmtree(src, ignore_errors=True)
"""
            subprocess.run(
                ["ssh", remote.ssh_target, f"python3 -c {repr(unpack_script)}"],
                check=True, capture_output=True, timeout=60,
            )
            # Update remote ref
            subprocess.run(
                ["ssh", remote.ssh_target,
                 f"mkdir -p {remote.gitdb_dir}/refs/heads && echo '{local_head}' > {remote.gitdb_dir}/refs/heads/{branch}"],
                check=True, capture_output=True, timeout=30,
            )
        else:
            # Local remote — unpack directly
            remote_objects = ObjectStore(Path(remote.gitdb_dir))
            _unpack_objects(pack_dir, remote_objects)
            # Update ref
            remote_ref_store = RefStore(Path(remote.gitdb_dir))
            remote_ref_store.set_branch(branch, local_head)

    return {"status": "pushed", "objects_pushed": len(missing), "branch": branch}


def pull(
    local_objects: ObjectStore,
    local_refs: RefStore,
    remote: Remote,
    branch: str,
) -> Dict[str, any]:
    """Pull a branch from a remote (fetch + update local ref)."""
    result = fetch(local_objects, remote, branch)

    if result["status"] == "up-to-date":
        return result

    # Update local branch ref to match remote
    remote_refs = _read_remote_refs(remote)
    remote_head = remote_refs.get(branch)
    if remote_head:
        local_refs.set_branch(branch, remote_head)
        result["status"] = "pulled"

    return result


def fetch(
    local_objects: ObjectStore,
    remote: Remote,
    branch: str,
) -> Dict[str, any]:
    """Fetch objects from remote without updating local refs."""
    remote_refs = _read_remote_refs(remote)
    remote_head = remote_refs.get(branch)

    if remote_head is None:
        return {"status": "branch-not-found", "objects_fetched": 0}

    # Check if we already have the remote head
    if local_objects.has(remote_head):
        return {"status": "up-to-date", "objects_fetched": 0}

    if remote._is_ssh:
        # Fetch via SSH: ask remote to pack objects we don't have
        with tempfile.TemporaryDirectory() as tmpdir:
            remote_tmp = f"/tmp/gitdb_fetch_{os.getpid()}"
            # Walk remote history to find what we need
            pack_script = f"""
import json, os
obj_dir = '{remote.gitdb_dir}/objects'
refs_dir = '{remote.gitdb_dir}/refs/heads'
branch_file = os.path.join(refs_dir, '{branch}')
if not os.path.exists(branch_file):
    print(json.dumps({{"hashes": []}}))
else:
    head = open(branch_file).read().strip()
    hashes = []
    current = head
    for _ in range(10000):
        if current is None:
            break
        prefix = current[:2]
        rest = current[2:]
        obj_path = os.path.join(obj_dir, prefix, rest)
        if not os.path.exists(obj_path):
            break
        hashes.append(current)
        data = open(obj_path, 'rb').read()
        try:
            d = json.loads(data.decode('utf-8'))
            delta_hash = d.get('delta_hash', '')
            if delta_hash:
                hashes.append(delta_hash)
            current = d.get('parent')
        except:
            current = None
    print(json.dumps({{"hashes": hashes}}))
"""
            result = subprocess.run(
                ["ssh", remote.ssh_target, f"python3 -c {repr(pack_script)}"],
                capture_output=True, text=True, timeout=60,
            )
            remote_hashes = json.loads(result.stdout)["hashes"]

            # Filter to what we don't have
            needed = [h for h in remote_hashes if not local_objects.has(h)]
            if not needed:
                return {"status": "up-to-date", "objects_fetched": 0}

            # Fetch needed objects
            fetch_script = f"""
import os, json, sys, shutil, tempfile
obj_dir = '{remote.gitdb_dir}/objects'
dest = '{remote_tmp}'
os.makedirs(dest, exist_ok=True)
needed = {json.dumps(needed)}
for h in needed:
    src = os.path.join(obj_dir, h[:2], h[2:])
    if os.path.exists(src):
        d = os.path.join(dest, h[:2])
        os.makedirs(d, exist_ok=True)
        shutil.copy2(src, os.path.join(d, h[2:]))
print(json.dumps({{"packed": len(needed)}}))
"""
            subprocess.run(
                ["ssh", remote.ssh_target, f"python3 -c {repr(fetch_script)}"],
                check=True, capture_output=True, timeout=60,
            )

            # SCP the pack back
            local_pack = Path(tmpdir) / "pack"
            subprocess.run(
                ["scp", "-r", f"{remote.ssh_target}:{remote_tmp}", str(local_pack)],
                check=True, capture_output=True, timeout=120,
            )

            # Unpack locally
            _unpack_objects(local_pack, local_objects)

            # Cleanup remote tmp
            subprocess.run(
                ["ssh", remote.ssh_target, f"rm -rf {remote_tmp}"],
                capture_output=True, timeout=30,
            )

            return {"status": "fetched", "objects_fetched": len(needed)}
    else:
        # Local remote — copy objects directly
        remote_objects_dir = Path(remote.gitdb_dir) / "objects"
        fetched = 0

        # Walk remote branch history
        current = remote_head
        while current is not None:
            if local_objects.has(current):
                break

            # Copy commit object
            remote_path = remote_objects_dir / current[:2] / current[2:]
            if remote_path.exists():
                data = remote_path.read_bytes()
                local_objects.write(data)
                # Also store by commit hash
                local_path = local_objects._path(current)
                if not local_path.exists():
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(data)
                fetched += 1

                try:
                    d = json.loads(data.decode("utf-8"))
                    delta_hash = d.get("delta_hash", "")
                    if delta_hash and not local_objects.has(delta_hash):
                        delta_path = remote_objects_dir / delta_hash[:2] / delta_hash[2:]
                        if delta_path.exists():
                            delta_data = delta_path.read_bytes()
                            local_objects.write(delta_data)
                            fetched += 1
                    current = d.get("parent")
                except Exception:
                    current = None
            else:
                break

        return {"status": "fetched" if fetched > 0 else "up-to-date", "objects_fetched": fetched}
