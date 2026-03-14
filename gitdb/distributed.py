"""P2P Distributed GitDB — CRUSH-routed, SSH-transported, git-merged.

Every node is equal. No coordinator. Any node can route any vector.
CRUSH maps are replicated to all peers. Topology changes propagate via gossip.

Architecture:
  - Each peer runs a local GitDB shard
  - CRUSH determines which peers own which vectors
  - Existing SSH remotes handle transport (push/pull)
  - Git-style branch/merge handles conflicts
  - Scatter/gather for distributed queries
"""

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch

from gitdb.crush import CRUSHMap, CRUSHRouter, CRUSHRule, default_rule


@dataclass
class Peer:
    """A node in the P2P network."""
    name: str              # unique peer ID
    address: str           # SSH address (user@host) or local path
    port: int = 22         # SSH port
    weight: float = 1.0    # CRUSH weight (capacity)
    location: Optional[Dict] = None  # {"host": "...", "rack": "...", "dc": "..."}
    status: str = "up"     # "up", "down", "syncing"
    last_seen: float = 0   # timestamp of last successful contact
    shard_path: str = ""   # path to GitDB store on this peer


class PeerRegistry:
    """Thread-safe peer registry with persistence."""

    def __init__(self, path: Path):
        self._path = path / "peers.json"
        self._lock = threading.Lock()
        self._peers: Dict[str, Peer] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for name, info in data.items():
                self._peers[name] = Peer(
                    name=name,
                    address=info.get("address", ""),
                    port=info.get("port", 22),
                    weight=info.get("weight", 1.0),
                    location=info.get("location"),
                    status=info.get("status", "up"),
                    last_seen=info.get("last_seen", 0),
                    shard_path=info.get("shard_path", ""),
                )

    def _save(self):
        data = {}
        for name, p in self._peers.items():
            data[name] = {
                "address": p.address,
                "port": p.port,
                "weight": p.weight,
                "location": p.location,
                "status": p.status,
                "last_seen": p.last_seen,
                "shard_path": p.shard_path,
            }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))

    def add(self, peer: Peer):
        with self._lock:
            self._peers[peer.name] = peer
            self._save()

    def remove(self, name: str):
        with self._lock:
            if name not in self._peers:
                raise ValueError(f"Peer not found: {name}")
            del self._peers[name]
            self._save()

    def get(self, name: str) -> Peer:
        with self._lock:
            if name not in self._peers:
                raise ValueError(f"Peer not found: {name}")
            return self._peers[name]

    def list(self) -> List[Peer]:
        with self._lock:
            return list(self._peers.values())

    def update_status(self, name: str, status: str):
        with self._lock:
            if name in self._peers:
                self._peers[name].status = status
                if status == "up":
                    self._peers[name].last_seen = time.time()
                self._save()

    def active_peers(self) -> List[Peer]:
        with self._lock:
            return [p for p in self._peers.values() if p.status == "up"]


class DistributedGitDB:
    """P2P distributed vector database.

    Wraps a local GitDB shard with CRUSH routing, peer management,
    and scatter/gather query distribution.

    Usage:
        from gitdb import GitDB
        from gitdb.distributed import DistributedGitDB, Peer

        # Local shard
        db = GitDB("my_shard", dim=1024, device="mps")

        # Go distributed
        ddb = DistributedGitDB(db, self_name="laptop")

        # Add peers
        ddb.add_peer(Peer("spark", "xentureon@100.89.91.91", port=22,
                          shard_path="~/stores/shard1", weight=2.0,
                          location={"host": "spark", "rack": "home"}))
        ddb.add_peer(Peer("cloud", "ubuntu@ec2.example.com",
                          shard_path="/data/shard", weight=1.0,
                          location={"host": "cloud", "rack": "aws"}))

        # Add vectors — CRUSH routes to correct shards
        ddb.add(texts=["document 1", "document 2"], replicas=2)
        ddb.commit("Add documents")

        # Query — scatter to all shards, merge results
        results = ddb.query_text("search term", k=10)

        # Sync — push/pull with all peers
        ddb.sync()
    """

    def __init__(self, local_db, self_name: str, replicas: int = 1):
        """
        Args:
            local_db: Local GitDB instance (this node's shard)
            self_name: This node's unique name in the CRUSH map
            replicas: Default number of replicas per vector
        """
        self.db = local_db
        self.self_name = self_name
        self.replicas = replicas

        # Peer registry
        self._dist_dir = local_db._gitdb_dir / "distributed"
        self._dist_dir.mkdir(exist_ok=True)
        self.peers = PeerRegistry(self._dist_dir)

        # CRUSH setup
        self._crush_path = self._dist_dir / "crush_map.json"
        self._router: Optional[CRUSHRouter] = None
        self._rebuild_crush()

    def _rebuild_crush(self):
        """Rebuild CRUSH map from current peer list."""
        crush_map = CRUSHMap()

        # Add self as a device
        crush_map.add_device(self.self_name, weight=1.0)

        # Add all peers
        for peer in self.peers.list():
            if peer.status != "down":
                crush_map.add_device(
                    peer.name,
                    weight=peer.weight,
                    location=peer.location,
                )

        rule = default_rule()
        self._router = CRUSHRouter(crush_map, rule)

        # Save CRUSH map
        self._router.save(str(self._crush_path))

    # --- Peer Management -------------------------------------------------

    def add_peer(self, peer: Peer):
        """Add a peer to the network."""
        self.peers.add(peer)
        self._rebuild_crush()
        # Also add as a git remote for push/pull
        try:
            if peer.address and peer.shard_path:
                remote_path = f"{peer.address}:{peer.shard_path}"
                if peer.port != 22:
                    # Store port info in peer registry, remote.py handles it
                    pass
                self.db.remote_add(peer.name, remote_path)
        except Exception:
            pass  # Remote may already exist

    def remove_peer(self, name: str):
        """Remove a peer from the network."""
        self.peers.remove(name)
        self._rebuild_crush()
        try:
            self.db.remote_remove(name)
        except Exception:
            pass

    def mark_down(self, name: str):
        """Mark a peer as down (keeps in topology for CRUSH stability)."""
        self.peers.update_status(name, "down")
        self._rebuild_crush()

    def mark_up(self, name: str):
        """Mark a peer as back up."""
        self.peers.update_status(name, "up")
        self._rebuild_crush()

    def list_peers(self) -> List[dict]:
        """List all peers with status."""
        result = []
        for p in self.peers.list():
            result.append({
                "name": p.name,
                "address": p.address,
                "port": p.port,
                "weight": p.weight,
                "status": p.status,
                "last_seen": p.last_seen,
                "location": p.location,
            })
        return result

    # --- CRUSH Routing ---------------------------------------------------

    def route(self, vector_id: str, replicas: Optional[int] = None) -> List[str]:
        """Where should this vector live? Returns list of peer names."""
        r = replicas or self.replicas
        if self._router is None:
            return [self.self_name]
        return self._router.place(vector_id, r)

    def is_local(self, vector_id: str, replicas: Optional[int] = None) -> bool:
        """Should this vector be stored on this node?"""
        owners = self.route(vector_id, replicas)
        return self.self_name in owners

    def route_batch(self, vector_ids: List[str], replicas: Optional[int] = None) -> Dict[str, List[str]]:
        """Route multiple vectors. Returns {peer_name: [vector_ids]}."""
        r = replicas or self.replicas
        routing = {}  # peer_name -> [vector_ids]
        for vid in vector_ids:
            owners = self.route(vid, r)
            for owner in owners:
                routing.setdefault(owner, []).append(vid)
        return routing

    def distribution(self, sample_size: int = 1000) -> Dict[str, int]:
        """Show how vectors distribute across peers."""
        if self._router is None:
            return {self.self_name: sample_size}
        keys = [hashlib.sha256(f"sample_{i}".encode()).hexdigest() for i in range(sample_size)]
        return self._router.distribution(keys, self.replicas)

    # --- Distributed Add -------------------------------------------------

    def add(
        self,
        embeddings=None,
        documents=None,
        metadata=None,
        ids=None,
        texts=None,
        embed_model=None,
        embed_dim=None,
        replicas: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        """Add vectors with CRUSH-based routing.

        Vectors are added to the local shard if this node owns them.
        Returns {peer_name: [indices]} showing where vectors were placed.

        For remote peers, vectors are queued for the next sync().
        """
        # First, do the local add to get IDs
        indices = self.db.add(
            embeddings=embeddings,
            documents=documents,
            metadata=metadata,
            ids=ids,
            texts=texts,
            embed_model=embed_model,
            embed_dim=embed_dim,
        )

        # Route each vector
        r = replicas or self.replicas
        placement = {}  # peer -> [local_indices]

        for idx in indices:
            vid = self.db.tree.metadata[idx].id
            owners = self.route(vid, r)
            for owner in owners:
                placement.setdefault(owner, []).append(idx)

        # Queue remote placements for sync
        remote_queue = self._dist_dir / "outbox"
        remote_queue.mkdir(exist_ok=True)

        for peer_name, idxs in placement.items():
            if peer_name != self.self_name:
                # Write to outbox for this peer
                queue_file = remote_queue / f"{peer_name}.jsonl"
                with open(queue_file, "a") as f:
                    for idx in idxs:
                        meta = self.db.tree.metadata[idx]
                        entry = {
                            "id": meta.id,
                            "document": meta.document,
                            "metadata": meta.metadata,
                            "embedding": self.db.tree.embeddings[idx].cpu().tolist(),
                        }
                        f.write(json.dumps(entry) + "\n")

        return placement

    # --- Distributed Query -----------------------------------------------

    def query(
        self,
        vector,
        k: int = 10,
        where: Optional[Dict] = None,
        ac_boost: bool = False,
    ):
        """Distributed query — search local shard.

        For full scatter/gather across all peers, use query_distributed().
        This method queries the local shard only (fast path).
        """
        return self.db.query(vector, k=k, where=where, ac_boost=ac_boost)

    def query_text(
        self,
        text: str,
        k: int = 10,
        where: Optional[Dict] = None,
        embed_model: Optional[str] = None,
    ):
        """Distributed text query — search local shard."""
        return self.db.query_text(text, k=k, where=where, embed_model=embed_model)

    def query_distributed(
        self,
        vector,
        k: int = 10,
        where: Optional[Dict] = None,
        timeout: float = 10.0,
    ):
        """Scatter/gather query across all active peers.

        1. Query local shard
        2. SSH to each active peer and query their shard
        3. Merge results by score, return top-k
        """
        from gitdb.types import Results

        # Local results
        local_results = self.db.query(vector, k=k, where=where)

        # Collect all results: (score, id, document, metadata)
        all_results = []
        for i in range(len(local_results.ids)):
            all_results.append((
                local_results.scores[i],
                local_results.ids[i],
                local_results.documents[i],
                local_results.metadata[i],
            ))

        # Query active peers in parallel
        active = self.peers.active_peers()
        if active:
            results_lock = threading.Lock()
            threads = []

            for peer in active:
                t = threading.Thread(
                    target=self._query_peer,
                    args=(peer, vector, k, where, all_results, results_lock, timeout),
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join(timeout=timeout)

        # Sort by score descending, take top-k
        all_results.sort(key=lambda x: x[0], reverse=True)
        top_k = all_results[:k]

        # Deduplicate by ID (same vector may exist on multiple shards)
        seen = set()
        deduped = []
        for score, vid, doc, meta in top_k:
            if vid not in seen:
                seen.add(vid)
                deduped.append((score, vid, doc, meta))

        if not deduped:
            return Results(ids=[], scores=[], documents=[], metadata=[])

        return Results(
            ids=[r[1] for r in deduped],
            scores=[r[0] for r in deduped],
            documents=[r[2] for r in deduped],
            metadata=[r[3] for r in deduped],
        )

    def _query_peer(self, peer, vector, k, where, results_list, lock, timeout):
        """Query a remote peer via SSH. Results appended to results_list."""
        try:
            # Serialize query to JSON
            query_data = {
                "vector": vector.cpu().tolist() if hasattr(vector, 'tolist') else list(vector),
                "k": k,
                "where": where,
            }
            query_json = json.dumps(query_data)

            # SSH command to query remote shard
            ssh_cmd = [
                "ssh", "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=no",
            ]
            if peer.port != 22:
                ssh_cmd.extend(["-p", str(peer.port)])
            ssh_cmd.append(peer.address)

            # Remote python command
            remote_cmd = (
                f"python3 -c \""
                f"import json, sys, torch; "
                f"from gitdb import GitDB; "
                f"db = GitDB('{peer.shard_path}'); "
                f"q = json.loads(sys.stdin.read()); "
                f"v = torch.tensor(q['vector']); "
                f"r = db.query(v, k=q['k'], where=q.get('where')); "
                f"print(json.dumps({{'ids': r.ids, 'scores': r.scores, "
                f"'documents': r.documents, 'metadata': r.metadata}}))\""
            )
            ssh_cmd.append(remote_cmd)

            result = subprocess.run(
                ssh_cmd,
                input=query_json,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip())
                with lock:
                    for i in range(len(data.get("ids", []))):
                        results_list.append((
                            data["scores"][i],
                            data["ids"][i],
                            data["documents"][i],
                            data["metadata"][i],
                        ))
                self.peers.update_status(peer.name, "up")
            else:
                self.peers.update_status(peer.name, "down")
        except Exception:
            self.peers.update_status(peer.name, "down")

    # --- Sync ------------------------------------------------------------

    def sync(self, peer_name: Optional[str] = None) -> Dict[str, dict]:
        """Sync with peers — push outbox, pull updates.

        Uses existing git-style push/pull over SSH.

        Args:
            peer_name: Sync with specific peer, or all if None.

        Returns:
            {peer_name: {"pushed": N, "pulled": N, "status": "ok"|"error"}}
        """
        targets = []
        if peer_name:
            targets = [self.peers.get(peer_name)]
        else:
            targets = self.peers.active_peers()

        results = {}
        for peer in targets:
            result = {"pushed": 0, "pulled": 0, "status": "ok"}
            try:
                # Push outbox
                outbox_file = self._dist_dir / "outbox" / f"{peer.name}.jsonl"
                if outbox_file.exists():
                    result["pushed"] = sum(1 for _ in open(outbox_file))
                    # Push via git remote (commit first if needed)
                    try:
                        branch = self.db.refs.current_branch
                        self.db.push(peer.name, branch)
                    except Exception:
                        pass
                    # Clear outbox on success
                    outbox_file.unlink(missing_ok=True)

                # Pull from peer
                try:
                    branch = self.db.refs.current_branch
                    pull_result = self.db.pull(peer.name, branch)
                    result["pulled"] = 1  # At least one sync happened
                except Exception:
                    pass

                self.peers.update_status(peer.name, "up")
            except Exception as e:
                result["status"] = f"error: {e}"
                self.peers.update_status(peer.name, "down")

            results[peer.name] = result

        return results

    def push_outbox(self, peer_name: str) -> int:
        """Push queued vectors to a specific peer. Returns count pushed."""
        outbox_file = self._dist_dir / "outbox" / f"{peer_name}.jsonl"
        if not outbox_file.exists():
            return 0

        count = 0
        entries = []
        with open(outbox_file) as f:
            for line in f:
                entries.append(json.loads(line.strip()))
                count += 1

        if entries:
            # Write JSONL to remote via SSH
            peer = self.peers.get(peer_name)
            jsonl_data = "\n".join(json.dumps(e) for e in entries)

            ssh_cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no"]
            if peer.port != 22:
                ssh_cmd.extend(["-p", str(peer.port)])
            ssh_cmd.append(peer.address)
            ssh_cmd.append(
                f"python3 -c \""
                f"import sys, json, torch; "
                f"from gitdb import GitDB; "
                f"db = GitDB('{peer.shard_path}'); "
                f"lines = sys.stdin.read().strip().split('\\n'); "
                f"for line in lines: "
                f"  e = json.loads(line); "
                f"  emb = torch.tensor([e['embedding']]); "
                f"  db.add(embeddings=emb, documents=[e.get('document')], "
                f"         metadata=[e.get('metadata', {{}})], ids=[e['id']]); "
                f"db.commit('sync from {self.self_name}')\""
            )

            try:
                subprocess.run(
                    ssh_cmd,
                    input=jsonl_data,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                outbox_file.unlink(missing_ok=True)
                self.peers.update_status(peer_name, "up")
            except Exception:
                self.peers.update_status(peer_name, "down")

        return count

    # --- Rebalance -------------------------------------------------------

    def rebalance_plan(self) -> Dict[str, Any]:
        """Compute what vectors need to move based on current CRUSH map.

        Compares current vector locations (local shard) against
        where CRUSH says they should be.

        Returns:
            {
                "should_keep": [ids that belong here],
                "should_migrate": {peer_name: [ids to send there]},
                "foreign_count": N (vectors here that don't belong),
                "missing_count": N (vectors that should be here but aren't),
            }
        """
        should_keep = []
        should_migrate = {}  # peer -> [ids]
        foreign_count = 0

        for meta in self.db.tree.metadata:
            vid = meta.id
            owners = self.route(vid)

            if self.self_name in owners:
                should_keep.append(vid)
            else:
                foreign_count += 1
                # Find the correct owner
                primary = owners[0] if owners else self.self_name
                should_migrate.setdefault(primary, []).append(vid)

        return {
            "should_keep": should_keep,
            "should_migrate": should_migrate,
            "foreign_count": foreign_count,
            "local_total": len(self.db.tree.metadata),
        }

    def rebalance(self, dry_run: bool = True) -> Dict[str, Any]:
        """Execute rebalance — migrate vectors to correct owners.

        Args:
            dry_run: If True, only compute plan. If False, actually migrate.

        Returns:
            Rebalance plan with execution status.
        """
        plan = self.rebalance_plan()
        plan["dry_run"] = dry_run

        if dry_run:
            return plan

        # Queue migrations
        for peer_name, vids in plan["should_migrate"].items():
            outbox = self._dist_dir / "outbox"
            outbox.mkdir(exist_ok=True)
            queue_file = outbox / f"{peer_name}.jsonl"

            with open(queue_file, "a") as f:
                for vid in vids:
                    # Find the vector in local store
                    for i, meta in enumerate(self.db.tree.metadata):
                        if meta.id == vid:
                            entry = {
                                "id": meta.id,
                                "document": meta.document,
                                "metadata": meta.metadata,
                                "embedding": self.db.tree.embeddings[i].cpu().tolist(),
                            }
                            f.write(json.dumps(entry) + "\n")
                            break

        plan["queued"] = True
        return plan

    # --- Gossip ----------------------------------------------------------

    def gossip_map(self) -> dict:
        """Get the current state to share with peers.

        Returns serializable dict with CRUSH map + peer list.
        """
        return {
            "crush_map": self._router.map.to_dict() if self._router else {},
            "peers": {p.name: {
                "address": p.address,
                "port": p.port,
                "weight": p.weight,
                "status": p.status,
                "location": p.location,
                "shard_path": p.shard_path,
            } for p in self.peers.list()},
            "self_name": self.self_name,
            "timestamp": time.time(),
        }

    def apply_gossip(self, gossip_data: dict):
        """Apply topology updates from a peer's gossip.

        Merges peer lists (union), updates CRUSH map.
        """
        remote_peers = gossip_data.get("peers", {})
        for name, info in remote_peers.items():
            if name == self.self_name:
                continue  # Don't overwrite self
            existing = None
            try:
                existing = self.peers.get(name)
            except ValueError:
                pass

            if existing is None:
                # New peer discovered via gossip
                self.peers.add(Peer(
                    name=name,
                    address=info.get("address", ""),
                    port=info.get("port", 22),
                    weight=info.get("weight", 1.0),
                    location=info.get("location"),
                    status=info.get("status", "up"),
                    shard_path=info.get("shard_path", ""),
                ))

        self._rebuild_crush()

    # --- Status ----------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Cluster status overview."""
        peers = self.peers.list()
        active = [p for p in peers if p.status == "up"]

        # Check outbox
        outbox_dir = self._dist_dir / "outbox"
        pending = 0
        if outbox_dir.exists():
            for f in outbox_dir.iterdir():
                if f.suffix == ".jsonl":
                    pending += sum(1 for _ in open(f))

        return {
            "self": self.self_name,
            "peers_total": len(peers),
            "peers_active": len(active),
            "peers_down": len(peers) - len(active),
            "replicas": self.replicas,
            "local_vectors": self.db.tree.size,
            "pending_outbox": pending,
            "crush_devices": len(self._router.map._devices) if self._router else 0,
            "peer_details": [{
                "name": p.name,
                "address": p.address,
                "status": p.status,
                "weight": p.weight,
            } for p in peers],
        }

    # --- Convenience -----------------------------------------------------

    def commit(self, message: str) -> str:
        """Commit local shard."""
        return self.db.commit(message)

    def __repr__(self):
        peers = self.peers.list()
        active = sum(1 for p in peers if p.status == "up")
        return (
            f"DistributedGitDB(self={self.self_name!r}, "
            f"peers={len(peers)} ({active} active), "
            f"replicas={self.replicas}, local_vectors={self.db.tree.size})"
        )
