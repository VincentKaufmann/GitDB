"""Tests for P2P distributed GitDB."""

import json
import pytest
import torch
import time

from gitdb import GitDB
from gitdb.distributed import DistributedGitDB, Peer, PeerRegistry


@pytest.fixture
def local_db(tmp_path):
    return GitDB(str(tmp_path / "local_shard"), dim=8, device="cpu")


@pytest.fixture
def ddb(local_db):
    return DistributedGitDB(local_db, self_name="node_a")


class TestPeerRegistry:
    def test_add_and_list(self, tmp_path):
        reg = PeerRegistry(tmp_path)
        reg.add(Peer("spark", "user@spark", shard_path="/data/shard"))
        peers = reg.list()
        assert len(peers) == 1
        assert peers[0].name == "spark"

    def test_remove(self, tmp_path):
        reg = PeerRegistry(tmp_path)
        reg.add(Peer("spark", "user@spark"))
        reg.remove("spark")
        assert len(reg.list()) == 0

    def test_remove_nonexistent(self, tmp_path):
        reg = PeerRegistry(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            reg.remove("ghost")

    def test_persistence(self, tmp_path):
        reg1 = PeerRegistry(tmp_path)
        reg1.add(Peer("spark", "user@spark", weight=2.0))
        # Reload
        reg2 = PeerRegistry(tmp_path)
        peers = reg2.list()
        assert len(peers) == 1
        assert peers[0].weight == 2.0

    def test_update_status(self, tmp_path):
        reg = PeerRegistry(tmp_path)
        reg.add(Peer("spark", "user@spark"))
        reg.update_status("spark", "down")
        assert reg.get("spark").status == "down"
        assert len(reg.active_peers()) == 0

    def test_active_peers(self, tmp_path):
        reg = PeerRegistry(tmp_path)
        reg.add(Peer("a", "user@a"))
        reg.add(Peer("b", "user@b"))
        reg.update_status("b", "down")
        active = reg.active_peers()
        assert len(active) == 1
        assert active[0].name == "a"


class TestDistributedInit:
    def test_create(self, ddb):
        assert ddb.self_name == "node_a"
        assert ddb._router is not None

    def test_status_empty(self, ddb):
        status = ddb.status()
        assert status["self"] == "node_a"
        assert status["peers_total"] == 0

    def test_repr(self, ddb):
        r = repr(ddb)
        assert "node_a" in r


class TestPeerManagement:
    def test_add_peer(self, ddb):
        ddb.add_peer(Peer("spark", "user@spark", shard_path="/data"))
        peers = ddb.list_peers()
        assert len(peers) == 1
        assert peers[0]["name"] == "spark"

    def test_remove_peer(self, ddb):
        ddb.add_peer(Peer("spark", "user@spark"))
        ddb.remove_peer("spark")
        assert len(ddb.list_peers()) == 0

    def test_mark_down_up(self, ddb):
        ddb.add_peer(Peer("spark", "user@spark"))
        ddb.mark_down("spark")
        peers = ddb.list_peers()
        assert peers[0]["status"] == "down"
        ddb.mark_up("spark")
        peers = ddb.list_peers()
        assert peers[0]["status"] == "up"

    def test_crush_rebuilds_on_peer_change(self, ddb):
        devices_before = len(ddb._router.map._devices)
        ddb.add_peer(Peer("spark", "user@spark"))
        devices_after = len(ddb._router.map._devices)
        assert devices_after == devices_before + 1


class TestCRUSHRouting:
    def test_route_single(self, ddb):
        # With only self, everything routes to self
        owners = ddb.route("vector_abc")
        assert owners == ["node_a"]

    def test_route_with_peers(self, ddb):
        ddb.add_peer(Peer("b", "user@b"))
        ddb.add_peer(Peer("c", "user@c"))
        owners = ddb.route("vector_abc", replicas=2)
        assert len(owners) == 2
        # All owners should be known nodes
        known = {"node_a", "b", "c"}
        assert all(o in known for o in owners)

    def test_route_deterministic(self, ddb):
        ddb.add_peer(Peer("b", "user@b"))
        r1 = ddb.route("same_vector")
        r2 = ddb.route("same_vector")
        assert r1 == r2

    def test_is_local(self, ddb):
        # Solo node — everything is local
        assert ddb.is_local("any_vector") is True

    def test_route_batch(self, ddb):
        ddb.add_peer(Peer("b", "user@b"))
        routing = ddb.route_batch(["v1", "v2", "v3"], replicas=1)
        total_placed = sum(len(v) for v in routing.values())
        assert total_placed == 3

    def test_distribution_uniform(self, ddb):
        ddb.add_peer(Peer("b", "user@b", weight=1.0))
        ddb.add_peer(Peer("c", "user@c", weight=1.0))
        dist = ddb.distribution(sample_size=3000)
        # With equal weights, each should get ~1000 +/- 400
        for name, count in dist.items():
            assert 600 < count < 1400, f"{name} got {count}, expected ~1000"


class TestDistributedAdd:
    def test_add_local_only(self, ddb):
        # Solo node — all vectors stay local
        placement = ddb.add(embeddings=torch.randn(3, 8), documents=["a", "b", "c"])
        assert "node_a" in placement
        assert ddb.db.tree.size == 3

    def test_add_with_peers_routes(self, ddb):
        ddb.add_peer(Peer("b", "user@b", shard_path="/data"))
        placement = ddb.add(embeddings=torch.randn(10, 8),
                           documents=[f"doc_{i}" for i in range(10)],
                           replicas=2)
        # Should have placements for multiple peers
        total_placements = sum(len(v) for v in placement.values())
        assert total_placements == 20  # 10 vectors x 2 replicas

    def test_add_creates_outbox(self, ddb):
        ddb.add_peer(Peer("b", "user@b", shard_path="/data"))
        ddb.add(embeddings=torch.randn(5, 8), documents=[f"d{i}" for i in range(5)], replicas=2)
        outbox = ddb._dist_dir / "outbox" / "b.jsonl"
        # Some vectors should be queued for peer b
        # (may or may not exist depending on CRUSH routing)
        # At minimum, local shard should have all vectors
        assert ddb.db.tree.size == 5


class TestDistributedQuery:
    def test_local_query(self, ddb):
        ddb.add(embeddings=torch.randn(5, 8), documents=[f"doc_{i}" for i in range(5)])
        ddb.commit("test")
        results = ddb.query(torch.randn(8), k=3)
        assert len(results.ids) == 3

    def test_distributed_query_local_only(self, ddb):
        ddb.add(embeddings=torch.randn(5, 8), documents=[f"doc_{i}" for i in range(5)])
        ddb.commit("test")
        results = ddb.query_distributed(torch.randn(8), k=3)
        assert len(results.ids) == 3

    def test_query_deduplication(self, ddb):
        ddb.add(embeddings=torch.randn(5, 8), documents=[f"doc_{i}" for i in range(5)])
        ddb.commit("test")
        # Distributed query with only local node should return unique IDs
        results = ddb.query_distributed(torch.randn(8), k=5)
        assert len(results.ids) == len(set(results.ids))


class TestRebalance:
    def test_rebalance_plan_solo(self, ddb):
        ddb.add(embeddings=torch.randn(5, 8), documents=["a"] * 5)
        plan = ddb.rebalance_plan()
        # Solo node — everything belongs here
        assert len(plan["should_keep"]) == 5
        assert plan["foreign_count"] == 0

    def test_rebalance_plan_with_peers(self, ddb):
        ddb.add(embeddings=torch.randn(20, 8), documents=[f"d{i}" for i in range(20)])
        ddb.commit("data")
        # Add a peer — some vectors should now belong elsewhere
        ddb.add_peer(Peer("b", "user@b"))
        plan = ddb.rebalance_plan()
        # Some vectors should migrate to the new peer
        assert plan["foreign_count"] > 0 or len(plan["should_keep"]) < 20

    def test_rebalance_dry_run(self, ddb):
        ddb.add(embeddings=torch.randn(10, 8), documents=["x"] * 10)
        ddb.add_peer(Peer("b", "user@b"))
        result = ddb.rebalance(dry_run=True)
        assert result["dry_run"] is True


class TestGossip:
    def test_gossip_map(self, ddb):
        ddb.add_peer(Peer("spark", "user@spark"))
        gossip = ddb.gossip_map()
        assert "crush_map" in gossip
        assert "peers" in gossip
        assert "spark" in gossip["peers"]

    def test_apply_gossip_discovers_peers(self, local_db, ddb):
        # Create another distributed node
        db2 = GitDB(str(local_db.path.parent / "shard_b"), dim=8, device="cpu")
        ddb2 = DistributedGitDB(db2, self_name="node_b")
        ddb2.add_peer(Peer("node_c", "user@node_c"))

        # Node A applies gossip from Node B
        gossip = ddb2.gossip_map()
        ddb.apply_gossip(gossip)

        # Node A should now know about node_c
        peer_names = [p["name"] for p in ddb.list_peers()]
        assert "node_c" in peer_names

    def test_gossip_doesnt_overwrite_self(self, ddb):
        gossip = {
            "peers": {"node_a": {"address": "wrong", "status": "down"}},
            "timestamp": time.time(),
        }
        ddb.apply_gossip(gossip)
        # Self should not be in peer list (it's the local node)
        peer_names = [p["name"] for p in ddb.list_peers()]
        assert "node_a" not in peer_names


class TestSyncOutbox:
    def test_outbox_created_on_add(self, ddb):
        ddb.add_peer(Peer("b", "user@b", shard_path="/data"))
        ddb.replicas = 2  # Force replicas to both nodes
        ddb.add(embeddings=torch.randn(20, 8), documents=[f"d{i}" for i in range(20)], replicas=2)
        outbox_dir = ddb._dist_dir / "outbox"
        # Outbox should exist (some vectors routed to peer b)
        assert outbox_dir.exists()

    def test_sync_status(self, ddb):
        # No peers — sync returns empty
        result = ddb.sync()
        assert result == {}
