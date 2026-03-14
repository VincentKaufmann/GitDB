"""Tests for CRUSH — Controlled Replication Under Scalable Hashing."""

import json
import os
import tempfile
import uuid

import pytest

from gitdb.crush import (
    CRUSHMap,
    CRUSHNode,
    CRUSHRouter,
    CRUSHRule,
    _crush_hash,
    _straw2_select,
    crush_select,
    default_rule,
    host_spread_rule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_simple_map(n_devices=4):
    """Create a flat map with N devices under root."""
    m = CRUSHMap()
    for i in range(n_devices):
        m.add_device(f"dev{i}", weight=1.0)
    return m


def make_hierarchical_map():
    """Create a map with 2 racks, 2 hosts each, 2 devices each = 8 devices."""
    m = CRUSHMap()
    for rack_i in range(2):
        for host_i in range(2):
            for dev_i in range(2):
                name = f"dev-r{rack_i}h{host_i}d{dev_i}"
                m.add_device(
                    name,
                    weight=1.0,
                    location={
                        "dc": "us-east",
                        "rack": f"rack{rack_i}",
                        "host": f"host{rack_i}-{host_i}",
                    },
                )
    return m


# ---------------------------------------------------------------------------
# TestCRUSHMap
# ---------------------------------------------------------------------------


class TestCRUSHMap:
    def test_add_device(self):
        m = CRUSHMap()
        m.add_device("ssd0", weight=2.0)
        assert "ssd0" in m._devices
        dev = m.get_device("ssd0")
        assert dev.weight == 2.0
        assert dev.status == "up"
        assert dev.is_leaf

    def test_add_device_duplicate_raises(self):
        m = CRUSHMap()
        m.add_device("ssd0")
        with pytest.raises(ValueError, match="already exists"):
            m.add_device("ssd0")

    def test_add_device_with_location(self):
        m = CRUSHMap()
        m.add_device(
            "ssd0",
            weight=1.0,
            location={"host": "node1", "rack": "rack1", "dc": "us-east"},
        )
        devices = m.list_devices()
        assert len(devices) == 1
        assert devices[0]["location"] == {
            "host": "node1",
            "rack": "rack1",
            "dc": "us-east",
        }
        # Verify tree structure: root -> dc -> rack -> host -> device
        assert len(m.root.children) == 1  # us-east
        dc = m.root.children[0]
        assert dc.name == "us-east"
        assert dc.node_type == "dc"
        rack = dc.children[0]
        assert rack.name == "rack1"
        host = rack.children[0]
        assert host.name == "node1"
        assert host.children[0].name == "ssd0"

    def test_remove_device(self):
        m = make_simple_map(2)
        m.remove_device("dev0")
        dev = m.get_device("dev0")
        assert dev.status == "down"
        # Still in the tree
        assert "dev0" in m._devices

    def test_remove_nonexistent_raises(self):
        m = CRUSHMap()
        with pytest.raises(KeyError):
            m.remove_device("nope")

    def test_reweight(self):
        m = make_simple_map(2)
        m.reweight("dev0", 3.0)
        assert m.get_device("dev0").weight == 3.0
        # Root weight should update
        assert m.root.weight == 4.0  # 3.0 + 1.0

    def test_list_devices(self):
        m = make_simple_map(3)
        devices = m.list_devices()
        assert len(devices) == 3
        names = {d["name"] for d in devices}
        assert names == {"dev0", "dev1", "dev2"}

    def test_serialize_deserialize(self):
        m = make_hierarchical_map()
        data = m.to_dict()
        m2 = CRUSHMap.from_dict(data)
        assert len(m2._devices) == 8
        assert m2.root.weight == m.root.weight
        # Round-trip preserves device names
        orig_names = {d["name"] for d in m.list_devices()}
        new_names = {d["name"] for d in m2.list_devices()}
        assert orig_names == new_names

    def test_serialize_json_roundtrip(self):
        m = make_hierarchical_map()
        s = json.dumps(m.to_dict())
        m2 = CRUSHMap.from_dict(json.loads(s))
        assert len(m2._devices) == 8


# ---------------------------------------------------------------------------
# TestStraw2
# ---------------------------------------------------------------------------


class TestStraw2:
    def test_deterministic(self):
        """Same inputs must always produce the same output."""
        children = [
            CRUSHNode(f"d{i}", "device", 1.0) for i in range(5)
        ]
        results = []
        for _ in range(100):
            picked = _straw2_select(children, 0, 42, "test-key")
            results.append(picked.name)
        # All 100 calls should pick the same node
        assert len(set(results)) == 1

    def test_weight_proportional(self):
        """A node with 10x weight should get ~10x more selections."""
        heavy = CRUSHNode("heavy", "device", 10.0)
        light = CRUSHNode("light", "device", 1.0)
        children = [heavy, light]

        heavy_count = 0
        n = 5000
        for i in range(n):
            picked = _straw2_select(children, 0, i, f"key-{i}")
            if picked.name == "heavy":
                heavy_count += 1

        ratio = heavy_count / (n - heavy_count) if heavy_count < n else float("inf")
        # Should be roughly 10:1 — accept 5:1 to 20:1
        assert 5 < ratio < 20, f"Ratio was {ratio:.1f}, expected ~10"

    def test_uniform_distribution(self):
        """Equal weights should give roughly equal distribution."""
        children = [
            CRUSHNode(f"d{i}", "device", 1.0) for i in range(4)
        ]
        counts = {f"d{i}": 0 for i in range(4)}
        n = 10000
        for i in range(n):
            picked = _straw2_select(children, 0, i, f"k-{i}")
            counts[picked.name] += 1

        expected = n / 4
        for name, count in counts.items():
            # Within 30% of expected
            assert abs(count - expected) / expected < 0.30, (
                f"{name}: {count} vs expected {expected}"
            )

    def test_zero_weight_excluded(self):
        """Nodes with weight 0 should never be selected."""
        children = [
            CRUSHNode("zero", "device", 0.0),
            CRUSHNode("one", "device", 1.0),
        ]
        for i in range(100):
            picked = _straw2_select(children, 0, i, f"key-{i}")
            assert picked.name == "one"

    def test_down_node_excluded(self):
        """Down nodes (effective_weight=0) should never be selected."""
        down = CRUSHNode("down", "device", 1.0, status="down")
        up = CRUSHNode("up", "device", 1.0)
        for i in range(100):
            picked = _straw2_select([down, up], 0, i, f"key-{i}")
            assert picked.name == "up"


# ---------------------------------------------------------------------------
# TestCRUSHSelect
# ---------------------------------------------------------------------------


class TestCRUSHSelect:
    def test_single_replica(self):
        m = make_simple_map(4)
        rule = default_rule()
        result = crush_select("my-vector", 1, m, rule)
        assert len(result) == 1
        assert result[0].startswith("dev")

    def test_multiple_replicas_different_devices(self):
        m = make_simple_map(4)
        rule = default_rule()
        result = crush_select("my-vector", 3, m, rule)
        assert len(result) == 3
        assert len(set(result)) == 3  # all different

    def test_replicas_on_different_hosts(self):
        m = make_hierarchical_map()  # 2 racks, 2 hosts each, 2 devs each
        rule = host_spread_rule()
        result = crush_select("my-vector", 3, m, rule)
        assert len(result) == 3
        # Verify they're on different hosts
        hosts = set()
        for dev_name in result:
            dev_info = next(d for d in m.list_devices() if d["name"] == dev_name)
            hosts.add(dev_info["location"]["host"])
        assert len(hosts) == 3

    def test_deterministic_placement(self):
        m = make_simple_map(4)
        rule = default_rule()
        r1 = crush_select("vec-abc", 2, m, rule)
        r2 = crush_select("vec-abc", 2, m, rule)
        assert r1 == r2

    def test_handles_down_device(self):
        m = make_simple_map(4)
        rule = default_rule()
        # Get placement
        result_before = crush_select("vec-1", 2, m, rule)
        # Mark one of the selected devices as down
        m.remove_device(result_before[0])
        result_after = crush_select("vec-1", 2, m, rule)
        # Should still get 2 replicas, but not the down device
        assert len(result_after) == 2
        assert result_before[0] not in result_after

    def test_weight_zero_excluded(self):
        m = make_simple_map(4)
        m.reweight("dev0", 0)
        rule = default_rule()
        # dev0 should never appear
        for i in range(200):
            result = crush_select(f"key-{i}", 1, m, rule)
            assert "dev0" not in result

    def test_more_replicas_than_devices(self):
        m = make_simple_map(3)
        rule = default_rule()
        result = crush_select("vec-1", 5, m, rule)
        # Should cap at number of available devices
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# TestCRUSHRouter
# ---------------------------------------------------------------------------


class TestCRUSHRouter:
    def test_place_and_locate_same(self):
        m = make_simple_map(4)
        router = CRUSHRouter(m)
        placed = router.place("vec-123", replicas=2)
        located = router.locate("vec-123", replicas=2)
        assert placed == located

    def test_place_batch(self):
        m = make_simple_map(4)
        router = CRUSHRouter(m)
        ids = [f"vec-{i}" for i in range(100)]
        batch = router.place_batch(ids, replicas=1)
        assert len(batch) == 100
        for vid, devs in batch.items():
            assert len(devs) == 1

    def test_distribution_uniform(self):
        m = make_simple_map(4)
        router = CRUSHRouter(m)
        keys = [f"key-{i}" for i in range(10000)]
        dist = router.distribution(keys, replicas=1)
        expected = 10000 / 4
        for dev, count in dist.items():
            assert abs(count - expected) / expected < 0.30, (
                f"{dev}: {count} vs expected {expected}"
            )

    def test_rebalance_minimal_movement(self):
        """Adding one node to N nodes should move ~1/(N+1) of data."""
        m_old = make_simple_map(4)
        m_new = make_simple_map(4)
        m_new.add_device("dev4", weight=1.0)

        router_new = CRUSHRouter(m_new)
        keys = [f"sample-{i}" for i in range(10000)]
        plan = router_new.rebalance_plan(m_old, sample_keys=keys, replicas=1)

        # Expected: ~20% movement (1/5). Accept 10-35%.
        pct = plan["percent"]
        assert 10 < pct < 35, f"Movement was {pct}%, expected ~20%"

    def test_save_and_load(self):
        m = make_hierarchical_map()
        rule = host_spread_rule()
        router = CRUSHRouter(m, rule)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            router.save(path)
            loaded = CRUSHRouter.load(path)
            # Same placement
            for i in range(50):
                vid = f"vec-{i}"
                assert router.place(vid, 2) == loaded.place(vid, 2)
            assert loaded.rule.name == "host_spread"
        finally:
            os.unlink(path)

    def test_default_rule(self):
        r = default_rule()
        assert r.name == "default"
        assert r.min_size == 1
        assert len(r.steps) == 3

    def test_router_with_replicas(self):
        m = make_simple_map(6)
        router = CRUSHRouter(m)
        result = router.place("vec-x", replicas=3)
        assert len(result) == 3
        assert len(set(result)) == 3


# ---------------------------------------------------------------------------
# TestCRUSHIntegration
# ---------------------------------------------------------------------------


class TestCRUSHIntegration:
    def test_crush_with_gitdb_vector_ids(self):
        """Use realistic GitDB vector IDs (UUIDs and SHA-style hashes)."""
        m = make_hierarchical_map()
        router = CRUSHRouter(m)

        # UUID-style IDs
        uuid_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"vec-{i}")) for i in range(100)]
        batch = router.place_batch(uuid_ids, replicas=2)
        for vid, devs in batch.items():
            assert len(devs) == 2
            assert len(set(devs)) == 2

        # SHA-style IDs (like git commit hashes)
        import hashlib as hl
        sha_ids = [hl.sha1(f"commit-{i}".encode()).hexdigest() for i in range(100)]
        batch2 = router.place_batch(sha_ids, replicas=3)
        for vid, devs in batch2.items():
            assert len(devs) == 3

    def test_crush_stability_across_key_formats(self):
        """Different key formats should all produce valid deterministic placements."""
        m = make_simple_map(4)
        router = CRUSHRouter(m)

        keys = [
            "simple-string",
            "with/slashes/like/git/paths",
            "with spaces",
            "unicode-\u00e9\u00e8\u00ea",
            "very-long-" + "x" * 1000,
            "",  # empty string
        ]
        for key in keys:
            r1 = router.place(key, replicas=2)
            r2 = router.place(key, replicas=2)
            assert r1 == r2, f"Non-deterministic for key: {key!r}"
            assert len(r1) == 2

    def test_crush_large_cluster(self):
        """Test with a larger cluster: 3 DCs, 4 racks each, 3 hosts, 4 devices."""
        m = CRUSHMap()
        for dc_i in range(3):
            for rack_i in range(4):
                for host_i in range(3):
                    for dev_i in range(4):
                        m.add_device(
                            f"d{dc_i}r{rack_i}h{host_i}d{dev_i}",
                            weight=1.0,
                            location={
                                "dc": f"dc-{dc_i}",
                                "rack": f"rack-{dc_i}-{rack_i}",
                                "host": f"host-{dc_i}-{rack_i}-{host_i}",
                            },
                        )
        # 3*4*3*4 = 144 devices
        assert len(m._devices) == 144

        router = CRUSHRouter(m)
        keys = [f"vec-{i}" for i in range(5000)]
        dist = router.distribution(keys, replicas=1)

        # All 144 devices should get at least some keys
        assert len(dist) > 100, f"Only {len(dist)} devices got keys out of 144"

    def test_topology_change_determinism(self):
        """After marking a device down, other keys should mostly stay put."""
        m = make_simple_map(8)
        router = CRUSHRouter(m)

        # Record placements for 1000 keys
        keys = [f"k-{i}" for i in range(1000)]
        before = {k: router.place(k, 1)[0] for k in keys}

        # Mark one device down
        m.remove_device("dev3")

        # Only keys that were on dev3 should move
        moved = 0
        for k in keys:
            after = router.place(k, 1)[0]
            if before[k] == "dev3":
                assert after != "dev3"  # must move away
            elif before[k] != after:
                moved += 1

        # Very few non-dev3 keys should move (ideally zero for straw2)
        assert moved < 50, f"{moved} non-dev3 keys moved unexpectedly"
