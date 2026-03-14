"""CRUSH — Controlled Replication Under Scalable Hashing.

Deterministic data placement algorithm adapted from CEPH for GitDB.
Maps vector IDs to storage nodes/shards without a central lookup table.

The key properties:
- Deterministic: same key + same map = same placement, always
- Weight-proportional: heavier nodes get proportionally more data
- Minimal disruption: topology changes move only the minimum necessary data
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Topology nodes
# ---------------------------------------------------------------------------

# Hierarchy levels, ordered from leaf to root
TYPE_ORDER = ["device", "host", "rack", "dc", "root"]


@dataclass
class CRUSHNode:
    """A node in the CRUSH hierarchy (device, host, rack, datacenter, root)."""

    name: str
    node_type: str  # "device", "host", "rack", "dc", "root"
    weight: float  # capacity weight (1.0 = baseline)
    children: List["CRUSHNode"] = field(default_factory=list)
    status: str = "up"  # "up" or "down"

    @property
    def is_leaf(self) -> bool:
        return self.node_type == "device"

    @property
    def effective_weight(self) -> float:
        if self.status == "down" or self.weight <= 0:
            return 0.0
        if self.is_leaf:
            return self.weight
        return sum(c.effective_weight for c in self.children)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "node_type": self.node_type,
            "weight": self.weight,
            "status": self.status,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CRUSHNode":
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            name=data["name"],
            node_type=data["node_type"],
            weight=data["weight"],
            children=children,
            status=data.get("status", "up"),
        )


# ---------------------------------------------------------------------------
# CRUSHMap — the cluster topology tree
# ---------------------------------------------------------------------------


class CRUSHMap:
    """The cluster topology tree."""

    def __init__(self):
        self.root = CRUSHNode("root", "root", 0, [], "up")
        self._devices: Dict[str, CRUSHNode] = {}

    def add_device(
        self,
        name: str,
        weight: float = 1.0,
        location: Optional[Dict[str, str]] = None,
    ) -> CRUSHNode:
        """Add a storage device.

        Args:
            name: Unique device name.
            weight: Capacity weight (1.0 = baseline).
            location: Placement in the hierarchy, e.g.
                      {"host": "node1", "rack": "rack1", "dc": "us-east"}
        """
        if name in self._devices:
            raise ValueError(f"Device {name!r} already exists")

        device = CRUSHNode(name, "device", weight, [], "up")

        if location is None:
            location = {}

        # Build/find the path from root down to the device.
        # Levels (excluding root and device): dc -> rack -> host
        parent = self.root
        for level in ["dc", "rack", "host"]:
            bucket_name = location.get(level)
            if bucket_name is None:
                continue
            # Find or create the intermediate bucket
            child = self._find_child(parent, bucket_name)
            if child is None:
                child = CRUSHNode(bucket_name, level, 0, [], "up")
                parent.children.append(child)
            parent = child

        parent.children.append(device)
        self._devices[name] = device
        self._update_weights()
        return device

    def remove_device(self, name: str):
        """Mark device as down (don't remove — CRUSH needs stable topology)."""
        dev = self._devices.get(name)
        if dev is None:
            raise KeyError(f"Device {name!r} not found")
        dev.status = "down"
        self._update_weights()

    def reweight(self, name: str, weight: float):
        """Change device weight (for capacity changes)."""
        dev = self._devices.get(name)
        if dev is None:
            raise KeyError(f"Device {name!r} not found")
        dev.weight = weight
        self._update_weights()

    def get_device(self, name: str) -> CRUSHNode:
        dev = self._devices.get(name)
        if dev is None:
            raise KeyError(f"Device {name!r} not found")
        return dev

    def list_devices(self) -> List[dict]:
        """Return all devices with their status and weight."""
        result = []
        for name, dev in sorted(self._devices.items()):
            result.append(
                {
                    "name": dev.name,
                    "weight": dev.weight,
                    "status": dev.status,
                    "location": self._device_location(dev),
                }
            )
        return result

    def to_dict(self) -> dict:
        return {"root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "CRUSHMap":
        m = cls()
        m.root = CRUSHNode.from_dict(data["root"])
        m._devices = {}
        m._index_devices(m.root)
        return m

    # -- internal helpers --

    def _find_child(self, parent: CRUSHNode, name: str) -> Optional[CRUSHNode]:
        for c in parent.children:
            if c.name == name:
                return c
        return None

    def _index_devices(self, node: CRUSHNode):
        if node.is_leaf:
            self._devices[node.name] = node
        else:
            for c in node.children:
                self._index_devices(c)

    def _update_weights(self):
        """Propagate leaf weights up the tree."""
        self._propagate_weight(self.root)

    def _propagate_weight(self, node: CRUSHNode):
        if node.is_leaf:
            return
        for c in node.children:
            self._propagate_weight(c)
        node.weight = sum(c.effective_weight for c in node.children)

    def _device_location(self, target: CRUSHNode) -> Dict[str, str]:
        """Walk the tree to find the path to a device."""
        path: Dict[str, str] = {}

        def _walk(node: CRUSHNode) -> bool:
            if node is target:
                return True
            for c in node.children:
                if _walk(c):
                    if node.node_type not in ("root", "device"):
                        path[node.node_type] = node.name
                    return True
            return False

        _walk(self.root)
        return path


# ---------------------------------------------------------------------------
# CRUSHRule — placement rules
# ---------------------------------------------------------------------------


@dataclass
class CRUSHRule:
    """A placement rule — controls replica placement."""

    name: str
    min_size: int  # minimum replicas
    max_size: int  # maximum replicas
    steps: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "steps": list(self.steps),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CRUSHRule":
        return cls(
            name=data["name"],
            min_size=data["min_size"],
            max_size=data["max_size"],
            steps=data.get("steps", []),
        )


def default_rule() -> CRUSHRule:
    """Default rule: pick N different devices under root."""
    return CRUSHRule(
        name="default",
        min_size=1,
        max_size=10,
        steps=[
            {"op": "take", "item": "root"},
            {"op": "choose_firstn", "num": 0, "type": "device"},
            {"op": "emit"},
        ],
    )


def host_spread_rule() -> CRUSHRule:
    """Spread replicas across different hosts, one device per host."""
    return CRUSHRule(
        name="host_spread",
        min_size=1,
        max_size=10,
        steps=[
            {"op": "take", "item": "root"},
            {"op": "choose_firstn", "num": 0, "type": "host"},
            {"op": "choose_firstn", "num": 1, "type": "device"},
            {"op": "emit"},
        ],
    )


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _crush_hash(key: str, r: int, node_name: str) -> int:
    """Deterministic hash: SHA-256 of (key, replica_index, node_name) -> 64-bit uint."""
    h = hashlib.sha256(f"{key}|{r}|{node_name}".encode()).digest()
    return struct.unpack("<Q", h[:8])[0]


# ---------------------------------------------------------------------------
# Straw2 bucket selection
# ---------------------------------------------------------------------------

# Large value used when weight is zero to ensure -inf draw
_HASH_MAX = (1 << 64) - 1


def _straw2_select(
    children: List[CRUSHNode], key_hash: int, r: int, key: str
) -> CRUSHNode:
    """Straw2 algorithm — each child draws a weighted straw, longest wins.

    For each child:
        draw = hash(key, r, child.name)
        straw = ln(draw / 2^64) / weight

    Highest straw wins.  This gives weight-proportional, deterministic selection
    with minimal disruption when weights change.
    """
    best_node: Optional[CRUSHNode] = None
    best_draw = -math.inf

    for child in children:
        w = child.effective_weight
        if w <= 0:
            continue

        h = _crush_hash(key, r, child.name)
        # Normalize hash to (0, 1) — avoid 0 for ln()
        u = (h & 0xFFFFFFFFFFFFFFFF) / (1 << 64)
        if u == 0:
            u = 1e-18

        # Straw = ln(u) / weight  (higher weight → less negative → wins more)
        draw = math.log(u) / w

        if draw > best_draw:
            best_draw = draw
            best_node = child

    if best_node is None:
        raise RuntimeError("No available nodes with positive weight")

    return best_node


# ---------------------------------------------------------------------------
# Core select — walks the rule steps
# ---------------------------------------------------------------------------


def _collect_type(node: CRUSHNode, target_type: str) -> List[CRUSHNode]:
    """Collect all descendants of a given type (BFS)."""
    if node.node_type == target_type:
        return [node]
    results = []
    for c in node.children:
        results.extend(_collect_type(c, target_type))
    return results


def _find_node_by_name(root: CRUSHNode, name: str) -> Optional[CRUSHNode]:
    """Find a node anywhere in the tree by name."""
    if root.name == name:
        return root
    for c in root.children:
        result = _find_node_by_name(c, name)
        if result is not None:
            return result
    return None


def _select_from_bucket(
    bucket: CRUSHNode,
    target_type: str,
    num: int,
    key: str,
    firstn: bool,
) -> List[CRUSHNode]:
    """Select `num` distinct items of `target_type` from `bucket`.

    Uses straw2 at each level of the tree, walking down from the bucket
    to the target type.  Retries on collisions and down nodes.
    """
    # Collect candidate subtrees of the target type
    candidates = _collect_type(bucket, target_type)
    candidates = [c for c in candidates if c.effective_weight > 0]

    if num <= 0 or num > len(candidates):
        num = len(candidates)

    selected: List[CRUSHNode] = []
    selected_names: set = set()
    max_retries = 50

    for rep in range(num):
        retry = 0
        while retry < max_retries:
            r = rep + retry * num  # vary the hash input on retry

            if firstn:
                # firstn: walk down from bucket using straw2 at each level
                node = bucket
                while node.node_type != target_type:
                    alive = [c for c in node.children if c.effective_weight > 0]
                    if not alive:
                        break
                    node = _straw2_select(alive, 0, r, key)
                    # If we overshot (node is below target_type), collect upward
                    if TYPE_ORDER.index(node.node_type) < TYPE_ORDER.index(target_type):
                        break
            else:
                # choose_indep: same algorithm, just different retry semantics
                node = bucket
                while node.node_type != target_type:
                    alive = [c for c in node.children if c.effective_weight > 0]
                    if not alive:
                        break
                    node = _straw2_select(alive, 0, r, key)

            if node.node_type != target_type:
                retry += 1
                continue

            if node.effective_weight <= 0:
                retry += 1
                continue

            if node.name in selected_names:
                retry += 1
                continue

            selected.append(node)
            selected_names.add(node.name)
            break

    return selected


def crush_select(
    key: str,
    num_replicas: int,
    crush_map: CRUSHMap,
    rule: CRUSHRule,
) -> List[str]:
    """Deterministic placement — returns list of device names.

    Uses the CRUSH algorithm:
    1. Hash the key to get a seed
    2. Walk the topology tree according to rule steps
    3. At each level, use straw2 bucket selection
    4. Handle collisions (same device picked twice) with retry
    5. Handle down devices with retry

    Same key + same map + same rule = same output. Always.
    """
    if num_replicas < rule.min_size:
        num_replicas = rule.min_size
    if num_replicas > rule.max_size:
        num_replicas = rule.max_size

    # The working set: a list of buckets we're currently selecting from
    working: List[CRUSHNode] = []
    output: List[CRUSHNode] = []

    for step in rule.steps:
        op = step["op"]

        if op == "take":
            item_name = step["item"]
            node = _find_node_by_name(crush_map.root, item_name)
            if node is None:
                raise ValueError(f"Item {item_name!r} not found in CRUSH map")
            working = [node]

        elif op in ("choose_firstn", "choose_indep"):
            target_type = step["type"]
            num = step.get("num", 0)
            if num <= 0:
                num = num_replicas

            firstn = op == "choose_firstn"
            new_working = []
            for bucket in working:
                selected = _select_from_bucket(
                    bucket, target_type, num, key, firstn
                )
                new_working.extend(selected)
            working = new_working

        elif op == "emit":
            output.extend(working)
            working = []

    # Extract device names — if output nodes aren't devices, descend to devices
    result = []
    for node in output:
        if node.is_leaf:
            if node.status == "up" and node.weight > 0:
                result.append(node.name)
        else:
            # Pick one device from this subtree
            devices = _collect_type(node, "device")
            devices = [d for d in devices if d.effective_weight > 0]
            if devices:
                picked = _straw2_select(devices, 0, 0, key)
                result.append(picked.name)

    # Deduplicate while preserving order
    seen: set = set()
    deduped = []
    for name in result:
        if name not in seen:
            seen.add(name)
            deduped.append(name)

    return deduped[:num_replicas]


# ---------------------------------------------------------------------------
# CRUSHRouter — high-level interface for GitDB
# ---------------------------------------------------------------------------


class CRUSHRouter:
    """Routes vectors to GitDB shards using CRUSH."""

    def __init__(self, crush_map: CRUSHMap, rule: CRUSHRule = None):
        self.map = crush_map
        self.rule = rule or default_rule()

    def place(self, vector_id: str, replicas: int = 1) -> List[str]:
        """Where should this vector go? Returns device names."""
        return crush_select(vector_id, replicas, self.map, self.rule)

    def place_batch(
        self, vector_ids: List[str], replicas: int = 1
    ) -> Dict[str, List[str]]:
        """Batch placement for multiple vectors."""
        return {vid: self.place(vid, replicas) for vid in vector_ids}

    def locate(self, vector_id: str, replicas: int = 1) -> List[str]:
        """Where IS this vector? Same as place() — deterministic."""
        return self.place(vector_id, replicas)

    def rebalance_plan(
        self, old_map: CRUSHMap, sample_keys: Optional[List[str]] = None, replicas: int = 1
    ) -> Dict[str, object]:
        """Compute what needs to move when topology changes.

        Compares placement under old_map vs current map for a sample of keys.
        Returns stats: {"moves": N, "total": M, "percent": P, "details": [...]}
        """
        if sample_keys is None:
            sample_keys = [f"sample-{i}" for i in range(10000)]

        old_router = CRUSHRouter(old_map, self.rule)
        moves = 0
        details = []

        for key in sample_keys:
            old_placement = old_router.place(key, replicas)
            new_placement = self.place(key, replicas)
            if set(old_placement) != set(new_placement):
                moves += 1
                if len(details) < 100:  # cap detail output
                    details.append(
                        {
                            "key": key,
                            "from": old_placement,
                            "to": new_placement,
                            "action": "migrate",
                        }
                    )

        total = len(sample_keys)
        return {
            "moves": moves,
            "total": total,
            "percent": round(moves / total * 100, 2) if total else 0,
            "details": details,
        }

    def distribution(
        self, sample_keys: List[str], replicas: int = 1
    ) -> Dict[str, int]:
        """Show how keys distribute across devices (for testing uniformity)."""
        counts: Dict[str, int] = {}
        for key in sample_keys:
            placements = self.place(key, replicas)
            for dev in placements:
                counts[dev] = counts.get(dev, 0) + 1
        return counts

    def save(self, path: str):
        """Save CRUSH map + rule to JSON."""
        data = {
            "crush_map": self.map.to_dict(),
            "rule": self.rule.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CRUSHRouter":
        """Load from JSON."""
        with open(path) as f:
            data = json.load(f)
        crush_map = CRUSHMap.from_dict(data["crush_map"])
        rule = CRUSHRule.from_dict(data["rule"])
        return cls(crush_map, rule)
