"""WatchManager — subscribe to changes on vectors/metadata patterns."""

import threading
from typing import Any, Callable, Dict, List, Optional


class WatchManager:
    """Subscribe to changes on vectors/metadata patterns.

    Callbacks fire when matching data changes during commit or push/pull.
    """

    def __init__(self):
        self._watches: Dict[int, dict] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def watch(self, where: dict, callback: Callable) -> int:
        """Subscribe to changes matching a metadata filter. Returns watch_id."""
        with self._lock:
            wid = self._next_id
            self._next_id += 1
            self._watches[wid] = {
                "id": wid,
                "type": "where",
                "where": where,
                "callback": callback,
                "branch": None,
            }
            return wid

    def watch_branch(self, branch: str, callback: Callable) -> int:
        """Subscribe to changes on a specific branch. Returns watch_id."""
        with self._lock:
            wid = self._next_id
            self._next_id += 1
            self._watches[wid] = {
                "id": wid,
                "type": "branch",
                "where": None,
                "branch": branch,
                "callback": callback,
            }
            return wid

    def unwatch(self, watch_id: int) -> None:
        """Remove a watch by ID."""
        with self._lock:
            if watch_id not in self._watches:
                raise ValueError(f"Watch not found: {watch_id}")
            del self._watches[watch_id]

    def check(self, event: str, **context) -> None:
        """Fire matching watches. Called internally on commit/push/pull.

        Args:
            event: "commit", "push", or "pull"
            **context: event-specific data:
                - commit: added_metadata, removed_metadata, modified_count, branch, commit_hash, message
                - push/pull: branch
        """
        with self._lock:
            watches = list(self._watches.values())

        for w in watches:
            try:
                if w["type"] == "branch":
                    branch = context.get("branch")
                    if branch and branch == w["branch"]:
                        w["callback"](event, context)
                elif w["type"] == "where":
                    if self._matches_context(w["where"], event, context):
                        w["callback"](event, context)
            except Exception:
                pass  # Don't let a bad callback break the system

    def _matches_context(self, where: dict, event: str, context: dict) -> bool:
        """Check if any added/removed/modified metadata matches the where filter."""
        from gitdb.structured import matches as struct_matches
        from gitdb.types import VectorMeta

        # Check added metadata
        for meta in context.get("added_metadata", []):
            if isinstance(meta, VectorMeta):
                if struct_matches(meta, where):
                    return True
            elif isinstance(meta, dict):
                # Build a temporary VectorMeta for matching
                vm = VectorMeta(id="", document=None, metadata=meta)
                if struct_matches(vm, where):
                    return True

        # Check removed metadata
        for meta in context.get("removed_metadata", []):
            if isinstance(meta, VectorMeta):
                if struct_matches(meta, where):
                    return True
            elif isinstance(meta, dict):
                vm = VectorMeta(id="", document=None, metadata=meta)
                if struct_matches(vm, where):
                    return True

        # If nothing matched but there were modifications, check if
        # the where filter matches broadly (modified_count > 0 means something changed)
        if context.get("modified_count", 0) > 0:
            # For modifications we don't have the metadata readily available,
            # so fire if the event happened at all
            return True

        return False

    def list_watches(self) -> List[dict]:
        """List all active watches."""
        with self._lock:
            result = []
            for w in self._watches.values():
                result.append({
                    "id": w["id"],
                    "type": w["type"],
                    "where": w["where"],
                    "branch": w["branch"],
                })
            return result
