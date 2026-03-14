"""GitDB Hooks — pre/post event hooks for commit, merge, push, and drift."""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Set


VALID_EVENTS = frozenset([
    "pre-commit", "post-commit",
    "pre-merge", "post-merge",
    "pre-push", "post-push",
    "on-drift",
])


class HookManager:
    """Registry for event hooks.

    Pre-hooks (pre-commit, pre-merge, pre-push) can reject an operation
    by returning False. Post-hooks and on-drift are fire-and-forget.
    """

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)

    def register(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        if event not in VALID_EVENTS:
            raise ValueError(f"Unknown event: {event!r}. Valid: {sorted(VALID_EVENTS)}")
        if callback in self._hooks[event]:
            return  # already registered, idempotent
        self._hooks[event].append(callback)

    def unregister(self, event: str, callback: Callable) -> None:
        """Remove a callback for an event."""
        if event not in VALID_EVENTS:
            raise ValueError(f"Unknown event: {event!r}. Valid: {sorted(VALID_EVENTS)}")
        try:
            self._hooks[event].remove(callback)
        except ValueError:
            pass  # not registered, no-op

    def fire(self, event: str, **context) -> bool:
        """Fire all callbacks for an event.

        For pre-* events, returns False if any callback returns False
        (the operation should be aborted). For post-* events, always
        returns True.
        """
        if event not in VALID_EVENTS:
            raise ValueError(f"Unknown event: {event!r}. Valid: {sorted(VALID_EVENTS)}")

        is_pre = event.startswith("pre-")
        for callback in self._hooks[event]:
            result = callback(**context)
            if is_pre and result is False:
                return False
        return True

    def list_hooks(self) -> Dict[str, List[str]]:
        """Return a dict of event → list of callback names."""
        out = {}
        for event in sorted(VALID_EVENTS):
            callbacks = self._hooks.get(event, [])
            if callbacks:
                out[event] = [_cb_name(cb) for cb in callbacks]
        return out

    def clear(self, event: str = None):
        """Clear hooks for an event, or all hooks if event is None."""
        if event is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event, None)


def _cb_name(cb: Callable) -> str:
    """Best-effort name for a callback."""
    return getattr(cb, "__qualname__", None) or getattr(cb, "__name__", repr(cb))
