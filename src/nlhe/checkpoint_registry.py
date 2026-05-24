"""Registry of trained checkpoints available as training opponents or eval targets.

A CheckpointRegistry is a YAML/JSON-backed catalog. Each entry has:
  - name: a stable string identifier (e.g. "dcfr-overnight-3400")
  - path: filesystem path to the .pt checkpoint
  - metadata: free-form dict (training config tag, iter count, date, notes)
  - tags: list of strings for query filtering (e.g. ["dcfr", "overnight"])

The registry is the source of truth for "which checkpoints does the project
know about as opponents." LeaguePool wraps it for sampling during training;
scripts/eval_pool.py can also use it to construct opponent lists by tag.

Design choices:
  - JSON over YAML for the persisted format. Less ambiguity, no quoting bugs,
    and we already have json in stdlib. The format is human-readable enough
    for hand-editing if needed.
  - Paths stored as strings, not Path objects. Survives JSON round-trip
    without custom serializers.
  - No automatic discovery from runs/ — caller registers explicitly. Keeps
    the registry intentional rather than scooping up every stray checkpoint.
  - Idempotent register: same name re-registered with same content is a no-op;
    same name with different content raises. Prevents silent overwrites.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class CheckpointEntry:
    """One entry in the registry."""
    name: str
    path: str
    metadata: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointEntry":
        return cls(
            name=d["name"],
            path=d["path"],
            metadata=dict(d.get("metadata", {})),
            tags=list(d.get("tags", [])),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, CheckpointEntry):
            return False
        return (
            self.name == other.name
            and self.path == other.path
            and self.metadata == other.metadata
            and sorted(self.tags) == sorted(other.tags)
        )


DEFAULT_REGISTRY_PATH = "runs/league/registry.json"


class CheckpointRegistry:
    """Catalog of named checkpoints with metadata and tag-based query."""

    def __init__(self, entries: Optional[list] = None):
        self._entries: dict = {}
        if entries:
            for e in entries:
                self._entries[e.name] = e

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self):
        return iter(self._entries.values())

    def register(
        self,
        name: str,
        path: str,
        metadata: Optional[dict] = None,
        tags: Optional[list] = None,
    ) -> CheckpointEntry:
        """Add an entry. Idempotent on identical re-registration.

        Raises ValueError if `name` already exists with different content.
        """
        entry = CheckpointEntry(
            name=name,
            path=str(path),
            metadata=dict(metadata or {}),
            tags=list(tags or []),
        )
        if name in self._entries:
            existing = self._entries[name]
            if existing == entry:
                return existing
            raise ValueError(
                f"checkpoint {name!r} already registered with different content; "
                f"use unregister() first if intentional"
            )
        self._entries[name] = entry
        return entry

    def unregister(self, name: str) -> None:
        """Remove an entry. KeyError if not present."""
        del self._entries[name]

    def get(self, name: str) -> CheckpointEntry:
        """Fetch an entry by name. KeyError if not present."""
        return self._entries[name]

    def names(self) -> list:
        """All registered names, in registration order."""
        return list(self._entries.keys())

    def query(self, tags: Optional[list] = None, match: str = "any") -> list:
        """Return entries matching tag filter.

        Args:
            tags: list of tag strings to filter on. None or empty returns all.
            match: "any" (default) returns entries with at least one matching tag;
                "all" returns entries with every requested tag.

        Returns:
            List of CheckpointEntry, in registration order.
        """
        if not tags:
            return list(self._entries.values())
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all', got {match!r}")
        tag_set = set(tags)
        out = []
        for entry in self._entries.values():
            entry_tags = set(entry.tags)
            if match == "any" and entry_tags & tag_set:
                out.append(entry)
            elif match == "all" and tag_set <= entry_tags:
                out.append(entry)
        return out

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "entries": [e.to_dict() for e in self._entries.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointRegistry":
        if d.get("version") != 1:
            raise ValueError(f"unsupported registry version: {d.get('version')!r}")
        entries = [CheckpointEntry.from_dict(e) for e in d.get("entries", [])]
        return cls(entries=entries)

    def save(self, path: Optional[str] = None) -> str:
        """Persist to JSON. Returns the path written."""
        path = str(path or DEFAULT_REGISTRY_PATH)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "CheckpointRegistry":
        """Load from JSON. Returns an empty registry if path does not exist."""
        path = str(path or DEFAULT_REGISTRY_PATH)
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        return cls.from_dict(d)
