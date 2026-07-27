"""Cross-session memory.

A flat key/value store on disk. Facts the agent learns in one session are
available in the next one, which is what makes a long-lived agent feel
continuous rather than amnesiac.

Do not store credentials here — the file is plain JSON and its contents are
replayed into future model contexts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Tool, ToolError, obj_schema

MAX_VALUE_CHARS = 20_000
MAX_ENTRIES = 500


class MemoryStore:
    """JSON-backed key/value store. Reads and writes are atomic per call."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A corrupt store should not brick the agent; start clean but keep
            # the old file so nothing is silently destroyed.
            self.path.replace(self.path.with_suffix(".corrupt.json"))
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def all(self) -> dict[str, str]:
        return self._load()

    def get(self, key: str) -> str | None:
        return self._load().get(key)

    def set(self, key: str, value: str) -> None:
        data = self._load()
        if key not in data and len(data) >= MAX_ENTRIES:
            raise ToolError(
                f"memory is full ({MAX_ENTRIES} entries). Delete something first."
            )
        data[key] = value
        self._save(data)

    def delete(self, key: str) -> bool:
        data = self._load()
        if key not in data:
            return False
        del data[key]
        self._save(data)
        return True


def _validate_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        raise ToolError("key must be a non-empty string.")
    if len(key) > 200:
        raise ToolError("key must be 200 characters or fewer.")
    return key


class RememberTool(Tool):
    name = "remember"
    description = (
        "Save a durable fact, preference, or conclusion under a short key. "
        "Persists across sessions. Use it for things worth carrying forward — "
        "user preferences, project context, findings you will need again. "
        "Never store passwords, API keys, or tokens."
    )
    input_schema: dict[str, Any] = obj_schema(
        {
            "key": {
                "type": "string",
                "description": "Short stable identifier, e.g. 'user.timezone'.",
            },
            "value": {"type": "string", "description": "The content to remember."},
        },
        required=["key", "value"],
    )

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(self, key: str, value: str) -> str:  # type: ignore[override]
        key = _validate_key(key)
        if len(value) > MAX_VALUE_CHARS:
            raise ToolError(
                f"value is {len(value)} characters, over the {MAX_VALUE_CHARS} limit. "
                "Write long content to a workspace file and remember the path instead."
            )
        self.store.set(key, value)
        return f"Remembered {key!r}."


class RecallTool(Tool):
    name = "recall"
    description = (
        "Look up remembered facts. Omit `key` to list every key with a preview, "
        "or pass one to read a single entry in full. Check this before starting "
        "non-trivial work."
    )
    input_schema: dict[str, Any] = obj_schema(
        {
            "key": {
                "type": "string",
                "description": "Key to read. Omit to list all keys.",
            }
        }
    )

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(self, key: str | None = None) -> str:  # type: ignore[override]
        data = self.store.all()
        if not data:
            return "Memory is empty."
        if key:
            value = data.get(key.strip())
            if value is None:
                available = ", ".join(sorted(data)) or "(none)"
                raise ToolError(f"No memory under {key!r}. Known keys: {available}")
            return value
        lines = []
        for k in sorted(data):
            preview = data[k].replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            lines.append(f"{k}: {preview}")
        return "\n".join(lines)


class ForgetTool(Tool):
    name = "forget"
    description = (
        "Delete a remembered entry. Use when a fact is superseded or wrong."
    )
    input_schema: dict[str, Any] = obj_schema(
        {"key": {"type": "string", "description": "Key to delete."}},
        required=["key"],
    )

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(self, key: str) -> str:  # type: ignore[override]
        key = _validate_key(key)
        if not self.store.delete(key):
            raise ToolError(f"No memory under {key!r}; nothing deleted.")
        return f"Forgot {key!r}."
