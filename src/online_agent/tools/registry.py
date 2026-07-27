"""Tool registry — owns the local tool set and dispatches calls to it."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import AgentConfig
from .base import Tool, ToolError
from .files import ListFilesTool, ReadFileTool, WriteFileTool
from .memory import ForgetTool, MemoryStore, RecallTool, RememberTool

MEMORY_FILENAME = ".agent_memory.json"


class ToolRegistry:
    """Holds local tools and runs them off the event loop."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    @classmethod
    def default(cls, config: AgentConfig) -> ToolRegistry:
        """The standard tool set: workspace files plus cross-session memory."""
        workspace = config.workspace
        workspace.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(workspace / MEMORY_FILENAME)
        return cls(
            [
                ReadFileTool(workspace),
                WriteFileTool(workspace),
                ListFilesTool(workspace),
                RememberTool(store),
                RecallTool(store),
                ForgetTool(store),
            ]
        )

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self._tools.values()]

    async def call(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool. Returns (output, is_error).

        Tool failures are returned rather than raised: the model gets an error
        tool_result and can correct itself, which is the whole point of the
        `is_error` flag.
        """
        tool = self._tools.get(name)
        if tool is None:
            return (
                f"Unknown tool {name!r}. Available tools: {', '.join(self.names())}.",
                True,
            )
        if not isinstance(tool_input, dict):
            return (f"Tool input for {name!r} must be an object.", True)

        try:
            # Tools are synchronous and may touch the filesystem — keep them off
            # the event loop so concurrent calls actually overlap.
            output = await asyncio.to_thread(tool.run, **tool_input)
        except ToolError as exc:
            return (str(exc), True)
        except TypeError as exc:
            # Wrong or missing arguments from the model.
            return (f"Invalid arguments for {name!r}: {exc}", True)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            return (f"{name!r} failed: {type(exc).__name__}: {exc}", True)
        return (output if output else "(no output)", False)


__all__ = [
    "ListFilesTool",
    "MemoryStore",
    "Path",
    "ReadFileTool",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "WriteFileTool",
]
