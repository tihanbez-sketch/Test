"""Events emitted by the agent loop.

The agent is a generator of these. The CLI renders them to a terminal and the
HTTP server relays them as server-sent events, so both frontends share one
contract and neither needs to know about the Messages API wire format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class AgentEvent:
    """Base class for everything the agent yields."""

    type: str = field(init=False, default="event")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass(slots=True)
class TextDelta(AgentEvent):
    """A chunk of user-facing assistant text."""

    text: str

    def __post_init__(self) -> None:
        self.type = "text_delta"


@dataclass(slots=True)
class ThinkingDelta(AgentEvent):
    """A chunk of summarised reasoning (empty when display is 'omitted')."""

    text: str

    def __post_init__(self) -> None:
        self.type = "thinking_delta"


@dataclass(slots=True)
class ServerToolUse(AgentEvent):
    """Anthropic ran a hosted tool (web search / web fetch) on our behalf."""

    name: str
    query: str | None = None

    def __post_init__(self) -> None:
        self.type = "server_tool_use"


@dataclass(slots=True)
class Citation(AgentEvent):
    """A source the model cited in its answer."""

    url: str
    title: str | None = None

    def __post_init__(self) -> None:
        self.type = "citation"


@dataclass(slots=True)
class ToolStart(AgentEvent):
    """A local tool is about to run."""

    id: str
    name: str
    input: dict[str, Any]

    def __post_init__(self) -> None:
        self.type = "tool_start"


@dataclass(slots=True)
class ToolEnd(AgentEvent):
    """A local tool finished. `output` is truncated for display only."""

    id: str
    name: str
    output: str
    is_error: bool = False

    def __post_init__(self) -> None:
        self.type = "tool_end"


@dataclass(slots=True)
class ApprovalDenied(AgentEvent):
    """The caller refused a tool call that required approval."""

    name: str
    reason: str

    def __post_init__(self) -> None:
        self.type = "approval_denied"


@dataclass(slots=True)
class Compacted(AgentEvent):
    """The server summarised earlier conversation to stay inside the context window."""

    summary: str | None = None

    def __post_init__(self) -> None:
        self.type = "compacted"


@dataclass(slots=True)
class TurnEnd(AgentEvent):
    """The agent finished responding to one user message."""

    stop_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    steps: int = 0

    def __post_init__(self) -> None:
        self.type = "turn_end"


@dataclass(slots=True)
class AgentError(AgentEvent):
    """Something went wrong, or the model declined the request."""

    message: str
    kind: Literal["refusal", "api", "limit", "internal"] = "internal"
    detail: str | None = None

    def __post_init__(self) -> None:
        self.type = "error"
