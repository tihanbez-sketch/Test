"""The agent loop.

This is a hand-written agentic loop rather than the SDK's `tool_runner`. The
runner is the right default for a plain custom-tool agent, but this agent needs
three things it does not expose:

* `pause_turn` resumption. Long server-tool turns (web search / fetch) stop with
  `stop_reason: "pause_turn"`, and the Python runner exits instead of resuming.
* Per-event streaming out to two different frontends (terminal and SSE).
* An approval gate that can deny a call and hand the model an explanatory error
  rather than executing it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import anthropic

from .config import AgentConfig
from .events import (
    AgentError,
    AgentEvent,
    ApprovalDenied,
    Citation,
    Compacted,
    ServerToolUse,
    TextDelta,
    ThinkingDelta,
    ToolEnd,
    ToolStart,
    TurnEnd,
)
from .tools.registry import ToolRegistry

# Beta flags for the optional features. Each is dropped independently if the
# API rejects it, so the agent still runs on accounts without the beta.
COMPACTION_BETA = "compact-2026-01-12"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Approval callback: return True to allow, or False / a reason string to deny.
Approver = Callable[[str, dict[str, Any]], Awaitable[bool | str]]

TOOL_OUTPUT_PREVIEW = 2_000


async def allow_all(name: str, tool_input: dict[str, Any]) -> bool:
    """Default approver — allows everything. Pass your own to gate writes."""
    return True


class Agent:
    """A stateful, web-connected agent over the Claude Messages API."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        registry: ToolRegistry | None = None,
        client: Any | None = None,
        approver: Approver | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.registry = registry or ToolRegistry.default(self.config)
        # A bare constructor resolves credentials from the environment:
        # ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile.
        self.client = client or anthropic.AsyncAnthropic()
        self.approver = approver or allow_all
        self.messages: list[dict[str, Any]] = []

        # Disabled at runtime if the API rejects them (see _handle_bad_request).
        self._compaction = self.config.compaction
        self._fallbacks = self.config.refusal_fallbacks

    # ---------------------------------------------------------------- state

    def reset(self) -> None:
        """Drop conversation history. Memory and workspace files survive."""
        self.messages = []

    def save(self, path: Path) -> None:
        """Persist the conversation so a later process can resume it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.config.model, "messages": self._serialisable_messages()}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Restore a conversation saved by :meth:`save`."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self.messages = data.get("messages", [])

    def _serialisable_messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in self.messages:
            content = message["content"]
            if isinstance(content, str):
                out.append({"role": message["role"], "content": content})
                continue
            blocks = [
                block if isinstance(block, dict) else block.model_dump(exclude_none=True)
                for block in content
            ]
            out.append({"role": message["role"], "content": blocks})
        return out

    # -------------------------------------------------------------- request

    def _tools(self) -> list[dict[str, Any]]:
        return self.config.server_tools() + self.registry.definitions()

    def _request_kwargs(self) -> dict[str, Any]:
        cfg = self.config
        betas: list[str] = []
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            # cache_control on the system prompt caches tools + system together;
            # tools render before system, so one breakpoint covers both.
            "system": [
                {
                    "type": "text",
                    "text": cfg.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": self.messages,
            "tools": self._tools(),
            "thinking": {"type": "adaptive", "display": cfg.thinking_display},
            "output_config": {"effort": cfg.effort},
        }
        if self._compaction:
            betas.append(COMPACTION_BETA)
            kwargs["context_management"] = {"edits": [{"type": "compact_20260112"}]}
        if self._fallbacks:
            betas.append(FALLBACK_BETA)
            # "default" routes by refusal category instead of pinning a model,
            # so this needs no maintenance when fallback targets change.
            kwargs["fallbacks"] = "default"
        if betas:
            kwargs["betas"] = betas
        return kwargs

    def _handle_bad_request(self, exc: Exception) -> bool:
        """Disable whichever optional feature the API rejected. True if retryable."""
        message = str(exc).lower()
        disabled = False
        if self._compaction and ("compact" in message or "context_management" in message):
            self._compaction = False
            disabled = True
        if self._fallbacks and "fallback" in message:
            self._fallbacks = False
            disabled = True
        if not disabled and (self._compaction or self._fallbacks):
            # Unattributable 400 — drop every optional feature once and retry
            # with a plain request before giving up.
            self._compaction = False
            self._fallbacks = False
            disabled = True
        return disabled

    # ----------------------------------------------------------------- loop

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Respond to one user message, yielding events as work happens."""
        self.messages.append({"role": "user", "content": user_input})

        totals = {"input": 0, "output": 0, "cache_read": 0}
        pause_resumes = 0
        stop_reason: str | None = None
        steps = 0

        for step in range(self.config.max_steps):
            steps = step + 1
            try:
                final = None
                async for event in self._stream_turn():
                    if isinstance(event, _FinalMessage):
                        final = event.message
                    else:
                        yield event
            except anthropic.APIStatusError as exc:
                yield AgentError(
                    message=_friendly_api_error(exc),
                    kind="api",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return
            except anthropic.APIConnectionError as exc:
                yield AgentError(
                    message="Could not reach the Claude API. Check your connection and retry.",
                    kind="api",
                    detail=str(exc),
                )
                return

            if final is None:  # pragma: no cover - stream always yields a final message
                yield AgentError(message="The model returned no response.", kind="internal")
                return

            usage = getattr(final, "usage", None)
            if usage is not None:
                totals["input"] += getattr(usage, "input_tokens", 0) or 0
                totals["output"] += getattr(usage, "output_tokens", 0) or 0
                totals["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0

            content = list(final.content)
            for block in content:
                if _block_type(block) == "compaction":
                    yield Compacted(summary=getattr(block, "content", None))

            self.messages.append(
                {"role": "assistant", "content": _echo_content(content)}
            )
            for citation in _citations(content):
                yield citation

            stop_reason = getattr(final, "stop_reason", None)

            if stop_reason == "refusal":
                yield AgentError(
                    message=_refusal_message(final),
                    kind="refusal",
                    detail=_refusal_detail(final),
                )
                break

            if stop_reason == "pause_turn":
                # A server tool hit its per-turn iteration limit. Re-sending the
                # conversation (the paused assistant turn is already appended)
                # resumes it server-side; do not add a "continue" message.
                pause_resumes += 1
                if pause_resumes > self.config.max_pause_resumes:
                    yield AgentError(
                        message=(
                            f"Gave up after resuming a paused turn "
                            f"{self.config.max_pause_resumes} times."
                        ),
                        kind="limit",
                    )
                    break
                continue

            if stop_reason == "max_tokens":
                yield AgentError(
                    message=(
                        "Response hit the max_tokens ceiling and was cut off. "
                        "Raise AGENT_MAX_TOKENS or narrow the request."
                    ),
                    kind="limit",
                )
                break

            tool_uses = [b for b in content if _block_type(b) == "tool_use"]
            if not tool_uses:
                break

            results: list[dict[str, Any]] = []
            for block in tool_uses:
                async for event in self._run_tool(block, results):
                    yield event

            # All results for one assistant turn go back in a single user
            # message — splitting them trains the model out of parallel calls.
            self.messages.append({"role": "user", "content": results})
        else:
            yield AgentError(
                message=f"Stopped after {self.config.max_steps} steps without finishing.",
                kind="limit",
            )

        yield TurnEnd(
            stop_reason=stop_reason,
            input_tokens=totals["input"],
            output_tokens=totals["output"],
            cache_read_tokens=totals["cache_read"],
            steps=steps,
        )

    async def _stream_turn(self) -> AsyncIterator[AgentEvent | "_FinalMessage"]:
        """Stream one assistant turn, retrying once if an optional beta is rejected."""
        attempt = 0
        while True:
            try:
                async with self.client.beta.messages.stream(
                    **self._request_kwargs()
                ) as stream:
                    async for event in stream:
                        translated = _translate(event)
                        if translated is not None:
                            yield translated
                    yield _FinalMessage(await stream.get_final_message())
                return
            except anthropic.BadRequestError as exc:
                attempt += 1
                if attempt > 2 or not self._handle_bad_request(exc):
                    raise

    async def _run_tool(
        self, block: Any, results: list[dict[str, Any]]
    ) -> AsyncIterator[AgentEvent]:
        name = block.name
        tool_input = block.input if isinstance(block.input, dict) else {}
        yield ToolStart(id=block.id, name=name, input=tool_input)

        if name in self.config.approval_required:
            verdict = await self.approver(name, tool_input)
            if verdict is not True:
                reason = (
                    verdict
                    if isinstance(verdict, str) and verdict.strip()
                    else "The user declined this action."
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"{reason} Do not retry it; continue without it "
                        "or propose an alternative.",
                        "is_error": True,
                    }
                )
                yield ApprovalDenied(name=name, reason=reason)
                return

        output, is_error = await self.registry.call(name, tool_input)
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
                "is_error": is_error,
            }
        )
        preview = output if len(output) <= TOOL_OUTPUT_PREVIEW else (
            output[:TOOL_OUTPUT_PREVIEW] + f"... [{len(output)} chars total]"
        )
        yield ToolEnd(id=block.id, name=name, output=preview, is_error=is_error)


class _FinalMessage:
    """Internal marker so _stream_turn can hand back the accumulated message."""

    __slots__ = ("message",)

    def __init__(self, message: Any) -> None:
        self.message = message


# --------------------------------------------------------------- translation


def _block_type(block: Any) -> str | None:
    return getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")


def _translate(event: Any) -> AgentEvent | None:
    """Map a raw stream event onto an AgentEvent, or None to ignore it."""
    kind = getattr(event, "type", None)

    if kind == "content_block_delta":
        delta = event.delta
        delta_type = getattr(delta, "type", None)
        if delta_type == "text_delta":
            return TextDelta(text=delta.text)
        if delta_type == "thinking_delta":
            return ThinkingDelta(text=delta.thinking)
        return None

    if kind == "content_block_start":
        block = event.content_block
        if _block_type(block) == "server_tool_use":
            # Input streams in separately, so only the tool name is known here.
            return ServerToolUse(name=getattr(block, "name", "server_tool"))
        return None

    return None


def _citations(content: list[Any]) -> list[Citation]:
    """Pull deduplicated citations out of the assistant's text blocks."""
    seen: set[str] = set()
    out: list[Citation] = []
    for block in content:
        if _block_type(block) != "text":
            continue
        for citation in getattr(block, "citations", None) or []:
            url = getattr(citation, "url", None)
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(Citation(url=url, title=getattr(citation, "title", None)))
    return out


def _echo_content(content: list[Any]) -> list[Any]:
    """Prepare assistant content for the next request.

    Normally the response content is echoed back verbatim — thinking blocks and
    compaction blocks must survive unmodified. The exception is a mid-output
    refusal fallback: blocks before the final `fallback` marker came from the
    model that declined, and thinking / tool_use blocks from that partial turn
    must not be replayed.
    """
    boundary = -1
    for index, block in enumerate(content):
        if _block_type(block) == "fallback":
            boundary = index
    if boundary < 0:
        return content

    dropped = {"thinking", "redacted_thinking", "tool_use"}
    return [
        block
        for index, block in enumerate(content)
        if index > boundary or _block_type(block) not in dropped
    ]


def _refusal_message(final: Any) -> str:
    details = getattr(final, "stop_details", None)
    category = getattr(details, "category", None) if details else None
    if category:
        return (
            f"The model declined this request ({category}). Rephrasing it, or "
            "narrowing it to the specific task you need, may help."
        )
    return "The model declined this request."


def _refusal_detail(final: Any) -> str | None:
    details = getattr(final, "stop_details", None)
    return getattr(details, "explanation", None) if details else None


def _friendly_api_error(exc: anthropic.APIStatusError) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return (
            "Authentication failed. Set ANTHROPIC_API_KEY, or run `ant auth login` "
            "and leave it unset."
        )
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = exc.response.headers.get("retry-after") if exc.response else None
        suffix = f" Retry after {retry_after}s." if retry_after else ""
        return f"Rate limited by the API.{suffix}"
    if isinstance(exc, anthropic.NotFoundError):
        return "Model or endpoint not found. Check AGENT_MODEL."
    if exc.status_code >= 500:
        return f"The API returned a server error ({exc.status_code}). Retry shortly."
    return f"API error {exc.status_code}: {exc.message}"
