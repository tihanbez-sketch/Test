"""Fakes for the Messages API streaming surface.

The agent only touches `client.beta.messages.stream(...)`, so these fakes model
exactly that: an async context manager that is async-iterable over raw stream
events and exposes `get_final_message()`.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


class Block:
    """A response content block (text, thinking, tool_use, ...)."""

    def __init__(self, type: str, **fields: Any) -> None:
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        data = {"type": self.type, **{k: v for k, v in vars(self).items() if k != "type"}}
        return {k: v for k, v in data.items() if v is not None} if exclude_none else data


class Usage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class StopDetails:
    def __init__(self, category: str | None = None, explanation: str | None = None):
        self.category = category
        self.explanation = explanation


class Message:
    def __init__(
        self,
        content: list[Block],
        stop_reason: str = "end_turn",
        usage: Usage | None = None,
        stop_details: StopDetails | None = None,
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or Usage(10, 5, 0)
        self.stop_details = stop_details


class Delta:
    def __init__(self, type: str, **fields: Any) -> None:
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


class StreamEvent:
    def __init__(self, type: str, **fields: Any) -> None:
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


def text_events(text: str) -> list[StreamEvent]:
    return [StreamEvent("content_block_delta", delta=Delta("text_delta", text=text))]


class Turn:
    """One scripted assistant turn: the events to stream, then the final message."""

    def __init__(self, message: Message, events: list[StreamEvent] | None = None) -> None:
        self.message = message
        self.events = events or []


class FakeStream:
    def __init__(self, turn: Turn) -> None:
        self._turn = turn

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def __aiter__(self):
        for event in self._turn.events:
            yield event

    async def get_final_message(self) -> Message:
        return self._turn.message


class FakeMessages:
    def __init__(self, turns: list[Turn | Exception]) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if not self.turns:
            raise AssertionError("the agent made more requests than the test scripted")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return FakeStream(turn)


class FakeClient:
    """Stands in for anthropic.AsyncAnthropic."""

    def __init__(self, turns: list[Turn | Exception]) -> None:
        self.messages = FakeMessages(turns)
        self.beta = type("Beta", (), {"messages": self.messages})()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


def bad_request(message: str) -> Exception:
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


@pytest.fixture()
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws
