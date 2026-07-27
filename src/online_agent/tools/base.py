"""Local tool protocol.

Local tools run in this process. Everything the model puts in `input` is
untrusted — validate before acting on it.
"""

from __future__ import annotations

import abc
from typing import Any


class ToolError(Exception):
    """Raised by a tool when the call is invalid or fails.

    The message is returned to the model as an error tool_result, so write it
    for the model: say what went wrong and what a valid call looks like.
    """


class Tool(abc.ABC):
    """A tool the agent can call locally."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return its result as text."""

    def definition(self) -> dict[str, Any]:
        """The tool definition sent to the Messages API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def obj_schema(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    """Build a JSON Schema object with the fields the API expects."""
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
