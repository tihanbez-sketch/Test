"""Configuration for the online agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Effort levels accepted by output_config.effort.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

DEFAULT_SYSTEM_PROMPT = """\
You are an online research and work agent. You have live web access and a small \
local workspace, and you are expected to use them rather than answering from memory.

How to work:
- When the answer depends on current information (recent events, prices, versions, \
release dates, anything the user flags as time-sensitive), search before answering. \
For open-ended research, start searching immediately instead of asking a scoping \
question, unless the request is genuinely ambiguous.
- Read the sources you cite. A search snippet is a pointer, not evidence — fetch the \
page when a claim carries weight.
- Cite what you used. Give the URL inline for every non-obvious factual claim, and say \
plainly when sources disagree or when you could not verify something.
- Use the workspace to write notes, drafts, and deliverables. Use memory to carry facts \
and preferences across sessions; check it before starting anything non-trivial.

How to communicate:
- Lead with the outcome. Your first sentence should answer what you found or what you \
did; supporting detail comes after.
- Keep responses focused and concise. Being readable matters more than being short: \
write complete sentences, spell terms out, and skip detail that would not change what \
the reader does next.
- Deliver what was asked at the scope intended. Make routine judgment calls yourself \
and check in only when different readings lead to materially different work.
- Report faithfully: if a fetch failed or a step was skipped, say so.
"""


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class AgentConfig:
    """Runtime settings for an :class:`~online_agent.agent.Agent`."""

    model: str = "claude-opus-5"
    max_tokens: int = 64_000
    effort: str = "xhigh"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Workspace the file tools are confined to. Nothing outside it is readable
    # or writable, no matter what path the model asks for.
    workspace: Path = field(default_factory=lambda: Path.cwd() / "workspace")

    # Loop bounds. max_steps caps assistant turns per user message; max_pause_resumes
    # caps how many times a server-tool `pause_turn` is resumed.
    max_steps: int = 24
    max_pause_resumes: int = 5

    # Server-side web tools.
    web_search: bool = True
    web_fetch: bool = True
    max_web_search_uses: int = 12
    max_web_fetch_uses: int = 12
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None

    # Optional API features. Each degrades independently if the account or model
    # does not have it — see Agent._request_kwargs.
    thinking_display: str = "summarized"  # "summarized" | "omitted"
    compaction: bool = True  # server-side summarisation of long histories
    refusal_fallbacks: bool = True  # re-serve policy declines on a fallback model

    # Every tool named here must be approved by the caller before it runs.
    approval_required: frozenset[str] = frozenset({"write_file"})

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Build a config from environment variables (see .env.example)."""
        workspace = Path(
            os.environ.get("AGENT_WORKSPACE", str(Path.cwd() / "workspace"))
        ).expanduser()

        effort = os.environ.get("AGENT_EFFORT", "xhigh").strip().lower()
        if effort not in EFFORT_LEVELS:
            raise ValueError(
                f"AGENT_EFFORT must be one of {', '.join(EFFORT_LEVELS)}; got {effort!r}"
            )

        def _domains(name: str) -> list[str] | None:
            raw = os.environ.get(name, "").strip()
            if not raw:
                return None
            return [d.strip() for d in raw.split(",") if d.strip()]

        return cls(
            model=os.environ.get("AGENT_MODEL", "claude-opus-5"),
            max_tokens=_env_int("AGENT_MAX_TOKENS", 64_000),
            effort=effort,
            workspace=workspace,
            max_steps=_env_int("AGENT_MAX_STEPS", 24),
            web_search=_env_flag("AGENT_WEB_SEARCH", True),
            web_fetch=_env_flag("AGENT_WEB_FETCH", True),
            allowed_domains=_domains("AGENT_ALLOWED_DOMAINS"),
            blocked_domains=_domains("AGENT_BLOCKED_DOMAINS"),
            thinking_display=os.environ.get("AGENT_THINKING_DISPLAY", "summarized"),
            compaction=_env_flag("AGENT_COMPACTION", True),
            refusal_fallbacks=_env_flag("AGENT_REFUSAL_FALLBACKS", True),
        )

    def server_tools(self) -> list[dict]:
        """Anthropic-hosted tool definitions. These are never executed locally."""
        tools: list[dict] = []
        if self.web_search:
            tool: dict = {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": self.max_web_search_uses,
            }
            if self.allowed_domains:
                tool["allowed_domains"] = self.allowed_domains
            elif self.blocked_domains:
                tool["blocked_domains"] = self.blocked_domains
            tools.append(tool)
        if self.web_fetch:
            tool = {
                "type": "web_fetch_20260209",
                "name": "web_fetch",
                "max_uses": self.max_web_fetch_uses,
                "citations": {"enabled": True},
            }
            if self.allowed_domains:
                tool["allowed_domains"] = self.allowed_domains
            elif self.blocked_domains:
                tool["blocked_domains"] = self.blocked_domains
            tools.append(tool)
        return tools
