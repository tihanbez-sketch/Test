"""Interactive terminal frontend."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .agent import Agent
from .config import EFFORT_LEVELS, AgentConfig
from .events import (
    AgentError,
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

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

BANNER = f"""{BOLD}online-agent{RESET} — web-connected agent on the Claude API
{DIM}/help for commands, /exit to quit{RESET}
"""

HELP = f"""{BOLD}Commands{RESET}
  /help              show this
  /reset             clear conversation history (memory and files survive)
  /save <path>       write the conversation to disk
  /load <path>       restore a saved conversation
  /tools             list local tools
  /workspace         show the workspace path
  /exit              quit
"""


def _colour(enabled: bool) -> dict[str, str]:
    if enabled:
        return {
            "reset": RESET, "dim": DIM, "bold": BOLD,
            "blue": BLUE, "green": GREEN, "yellow": YELLOW, "red": RED,
        }
    return dict.fromkeys(
        ["reset", "dim", "bold", "blue", "green", "yellow", "red"], ""
    )


async def _confirm(name: str, tool_input: dict[str, Any]) -> bool | str:
    """Ask the operator before a gated tool runs."""
    detail = ", ".join(f"{k}={_short(v)}" for k, v in tool_input.items())
    prompt = f"\n{YELLOW}Allow {name}({detail})? [y/N] {RESET}"
    answer = await asyncio.to_thread(input, prompt)
    if answer.strip().lower() in {"y", "yes"}:
        return True
    return "The user declined to run this tool."


def _short(value: Any, limit: int = 60) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


async def _converse(agent: Agent, message: str, colour: dict[str, str]) -> None:
    """Run one turn and render its events."""
    thinking_open = False
    text_started = False

    async for event in agent.run(message):
        if isinstance(event, ThinkingDelta):
            if not event.text:
                continue
            if not thinking_open:
                sys.stdout.write(f"\n{colour['dim']}thinking: ")
                thinking_open = True
            sys.stdout.write(event.text)
            sys.stdout.flush()
            continue

        if thinking_open:
            sys.stdout.write(colour["reset"] + "\n")
            thinking_open = False

        if isinstance(event, TextDelta):
            if not text_started:
                sys.stdout.write("\n")
                text_started = True
            sys.stdout.write(event.text)
            sys.stdout.flush()
            continue

        # Anything that is not assistant text starts on its own line.
        if text_started:
            sys.stdout.write("\n")
            text_started = False

        if isinstance(event, ServerToolUse):
            print(f"{colour['blue']}▸ {event.name}{colour['reset']}")
        elif isinstance(event, ToolStart):
            args = ", ".join(f"{k}={_short(v, 40)}" for k, v in event.input.items())
            print(f"{colour['blue']}▸ {event.name}({args}){colour['reset']}")
        elif isinstance(event, ToolEnd):
            mark = colour["red"] + "✗" if event.is_error else colour["green"] + "✓"
            print(f"  {mark} {_short(event.output, 100)}{colour['reset']}")
        elif isinstance(event, ApprovalDenied):
            print(f"{colour['yellow']}  skipped {event.name}{colour['reset']}")
        elif isinstance(event, Compacted):
            print(f"{colour['dim']}  [history compacted]{colour['reset']}")
        elif isinstance(event, Citation):
            title = f" — {event.title}" if event.title else ""
            print(f"{colour['dim']}  source: {event.url}{title}{colour['reset']}")
        elif isinstance(event, AgentError):
            print(f"{colour['red']}{event.message}{colour['reset']}")
            if event.detail:
                print(f"{colour['dim']}{event.detail}{colour['reset']}")
        elif isinstance(event, TurnEnd):
            print(
                f"{colour['dim']}[{event.steps} step(s) · "
                f"{event.input_tokens:,} in / {event.output_tokens:,} out · "
                f"{event.cache_read_tokens:,} cached]{colour['reset']}"
            )


def _handle_command(agent: Agent, line: str, colour: dict[str, str]) -> bool:
    """Handle a /command. Returns False to exit the REPL."""
    parts = line.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in {"/exit", "/quit"}:
        return False
    if command == "/help":
        print(HELP)
    elif command == "/reset":
        agent.reset()
        print(f"{colour['dim']}History cleared.{colour['reset']}")
    elif command == "/tools":
        print("  " + "\n  ".join(agent.registry.names()))
    elif command == "/workspace":
        print(f"  {agent.config.workspace.resolve()}")
    elif command == "/save":
        if not argument:
            print(f"{colour['red']}Usage: /save <path>{colour['reset']}")
        else:
            agent.save(Path(argument))
            print(f"{colour['dim']}Saved to {argument}.{colour['reset']}")
    elif command == "/load":
        if not argument:
            print(f"{colour['red']}Usage: /load <path>{colour['reset']}")
        else:
            try:
                agent.load(Path(argument))
                print(f"{colour['dim']}Loaded {argument}.{colour['reset']}")
            except (OSError, ValueError) as exc:
                print(f"{colour['red']}Could not load {argument}: {exc}{colour['reset']}")
    else:
        print(f"{colour['red']}Unknown command {command}. /help for the list.{colour['reset']}")
    return True


async def _repl(agent: Agent, colour: dict[str, str]) -> None:
    print(BANNER if colour["bold"] else "online-agent\n")
    print(f"{colour['dim']}workspace: {agent.config.workspace.resolve()}{colour['reset']}\n")

    while True:
        try:
            line = await asyncio.to_thread(input, f"{colour['bold']}› {colour['reset']}")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            if not _handle_command(agent, line, colour):
                return
            continue

        try:
            await _converse(agent, line, colour)
        except KeyboardInterrupt:
            print(f"\n{colour['yellow']}Interrupted.{colour['reset']}")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="online-agent", description="A web-connected agent on the Claude API."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Run one prompt and exit. Omit for an interactive session.",
    )
    parser.add_argument("--model", help="Model id (default: claude-opus-5).")
    parser.add_argument("--effort", choices=EFFORT_LEVELS, help="Reasoning effort.")
    parser.add_argument("--workspace", type=Path, help="Directory the file tools use.")
    parser.add_argument(
        "--no-web", action="store_true", help="Disable web search and fetch."
    )
    parser.add_argument(
        "--yes", action="store_true", help="Auto-approve gated tools (no prompts)."
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    try:
        config = AgentConfig.from_env()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort
    if args.workspace:
        config.workspace = args.workspace
    if args.no_web:
        config.web_search = False
        config.web_fetch = False
    if args.yes:
        config.approval_required = frozenset()

    colour = _colour(not args.no_color and sys.stdout.isatty())
    agent = Agent(config, approver=None if args.yes else _confirm)

    try:
        if args.prompt:
            asyncio.run(_converse(agent, " ".join(args.prompt), colour))
            print()
        else:
            asyncio.run(_repl(agent, colour))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
