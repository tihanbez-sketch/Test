# online-agent

An advanced, web-connected agent built on the Claude API. It searches and reads
the live web, keeps a workspace of files, carries memory across sessions, and
streams everything it does — from a terminal REPL or an HTTP API.

Built on **Claude Opus 5** with adaptive thinking, server-side web search and
fetch, prompt caching, and server-side history compaction.

## What it does

| Capability | How |
| --- | --- |
| Live web search | Anthropic-hosted `web_search_20260209` (dynamic filtering) |
| Reads pages it finds | Anthropic-hosted `web_fetch_20260209` with citations on |
| Cites sources | Citations are extracted from the response and surfaced as events |
| Works in files | Sandboxed `read_file` / `write_file` / `list_files` in a workspace dir |
| Remembers across runs | `remember` / `recall` / `forget` backed by a JSON store |
| Survives long sessions | Server-side compaction summarises old turns automatically |
| Streams its work | Thinking, tool calls, text, and citations arrive as they happen |
| Asks before acting | Configurable approval gate on destructive tools |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # then set ANTHROPIC_API_KEY
```

Credentials resolve the standard way: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`,
or an `ant auth login` profile. If you use a profile, leave the key unset — a set
(even empty) `ANTHROPIC_API_KEY` shadows it.

## Use it

### Terminal

```bash
online-agent                                   # interactive session
online-agent "What shipped in Python 3.13?"    # one-shot
online-agent --effort medium --no-web "..."    # cheaper, offline
online-agent --yes "Draft a brief on X"        # skip approval prompts
```

In a session: `/help`, `/reset`, `/tools`, `/save <path>`, `/load <path>`,
`/workspace`, `/exit`.

### HTTP

```bash
uvicorn --factory online_agent.server:app --port 8000
```

```bash
# Non-streaming — returns the finished answer plus citations and usage.
curl -sX POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message": "What is the latest Claude model?"}' | jq

# Streaming — server-sent events, one per agent event.
curl -NX POST localhost:8000/chat/stream \
  -H 'content-type: application/json' \
  -d '{"message": "Research X and write me a summary"}'
```

Pass `session_id` (returned by both endpoints) on later requests to continue a
conversation. `DELETE /sessions/{id}` drops one; idle sessions expire after an
hour. `GET /health` reports model and session count.

SSE event names match the event types below, plus a terminal `done`.

### Library

```python
import asyncio
from online_agent import Agent, AgentConfig
from online_agent.events import TextDelta

async def main():
    agent = Agent(AgentConfig(effort="high"))
    async for event in agent.run("Summarise this week in AI, with sources."):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)

asyncio.run(main())
```

## Configuration

Environment variables (all optional — see `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | API key, unless you use an `ant` profile |
| `AGENT_MODEL` | `claude-opus-5` | Model id |
| `AGENT_EFFORT` | `xhigh` | `low` / `medium` / `high` / `xhigh` / `max` |
| `AGENT_MAX_TOKENS` | `64000` | Output ceiling per turn (thinking included) |
| `AGENT_WORKSPACE` | `./workspace` | Directory the file tools are confined to |
| `AGENT_MAX_STEPS` | `24` | Assistant turns per user message |
| `AGENT_WEB_SEARCH` / `AGENT_WEB_FETCH` | `true` | Toggle the hosted web tools |
| `AGENT_ALLOWED_DOMAINS` | — | Comma-separated allowlist for web tools |
| `AGENT_BLOCKED_DOMAINS` | — | Comma-separated blocklist (ignored if an allowlist is set) |
| `AGENT_THINKING_DISPLAY` | `summarized` | `summarized` shows reasoning; `omitted` hides it |
| `AGENT_COMPACTION` | `true` | Server-side history compaction |
| `AGENT_REFUSAL_FALLBACKS` | `true` | Re-serve policy declines on a fallback model |

`effort` is the main cost/latency dial. `xhigh` suits agentic research; `medium`
is a good default for routine questions and materially cheaper.

## Events

`Agent.run()` yields these, in the order things happen:

`thinking_delta` · `text_delta` · `server_tool_use` · `tool_start` · `tool_end` ·
`approval_denied` · `citation` · `compacted` · `error` · `turn_end`

`turn_end` carries `stop_reason`, step count, and token usage (including cache
reads). `error` carries a `kind` of `refusal`, `api`, `limit`, or `internal`.

## Design notes

**Why a hand-written loop instead of the SDK tool runner.** The runner is the
right default for a plain custom-tool agent, but it does not resume
`stop_reason: "pause_turn"` in Python — a long web-search turn would end
silently truncated. The loop here also needs to emit per-event streams to two
different frontends and to deny a tool call with an explanatory error instead of
executing it. See the module docstring in `agent.py`.

**Graceful feature degradation.** Compaction and refusal fallbacks are beta
features. If the API rejects either, the agent disables that one feature and
retries the same request once, rather than failing the turn.

**Prompt caching.** The system prompt carries a `cache_control` breakpoint.
Because tools render before system, one breakpoint caches the tool definitions
and system prompt together; `turn_end.cache_read_tokens` shows it working. Keep
the system prompt and tool set stable across requests or the cache is lost.

## Security

- **File tools are confined to the workspace.** Every model-supplied path is
  resolved and checked against the workspace root; `..`, absolute paths, and
  symlinks pointing outside are all rejected.
- **`write_file` requires approval by default.** The CLI prompts; the HTTP
  server auto-approves, so change `approval_required` before exposing it to
  untrusted input.
- **Memory is plain JSON.** Never put credentials in it — entries are replayed
  into every future session's context.
- **Web content is untrusted.** Pages the agent fetches can contain text that
  tries to redirect it. The approval gate is the backstop; keep it on for
  anything with side effects.

## Tests

```bash
pytest
```

The suite runs entirely against a scripted fake client — no network, no API key.
It covers path confinement, memory persistence, tool dispatch and error
handling, the tool-result round trip, `pause_turn` resumption, refusal handling,
step caps, beta degradation, and session save/load.
