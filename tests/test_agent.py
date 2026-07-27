"""Tests for the agent loop, driven by a scripted fake client."""

from __future__ import annotations

import pytest
from conftest import (
    Block,
    Delta,
    FakeClient,
    Message,
    StopDetails,
    StreamEvent,
    Turn,
    Usage,
    bad_request,
    text_events,
)

from online_agent.agent import Agent
from online_agent.config import AgentConfig
from online_agent.events import (
    AgentError,
    ApprovalDenied,
    Citation,
    ServerToolUse,
    TextDelta,
    ThinkingDelta,
    ToolEnd,
    ToolStart,
    TurnEnd,
)


def make_agent(workspace, turns, **overrides):
    config = AgentConfig(workspace=workspace, **overrides)
    return Agent(config, client=FakeClient(turns))


async def collect(agent, message="hello"):
    return [event async for event in agent.run(message)]


def of_type(events, cls):
    return [e for e in events if isinstance(e, cls)]


# ------------------------------------------------------------- plain answers


async def test_streams_text_and_reports_usage(workspace):
    agent = make_agent(
        workspace,
        [
            Turn(
                Message([Block("text", text="Paris.")], usage=Usage(120, 30, 90)),
                text_events("Paris."),
            )
        ],
    )

    events = await collect(agent, "Capital of France?")

    assert "".join(e.text for e in of_type(events, TextDelta)) == "Paris."
    end = of_type(events, TurnEnd)[0]
    assert (end.stop_reason, end.steps) == ("end_turn", 1)
    assert (end.input_tokens, end.output_tokens, end.cache_read_tokens) == (120, 30, 90)


async def test_thinking_deltas_are_separated_from_text(workspace):
    agent = make_agent(
        workspace,
        [
            Turn(
                Message([Block("text", text="42")]),
                [
                    StreamEvent(
                        "content_block_delta",
                        delta=Delta("thinking_delta", thinking="weighing options"),
                    ),
                    *text_events("42"),
                ],
            )
        ],
    )

    events = await collect(agent)

    assert [e.text for e in of_type(events, ThinkingDelta)] == ["weighing options"]
    assert [e.text for e in of_type(events, TextDelta)] == ["42"]


async def test_conversation_history_accumulates(workspace):
    agent = make_agent(
        workspace,
        [
            Turn(Message([Block("text", text="one")])),
            Turn(Message([Block("text", text="two")])),
        ],
    )

    await collect(agent, "first")
    await collect(agent, "second")

    assert [m["role"] for m in agent.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    assert agent.messages[2]["content"] == "second"


# ------------------------------------------------------------------- tooling


async def test_tool_call_executes_and_feeds_the_result_back(workspace):
    tool_use = Block("tool_use", id="toolu_1", name="write_file",
                     input={"path": "note.md", "content": "hi"})
    agent = make_agent(
        workspace,
        [
            Turn(Message([tool_use], stop_reason="tool_use")),
            Turn(Message([Block("text", text="Done.")]), text_events("Done.")),
        ],
        approval_required=frozenset(),
    )

    events = await collect(agent, "write a note")

    assert of_type(events, ToolStart)[0].name == "write_file"
    assert of_type(events, ToolEnd)[0].is_error is False
    assert (workspace / "note.md").read_text(encoding="utf-8") == "hi"

    results_turn = agent.messages[2]
    assert results_turn["role"] == "user"
    assert results_turn["content"][0]["tool_use_id"] == "toolu_1"
    assert of_type(events, TurnEnd)[0].steps == 2


async def test_parallel_tool_results_go_back_in_one_user_message(workspace):
    calls = [
        Block("tool_use", id="a", name="write_file", input={"path": "a.md", "content": "a"}),
        Block("tool_use", id="b", name="write_file", input={"path": "b.md", "content": "b"}),
    ]
    agent = make_agent(
        workspace,
        [
            Turn(Message(calls, stop_reason="tool_use")),
            Turn(Message([Block("text", text="Both written.")])),
        ],
        approval_required=frozenset(),
    )

    await collect(agent, "write two notes")

    results_turn = agent.messages[2]
    assert len(results_turn["content"]) == 2
    assert [r["tool_use_id"] for r in results_turn["content"]] == ["a", "b"]


async def test_failing_tool_returns_an_error_result_rather_than_crashing(workspace):
    tool_use = Block("tool_use", id="t", name="read_file", input={"path": "../secrets"})
    agent = make_agent(
        workspace,
        [
            Turn(Message([tool_use], stop_reason="tool_use")),
            Turn(Message([Block("text", text="I could not read that.")])),
        ],
        approval_required=frozenset(),
    )

    events = await collect(agent, "read outside")

    assert of_type(events, ToolEnd)[0].is_error is True
    assert agent.messages[2]["content"][0]["is_error"] is True
    assert "outside the workspace" in agent.messages[2]["content"][0]["content"]


async def test_denied_tool_is_not_executed(workspace):
    tool_use = Block("tool_use", id="t", name="write_file",
                     input={"path": "note.md", "content": "hi"})

    async def deny(name, tool_input):
        return "Not allowed in this environment."

    config = AgentConfig(workspace=workspace, approval_required=frozenset({"write_file"}))
    agent = Agent(
        config,
        client=FakeClient(
            [
                Turn(Message([tool_use], stop_reason="tool_use")),
                Turn(Message([Block("text", text="Understood.")])),
            ]
        ),
        approver=deny,
    )

    events = await collect(agent, "write a note")

    assert not (workspace / "note.md").exists()
    assert of_type(events, ApprovalDenied)[0].reason == "Not allowed in this environment."
    result = agent.messages[2]["content"][0]
    assert result["is_error"] is True
    assert "Not allowed in this environment." in result["content"]


async def test_server_tool_use_is_surfaced_but_not_executed(workspace):
    agent = make_agent(
        workspace,
        [
            Turn(
                Message([Block("text", text="Found it.")]),
                [
                    StreamEvent(
                        "content_block_start",
                        content_block=Block("server_tool_use", name="web_search"),
                    ),
                    *text_events("Found it."),
                ],
            )
        ],
    )

    events = await collect(agent, "search for something")

    assert of_type(events, ServerToolUse)[0].name == "web_search"
    assert not of_type(events, ToolStart)


async def test_citations_are_deduplicated(workspace):
    cited = Block(
        "text",
        text="Per the docs.",
        citations=[
            Block("web_search_result_location", url="https://example.com/a", title="A"),
            Block("web_search_result_location", url="https://example.com/a", title="A"),
            Block("web_search_result_location", url="https://example.com/b", title="B"),
        ],
    )
    agent = make_agent(workspace, [Turn(Message([cited]))])

    citations = of_type(await collect(agent), Citation)

    assert [c.url for c in citations] == ["https://example.com/a", "https://example.com/b"]


# --------------------------------------------------------------- stop reasons


async def test_pause_turn_is_resumed_without_an_extra_user_message(workspace):
    agent = make_agent(
        workspace,
        [
            Turn(Message([Block("text", text="Searching...")], stop_reason="pause_turn")),
            Turn(Message([Block("text", text="Here it is.")])),
        ],
    )

    events = await collect(agent, "research something")

    assert of_type(events, TurnEnd)[0].stop_reason == "end_turn"
    # user, assistant (paused), assistant (resumed) — no synthetic "continue".
    assert [m["role"] for m in agent.messages] == ["user", "assistant", "assistant"]


async def test_pause_turn_resumption_is_bounded(workspace):
    paused = [
        Turn(Message([Block("text", text="...")], stop_reason="pause_turn"))
        for _ in range(4)
    ]
    agent = make_agent(workspace, paused, max_pause_resumes=2)

    errors = of_type(await collect(agent), AgentError)

    assert errors[0].kind == "limit"
    assert "resuming a paused turn" in errors[0].message


async def test_refusal_stops_the_turn_and_reports_the_category(workspace):
    agent = make_agent(
        workspace,
        [
            Turn(
                Message(
                    [],
                    stop_reason="refusal",
                    stop_details=StopDetails(category="cyber", explanation="policy"),
                )
            )
        ],
    )

    errors = of_type(await collect(agent), AgentError)

    assert errors[0].kind == "refusal"
    assert "cyber" in errors[0].message
    assert errors[0].detail == "policy"


async def test_max_tokens_is_reported_as_a_limit(workspace):
    agent = make_agent(
        workspace, [Turn(Message([Block("text", text="tru")], stop_reason="max_tokens"))]
    )

    errors = of_type(await collect(agent), AgentError)

    assert errors[0].kind == "limit"
    assert "max_tokens" in errors[0].message


async def test_step_cap_stops_a_runaway_loop(workspace):
    tool_use = Block("tool_use", id="t", name="list_files", input={})
    agent = make_agent(
        workspace,
        [Turn(Message([tool_use], stop_reason="tool_use")) for _ in range(5)],
        max_steps=3,
        approval_required=frozenset(),
    )

    events = await collect(agent, "loop forever")

    errors = of_type(events, AgentError)
    assert errors[0].kind == "limit"
    assert "3 steps" in errors[0].message
    assert of_type(events, TurnEnd)[0].steps == 3


# ------------------------------------------------------------------ requests


async def test_request_carries_web_tools_thinking_effort_and_cache_control(workspace):
    agent = make_agent(workspace, [Turn(Message([Block("text", text="ok")]))])

    await collect(agent)

    request = agent.client.calls[0]
    tool_types = {t.get("type") for t in request["tools"]}
    assert {"web_search_20260209", "web_fetch_20260209"} <= tool_types
    assert {"read_file", "write_file"} <= {t.get("name") for t in request["tools"]}
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert request["output_config"] == {"effort": "xhigh"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["fallbacks"] == "default"
    assert "compact-2026-01-12" in request["betas"]


async def test_web_tools_can_be_disabled(workspace):
    agent = make_agent(
        workspace,
        [Turn(Message([Block("text", text="ok")]))],
        web_search=False,
        web_fetch=False,
    )

    await collect(agent)

    tool_types = {t.get("type") for t in agent.client.calls[0]["tools"]}
    assert "web_search_20260209" not in tool_types


async def test_rejected_beta_is_dropped_and_the_request_retried(workspace):
    agent = make_agent(
        workspace,
        [
            bad_request("compact-2026-01-12 is not supported for this account"),
            Turn(Message([Block("text", text="ok")]), text_events("ok")),
        ],
    )

    events = await collect(agent)

    assert [e.text for e in of_type(events, TextDelta)] == ["ok"]
    assert "context_management" not in agent.client.calls[1]
    assert "compact-2026-01-12" not in agent.client.calls[1].get("betas", [])


async def test_persistent_bad_request_surfaces_as_an_error(workspace):
    agent = make_agent(
        workspace,
        [bad_request("bad") for _ in range(3)],
    )

    errors = of_type(await collect(agent), AgentError)

    assert errors[0].kind == "api"


# ------------------------------------------------------------------- session


async def test_conversation_round_trips_through_disk(workspace, tmp_path):
    agent = make_agent(workspace, [Turn(Message([Block("text", text="remembered")]))])
    await collect(agent, "note this")

    path = tmp_path / "session.json"
    agent.save(path)

    restored = make_agent(workspace, [])
    restored.load(path)

    assert [m["role"] for m in restored.messages] == ["user", "assistant"]
    assert restored.messages[1]["content"][0]["text"] == "remembered"


async def test_reset_clears_history(workspace):
    agent = make_agent(workspace, [Turn(Message([Block("text", text="ok")]))])
    await collect(agent)

    agent.reset()

    assert agent.messages == []


def test_invalid_effort_is_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_EFFORT", "turbo")
    with pytest.raises(ValueError, match="AGENT_EFFORT"):
        AgentConfig.from_env()
