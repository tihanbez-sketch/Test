"""Tests for the local tool set — path confinement, memory, and dispatch."""

from __future__ import annotations

import pytest

from online_agent.config import AgentConfig
from online_agent.tools import (
    ListFilesTool,
    MemoryStore,
    ReadFileTool,
    RecallTool,
    RememberTool,
    ToolError,
    ToolRegistry,
    WriteFileTool,
    resolve_in_workspace,
)
from online_agent.tools.memory import ForgetTool


@pytest.fixture()
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


# ------------------------------------------------------------ path confinement


@pytest.mark.parametrize(
    "path",
    [
        "../escape.txt",
        "notes/../../escape.txt",
        "/etc/passwd",
        "subdir/../../../etc/passwd",
    ],
)
def test_paths_outside_workspace_are_rejected(workspace, path):
    with pytest.raises(ToolError, match="outside the workspace"):
        resolve_in_workspace(workspace, path)


def test_symlink_escaping_the_workspace_is_rejected(workspace, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("classified", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)

    with pytest.raises(ToolError, match="outside the workspace"):
        resolve_in_workspace(workspace, "link.txt")


def test_relative_paths_resolve_inside_the_workspace(workspace):
    resolved = resolve_in_workspace(workspace, "notes/today.md")
    assert resolved == (workspace.resolve() / "notes" / "today.md")


def test_empty_path_is_rejected(workspace):
    with pytest.raises(ToolError):
        resolve_in_workspace(workspace, "   ")


# -------------------------------------------------------------------- files


def test_write_then_read_round_trips(workspace):
    write = WriteFileTool(workspace)
    read = ReadFileTool(workspace)

    result = write.run(path="notes/summary.md", content="# Findings\n")
    assert "notes/summary.md" in result
    assert read.run(path="notes/summary.md") == "# Findings\n"


def test_read_missing_file_explains_the_next_step(workspace):
    with pytest.raises(ToolError, match="list_files"):
        ReadFileTool(workspace).run(path="nope.md")


def test_list_files_reports_directories_and_sizes(workspace):
    WriteFileTool(workspace).run(path="a.txt", content="hello")
    WriteFileTool(workspace).run(path="sub/b.txt", content="world")

    listing = ListFilesTool(workspace).run(path=".")
    assert "sub/" in listing
    assert "a.txt (5 bytes)" in listing


def test_list_files_hides_dotfiles(workspace):
    (workspace / ".agent_memory.json").write_text("{}", encoding="utf-8")
    assert ListFilesTool(workspace).run(path=".") == ". is empty."


def test_oversized_write_is_rejected(workspace, monkeypatch):
    monkeypatch.setattr("online_agent.tools.files.MAX_WRITE_BYTES", 10)
    with pytest.raises(ToolError, match="over the"):
        WriteFileTool(workspace).run(path="big.txt", content="x" * 11)


# ------------------------------------------------------------------- memory


def test_memory_persists_across_store_instances(tmp_path):
    path = tmp_path / "memory.json"
    RememberTool(MemoryStore(path)).run(key="user.timezone", value="Europe/Berlin")

    # A fresh store reading the same file is what a new session looks like.
    assert RecallTool(MemoryStore(path)).run(key="user.timezone") == "Europe/Berlin"


def test_recall_without_key_lists_previews(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    RememberTool(store).run(key="b", value="second")
    RememberTool(store).run(key="a", value="first")

    assert RecallTool(store).run() == "a: first\nb: second"


def test_recall_unknown_key_lists_what_exists(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    RememberTool(store).run(key="known", value="x")

    with pytest.raises(ToolError, match="known"):
        RecallTool(store).run(key="missing")


def test_forget_removes_an_entry(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    RememberTool(store).run(key="temp", value="x")

    assert "temp" in ForgetTool(store).run(key="temp")
    assert store.get("temp") is None


def test_corrupt_memory_file_is_quarantined_not_fatal(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json", encoding="utf-8")
    store = MemoryStore(path)

    assert store.all() == {}
    assert path.with_suffix(".corrupt.json").exists()


# ----------------------------------------------------------------- registry


async def test_registry_dispatches_to_a_tool(workspace):
    registry = ToolRegistry.default(AgentConfig(workspace=workspace))

    output, is_error = await registry.call("write_file", {"path": "x.md", "content": "hi"})

    assert is_error is False
    assert (workspace / "x.md").read_text(encoding="utf-8") == "hi"
    assert "Wrote 2 bytes" in output


async def test_registry_returns_errors_instead_of_raising(workspace):
    registry = ToolRegistry.default(AgentConfig(workspace=workspace))

    output, is_error = await registry.call("read_file", {"path": "../escape"})

    assert is_error is True
    assert "outside the workspace" in output


async def test_registry_reports_unknown_tools(workspace):
    registry = ToolRegistry.default(AgentConfig(workspace=workspace))

    output, is_error = await registry.call("nonexistent", {})

    assert is_error is True
    assert "Unknown tool" in output


async def test_registry_reports_bad_arguments(workspace):
    registry = ToolRegistry.default(AgentConfig(workspace=workspace))

    output, is_error = await registry.call("write_file", {"wrong": "arg"})

    assert is_error is True
    assert "Invalid arguments" in output


def test_default_registry_exposes_the_expected_tools(workspace):
    registry = ToolRegistry.default(AgentConfig(workspace=workspace))

    assert registry.names() == [
        "forget",
        "list_files",
        "read_file",
        "recall",
        "remember",
        "write_file",
    ]


def test_tool_definitions_are_valid_json_schema_objects(workspace):
    registry = ToolRegistry.default(AgentConfig(workspace=workspace))

    for definition in registry.definitions():
        assert set(definition) == {"name", "description", "input_schema"}
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        for field in schema["required"]:
            assert field in schema["properties"]
