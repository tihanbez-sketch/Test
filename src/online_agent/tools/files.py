"""Workspace file tools.

Every path the model supplies is resolved and checked against the workspace
root before any filesystem call. Traversal (`..`), absolute paths outside the
root, and symlinks that escape it are all rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool, ToolError, obj_schema

MAX_READ_BYTES = 400_000
MAX_WRITE_BYTES = 2_000_000
MAX_LISTING = 500


def resolve_in_workspace(workspace: Path, raw_path: str) -> Path:
    """Resolve `raw_path` inside `workspace`, or raise ToolError.

    The workspace root itself is resolved too, so a symlinked workspace still
    compares correctly.
    """
    if not raw_path or not raw_path.strip():
        raise ToolError("path must be a non-empty string.")

    root = workspace.resolve()
    candidate = Path(raw_path.strip())
    target = candidate if candidate.is_absolute() else root / candidate

    # strict=False so we can resolve paths that don't exist yet (writes).
    resolved = target.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ToolError(
            f"path {raw_path!r} is outside the workspace. "
            "Use a relative path such as 'notes/summary.md'."
        )
    return resolved


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the agent workspace. Use this to re-read "
        "notes, drafts, or data you wrote earlier in the session."
    )
    input_schema: dict[str, Any] = obj_schema(
        {
            "path": {
                "type": "string",
                "description": "Workspace-relative path, e.g. 'notes/research.md'.",
            }
        },
        required=["path"],
    )

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def run(self, path: str) -> str:  # type: ignore[override]
        target = resolve_in_workspace(self.workspace, path)
        if not target.exists():
            raise ToolError(f"{path!r} does not exist. Call list_files to see what does.")
        if target.is_dir():
            raise ToolError(f"{path!r} is a directory. Call list_files on it instead.")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ToolError(
                f"{path!r} is larger than {MAX_READ_BYTES} bytes. "
                "Read a smaller file, or process it in chunks with a script."
            )
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"{path!r} is not UTF-8 text.") from exc


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write a UTF-8 text file into the agent workspace, creating parent "
        "directories as needed. Use this for notes, drafts, reports, and any "
        "deliverable the user should be able to open afterwards. Overwrites an "
        "existing file at the same path."
    )
    input_schema: dict[str, Any] = obj_schema(
        {
            "path": {
                "type": "string",
                "description": "Workspace-relative path, e.g. 'report.md'.",
            },
            "content": {"type": "string", "description": "Full file contents."},
        },
        required=["path", "content"],
    )

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def run(self, path: str, content: str) -> str:  # type: ignore[override]
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            raise ToolError(
                f"content is {len(encoded)} bytes, over the {MAX_WRITE_BYTES}-byte limit. "
                "Split it across multiple files."
            )
        target = resolve_in_workspace(self.workspace, path)
        if target.is_dir():
            raise ToolError(f"{path!r} is an existing directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        rel = target.relative_to(self.workspace.resolve())
        return f"Wrote {len(encoded)} bytes to {rel}."


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files and directories in the agent workspace."
    input_schema: dict[str, Any] = obj_schema(
        {
            "path": {
                "type": "string",
                "description": "Workspace-relative directory. Defaults to the root.",
            }
        }
    )

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def run(self, path: str = ".") -> str:  # type: ignore[override]
        target = resolve_in_workspace(self.workspace, path or ".")
        if not target.exists():
            return f"{path} is empty (does not exist yet)."
        if not target.is_dir():
            raise ToolError(f"{path!r} is a file, not a directory.")

        root = self.workspace.resolve()
        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if child.name.startswith("."):
                continue
            rel = child.relative_to(root)
            if child.is_dir():
                entries.append(f"{rel}/")
            else:
                entries.append(f"{rel} ({child.stat().st_size} bytes)")
            if len(entries) >= MAX_LISTING:
                entries.append(f"... truncated at {MAX_LISTING} entries")
                break
        return "\n".join(entries) if entries else f"{path} is empty."
