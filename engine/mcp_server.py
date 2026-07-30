"""
MCP server — exposes ulcagent's file powers so DensAssistant can act, not just recall.

This is the second half of the DensAssistant integration. `densassistant_bridge`
lets ulcagent READ personal memory; this lets DensAssistant DO things — find a
file anywhere on the machine, create one, move one, edit one — by launching this
module as an MCP stdio server from its existing MCP panel.

Protocol is matched to `densassistant/mcp/client.py` exactly: newline-delimited
JSON-RPC 2.0 on stdin/stdout, `initialize` -> `notifications/initialized` ->
`tools/list` -> `tools/call`, results as `{"content":[{"type":"text",...}],
"isError": bool}`.

Two hard rules, because the caller here is another program rather than a human
watching a prompt:

  * The SAME `write_policy` allowlist governs every mutation. There is no
    confirm hook on this path, so the allowlist is the only control — and it is
    the one that refuses rather than asks.
  * stdout is the protocol. Nothing may print to it but JSON-RPC frames, so all
    diagnostics go to stderr.

Run:  python -m engine.mcp_server
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ulcagent-files"
SERVER_VERSION = "1.0"

_MAX_LINE = 8 * 1024 * 1024


def _log(msg: str) -> None:
    """Diagnostics go to stderr — stdout carries the protocol."""
    print(f"[mcp_server] {msg}", file=sys.stderr, flush=True)


# ── tool implementations ─────────────────────────────────────────

def _tool_locate(query: str, kind: str = "any", limit: int = 20) -> str:
    from . import file_index
    st = file_index.status()
    if not st["exists"] or not st["entries"]:
        return ("No file index has been built yet. Run `ulcagent --reindex` once "
                "(about 15 seconds) to enable locate.")
    hits = file_index.locate(query, kind=kind, limit=limit)
    if not hits:
        return f"No match for {query!r}."
    return "\n".join(f"[{h.kind}] {h.path}" for h in hits)


def _policy(workspace: Path):
    from .write_policy import WritePolicy
    return WritePolicy.load(workspace)


def _guard(path: str, op: str, workspace: Path) -> Path:
    from .write_policy import WritePolicy
    pol = WritePolicy.load(workspace)
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = (workspace / target)
    verdict = pol.check(target, op)
    if not verdict.allowed:
        raise PermissionError(f"{op} refused: {verdict.reason}")
    return target.resolve()


def _tool_read_file(path: str, workspace: Path, max_chars: int = 20000) -> str:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = workspace / target
    if not target.is_file():
        raise FileNotFoundError(f"Not a file: {target}")
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
    return text


def _tool_create_file(path: str, content: str, workspace: Path,
                      overwrite: bool = False) -> str:
    from . import write_policy as wp
    target = _guard(path, "write", workspace)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists. Pass overwrite=true to replace.")
    backup = wp._backup(target) if target.is_file() else None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    wp.record("overwrite" if backup else "create", target, backup=backup)
    return f"Wrote {len(content)} chars to {target}"


def _tool_move_path(source: str, destination: str, workspace: Path,
                    overwrite: bool = False) -> str:
    import shutil
    from . import write_policy as wp
    src = _guard(source, "move", workspace)
    dst = _guard(destination, "move", workspace)
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if dst.is_dir() and src.name != dst.name:
        dst = dst / src.name
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists. Pass overwrite=true to replace.")
    backup = wp._backup(dst) if dst.is_file() else None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    wp.record("move", src, dest=dst, backup=backup)
    return f"Moved {src} -> {dst}"


def _tool_edit_file(path: str, old_string: str, new_string: str, workspace: Path,
                    replace_all: bool = False) -> str:
    from . import write_policy as wp
    target = _guard(path, "edit", workspace)
    if not target.is_file():
        raise FileNotFoundError(f"Not a file: {target}")
    text = target.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)
    if count == 0:
        raise ValueError("old_string not found in the file (must match exactly).")
    if count > 1 and not replace_all:
        raise ValueError(f"old_string appears {count} times; pass replace_all=true "
                         f"or give a longer unique string.")
    backup = wp._backup(target)
    target.write_text(text.replace(old_string, new_string), encoding="utf-8")
    wp.record("edit", target, backup=backup)
    return f"Replaced {count if replace_all else 1} occurrence(s) in {target}"


def _tool_write_roots(workspace: Path) -> str:
    pol = _policy(workspace)
    return ("ulcagent may create/move/edit only inside these roots:\n"
            + pol.describe()
            + "\nAnything else is refused. Add one with: ulcagent --write-root \"<path>\"")


# ── tool table ───────────────────────────────────────────────────

def _tools(workspace: Path) -> dict[str, dict[str, Any]]:
    return {
        "locate": {
            "description": ("Find a file or folder anywhere on this computer by "
                            "name, from a prebuilt index. Fast."),
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {"type": "string", "enum": ["any", "dir", "file"]},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            "fn": lambda query, kind="any", limit=20: _tool_locate(query, kind, limit),
        },
        "read_file": {
            "description": "Read a text file's contents.",
            "schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
            "fn": lambda path: _tool_read_file(path, workspace),
        },
        "create_file": {
            "description": ("Create a file with the given content. Refuses paths "
                            "outside the configured write roots."),
            "schema": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"},
                                      "overwrite": {"type": "boolean"}},
                       "required": ["path", "content"]},
            "fn": lambda path, content, overwrite=False: _tool_create_file(
                path, content, workspace, overwrite),
        },
        "move_path": {
            "description": ("Move or rename a file or folder. Refuses paths "
                            "outside the configured write roots."),
            "schema": {"type": "object",
                       "properties": {"source": {"type": "string"},
                                      "destination": {"type": "string"},
                                      "overwrite": {"type": "boolean"}},
                       "required": ["source", "destination"]},
            "fn": lambda source, destination, overwrite=False: _tool_move_path(
                source, destination, workspace, overwrite),
        },
        "edit_file": {
            "description": ("Replace an exact string in a file. Fails if the string "
                            "is absent or ambiguous."),
            "schema": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "old_string": {"type": "string"},
                                      "new_string": {"type": "string"},
                                      "replace_all": {"type": "boolean"}},
                       "required": ["path", "old_string", "new_string"]},
            "fn": lambda path, old_string, new_string, replace_all=False:
                _tool_edit_file(path, old_string, new_string, workspace, replace_all),
        },
        "write_roots": {
            "description": "Show which directories may be written to.",
            "schema": {"type": "object", "properties": {}},
            "fn": lambda: _tool_write_roots(workspace),
        },
    }


# ── JSON-RPC plumbing ────────────────────────────────────────────

def _ok(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


class Server:
    def __init__(self, workspace: Path, out=None, err_log: Callable[[str], None] = _log):
        self.workspace = workspace
        self.tools = _tools(workspace)
        self._out = out or sys.stdout
        self._log = err_log

    def _send(self, obj: dict) -> None:
        self._out.write(json.dumps(obj) + "\n")
        self._out.flush()

    def handle(self, msg: dict) -> dict | None:
        """Return a response dict, or None for notifications."""
        method = msg.get("method", "")
        mid = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": mid,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            }

        if method.startswith("notifications/"):
            return None                      # nothing to acknowledge

        if method == "tools/list":
            listed = [{"name": name, "description": spec["description"],
                       "inputSchema": spec["schema"]}
                      for name, spec in self.tools.items()]
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": listed}}

        if method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments") or {}
            spec = self.tools.get(name)
            if spec is None:
                return {"jsonrpc": "2.0", "id": mid,
                        "result": _err(f"Unknown tool {name!r}. "
                                       f"Available: {sorted(self.tools)}")}
            try:
                text = spec["fn"](**args)
                return {"jsonrpc": "2.0", "id": mid, "result": _ok(str(text))}
            except TypeError as exc:
                return {"jsonrpc": "2.0", "id": mid,
                        "result": _err(f"Bad arguments for {name}: {exc}")}
            except (PermissionError, FileNotFoundError, FileExistsError,
                    IsADirectoryError, ValueError, OSError) as exc:
                # Expected, actionable failures — reported as tool errors so the
                # caller can correct, not as protocol errors.
                return {"jsonrpc": "2.0", "id": mid,
                        "result": _err(f"{type(exc).__name__}: {exc}")}
            except Exception as exc:                     # genuinely unexpected
                self._log(f"unhandled error in {name}: {traceback.format_exc()}")
                return {"jsonrpc": "2.0", "id": mid,
                        "result": _err(f"Internal error in {name}: {exc}")}

        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}

    def serve(self, stream=None) -> None:
        stream = stream or sys.stdin
        self._log(f"serving {len(self.tools)} tools; workspace={self.workspace}")
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if len(line) > _MAX_LINE:
                self._log("dropping oversized frame")
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self._log("dropping unparseable frame")
                continue
            try:
                response = self.handle(msg)
            except Exception:
                self._log(f"handler crashed: {traceback.format_exc()}")
                continue
            if response is not None:
                self._send(response)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    workspace = Path.cwd()
    if "--workspace" in argv:
        i = argv.index("--workspace")
        if i + 1 < len(argv):
            workspace = Path(argv[i + 1]).expanduser().resolve()
    Server(workspace).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
