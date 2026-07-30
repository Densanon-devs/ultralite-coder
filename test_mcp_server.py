"""
Tests for engine.mcp_server — the DensAssistant -> ulcagent direction.

Driven in-process against the Server class (no subprocess), so the JSON-RPC
contract is asserted directly. The contract is copied from
densassistant/mcp/client.py: newline-delimited JSON-RPC 2.0, initialize ->
notifications/initialized -> tools/list -> tools/call, results shaped
{"content":[{"type":"text","text":...}], "isError": bool}.

Two properties matter most here and neither is about happy paths:
  * The caller is a PROGRAM, not a human at a y/N prompt, so the write-policy
    allowlist is the only control on this path. Refusals must come back as tool
    errors, never as silent successes.
  * stdout is the protocol. A diagnostic printed to stdout corrupts the stream,
    so logging must go to stderr.

Run: python -m pytest test_mcp_server.py -v
     OR just: python test_mcp_server.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine import mcp_server as ms
from engine import write_policy as wp


def _server(workspace: Path) -> tuple[ms.Server, io.StringIO, list[str]]:
    out = io.StringIO()
    logs: list[str] = []
    return ms.Server(workspace, out=out, err_log=logs.append), out, logs


def _call(server: ms.Server, tool: str, args: dict) -> dict:
    resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": tool, "arguments": args}})
    return resp["result"]


def _text(result: dict) -> str:
    return "\n".join(b.get("text", "") for b in result.get("content", []))


def _isolate_policy(tmp: Path, allowed: Path) -> None:
    """Point the policy + journal at a temp sandbox."""
    wp._STATE_DIR = tmp / "state"
    wp._ROOTS_FILE = tmp / "state" / "write_roots.txt"
    wp._JOURNAL = tmp / "state" / "mutations.jsonl"
    wp._BACKUP_DIR = tmp / "state" / "backups"
    wp.save_write_roots([allowed])


# ── protocol ────────────────────────────────────────────────────

def test_initialize_returns_protocol_and_server_info():
    srv, _, _ = _server(ROOT)
    resp = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": ms.PROTOCOL_VERSION}})
    result = resp["result"]
    assert resp["jsonrpc"] == "2.0" and resp["id"] == 1
    assert result["protocolVersion"] == ms.PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == ms.SERVER_NAME
    assert "tools" in result["capabilities"]


def test_notifications_get_no_response():
    """Replying to a notification would desync the client's id matching."""
    srv, _, _ = _server(ROOT)
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list_shape():
    srv, _, _ = _server(ROOT)
    tools = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                        "params": {}})["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"locate", "read_file", "create_file", "move_path", "edit_file"} <= names
    for t in tools:
        assert t["description"] and isinstance(t["inputSchema"], dict)
        assert t["inputSchema"]["type"] == "object"


def test_unknown_method_is_a_jsonrpc_error():
    srv, _, _ = _server(ROOT)
    resp = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "nope/nope", "params": {}})
    assert resp["error"]["code"] == -32601


def test_unknown_tool_is_a_tool_error_not_a_crash():
    srv, _, _ = _server(ROOT)
    result = _call(srv, "does_not_exist", {})
    assert result["isError"] is True
    assert "Unknown tool" in _text(result)


def test_bad_arguments_are_reported():
    srv, _, _ = _server(ROOT)
    result = _call(srv, "locate", {"wrong_arg": 1})
    assert result["isError"] is True
    assert "Bad arguments" in _text(result)


def test_serve_loop_ignores_junk_and_answers_valid_frames():
    srv, out, logs = _server(ROOT)
    stream = io.StringIO(
        "\n"
        "not json at all\n"
        + json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/list", "params": {}}) + "\n"
    )
    srv.serve(stream)
    lines = [l for l in out.getvalue().splitlines() if l.strip()]
    assert len(lines) == 1, lines
    assert json.loads(lines[0])["id"] == 9
    assert any("unparseable" in m for m in logs)


def test_logging_never_touches_stdout():
    """stdout is the protocol stream — a stray print corrupts it."""
    srv, out, logs = _server(ROOT)
    srv.serve(io.StringIO("garbage\n"))
    assert out.getvalue() == "", f"log leaked into stdout: {out.getvalue()!r}"
    assert logs


# ── write policy on this path ───────────────────────────────────

def test_create_inside_allowed_root():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allowed = tmp / "ws"
        allowed.mkdir()
        _isolate_policy(tmp, allowed)
        srv, _, _ = _server(allowed)
        result = _call(srv, "create_file", {"path": "a.txt", "content": "hello"})
        assert result["isError"] is False, _text(result)
        assert (allowed / "a.txt").read_text() == "hello"


def test_create_outside_allowed_root_is_refused():
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as other:
        tmp = Path(td)
        allowed = tmp / "ws"
        allowed.mkdir()
        _isolate_policy(tmp, allowed)
        srv, _, _ = _server(allowed)
        victim = Path(other) / "sneaky.txt"
        result = _call(srv, "create_file", {"path": str(victim), "content": "x"})
        assert result["isError"] is True
        assert "refused" in _text(result)
        assert not victim.exists(), "refusal must not have written the file"


def test_system_path_is_refused():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allowed = tmp / "ws"
        allowed.mkdir()
        _isolate_policy(tmp, allowed)
        srv, _, _ = _server(allowed)
        result = _call(srv, "create_file",
                       {"path": "C:/Windows/system32/x.dll", "content": "x"})
        assert result["isError"] is True
        assert "protected system location" in _text(result)


def test_create_does_not_clobber_without_overwrite():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allowed = tmp / "ws"
        allowed.mkdir()
        _isolate_policy(tmp, allowed)
        (allowed / "keep.txt").write_text("precious")
        srv, _, _ = _server(allowed)
        result = _call(srv, "create_file", {"path": "keep.txt", "content": "clobber"})
        assert result["isError"] is True
        assert (allowed / "keep.txt").read_text() == "precious"


def test_move_and_edit_round_trip():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allowed = tmp / "ws"
        allowed.mkdir()
        _isolate_policy(tmp, allowed)
        (allowed / "one.txt").write_text("alpha beta")
        srv, _, _ = _server(allowed)

        moved = _call(srv, "move_path", {"source": "one.txt", "destination": "two.txt"})
        assert moved["isError"] is False, _text(moved)
        assert (allowed / "two.txt").exists() and not (allowed / "one.txt").exists()

        edited = _call(srv, "edit_file", {"path": "two.txt",
                                          "old_string": "beta", "new_string": "gamma"})
        assert edited["isError"] is False, _text(edited)
        assert (allowed / "two.txt").read_text() == "alpha gamma"


def test_edit_refuses_ambiguous_anchor():
    """A known 14B failure mode — must fail loudly, not edit the wrong line."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allowed = tmp / "ws"
        allowed.mkdir()
        _isolate_policy(tmp, allowed)
        (allowed / "d.txt").write_text("x\nx\n")
        srv, _, _ = _server(allowed)
        result = _call(srv, "edit_file",
                       {"path": "d.txt", "old_string": "x", "new_string": "y"})
        assert result["isError"] is True
        assert "appears 2 times" in _text(result)
        assert (allowed / "d.txt").read_text() == "x\nx\n"


def test_edit_missing_anchor_is_an_error():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allowed = tmp / "ws"
        allowed.mkdir()
        _isolate_policy(tmp, allowed)
        (allowed / "e.txt").write_text("hello")
        srv, _, _ = _server(allowed)
        result = _call(srv, "edit_file",
                       {"path": "e.txt", "old_string": "absent", "new_string": "z"})
        assert result["isError"] is True
        assert "not found" in _text(result)


def test_mutations_are_journaled():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allowed = tmp / "ws"
        allowed.mkdir()
        _isolate_policy(tmp, allowed)
        srv, _, _ = _server(allowed)
        _call(srv, "create_file", {"path": "j.txt", "content": "1"})
        ops = [e["op"] for e in wp.journal_entries(limit=10)]
        assert "create" in ops, ops


def test_write_roots_tool_reports_the_allowlist():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        allowed = tmp / "ws"
        allowed.mkdir()
        _isolate_policy(tmp, allowed)
        srv, _, _ = _server(allowed)
        text = _text(_call(srv, "write_roots", {}))
        assert str(allowed) in text
        assert "refused" in text.lower()


def test_read_file_missing_is_an_error():
    srv, _, _ = _server(ROOT)
    result = _call(srv, "read_file", {"path": "definitely_absent_xyz.txt"})
    assert result["isError"] is True


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                failures.append((name, f"{type(e).__name__}: {e}"))
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} failed")
        sys.exit(1)
    print("all passed")
