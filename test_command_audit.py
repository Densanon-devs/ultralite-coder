"""
Tests for engine.command_audit + the Agent._maybe_audit_destructive wiring.

Model-free and deterministic. Two layers:
  1. Module mechanics — snapshot_workspace / diff_snapshots / format /
     append_audit_log on a temp dir.
  2. End-to-end — drive Agent._execute_call with an APPROVED destructive
     run_bash whose stub function actually deletes a file, then confirm
     _maybe_audit_destructive threads a `command_audit` observation listing
     the real delta and writes the forensic log.

Run: python -m pytest test_command_audit.py -v
     OR just: python test_command_audit.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.command_audit import (
    AUDIT_LOG_NAME,
    CommandAudit,
    append_audit_log,
    diff_snapshots,
    format_audit_observation,
    snapshot_workspace,
)


def _write(tmp: Path, name: str, body: str = "x") -> Path:
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# ── snapshot_workspace ──

def test_snapshot_basic_and_relpaths():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "a.txt")
        _write(ws, "sub/b.txt")
        snap = snapshot_workspace(ws)
        assert snap is not None
        assert set(snap) == {"a.txt", "sub/b.txt"}


def test_snapshot_skips_ignore_dirs_and_audit_log():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "keep.txt")
        _write(ws, "node_modules/dep.js")
        _write(ws, "__pycache__/x.pyc")
        _write(ws, ".git/config")
        _write(ws, AUDIT_LOG_NAME, "old log\n")
        snap = snapshot_workspace(ws)
        assert snap is not None
        assert set(snap) == {"keep.txt"}


def test_snapshot_none_for_unset_or_missing_or_oversize():
    assert snapshot_workspace(None) is None
    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "nope"
        assert snapshot_workspace(missing) is None
        ws = Path(td)
        for i in range(5):
            _write(ws, f"f{i}.txt")
        # Cap below the file count → too large → None.
        assert snapshot_workspace(ws, max_entries=3) is None


# ── diff_snapshots ──

def test_diff_detects_delete_create_modify():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "stays.txt", "v1")
        _write(ws, "gone.txt", "bye")
        _write(ws, "changes.txt", "before")
        before = snapshot_workspace(ws)

        (ws / "gone.txt").unlink()
        _write(ws, "new.txt", "hello")
        # Force a different (mtime_ns, size) by changing content/length.
        _write(ws, "changes.txt", "after-longer")
        after = snapshot_workspace(ws)

        audit = diff_snapshots(before, after, "rm -rf gone.txt", ["rm_recursive"])
        assert audit.deleted == ["gone.txt"]
        assert audit.created == ["new.txt"]
        assert audit.modified == ["changes.txt"]
        assert audit.changed_any()
        assert audit.total_changes() == 3
        assert not audit.skipped


def test_diff_skipped_when_snapshot_none():
    audit = diff_snapshots(None, {"a": (1, 1)}, "rm -rf x", ["rm_recursive"])
    assert audit.skipped
    assert not audit.changed_any()
    audit2 = diff_snapshots({"a": (1, 1)}, None, "rm -rf x", ["rm_recursive"])
    assert audit2.skipped


def test_diff_no_change():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "a.txt")
        before = snapshot_workspace(ws)
        after = snapshot_workspace(ws)
        audit = diff_snapshots(before, after, "rm -rf nonexistent", ["rm_recursive"])
        assert not audit.changed_any()
        assert not audit.skipped


# ── formatting ──

def test_format_observation_variants():
    skipped = CommandAudit("rm -rf x", ["rm_recursive"], skipped=True, skip_reason="too large")
    assert "not captured" in format_audit_observation(skipped)

    nochange = CommandAudit("rm -rf x", ["rm_recursive"])
    assert "changed no files" in format_audit_observation(nochange)

    delta = CommandAudit(
        "rm -rf build", ["rm_recursive"], deleted=["build/a", "build/b"], created=[], modified=["x.py"]
    )
    obs = format_audit_observation(delta)
    assert "deleted (2)" in obs
    assert "build/a" in obs
    assert "modified (1)" in obs
    assert "NOT intended" in obs


def test_format_caps_long_lists():
    many = [f"f{i}" for i in range(120)]
    delta = CommandAudit("rm -rf .", ["rm_recursive"], deleted=many)
    obs = format_audit_observation(delta)
    assert "and 70 more" in obs  # 120 - 50 cap


# ── append_audit_log ──

def test_append_audit_log_writes_block():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        log = ws / AUDIT_LOG_NAME
        audit = CommandAudit("rm -rf old", ["rm_recursive", "rm_force"], deleted=["old/a"])
        append_audit_log(log, audit)
        append_audit_log(log, audit)  # appends, doesn't overwrite
        text = log.read_text(encoding="utf-8")
        assert text.count("command: rm -rf old") == 2
        assert "matched: rm_recursive, rm_force" in text
        assert "deleted: old/a" in text


def test_append_audit_log_swallows_bad_path():
    # Pointing at a directory path should not raise.
    with tempfile.TemporaryDirectory() as td:
        append_audit_log(Path(td), CommandAudit("x", []))  # td is a dir, not a file


# ── end-to-end through Agent._execute_call + _maybe_audit_destructive ──

class _StubModel:
    def generate(self, *_a, **_k):
        return ""


def _build_agent_with_deleting_bash(ws: Path, *, audit: bool = True, approve: bool = True):
    """Agent whose run_bash stub deletes <ws>/victim.txt when called."""
    from engine.agent import Agent
    from engine.agent_tools import ToolRegistry, ToolSchema

    def _fake_bash(command: str = "") -> str:
        # Simulate the destructive effect of the approved command.
        victim = ws / "victim.txt"
        if victim.exists():
            victim.unlink()
        return f"ran: {command}"

    reg = ToolRegistry()
    reg.register(
        ToolSchema(
            name="run_bash",
            description="run a shell command",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            function=_fake_bash,
            risky=True,
        )
    )
    return Agent(
        model=_StubModel(),
        registry=reg,
        workspace_root=ws,
        audit_destructive=audit,
        confirm_destructive=(lambda call, matches: approve),
    )


def test_e2e_destructive_audit_threads_observation_and_logs():
    from engine.agent_tools import ToolCall

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "victim.txt", "delete me")
        _write(ws, "keep.txt", "stay")
        agent = _build_agent_with_deleting_bash(ws)

        call = ToolCall(name="run_bash", arguments={"command": "rm -rf victim.txt"}, raw="")
        result = agent._execute_call(call)
        assert result.success, result.error
        # Snapshot was captured for the approved destructive command.
        assert id(call) in agent._pending_audit

        audit_result = agent._maybe_audit_destructive(call, result)
        assert audit_result is not None
        assert audit_result.name == "command_audit"
        assert "victim.txt" in audit_result.content
        assert "deleted (1)" in audit_result.content
        # Pending entry consumed.
        assert id(call) not in agent._pending_audit
        # Forensic log written.
        log = (ws / AUDIT_LOG_NAME).read_text(encoding="utf-8")
        assert "command: rm -rf victim.txt" in log
        assert "deleted: victim.txt" in log


def test_e2e_non_destructive_command_threads_nothing():
    from engine.agent_tools import ToolCall

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "victim.txt", "x")
        agent = _build_agent_with_deleting_bash(ws)
        # A benign command — destructive gate doesn't match, no snapshot captured.
        call = ToolCall(name="run_bash", arguments={"command": "echo hello"}, raw="")
        result = agent._execute_call(call)
        assert id(call) not in agent._pending_audit
        assert agent._maybe_audit_destructive(call, result) is None


def test_e2e_audit_disabled_captures_nothing():
    from engine.agent_tools import ToolCall

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "victim.txt", "x")
        agent = _build_agent_with_deleting_bash(ws, audit=False)
        call = ToolCall(name="run_bash", arguments={"command": "rm -rf victim.txt"}, raw="")
        result = agent._execute_call(call)
        # Disabled → no snapshot captured even though the command was destructive + approved.
        assert id(call) not in agent._pending_audit
        assert agent._maybe_audit_destructive(call, result) is None
        # And no forensic log written.
        assert not (ws / AUDIT_LOG_NAME).exists()


def test_e2e_denied_destructive_captures_nothing():
    from engine.agent_tools import ToolCall

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _write(ws, "victim.txt", "x")
        agent = _build_agent_with_deleting_bash(ws, approve=False)
        call = ToolCall(name="run_bash", arguments={"command": "rm -rf victim.txt"}, raw="")
        result = agent._execute_call(call)
        assert not result.success  # denied
        assert (ws / "victim.txt").exists()  # never ran
        assert id(call) not in agent._pending_audit


if __name__ == "__main__":
    import inspect

    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith("test_")]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
