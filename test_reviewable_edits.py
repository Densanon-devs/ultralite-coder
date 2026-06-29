"""
Tests for reviewable staged edits ("human-agent-in-the-loop", --review).

The feature: before each mutating file edit lands on disk, a `confirm_edit`
hook is shown the EXACT diff and approves/rejects it.

THE LOAD-BEARING INVARIANT under test: the diff passed to `confirm_edit` is
byte-identical to what actually gets written. The confirmation sits INSIDE the
tool's write path (engine.agent_builtins._apply_file_write), reusing the tool's
own content-computation logic, so preview == applied always.

Coverage:
- preview == applied for write_file (new + overwrite), edit_file (replace +
  empty-old_string prepend), insert_at_line.
- approve writes; reject does NOT write and surfaces a failure result the model
  can act on (EditRejected, which the registry turns into success=False).
- default off (confirm_edit=None) => byte-for-byte unchanged behavior +
  identical return strings.
- a raising confirm_edit callback => treated as reject (no write), like the
  other gates.
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

from engine.agent_builtins import (
    EditRejected,
    Workspace,
    _apply_file_write,
    _edit_file,
    _write_file,
    build_default_registry,
)


def _ws(td: str) -> Workspace:
    return Workspace(root=Path(td))


class _Recorder:
    """Approve-or-reject confirm_edit stub that records what it was shown."""

    def __init__(self, approve: bool = True):
        self.approve = approve
        self.calls: list[dict] = []

    def __call__(self, path, old_content, new_content, diff_text):
        self.calls.append(
            {
                "path": path,
                "old": old_content,
                "new": new_content,
                "diff": diff_text,
            }
        )
        return self.approve


# ── preview == applied (the load-bearing invariant) ───────────────


def test_preview_equals_applied_write_file_new():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        rec = _Recorder(approve=True)
        _write_file(ws, "new.py", content="x = 1\n", confirm_edit=rec)
        assert len(rec.calls) == 1
        shown_new = rec.calls[0]["new"]
        on_disk = (Path(td) / "new.py").read_text(encoding="utf-8")
        assert shown_new == on_disk  # byte-identical
        assert rec.calls[0]["old"] == ""  # new file => empty old


def test_preview_equals_applied_write_file_overwrite():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "x.py", content="x = 1\n")  # seed (no review)
        rec = _Recorder(approve=True)
        _write_file(ws, "x.py", content="x = 2\n", confirm_edit=rec)
        shown_new = rec.calls[0]["new"]
        on_disk = (Path(td) / "x.py").read_text(encoding="utf-8")
        assert shown_new == on_disk
        assert rec.calls[0]["old"] == "x = 1\n"
        # diff describes the real change
        assert "-x = 1" in rec.calls[0]["diff"]
        assert "+x = 2" in rec.calls[0]["diff"]


def test_preview_equals_applied_edit_file_replace():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "calc.py", content="def add(a, b):\n    return a + b\n")
        rec = _Recorder(approve=True)
        _edit_file(
            ws, "calc.py",
            old_string="return a + b",
            new_string="return a - b",
            confirm_edit=rec,
        )
        shown_new = rec.calls[0]["new"]
        on_disk = (Path(td) / "calc.py").read_text(encoding="utf-8")
        assert shown_new == on_disk
        assert "return a - b" in on_disk


def test_preview_equals_applied_edit_file_empty_old_prepend():
    # empty old_string + pure-import content => prepend. The forgiving
    # routing lives in _edit_file; the helper must show exactly that result.
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "mod.py", content="x = 1\n")
        rec = _Recorder(approve=True)
        _edit_file(
            ws, "mod.py",
            old_string="",
            new_string="import os\n",
            confirm_edit=rec,
        )
        shown_new = rec.calls[0]["new"]
        on_disk = (Path(td) / "mod.py").read_text(encoding="utf-8")
        assert shown_new == on_disk
        # prepend semantics actually happened
        assert on_disk.startswith("import os")
        assert "x = 1" in on_disk


def test_preview_equals_applied_insert_at_line():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        reg = build_default_registry(td)  # default: no review
        _write_file(ws, "f.py", content="a = 1\nb = 2\nc = 3\n")
        rec = _Recorder(approve=True)
        # Rebuild registry WITH the recorder so insert_at_line captures it.
        reg = build_default_registry(td, confirm_edit=rec)
        calls = reg.parse(
            '<tool_call>{"name":"insert_at_line","arguments":'
            '{"path":"f.py","line":2,"text":"INSERTED"}}</tool_call>'
        )
        result = reg.execute(calls[0])
        assert result.success, result.error
        shown_new = rec.calls[0]["new"]
        on_disk = (Path(td) / "f.py").read_text(encoding="utf-8")
        assert shown_new == on_disk
        assert on_disk == "a = 1\nINSERTED\nb = 2\nc = 3\n"


# ── approve writes; reject does NOT write ─────────────────────────


def test_approve_writes():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "x.py", content="old\n")
        rec = _Recorder(approve=True)
        _edit_file(ws, "x.py", old_string="old", new_string="new", confirm_edit=rec)
        assert (Path(td) / "x.py").read_text(encoding="utf-8") == "new\n"


def test_reject_does_not_write_and_signals_failure():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "x.py", content="old\n")
        rec = _Recorder(approve=False)
        raised = False
        try:
            _edit_file(ws, "x.py", old_string="old", new_string="new", confirm_edit=rec)
        except EditRejected as exc:
            raised = True
            assert "rejected" in str(exc).lower()
            assert "not written" in str(exc).lower()
        assert raised, "rejection must raise EditRejected"
        # nothing changed on disk
        assert (Path(td) / "x.py").read_text(encoding="utf-8") == "old\n"


def test_reject_through_registry_is_failure_result():
    # The registry must convert EditRejected (subclass of ValueError) into a
    # ToolResult(success=False, ...) the model can read and react to.
    with tempfile.TemporaryDirectory() as td:
        rec = _Recorder(approve=False)
        reg = build_default_registry(td, confirm_edit=rec)
        # write_file a new file, rejected
        calls = reg.parse(
            '<tool_call>{"name":"write_file","arguments":'
            '{"path":"new.py","content":"x = 1\\n"}}</tool_call>'
        )
        result = reg.execute(calls[0])
        assert not result.success
        assert "rejected" in (result.error or "").lower()
        assert not (Path(td) / "new.py").exists()


def test_reject_write_file_new_file_not_created():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        rec = _Recorder(approve=False)
        try:
            _write_file(ws, "ghost.py", content="x = 1\n", confirm_edit=rec)
        except EditRejected:
            pass
        assert not (Path(td) / "ghost.py").exists()


# ── default off (confirm_edit=None) => unchanged ──────────────────


def test_default_off_write_file_writes_immediately_same_return():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        # With None (default) the behavior + return string must match legacy.
        r_none = _write_file(ws, "a.py", content="x = 1\n")
        assert (Path(td) / "a.py").read_text(encoding="utf-8") == "x = 1\n"
        assert "Wrote" in r_none
        # Explicit None arg is identical.
        r_none2 = _write_file(ws, "b.py", content="y = 2\n", confirm_edit=None)
        assert (Path(td) / "b.py").read_text(encoding="utf-8") == "y = 2\n"
        assert "Wrote" in r_none2


def test_default_off_edit_file_return_string_unchanged():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "calc.py", content="def add(a, b):\n    return a + b\n")
        r = _edit_file(ws, "calc.py", old_string="a + b", new_string="a - b")
        # The trained-on success prefix must be intact (no review wording).
        assert r.startswith("Replaced")
        assert "rejected" not in r.lower()
        assert "review" not in r.lower()
        assert (Path(td) / "calc.py").read_text(encoding="utf-8").endswith("return a - b\n")


def test_default_off_registry_byte_for_byte():
    # Building the registry without confirm_edit must behave exactly as before:
    # tools write immediately, success results.
    with tempfile.TemporaryDirectory() as td:
        reg = build_default_registry(td)  # no confirm_edit
        calls = reg.parse(
            '<tool_call>{"name":"write_file","arguments":'
            '{"path":"z.py","content":"z = 9\\n"}}</tool_call>'
        )
        result = reg.execute(calls[0])
        assert result.success, result.error
        assert (Path(td) / "z.py").read_text(encoding="utf-8") == "z = 9\n"
        assert "Wrote" in result.content


# ── raising callback => treated as reject (no write) ──────────────


def test_raising_callback_treated_as_reject():
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        _write_file(ws, "x.py", content="old\n")

        def boom(path, old, new, diff):
            raise RuntimeError("callback exploded")

        raised = False
        try:
            _edit_file(ws, "x.py", old_string="old", new_string="new", confirm_edit=boom)
        except EditRejected:
            raised = True
        assert raised, "a raising confirm_edit must be treated as a rejection"
        # no write happened
        assert (Path(td) / "x.py").read_text(encoding="utf-8") == "old\n"


def test_apply_file_write_no_op_change_not_prompted():
    # If new content == current content there's nothing to review; the helper
    # writes (a no-op) without ever calling confirm_edit.
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        (Path(td) / "same.py").write_text("same\n", encoding="utf-8")
        rec = _Recorder(approve=False)  # would reject if called
        _apply_file_write(ws, "same.py", "same\n", confirm_edit=rec)
        assert len(rec.calls) == 0  # not prompted
        assert (Path(td) / "same.py").read_text(encoding="utf-8") == "same\n"


def test_diff_shown_is_full_not_truncated():
    # The reviewer must see the WHOLE change, not a max_lines summary.
    with tempfile.TemporaryDirectory() as td:
        ws = _ws(td)
        big_before = "\n".join(f"old_{i}" for i in range(60)) + "\n"
        big_after = "\n".join(f"new_{i}" for i in range(60)) + "\n"
        (Path(td) / "big.py").write_text(big_before, encoding="utf-8")
        rec = _Recorder(approve=True)
        _apply_file_write(ws, "big.py", big_after, confirm_edit=rec)
        diff = rec.calls[0]["diff"]
        assert "more diff lines elided" not in diff  # uncapped
        # every changed line present
        assert "-old_0" in diff and "-old_59" in diff
        assert "+new_0" in diff and "+new_59" in diff


if __name__ == "__main__":
    import inspect

    mod = sys.modules[__name__]
    fns = [
        (n, f)
        for n, f in inspect.getmembers(mod, inspect.isfunction)
        if n.startswith("test_")
    ]
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
            import traceback

            traceback.print_exc()
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
