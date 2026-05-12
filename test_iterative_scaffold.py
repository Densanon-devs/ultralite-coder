"""Unit tests for engine/iterative_scaffold.py.

Focused on the deterministic pieces (TODO detection, file snapshotting,
sub-agent construction). The full Phase 1+2+3 loop is exercised by the
soak comparison, not by unit tests (it requires a real model).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.iterative_scaffold import (
    TODO_RE,
    TODO_PRESENT_RE,
    _count_remaining_todos,
    _file_is_complete,
    _find_next_todo,
    _list_user_files,
    _scaffold_goal_wrapper,
    _fill_hint,
    SCAFFOLD_HINT,
)


class TestTodoRegex:
    def test_matches_basic_bracketed_todo(self):
        text = "    # TODO[1]: parse the iwlist output\n    pass\n"
        m = TODO_RE.search(text)
        assert m is not None
        assert m.group(2) == "1"
        assert "parse the iwlist" in m.group(3)

    def test_matches_unindented_todo(self):
        text = "# TODO[42]: top-level marker\n"
        m = TODO_RE.search(text)
        assert m is not None
        assert m.group(2) == "42"

    def test_matches_drift_formats(self):
        # The 14B drifts — must handle # TODO1:, # TODO 1:, # TODO-1:
        for variant in ("# TODO1: body", "# TODO 1: body", "# TODO-1: body",
                        "# TODO #1: body", "  # TODO[1] body"):
            m = TODO_RE.search(variant)
            assert m is not None, f"failed to match: {variant!r}"
            assert m.group(2) == "1", f"wrong number for {variant!r}"

    def test_matches_multiple_in_order(self):
        text = """
def a():
    # TODO[1]: a body
    pass

def b():
    # TODO2: b body
    pass
"""
        nums = [int(g[1]) for g in TODO_RE.findall(text)]
        assert nums == [1, 2]

    def test_does_not_match_unnumbered_todo(self):
        # Plain "# TODO: do X" without a number should NOT match
        assert TODO_RE.search("# TODO: do something later\n") is None

    def test_present_re_quick_check(self):
        assert TODO_PRESENT_RE.search("# TODO[1]: x")
        assert TODO_PRESENT_RE.search("# TODO1: x")
        assert TODO_PRESENT_RE.search("# todo 5: lowercase")
        assert not TODO_PRESENT_RE.search("# TODO: no number")


class TestFindNextTodo:
    def test_returns_none_for_no_todos(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("def x(): return 1\n", encoding="utf-8")
        assert _find_next_todo(f) is None

    def test_returns_first_todo_with_raw_line(self, tmp_path):
        f = tmp_path / "scaffold.py"
        f.write_text(
            'def parse_a():\n'
            '    """parse format A."""\n'
            '    # TODO[1]: parse format A\n'
            '    pass\n'
            '\n'
            'def parse_b():\n'
            '    # TODO[2]: parse format B\n'
            '    pass\n',
            encoding="utf-8",
        )
        result = _find_next_todo(f)
        assert result is not None
        num, text, raw_line, ctx = result
        assert num == 1
        assert "parse format A" in text
        # raw_line is the exact file line (includes indentation)
        assert raw_line.strip() == "# TODO[1]: parse format A"
        # Context should include the function signature
        assert "def parse_a" in ctx

    def test_raw_line_preserves_drift_format(self, tmp_path):
        f = tmp_path / "scaffold.py"
        f.write_text("def a():\n    # TODO1: drifted format\n    pass\n", encoding="utf-8")
        result = _find_next_todo(f)
        assert result is not None
        num, text, raw_line, ctx = result
        assert num == 1
        assert raw_line.strip() == "# TODO1: drifted format"

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _find_next_todo(tmp_path / "nope.py") is None


class TestFileIsComplete:
    def test_complete_when_compiles_no_todos(self, tmp_path):
        f = tmp_path / "done.py"
        f.write_text("def x():\n    return 1\n", encoding="utf-8")
        ok, why = _file_is_complete(f)
        assert ok, why

    def test_incomplete_with_todo(self, tmp_path):
        f = tmp_path / "skel.py"
        f.write_text("def x():\n    # TODO[1]: impl\n    pass\n", encoding="utf-8")
        ok, why = _file_is_complete(f)
        assert not ok
        assert "TODO" in why

    def test_incomplete_with_drift_todo(self, tmp_path):
        f = tmp_path / "skel.py"
        f.write_text("def x():\n    # TODO1: impl\n    pass\n", encoding="utf-8")
        ok, why = _file_is_complete(f)
        assert not ok

    def test_incomplete_with_syntax_error(self, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def x(:\n    pass\n", encoding="utf-8")
        ok, why = _file_is_complete(f)
        assert not ok
        assert "SyntaxError" in why

    def test_missing_file(self, tmp_path):
        ok, why = _file_is_complete(tmp_path / "nope.py")
        assert not ok


class TestCountRemaining:
    def test_zero_for_clean_file(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("def x(): return 1\n", encoding="utf-8")
        assert _count_remaining_todos(f) == 0

    def test_counts_correctly(self, tmp_path):
        f = tmp_path / "scaffold.py"
        f.write_text(
            "# TODO[1]: x\npass\n# TODO[2]: y\npass\n# TODO[3]: z\npass\n",
            encoding="utf-8",
        )
        assert _count_remaining_todos(f) == 3

    def test_zero_for_missing_file(self, tmp_path):
        assert _count_remaining_todos(tmp_path / "nope.py") == 0


class TestListUserFiles:
    def test_lists_top_level_files(self, tmp_path):
        (tmp_path / "a.py").write_text("a", encoding="utf-8")
        (tmp_path / "b.md").write_text("b", encoding="utf-8")
        (tmp_path / "c.json").write_text("{}", encoding="utf-8")
        files = _list_user_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"a.py", "b.md", "c.json"}

    def test_skips_hidden_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("a", encoding="utf-8")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "secret.py").write_text("s", encoding="utf-8")
        files = _list_user_files(tmp_path)
        names = {f.name for f in files}
        assert names == {"a.py"}

    def test_skips_pycache(self, tmp_path):
        (tmp_path / "a.py").write_text("a", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "ignore.pyc").write_text("x", encoding="utf-8")
        files = _list_user_files(tmp_path)
        assert {f.name for f in files} == {"a.py"}

    def test_handles_missing_workspace(self, tmp_path):
        files = _list_user_files(tmp_path / "nonexistent")
        assert files == set()


class TestPromptHelpers:
    def test_scaffold_goal_wrapper_includes_original(self):
        out = _scaffold_goal_wrapper("Build a port scanner")
        assert "Build a port scanner" in out
        assert "SCAFFOLD" in out.upper()
        assert "TODO[N]" in out
        assert "write_file" in out

    def test_fill_hint_includes_todo_details(self):
        out = _fill_hint("port_scan.py", 3, "implement banner grab",
                         "    # TODO[3]: implement banner grab", "def grab(): pass")
        assert "implement banner grab" in out
        assert "port_scan.py" in out
        assert "edit_file" in out
        # The old_string example MUST include the 4-space indent so the model
        # anchors on the full indented line (a short non-indented anchor causes
        # IndentationError when the multi-line replacement loses its indent).
        assert '"    # TODO[3]: implement banner grab\\n    pass"' in out
        # And it must tell the model EVERY new_string line needs >=4 spaces
        assert "4 spaces" in out

    def test_fill_hint_uses_raw_line_not_reconstructed(self):
        # If the scaffold drifted to "# TODO3:", the fill hint must use THAT
        # exact line (with its indent), not a reconstructed "# TODO[3]:".
        out = _fill_hint("x.py", 3, "do thing", "        # TODO3: do thing", "ctx")
        assert "# TODO3: do thing" in out
        # 8-space indent should be reflected
        assert '"        # TODO3: do thing\\n        pass"' in out
        assert "8 spaces" in out

    def test_scaffold_hint_forces_skeleton(self):
        assert "60 lines" in SCAFFOLD_HINT
        assert "TODO[N]" in SCAFFOLD_HINT or "TODO[1]" in SCAFFOLD_HINT
        assert "array" in SCAFFOLD_HINT.lower()
        # Must explicitly forbid implementing bodies + putting regex in skeleton
        assert "DO NOT implement" in SCAFFOLD_HINT
        assert "regex" in SCAFFOLD_HINT.lower()
