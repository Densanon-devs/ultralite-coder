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
    _count_remaining_todos,
    _find_next_todo,
    _list_user_files,
    _scaffold_goal_wrapper,
    _fill_hint,
    SCAFFOLD_HINT,
)


class TestTodoRegex:
    def test_matches_basic_todo(self):
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

    def test_matches_multiple_in_order(self):
        text = """
def a():
    # TODO[1]: a body
    pass

def b():
    # TODO[2]: b body
    pass
"""
        nums = [int(g[1]) for g in TODO_RE.findall(text)]
        assert nums == [1, 2]

    def test_does_not_match_random_todo_comment(self):
        # Plain "# TODO: do X" without numbered brackets should NOT match
        text = "# TODO: do something later\n"
        assert TODO_RE.search(text) is None


class TestFindNextTodo:
    def test_returns_none_for_no_todos(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("def x(): return 1\n", encoding="utf-8")
        assert _find_next_todo(f) is None

    def test_returns_first_todo(self, tmp_path):
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
        num, text, ctx = result
        assert num == 1
        assert "parse format A" in text
        # Context should include the function signature
        assert "def parse_a" in ctx

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _find_next_todo(tmp_path / "nope.py") is None


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
        assert "PHASE 1" in out
        assert "SCAFFOLD ONLY" in out

    def test_fill_hint_includes_todo_details(self):
        out = _fill_hint("port_scan.py", 3, "implement banner grab", "def grab(): pass")
        assert "TODO[3]" in out
        assert "implement banner grab" in out
        assert "port_scan.py" in out
        assert "edit_file" in out

    def test_scaffold_hint_caps_lines(self):
        # The hint must explicitly cap file size
        assert "60 lines" in SCAFFOLD_HINT
        assert "TODO[N]" in SCAFFOLD_HINT
