"""Tests for the A3 test-edit hint (Task A3, 2026-05-19)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from engine.agent import Agent
from engine.agent_tools import ToolCall, ToolRegistry, ToolResult, ToolSchema


def _stub_model():
    m = MagicMock()
    m.generate.return_value = "stub"
    return m


def _stub_registry():
    reg = ToolRegistry()
    reg.register(ToolSchema(
        name="write_file",
        description="write a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        function=lambda **kw: None,
    ))
    return reg


class TestPathPatternRecognition(unittest.TestCase):
    """The path-pattern recognizer must match all the conventional
    pytest layouts and reject non-test paths."""

    def test_tests_dir_unix(self):
        self.assertTrue(Agent._looks_like_test_file("tests/test_foo.py"))

    def test_tests_dir_windows(self):
        self.assertTrue(Agent._looks_like_test_file("tests\\test_foo.py"))

    def test_nested_tests_dir(self):
        self.assertTrue(Agent._looks_like_test_file("src/pkg/tests/utils.py"))

    def test_test_singular_dir(self):
        self.assertTrue(Agent._looks_like_test_file("test/test_foo.py"))

    def test_test_prefix_file(self):
        self.assertTrue(Agent._looks_like_test_file("test_main.py"))

    def test_test_suffix_file(self):
        self.assertTrue(Agent._looks_like_test_file("main_test.py"))

    def test_test_in_nested_path(self):
        self.assertTrue(Agent._looks_like_test_file("src/test_helper.py"))

    def test_non_test_python_file(self):
        self.assertFalse(Agent._looks_like_test_file("src/main.py"))

    def test_non_python_file_in_tests_dir(self):
        # tests/ directory but it's a yaml; still counts (the directory is
        # the signal). This is a deliberate design choice — the hint is
        # cheap and easy to ignore if the model decides it's not relevant.
        self.assertTrue(Agent._looks_like_test_file("tests/fixtures.yaml"))

    def test_empty_path(self):
        self.assertFalse(Agent._looks_like_test_file(""))

    def test_just_test_in_name_not_at_boundary(self):
        # "fastest.py" should NOT match — "test" must be at a path boundary
        self.assertFalse(Agent._looks_like_test_file("fastest.py"))
        self.assertFalse(Agent._looks_like_test_file("contest.py"))


class TestHintDefaultOff(unittest.TestCase):
    """The hint must not fire when the flag is at its default."""

    def test_default_is_off(self):
        agent = Agent(
            model=_stub_model(),
            registry=_stub_registry(),
        )
        self.assertFalse(agent.suggest_run_tests_on_test_edit)

    def test_no_hint_when_flag_off(self):
        agent = Agent(
            model=_stub_model(),
            registry=_stub_registry(),
            suggest_run_tests_on_test_edit=False,
        )
        call = ToolCall(
            name="write_file",
            arguments={"path": "tests/test_foo.py"},
            raw="",
        )
        result = ToolResult(name="write_file", success=True, content="ok")
        self.assertIsNone(agent._maybe_suggest_run_tests(call, result))


class TestHintFireConditions(unittest.TestCase):
    def _agent(self):
        return Agent(
            model=_stub_model(),
            registry=_stub_registry(),
            suggest_run_tests_on_test_edit=True,
        )

    def test_fires_on_test_write(self):
        agent = self._agent()
        call = ToolCall(
            name="write_file",
            arguments={"path": "tests/test_foo.py"},
            raw="",
        )
        result = ToolResult(name="write_file", success=True, content="ok")
        hint = agent._maybe_suggest_run_tests(call, result)
        self.assertIsNotNone(hint)
        self.assertEqual(hint.name, "test_edit_hint")
        self.assertTrue(hint.success)
        self.assertIn("run_tests", hint.content)
        self.assertIn("tests/test_foo.py", hint.content)

    def test_fires_on_test_edit(self):
        agent = self._agent()
        call = ToolCall(
            name="edit_file",
            arguments={"path": "test_main.py"},
            raw="",
        )
        result = ToolResult(name="edit_file", success=True, content="ok")
        self.assertIsNotNone(agent._maybe_suggest_run_tests(call, result))

    def test_no_hint_on_non_test_file(self):
        agent = self._agent()
        call = ToolCall(
            name="write_file",
            arguments={"path": "src/main.py"},
            raw="",
        )
        result = ToolResult(name="write_file", success=True, content="ok")
        self.assertIsNone(agent._maybe_suggest_run_tests(call, result))

    def test_no_hint_on_failed_call(self):
        agent = self._agent()
        call = ToolCall(
            name="write_file",
            arguments={"path": "tests/test_foo.py"},
            raw="",
        )
        result = ToolResult(name="write_file", success=False, error="disk full")
        self.assertIsNone(agent._maybe_suggest_run_tests(call, result))

    def test_no_hint_on_read_only_call(self):
        agent = self._agent()
        call = ToolCall(
            name="read_file",
            arguments={"path": "tests/test_foo.py"},
            raw="",
        )
        result = ToolResult(name="read_file", success=True, content="...")
        self.assertIsNone(agent._maybe_suggest_run_tests(call, result))

    def test_no_hint_on_missing_path_arg(self):
        agent = self._agent()
        call = ToolCall(name="write_file", arguments={}, raw="")
        result = ToolResult(name="write_file", success=True, content="ok")
        self.assertIsNone(agent._maybe_suggest_run_tests(call, result))

    def test_no_hint_on_non_string_path(self):
        agent = self._agent()
        call = ToolCall(
            name="write_file",
            arguments={"path": ["tests/test_foo.py"]},
            raw="",
        )
        result = ToolResult(name="write_file", success=True, content="ok")
        self.assertIsNone(agent._maybe_suggest_run_tests(call, result))


if __name__ == "__main__":
    unittest.main()
