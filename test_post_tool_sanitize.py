"""Tests for the PostToolUse-style output sanitizer (Task 5, 2026-05-19)."""

from __future__ import annotations

import json
import unittest

from engine.agent_tools import ToolRegistry, ToolCall, ToolSchema
from engine.post_tool_sanitize import (
    SanitizerConfig,
    sanitize_content,
    sanitize_string,
    sanitize_tool_result,
)


class TestSanitizeString(unittest.TestCase):
    def test_plain_text_passthrough(self):
        self.assertEqual(sanitize_string("hello world"), "hello world")

    def test_empty(self):
        self.assertEqual(sanitize_string(""), "")

    def test_lone_high_surrogate(self):
        # U+D83D is a leading surrogate; without a trailing surrogate it's lone.
        bad = "\ud83d"
        out = sanitize_string(bad)
        self.assertNotIn("\ud83d", out)
        self.assertEqual(out, "�")

    def test_lone_low_surrogate(self):
        bad = "before\udc00after"
        out = sanitize_string(bad)
        self.assertEqual(out, "before�after")

    def test_paired_surrogates_passthrough(self):
        # Properly-paired surrogates form a valid astral codepoint in Python's
        # internal representation only on narrow builds, but a Python str on
        # modern builds keeps U+1F600 as one char. We test that ordinary
        # high-codepoint chars aren't touched.
        s = "🤖 hello 🌍"
        self.assertEqual(sanitize_string(s), s)

    def test_nul_byte_escaped(self):
        bad = "a\x00b"
        out = sanitize_string(bad)
        self.assertEqual(out, "a\\x00b")

    def test_other_control_chars_escaped(self):
        # \x07 (bell), \x1f (unit separator)
        self.assertEqual(sanitize_string("x\x07y"), "x\\x07y")
        self.assertEqual(sanitize_string("a\x1fb"), "a\\x1fb")

    def test_whitespace_preserved(self):
        # Tab, newline, CR are meaningful — keep them
        self.assertEqual(sanitize_string("a\tb"), "a\tb")
        self.assertEqual(sanitize_string("a\nb"), "a\nb")
        self.assertEqual(sanitize_string("a\rb"), "a\rb")

    def test_truncation(self):
        cfg = SanitizerConfig(max_chars=10)
        out = sanitize_string("0123456789ABCDEF", cfg)
        self.assertTrue(out.startswith("0123456789"))
        self.assertIn("truncated by post-tool-sanitize at 10 chars", out)

    def test_no_truncation_when_under_cap(self):
        cfg = SanitizerConfig(max_chars=100)
        self.assertEqual(sanitize_string("short", cfg), "short")

    def test_disabled_returns_untouched(self):
        cfg = SanitizerConfig(enabled=False)
        bad = "lone\ud83d"
        self.assertEqual(sanitize_string(bad, cfg), bad)

    def test_idempotent(self):
        # Running it twice gives the same result as once.
        bad = "lone\ud83d and \x00 and " + ("x" * 10)
        cfg = SanitizerConfig(max_chars=12)
        once = sanitize_string(bad, cfg)
        twice = sanitize_string(once, cfg)
        self.assertEqual(once, twice)

    def test_output_is_json_safe(self):
        # The whole point: after sanitization, json.dumps should succeed.
        bad = "lone\ud83d high \udfff and \x00 nul"
        out = sanitize_string(bad)
        json.dumps(out)  # must not raise


class TestSanitizeContent(unittest.TestCase):
    def test_str_dispatch(self):
        self.assertEqual(sanitize_content("a\x00b"), "a\\x00b")

    def test_list(self):
        out = sanitize_content(["a", "b\x00c"])
        self.assertEqual(out, ["a", "b\\x00c"])

    def test_tuple_preserves_type(self):
        out = sanitize_content(("a", "b\x00"))
        self.assertIsInstance(out, tuple)
        self.assertEqual(out, ("a", "b\\x00"))

    def test_dict_values_sanitized(self):
        out = sanitize_content({"k": "v\x00", "n": 42})
        self.assertEqual(out, {"k": "v\\x00", "n": 42})

    def test_non_str_passthrough(self):
        self.assertEqual(sanitize_content(42), 42)
        self.assertEqual(sanitize_content(None), None)
        self.assertEqual(sanitize_content(True), True)

    def test_nested(self):
        nested = {"files": [{"path": "a\x00.py", "size": 100}]}
        out = sanitize_content(nested)
        self.assertEqual(out, {"files": [{"path": "a\\x00.py", "size": 100}]})


class TestSanitizeToolResult(unittest.TestCase):
    def test_opt_out_tool_passthrough(self):
        cfg = SanitizerConfig(opt_out_tools=frozenset({"my_tool"}))
        result = sanitize_tool_result("my_tool", "a\x00b", cfg)
        self.assertEqual(result, "a\x00b")  # untouched

    def test_default_opt_out_auto_verify(self):
        # auto_verify is in the default opt_out set
        cfg = SanitizerConfig()
        result = sanitize_tool_result("auto_verify", "a\x00b", cfg)
        self.assertEqual(result, "a\x00b")

    def test_non_opted_out_tool_sanitized(self):
        result = sanitize_tool_result("read_file", "a\x00b")
        self.assertEqual(result, "a\\x00b")


class TestRegistryIntegration(unittest.TestCase):
    """Confirm the sanitizer fires inside ToolRegistry.execute."""

    def _make_registry(self):
        reg = ToolRegistry()

        # Tool that returns a string containing a NUL byte
        def bad_str_tool():
            return "before\x00after"

        reg.register(ToolSchema(
            name="bad_str_tool",
            description="returns a NUL byte",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            function=bad_str_tool,
        ))

        # Tool that returns a dict whose value has a control char
        def bad_dict_tool():
            return {"files": ["a\x07b"], "count": 1}

        reg.register(ToolSchema(
            name="bad_dict_tool",
            description="returns control char in dict",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            function=bad_dict_tool,
        ))

        # Lone surrogate tool
        def surrogate_tool():
            return "a\ud83db"  # lone high surrogate

        reg.register(ToolSchema(
            name="surrogate_tool",
            description="lone surrogate",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            function=surrogate_tool,
        ))
        return reg

    def test_string_result_sanitized(self):
        reg = self._make_registry()
        r = reg.execute(ToolCall(name="bad_str_tool", arguments={}, raw=""))
        self.assertTrue(r.success)
        self.assertEqual(r.content, "before\\x00after")

    def test_dict_result_sanitized(self):
        reg = self._make_registry()
        r = reg.execute(ToolCall(name="bad_dict_tool", arguments={}, raw=""))
        self.assertTrue(r.success)
        self.assertEqual(r.content, {"files": ["a\\x07b"], "count": 1})

    def test_surrogate_stripped_in_result(self):
        reg = self._make_registry()
        r = reg.execute(ToolCall(name="surrogate_tool", arguments={}, raw=""))
        self.assertTrue(r.success)
        self.assertNotIn("\ud83d", r.content)
        self.assertIn("�", r.content)

    def test_format_for_model_is_json_safe(self):
        reg = self._make_registry()
        r = reg.execute(ToolCall(name="surrogate_tool", arguments={}, raw=""))
        # This used to risk crashing on some Python builds; must not now.
        text = r.format_for_model()
        # Round-trip
        json.loads(text)

    def test_opt_out_via_disable_flag(self):
        reg = self._make_registry()
        reg._sanitizer_config.enabled = False
        r = reg.execute(ToolCall(name="bad_str_tool", arguments={}, raw=""))
        self.assertEqual(r.content, "before\x00after")


if __name__ == "__main__":
    unittest.main()
