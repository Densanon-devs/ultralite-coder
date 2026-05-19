"""Tests for the GBNF tool-call grammar (Task A1, 2026-05-19)."""

from __future__ import annotations

import unittest

from engine.tool_call_grammar import (
    TOOL_CALL_ONLY_GBNF,
    _validates_as_tool_call_envelope,
    get_tool_call_only_gbnf,
    load_tool_call_grammar,
)


class TestGrammarText(unittest.TestCase):
    """The GBNF source is the load-bearing artifact; validate its shape."""

    def test_grammar_text_has_root_rule(self):
        self.assertIn("root ::=", TOOL_CALL_ONLY_GBNF)

    def test_grammar_envelope_present(self):
        # Hermes <tool_call> envelope must be encoded
        self.assertIn("<tool_call>", TOOL_CALL_ONLY_GBNF)
        self.assertIn("</tool_call>", TOOL_CALL_ONLY_GBNF)

    def test_grammar_has_json_rules(self):
        for rule in ("object", "pair", "value", "string", "number", "array"):
            self.assertIn(f"{rule}", TOOL_CALL_ONLY_GBNF)

    def test_grammar_escapes_match_rfc8259(self):
        # The escape rule must include EXACTLY the seven valid JSON escapes
        # plus the unicode form. No `\@`, no `\d`.
        self.assertIn('["\\\\/bfnrt]', TOOL_CALL_ONLY_GBNF)

    def test_get_tool_call_only_gbnf_returns_source(self):
        self.assertEqual(get_tool_call_only_gbnf(), TOOL_CALL_ONLY_GBNF)


class TestLazyLoad(unittest.TestCase):
    """load_tool_call_grammar must either return a LlamaGrammar object,
    or None if llama-cpp-python is unavailable. Never raise on absent dep."""

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            load_tool_call_grammar(variant="totally_invented")

    def test_known_variant_returns_grammar_or_none(self):
        result = load_tool_call_grammar("tool_call_only")
        # Either we got a real LlamaGrammar (llama-cpp-python present) or
        # None (not installed). NEVER raise.
        if result is None:
            return
        # If we got something, it should be a LlamaGrammar instance.
        from llama_cpp import LlamaGrammar
        self.assertIsInstance(result, LlamaGrammar)

    def test_repeated_load_returns_cached(self):
        a = load_tool_call_grammar("tool_call_only")
        b = load_tool_call_grammar("tool_call_only")
        # Identity equality — same cached object
        self.assertIs(a, b)


class TestValidatorAcceptsGoodInputs(unittest.TestCase):
    """Mirror-validator should accept what the grammar would."""

    def test_simple_call(self):
        self.assertTrue(_validates_as_tool_call_envelope(
            '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
        ))

    def test_whitespace_padded(self):
        self.assertTrue(_validates_as_tool_call_envelope(
            '\n  <tool_call>\n  {"name": "x", "arguments": {}}\n  </tool_call>\n'
        ))

    def test_nested_object_args(self):
        self.assertTrue(_validates_as_tool_call_envelope(
            '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old_string": "a", "new_string": "b"}}</tool_call>'
        ))

    def test_escaped_quote_in_string(self):
        # JSON-valid escaped quote: \"
        self.assertTrue(_validates_as_tool_call_envelope(
            '<tool_call>{"name": "write_file", "arguments": {"content": "say \\"hi\\""}}</tool_call>'
        ))

    def test_array_value(self):
        self.assertTrue(_validates_as_tool_call_envelope(
            '<tool_call>{"name": "write_file", "arguments": {"content": ["a", "b"]}}</tool_call>'
        ))


class TestValidatorRejectsBadInputs(unittest.TestCase):
    """The whole point of the grammar — exactly these cases that drive
    ceiling #1 must be unrepresentable."""

    def test_python_single_quoted_string_rejected(self):
        # JSON doesn't allow single-quoted strings.
        self.assertFalse(_validates_as_tool_call_envelope(
            "<tool_call>{'name': 'x', 'arguments': {}}</tool_call>"
        ))

    def test_invalid_backslash_escape_rejected(self):
        # \@ is not a valid JSON escape
        self.assertFalse(_validates_as_tool_call_envelope(
            '<tool_call>{"name": "x", "arguments": {"content": "\\@dataclass"}}</tool_call>'
        ))

    def test_unescaped_inner_quote_rejected(self):
        # Inner " not escaped — invalid JSON
        self.assertFalse(_validates_as_tool_call_envelope(
            '<tool_call>{"name": "x", "arguments": {"content": "say "hi""}}</tool_call>'
        ))

    def test_trailing_comma_rejected(self):
        self.assertFalse(_validates_as_tool_call_envelope(
            '<tool_call>{"name": "x", "arguments": {"a": 1,}}</tool_call>'
        ))

    def test_no_envelope_rejected(self):
        self.assertFalse(_validates_as_tool_call_envelope(
            '{"name": "x", "arguments": {}}'
        ))

    def test_unclosed_envelope_rejected(self):
        self.assertFalse(_validates_as_tool_call_envelope(
            '<tool_call>{"name": "x", "arguments": {}}'
        ))

    def test_python_repr_style_rejected(self):
        # Mixed quote styles
        self.assertFalse(_validates_as_tool_call_envelope(
            "<tool_call>{\"name\": \"x\", \"arguments\": {\"path\": 'foo.py'}}</tool_call>"
        ))


class TestRealGrammarRoundTrip(unittest.TestCase):
    """Skipped if llama-cpp-python isn't installed. When it IS installed,
    verify the grammar string is parseable by the real parser — catches
    syntax errors in the GBNF itself."""

    def test_real_grammar_parses(self):
        try:
            from llama_cpp import LlamaGrammar  # noqa: F401
        except ImportError:
            self.skipTest("llama-cpp-python not installed")
        # Just verifying it doesn't raise. If LlamaGrammar.from_string
        # rejects our GBNF, this raises and we catch the bug.
        grammar = load_tool_call_grammar("tool_call_only")
        self.assertIsNotNone(grammar)


if __name__ == "__main__":
    unittest.main()
