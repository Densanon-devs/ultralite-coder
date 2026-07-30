"""Tests for the C4 grammar-guided parse-fail retry (Task C4, 2026-05-19)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from engine.agent import Agent
from engine.agent_tools import ToolRegistry, ToolSchema


def _make_registry():
    reg = ToolRegistry()
    reg.register(ToolSchema(
        name="read_file",
        description="read",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        function=lambda **kw: "file contents",
    ))
    return reg


class _SequencedModel:
    """Generate-returning model that yields a different value per call.
    Useful for asserting the agent re-calls on parse-fail."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        # Total tokens generated (number of model.generate calls)
        self.call_count = 0
        self.last_kwargs: list[dict] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        self.last_kwargs.append(dict(kwargs))
        if not self.responses:
            return ""
        return self.responses.pop(0)


class TestGrammarRetryFires(unittest.TestCase):
    def test_default_off_means_no_retry(self):
        """If retry_with_grammar_on_parse_fail is False (default), the
        agent must NOT re-call model.generate on a parse failure."""
        model = _SequencedModel([
            # First (and only) generation: a malformed tool_call.
            '<tool_call>{"name": "read_file", "arguments": {bad json}}</tool_call>',
        ])
        agent = Agent(
            model=model,
            registry=_make_registry(),
            grammar=object(),  # truthy grammar but flag is off
            retry_with_grammar_on_parse_fail=False,
            max_iterations=1,
        )
        agent.run("test goal")
        self.assertEqual(model.call_count, 1)
        # No "grammar" kwarg was ever passed
        for kw in model.last_kwargs:
            self.assertNotIn("grammar", kw)

    def test_no_grammar_means_no_retry(self):
        """retry_with_grammar_on_parse_fail=True but grammar=None → no retry."""
        model = _SequencedModel([
            '<tool_call>{"name": "read_file", "arguments": {bad}}</tool_call>',
        ])
        agent = Agent(
            model=model,
            registry=_make_registry(),
            grammar=None,
            retry_with_grammar_on_parse_fail=True,
            max_iterations=1,
        )
        agent.run("test goal")
        self.assertEqual(model.call_count, 1)

    def test_retry_fires_when_both_set_and_parse_fails(self):
        sentinel_grammar = object()
        model = _SequencedModel([
            # First: malformed
            '<tool_call>{"name": "read_file", "arguments": {bad json}}</tool_call>',
            # Second (after grammar retry): valid
            '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>',
            # Third: model declares done
            "All done.",
        ])
        agent = Agent(
            model=model,
            registry=_make_registry(),
            grammar=sentinel_grammar,
            retry_with_grammar_on_parse_fail=True,
            max_iterations=3,
        )
        agent.run("read x.py")
        # First two calls in iter 1 (initial + grammar retry), third call
        # in iter 2 for the final-answer turn.
        self.assertGreaterEqual(model.call_count, 2)
        # The second call MUST include grammar=sentinel_grammar.
        # (The first call must NOT — the grammar only attaches on retry.)
        self.assertNotIn("grammar", model.last_kwargs[0])
        self.assertIs(model.last_kwargs[1].get("grammar"), sentinel_grammar)

    def test_retry_does_NOT_fire_when_first_pass_parses_clean(self):
        """If the first generation parses cleanly, no grammar retry. The
        cost stays at 1 call/iteration on the happy path."""
        model = _SequencedModel([
            '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>',
            "Done.",
        ])
        agent = Agent(
            model=model,
            registry=_make_registry(),
            grammar=object(),
            retry_with_grammar_on_parse_fail=True,
            max_iterations=2,
        )
        agent.run("read x.py")
        # First call is unguided; second is the final-answer turn —
        # NEITHER should have grammar attached.
        for kw in model.last_kwargs:
            self.assertNotIn("grammar", kw)

    def test_retry_fall_through_when_retry_also_fails(self):
        """If both first pass AND grammar retry fail to parse, we fall
        through to the existing parse_errors handling (synthetic
        parse_error ToolResult)."""
        model = _SequencedModel([
            "<tool_call>{bad json}</tool_call>",
            "<tool_call>{also bad}</tool_call>",
            "Giving up.",
        ])
        agent = Agent(
            model=model,
            registry=_make_registry(),
            grammar=object(),
            retry_with_grammar_on_parse_fail=True,
            max_iterations=2,
        )
        result = agent.run("read x.py")
        # The agent should have emitted at least one synthetic parse_error
        # ToolResult — which is the pre-C4 baseline behavior.
        parse_errors = [r for r in result.tool_results if r.name == "parse_error"]
        self.assertGreaterEqual(len(parse_errors), 1)

    def test_retry_model_exception_falls_through(self):
        """If grammar retry raises, we keep the original parse_errors path."""
        class _RaisingOnSecond:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt: str, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return "<tool_call>{bad}</tool_call>"
                raise RuntimeError("grammar build failed")

        model = _RaisingOnSecond()
        agent = Agent(
            model=model,
            registry=_make_registry(),
            grammar=object(),
            retry_with_grammar_on_parse_fail=True,
            max_iterations=2,
        )
        # MUST NOT raise — exceptions in the retry are caught and we
        # fall back to the original parse_errors path.
        result = agent.run("read x.py")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
