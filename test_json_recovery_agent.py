"""Integration test: json_recovery regeneration wired into the live Agent loop.

Drives the real Agent via the same StubModel pattern as test_self_heal_agent.
A first turn emits a tool call whose JSON is too broken for the in-parser
repair tiers to salvage (an unquoted value). With json_recovery=True the agent
should fire ONE regeneration pass, adopt the corrected turn, and run the call.
With it OFF (the default) the broken call is surfaced as a parse_error instead.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.agent import Agent
from engine.agent_builtins import build_default_registry


class _StubModel:
    """Minimal canned-response model — same shape as test_self_heal_agent."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 1024, stop=None, **kwargs):
        self.prompts.append(prompt)
        if self.calls >= len(self.responses):
            self.calls += 1
            return "(stub exhausted)"
        r = self.responses[self.calls]
        self.calls += 1
        return r


# Unquoted value `x.py` — fails lenient JSON, escape repair, brace/array
# recovery, AND ast.literal_eval (it parses as attribute access, not a
# literal), so the parser returns zero calls + one error. This is exactly
# the residual case json_recovery targets.
_BROKEN = '<tool_call>{"name": "read_file", "arguments": {"path": x.py}}</tool_call>'
_VALID = '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'


def test_json_recovery_fires_and_recovers():
    """With json_recovery=True, a broken tool call triggers one regeneration
    pass that recovers a valid call, which then executes."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "x.py").write_text("print('hello')\n", encoding="utf-8")
        reg = build_default_registry(tmp)
        model = _StubModel([_BROKEN, _VALID, "Done — file says hello."])
        agent = Agent(
            model, reg, workspace_root=Path(tmp),
            max_iterations=5, json_recovery=True,
        )
        result = agent.run("Read x.py.")

    assert agent._json_recovery_fired == 1, "recovery should fire once"
    assert agent._json_recovery_recovered == 1, "recovery should succeed"
    # The recovered read_file call actually ran.
    assert any(r.name == "read_file" and r.success for r in result.tool_results), \
        [r.name for r in result.tool_results]
    # No parse_error surfaced — recovery absorbed it transparently.
    assert not any(r.name == "parse_error" for r in result.tool_results)


def test_default_off_does_not_fire():
    """Default (json_recovery=False): the broken call is NOT recovered; it is
    surfaced as a parse_error and recovery counters stay at zero."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "x.py").write_text("print('hello')\n", encoding="utf-8")
        reg = build_default_registry(tmp)
        model = _StubModel([_BROKEN, _VALID, "Done."])
        agent = Agent(model, reg, workspace_root=Path(tmp), max_iterations=5)
        result = agent.run("Read x.py.")

    assert agent._json_recovery_fired == 0, "default should be OFF"
    assert any(r.name == "parse_error" for r in result.tool_results), \
        [r.name for r in result.tool_results]


def test_clean_run_never_fires():
    """A run where the first tool call parses fine must not invoke recovery
    even with the flag on (no-op when calls already parsed)."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "x.py").write_text("print('hello')\n", encoding="utf-8")
        reg = build_default_registry(tmp)
        model = _StubModel([_VALID, "Done."])
        agent = Agent(
            model, reg, workspace_root=Path(tmp),
            max_iterations=5, json_recovery=True,
        )
        result = agent.run("Read x.py.")

    assert agent._json_recovery_fired == 0
    assert any(r.name == "read_file" and r.success for r in result.tool_results)


def test_recovery_failure_falls_through():
    """If the regeneration also fails to parse, recovery returns None and the
    original parse_error path runs (fired counter up, recovered counter not)."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "x.py").write_text("print('hello')\n", encoding="utf-8")
        reg = build_default_registry(tmp)
        # Broken twice: first turn + the recovery regeneration both unparseable.
        model = _StubModel([_BROKEN, _BROKEN, "Giving up."])
        agent = Agent(
            model, reg, workspace_root=Path(tmp),
            max_iterations=5, json_recovery=True,
        )
        result = agent.run("Read x.py.")

    assert agent._json_recovery_fired >= 1
    assert agent._json_recovery_recovered == 0
    assert any(r.name == "parse_error" for r in result.tool_results)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
