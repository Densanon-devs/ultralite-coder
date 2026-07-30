"""Cache-stability regression tests for the agent's prompt prefix (Task A2, 2026-05-19).

Why this exists. Append-only KV reuse (B4 + llama.cpp's `n_keep`) only delivers
the 15-40% per-turn latency win if the prompt prefix is byte-identical across
turns. A single drifting byte — a timestamp, a re-ordered dict, a random ID —
invalidates the cache and the per-turn cost goes from O(suffix) back to
O(prefix + suffix).

These tests are the regression guard. They DON'T verify any specific prompt
content (that's the existing system prompt unit tests' job); they verify that
the prompt is deterministic.

Per the research synthesis (2026-05-19):
> Unsloth/Claude-Code finding (March 2026) showed a single shifting header
> destroys 100% of cache hits.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from engine.agent import Agent
from engine.agent_tools import ToolRegistry, ToolSchema


def _stub_model():
    """Return a model stub with a static generate() that never gets called
    by the prompt-stability tests anyway. _system_prompt + prefix_hash are
    pure and don't touch the model."""
    m = MagicMock()
    m.generate.return_value = "stub"
    return m


def _stub_registry(tool_names=("read_file", "edit_file", "write_file")):
    reg = ToolRegistry()
    for name in tool_names:
        reg.register(ToolSchema(
            name=name,
            description=f"stub for {name}",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            function=lambda **kw: None,
        ))
    return reg


def _make_agent(**overrides):
    kwargs = dict(model=_stub_model(), registry=_stub_registry())
    kwargs.update(overrides)
    return Agent(**kwargs)


class TestSystemPromptIsByteIdentical(unittest.TestCase):
    """The prefix MUST be byte-identical across repeated calls on a single
    Agent instance (within one run() call's lifetime)."""

    def test_two_consecutive_calls_match(self):
        a = _make_agent()
        first = a._system_prompt()
        second = a._system_prompt()
        self.assertEqual(first, second)

    def test_ten_consecutive_calls_all_match(self):
        a = _make_agent()
        outputs = {a._system_prompt() for _ in range(10)}
        self.assertEqual(len(outputs), 1)

    def test_prefix_hash_matches_sha256_of_system_prompt(self):
        import hashlib
        a = _make_agent()
        expected = hashlib.sha256(a._system_prompt().encode("utf-8")).hexdigest()
        self.assertEqual(a.prefix_hash(), expected)

    def test_prefix_hash_is_deterministic(self):
        a = _make_agent()
        hashes = {a.prefix_hash() for _ in range(20)}
        self.assertEqual(len(hashes), 1)


class TestPrefixDependsOnRealInputs(unittest.TestCase):
    """Sanity: if a *real* input changes (system_prompt_extra, memory_block,
    goal_augmentor_block, tool registry), the hash MUST change. Otherwise
    we'd be silently caching across semantically-different prompts."""

    def test_system_prompt_extra_change_breaks_hash(self):
        a = _make_agent(system_prompt_extra="hint A")
        h1 = a.prefix_hash()
        a.system_prompt_extra = "hint B"
        h2 = a.prefix_hash()
        self.assertNotEqual(h1, h2)

    def test_memory_block_change_breaks_hash(self):
        a = _make_agent()
        h1 = a.prefix_hash()
        a._memory_block = "# notes\n- something"
        h2 = a.prefix_hash()
        self.assertNotEqual(h1, h2)

    def test_goal_augmentor_block_change_breaks_hash(self):
        a = _make_agent()
        h1 = a.prefix_hash()
        a._goal_augmentor_block = "augmentor: blah"
        h2 = a.prefix_hash()
        self.assertNotEqual(h1, h2)

    def test_tool_change_breaks_hash(self):
        a1 = _make_agent(registry=_stub_registry(("read_file",)))
        a2 = _make_agent(registry=_stub_registry(("read_file", "write_file")))
        self.assertNotEqual(a1.prefix_hash(), a2.prefix_hash())


class TestRegistryDeterminism(unittest.TestCase):
    """Tool registries that hold identical tools in identical registration
    order must produce identical Hermes blocks."""

    def test_same_registration_order_produces_same_block(self):
        r1 = _stub_registry(("read_file", "edit_file"))
        r2 = _stub_registry(("read_file", "edit_file"))
        self.assertEqual(r1.hermes_system_block(), r2.hermes_system_block())

    def test_block_is_deterministic_across_calls(self):
        r = _stub_registry()
        outs = {r.hermes_system_block() for _ in range(10)}
        self.assertEqual(len(outs), 1)


class TestBuildPromptOnlyGrowsBySuffix(unittest.TestCase):
    """The full ChatML prompt MUST share its prefix character-for-character
    with the previous turn's prompt — only the suffix (new transcript entries)
    differs. This is what the KV cache needs to deliver the win."""

    def _prompt_for(self, agent: Agent, transcript: list) -> str:
        agent._transcript = transcript
        return agent._build_prompt()

    def test_appending_a_turn_only_extends(self):
        agent = _make_agent()
        t1 = [{"role": "user", "content": "first goal"}]
        t2 = t1 + [{"role": "assistant", "content": "first reply"}]
        p1 = self._prompt_for(agent, t1)
        p2 = self._prompt_for(agent, t2)
        # p2 must start with p1 minus the trailing "<|im_start|>assistant\n"
        # tail (which is the open assistant header). The system block, the
        # user turn, and everything up to that tail must be identical.
        # We strip the assistant header from p1 before comparing.
        common = p1[: -len("<|im_start|>assistant\n")]
        self.assertTrue(p2.startswith(common), "prefix drifted between turns")

    def test_multiple_appends_all_share_common_prefix(self):
        agent = _make_agent()
        transcripts = [
            [{"role": "user", "content": "goal"}],
            [{"role": "user", "content": "goal"},
             {"role": "assistant", "content": "reply 1"}],
            [{"role": "user", "content": "goal"},
             {"role": "assistant", "content": "reply 1"},
             {"role": "tool", "content": "tool result"}],
        ]
        prompts = [self._prompt_for(agent, t) for t in transcripts]
        # Strip the trailing open-assistant header from each, then verify
        # monotonic prefix containment.
        suffix = "<|im_start|>assistant\n"
        bodies = [p[: -len(suffix)] for p in prompts]
        for i in range(1, len(bodies)):
            self.assertTrue(
                bodies[i].startswith(bodies[i - 1]),
                f"prefix drifted at transcript step {i}",
            )


if __name__ == "__main__":
    unittest.main()
