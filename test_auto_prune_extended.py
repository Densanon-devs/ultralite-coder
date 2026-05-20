"""Tests for A4 promotion: auto-prune tools when extended_tools=True
(2026-05-19 PM)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from engine.agent import Agent
from engine.agent_builtins import build_default_registry
from engine.agent_tools import ToolRegistry, ToolSchema


def _stub_model():
    m = MagicMock()
    m.generate.return_value = "done"
    return m


def _tool(name, desc="d"):
    return ToolSchema(
        name=name,
        description=desc,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        function=lambda **kw: None,
    )


class TestRegistryExtendedFlag(unittest.TestCase):
    def test_lean_registry_flag_false(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            reg = build_default_registry(tmp, extended_tools=False)
            self.assertFalse(reg._extended_tools_enabled)

    def test_extended_registry_flag_true(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            reg = build_default_registry(tmp, extended_tools=True)
            self.assertTrue(reg._extended_tools_enabled)

    def test_bare_registry_default_false(self):
        reg = ToolRegistry()
        self.assertFalse(reg._extended_tools_enabled)


class TestResolvePruneK(unittest.TestCase):
    def _agent(self, registry, **kw):
        return Agent(model=_stub_model(), registry=registry, **kw)

    def test_lean_registry_no_pruning(self):
        reg = ToolRegistry()  # _extended_tools_enabled = False
        agent = self._agent(reg)
        self.assertIsNone(agent._resolve_prune_k())

    def test_extended_registry_auto_engages_default_k(self):
        reg = ToolRegistry()
        reg._extended_tools_enabled = True
        agent = self._agent(reg)
        self.assertEqual(agent._resolve_prune_k(), 15)

    def test_explicit_k_overrides(self):
        reg = ToolRegistry()
        reg._extended_tools_enabled = True
        agent = self._agent(reg, auto_prune_tools_k=8)
        self.assertEqual(agent._resolve_prune_k(), 8)

    def test_explicit_zero_disables_even_on_extended(self):
        reg = ToolRegistry()
        reg._extended_tools_enabled = True
        agent = self._agent(reg, auto_prune_tools_k=0)
        self.assertIsNone(agent._resolve_prune_k())

    def test_explicit_k_on_lean_registry(self):
        reg = ToolRegistry()  # not extended
        agent = self._agent(reg, auto_prune_tools_k=6)
        self.assertEqual(agent._resolve_prune_k(), 6)


class TestComputeToolBlockOverride(unittest.TestCase):
    def _extended_registry(self):
        # 8 floor + several extras = >15 so pruning actually drops some
        reg = ToolRegistry()
        reg._extended_tools_enabled = True
        for n in ("read_file", "write_file", "edit_file", "list_dir",
                  "glob", "grep", "run_tests", "run_bash"):
            reg.register(_tool(n))
        for n in ("git_status", "git_commit", "rename_symbol",
                  "read_function", "checkpoint", "restore_checkpoint",
                  "find_definition", "find_references", "format_code",
                  "lint_file"):
            reg.register(_tool(n, f"{n} description"))
        return reg

    def test_override_built_when_extended_and_drops_tools(self):
        reg = self._extended_registry()  # 18 tools total
        agent = Agent(model=_stub_model(), registry=reg)
        override = agent._compute_tool_block_override("rename a symbol")
        self.assertIsNotNone(override)
        # The override block must be shorter than the full block.
        full = reg.hermes_system_block()
        self.assertLess(len(override), len(full))

    def test_no_override_when_pruning_drops_nothing(self):
        # Lean registry with exactly 8 tools — k=15 prunes nothing,
        # so override should be None (use byte-identical full block).
        reg = ToolRegistry()
        reg._extended_tools_enabled = True
        for n in ("read_file", "write_file", "edit_file", "list_dir"):
            reg.register(_tool(n))
        agent = Agent(model=_stub_model(), registry=reg)
        self.assertIsNone(agent._compute_tool_block_override("anything"))

    def test_no_override_on_lean_registry(self):
        reg = ToolRegistry()  # not extended
        for n in ("read_file", "write_file"):
            reg.register(_tool(n))
        agent = Agent(model=_stub_model(), registry=reg)
        self.assertIsNone(agent._compute_tool_block_override("any goal"))

    def test_floor_tools_survive_pruning(self):
        reg = self._extended_registry()
        agent = Agent(model=_stub_model(), registry=reg)
        override = agent._compute_tool_block_override("unrelated xyzzy goal")
        # Floor tools must always be present even with no goal overlap.
        self.assertIn('"name": "read_file"', override)
        self.assertIn('"name": "edit_file"', override)
        self.assertIn('"name": "run_tests"', override)


class TestRunIntegration(unittest.TestCase):
    """End-to-end: a run() on an extended registry should set the
    override; a run() on a lean registry should not."""

    def _extended_registry(self):
        reg = ToolRegistry()
        reg._extended_tools_enabled = True
        for n in ("read_file", "write_file", "edit_file", "list_dir",
                  "glob", "grep", "run_tests", "run_bash",
                  "git_status", "git_commit", "rename_symbol",
                  "read_function", "checkpoint", "find_definition",
                  "find_references", "format_code", "lint_file"):
            reg.register(_tool(n, f"{n} does {n}"))
        return reg

    def test_run_sets_override_on_extended(self):
        reg = self._extended_registry()
        agent = Agent(model=_stub_model(), registry=reg, max_iterations=1)
        agent.run("rename the helper symbol")
        self.assertIsNotNone(agent._tool_block_override)

    def test_run_no_override_on_lean(self):
        reg = ToolRegistry()
        for n in ("read_file", "write_file", "edit_file"):
            reg.register(_tool(n))
        agent = Agent(model=_stub_model(), registry=reg, max_iterations=1)
        agent.run("do something")
        self.assertIsNone(agent._tool_block_override)

    def test_system_prompt_uses_override(self):
        reg = self._extended_registry()
        agent = Agent(model=_stub_model(), registry=reg, max_iterations=1)
        agent.run("rename a symbol")
        sysprompt_pruned = agent._system_prompt()
        # Force the override off and re-render to get the full-tool variant.
        agent._tool_block_override = None
        sysprompt_full = agent._system_prompt()
        # The pruned system prompt must be strictly shorter than the full one.
        self.assertLess(len(sysprompt_pruned), len(sysprompt_full))
        # Floor tools present in both.
        self.assertIn('"name": "read_file"', sysprompt_pruned)

    def test_explicit_disable_keeps_full_set_on_extended(self):
        reg = self._extended_registry()
        agent = Agent(
            model=_stub_model(), registry=reg,
            auto_prune_tools_k=0, max_iterations=1,
        )
        agent.run("rename a symbol")
        self.assertIsNone(agent._tool_block_override)


if __name__ == "__main__":
    unittest.main()
