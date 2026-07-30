"""Tests for goal-aware tool pruning (Task A4, 2026-05-19)."""

from __future__ import annotations

import unittest

from engine.agent_tools import (
    ToolRegistry,
    ToolSchema,
    _score_tool_against_goal,
    _tokenize_for_pruning,
)


def _tool(name, description, params=None):
    return ToolSchema(
        name=name,
        description=description,
        parameters=params or {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        function=lambda **kw: None,
    )


class TestTokenizer(unittest.TestCase):
    def test_basic_split(self):
        tokens = _tokenize_for_pruning("rename function helper")
        self.assertIn("rename", tokens)
        self.assertIn("helper", tokens)
        # "function" is a generic stopword
        self.assertNotIn("function", tokens)

    def test_underscore_kept_as_one_token(self):
        tokens = _tokenize_for_pruning("call run_tests after edit")
        self.assertIn("run_tests", tokens)
        self.assertIn("after", tokens)
        self.assertIn("edit", tokens)

    def test_short_tokens_dropped(self):
        tokens = _tokenize_for_pruning("do it on a py")
        # All ≤2-char or stopwords
        self.assertEqual(tokens, set())

    def test_case_insensitive(self):
        self.assertEqual(
            _tokenize_for_pruning("Rename HELPER"),
            _tokenize_for_pruning("rename helper"),
        )

    def test_empty(self):
        self.assertEqual(_tokenize_for_pruning(""), set())


class TestScoring(unittest.TestCase):
    def test_matching_token_in_description(self):
        t = _tool("git_status", "show uncommitted git changes")
        score = _score_tool_against_goal(
            t, _tokenize_for_pruning("check uncommitted changes")
        )
        # "uncommitted" + "changes" — both match
        self.assertEqual(score, 2)

    def test_matching_token_in_name(self):
        t = _tool("rename_symbol", "AST-aware rename")
        score = _score_tool_against_goal(
            t, _tokenize_for_pruning("rename foo to bar")
        )
        self.assertGreaterEqual(score, 1)

    def test_zero_when_no_overlap(self):
        t = _tool("frobnicate", "spin the widget")
        score = _score_tool_against_goal(
            t, _tokenize_for_pruning("fix authentication bug")
        )
        self.assertEqual(score, 0)

    def test_param_property_names_count(self):
        t = _tool(
            "search_index",
            "look up records",
            params={
                "type": "object",
                "properties": {
                    "regex_pattern": {"type": "string", "description": "PCRE regex"},
                },
            },
        )
        score = _score_tool_against_goal(
            t, _tokenize_for_pruning("write a regex_pattern")
        )
        self.assertGreaterEqual(score, 1)


class TestPruningFloorSet(unittest.TestCase):
    def _registry_with_floor_plus_extras(self, extras):
        reg = ToolRegistry()
        for name in ("read_file", "write_file", "edit_file", "list_dir",
                     "glob", "grep", "run_tests", "run_bash"):
            reg.register(_tool(name, f"do {name}"))
        for name, desc in extras:
            reg.register(_tool(name, desc))
        return reg

    def test_floor_always_included_even_with_no_overlap(self):
        reg = self._registry_with_floor_plus_extras([
            ("git_status", "show uncommitted git changes"),
            ("rename_symbol", "AST rename"),
        ])
        tools = reg.enabled_tools_for_goal("unrelated topic xyzzy", k=10)
        names = {t.name for t in tools}
        for floor_tool in ("read_file", "write_file", "edit_file"):
            self.assertIn(floor_tool, names)

    def test_relevant_extra_beats_irrelevant_extra(self):
        reg = self._registry_with_floor_plus_extras([
            ("git_status", "show uncommitted git changes"),
            ("rename_symbol", "AST rename of identifiers"),
            ("frobnicate", "spin the widget"),
        ])
        tools = reg.enabled_tools_for_goal(
            "rename helper to helperv2", k=10,
        )
        names = {t.name for t in tools}
        self.assertIn("rename_symbol", names)
        # All extras might fit at k=10 (8 floor + 3 extras = 11, but k=10 caps)
        # frobnicate should be the one excluded
        self.assertNotIn("frobnicate", names)

    def test_k_smaller_than_floor_still_returns_floor(self):
        # Floor set is 8 tools. Asking for k=3 should still return all 8
        # floor tools (floor is mandatory, not capped).
        reg = self._registry_with_floor_plus_extras([])
        tools = reg.enabled_tools_for_goal("anything", k=3)
        self.assertEqual(len(tools), 8)

    def test_custom_floor_set(self):
        reg = self._registry_with_floor_plus_extras([
            ("rename_symbol", "AST rename"),
        ])
        tools = reg.enabled_tools_for_goal(
            "rename helper", k=2, floor_set=frozenset({"read_file"}),
        )
        names = {t.name for t in tools}
        self.assertIn("read_file", names)
        # rename_symbol would be top-scored, but k=2 means 1 floor + 1 extra
        self.assertIn("rename_symbol", names)
        self.assertEqual(len(tools), 2)

    def test_deterministic_tiebreak(self):
        # Two extras with identical 0-overlap score must come out in
        # registration order
        reg = self._registry_with_floor_plus_extras([
            ("alpha_tool", "first extra"),
            ("beta_tool", "second extra"),
            ("gamma_tool", "third extra"),
        ])
        tools = reg.enabled_tools_for_goal(
            "completely unrelated query", k=11,
        )
        # Extras (after floor) should appear in registration order
        extra_names = [t.name for t in tools if t.name not in reg.DEFAULT_FLOOR_SET]
        self.assertEqual(extra_names, ["alpha_tool", "beta_tool", "gamma_tool"])

    def test_empty_registry_returns_empty(self):
        reg = ToolRegistry()
        self.assertEqual(reg.enabled_tools_for_goal("any goal"), [])


class TestHermesSystemBlockOptIn(unittest.TestCase):
    """Verify the pruning-aware system block matches the un-pruned form
    when no tools arg is passed (preserving the pre-A4 default), and
    that explicit tools= produces a smaller block."""

    def test_default_behavior_unchanged(self):
        reg = ToolRegistry()
        reg.register(_tool("read_file", "read"))
        reg.register(_tool("rename_symbol", "rename"))
        full = reg.hermes_system_block()
        explicit = reg.hermes_system_block(reg.enabled_tools())
        self.assertEqual(full, explicit)

    def test_pruned_block_is_smaller(self):
        reg = ToolRegistry()
        for n in ("read_file", "write_file", "edit_file", "list_dir",
                  "glob", "grep", "run_tests", "run_bash",
                  "git_status", "rename_symbol", "frobnicate"):
            reg.register(_tool(n, f"description for {n}"))
        full = reg.hermes_system_block()
        pruned_tools = reg.enabled_tools_for_goal("read a file", k=8)
        pruned = reg.hermes_system_block(pruned_tools)
        self.assertLess(len(pruned), len(full))
        # Pruned still contains the floor read_file
        self.assertIn('"name": "read_file"', pruned)
        # Pruned excludes frobnicate (no overlap, dropped at k=8)
        self.assertNotIn('"name": "frobnicate"', pruned)


if __name__ == "__main__":
    unittest.main()
