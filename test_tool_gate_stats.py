"""Tests for the G-STEP gate observability counters (Task 7, 2026-05-19)."""

from __future__ import annotations

import unittest

from engine.tool_gate import (
    GateState,
    GateStats,
    ToolGateConfig,
    check,
)


class TestGateStatsInit(unittest.TestCase):
    def test_starts_at_zero(self):
        s = GateStats()
        self.assertEqual(s.checks, 0)
        self.assertEqual(s.allowed, 0)
        self.assertEqual(s.total_rejected, 0)
        self.assertEqual(s.fire_rate, 0.0)
        self.assertEqual(s.rejected_by_tool, {})

    def test_gatestate_has_stats_field(self):
        st = GateState()
        self.assertIsInstance(st.stats, GateStats)


class TestCounterUpdates(unittest.TestCase):
    def test_disabled_gate_records_allowed_disabled(self):
        state = GateState()
        cfg = ToolGateConfig(enabled=False)
        check("read_file", {"path": "foo.py"}, state, cfg)
        self.assertEqual(state.stats.checks, 1)
        self.assertEqual(state.stats.allowed_disabled, 1)
        self.assertEqual(state.stats.allowed, 1)
        self.assertEqual(state.stats.total_rejected, 0)

    def test_action_tool_bypass_recorded(self):
        state = GateState()
        cfg = ToolGateConfig(enabled=True)
        check("write_file", {"path": "out.py", "content": "x = 1"}, state, cfg)
        self.assertEqual(state.stats.allowed_action_tool, 1)
        self.assertEqual(state.stats.total_rejected, 0)

    def test_anchor_ok_recorded(self):
        state = GateState(seen_corpus="please read calc.py")
        cfg = ToolGateConfig(enabled=True)
        check("read_file", {"path": "calc.py"}, state, cfg)
        self.assertEqual(state.stats.allowed_anchor_ok, 1)
        self.assertEqual(state.stats.total_rejected, 0)

    def test_no_key_arg_recorded(self):
        state = GateState()
        cfg = ToolGateConfig(enabled=True)
        # 'plan' isn't in EXPLORATION_TOOLS or ACTION_TOOLS, no key arg
        check("plan", {"action": "set", "items": ["a"]}, state, cfg)
        self.assertEqual(state.stats.allowed_no_key_arg, 1)

    def test_rejection_anchor_recorded(self):
        state = GateState(seen_corpus="goal mentions only foo.py")
        cfg = ToolGateConfig(enabled=True)
        check("read_file", {"path": "totally_unrelated.py"}, state, cfg)
        self.assertEqual(state.stats.rejected_anchor, 1)
        self.assertEqual(state.stats.total_rejected, 1)
        self.assertEqual(state.stats.rejected_by_tool["read_file"], 1)

    def test_rejection_exploration_cap_recorded(self):
        state = GateState(seen_corpus="read foo.py")
        state.consecutive_exploration = 4  # at the default cap
        cfg = ToolGateConfig(enabled=True)
        check("read_file", {"path": "foo.py"}, state, cfg)
        self.assertEqual(state.stats.rejected_exploration_cap, 1)
        self.assertEqual(state.stats.total_rejected, 1)


class TestFireRate(unittest.TestCase):
    def test_fire_rate_only_counts_considered(self):
        state = GateState(seen_corpus="known.py")
        cfg = ToolGateConfig(enabled=True)
        # 2 action-tool bypasses (denominator ignores these)
        check("write_file", {"path": "out.py", "content": "x"}, state, cfg)
        check("edit_file", {"path": "x.py", "old_string": "a", "new_string": "b"}, state, cfg)
        # 2 anchor-OK calls (denominator includes, numerator excludes)
        check("read_file", {"path": "known.py"}, state, cfg)
        check("read_file", {"path": "known.py"}, state, cfg)
        # 1 anchor rejection (numerator includes)
        check("read_file", {"path": "invented.py"}, state, cfg)
        # 5 checks, 2 action-tool bypassed, 3 considered, 1 rejected
        self.assertEqual(state.stats.fire_rate, 1 / 3)

    def test_fire_rate_zero_when_disabled(self):
        state = GateState()
        cfg = ToolGateConfig(enabled=False)
        for _ in range(10):
            check("read_file", {"path": "x.py"}, state, cfg)
        # 10 checks all allowed_disabled, 0 considered, fire_rate 0
        self.assertEqual(state.stats.fire_rate, 0.0)

    def test_fire_rate_zero_for_empty_run(self):
        s = GateStats()
        self.assertEqual(s.fire_rate, 0.0)


class TestSummary(unittest.TestCase):
    def test_summary_keys(self):
        s = GateStats()
        d = s.summary()
        for key in (
            "checks",
            "allowed",
            "allowed_disabled",
            "allowed_action_tool",
            "allowed_anchor_ok",
            "allowed_no_key_arg",
            "rejected_anchor",
            "rejected_exploration_cap",
            "total_rejected",
            "fire_rate",
            "rejected_by_tool",
        ):
            self.assertIn(key, d, f"summary() missing {key!r}")

    def test_summary_after_mixed_run(self):
        state = GateState(seen_corpus="goal references calc.py and tests/")
        cfg = ToolGateConfig(enabled=True)
        # Sequence: 2 anchored reads, 1 invented anchor, 1 write, then
        # one rejected-on-cap (but only after some streak rebuild)
        check("read_file", {"path": "calc.py"}, state, cfg)
        state.record_call("read_file")
        check("read_file", {"path": "calc.py"}, state, cfg)
        state.record_call("read_file")
        check("read_file", {"path": "invented.py"}, state, cfg)  # anchor reject
        check("write_file", {"path": "calc.py", "content": "x"}, state, cfg)
        state.record_call("write_file")

        d = state.stats.summary()
        self.assertEqual(d["checks"], 4)
        self.assertEqual(d["allowed_anchor_ok"], 2)
        self.assertEqual(d["rejected_anchor"], 1)
        self.assertEqual(d["allowed_action_tool"], 1)
        self.assertEqual(d["total_rejected"], 1)
        self.assertIn("read_file", d["rejected_by_tool"])

    def test_summary_is_json_serializable(self):
        import json
        s = GateStats()
        s.record_reject("read_file", "anchor")
        json.dumps(s.summary())


class TestStatsSurviveBackwardsCompat(unittest.TestCase):
    """Existing call sites construct GateState() with no args — verify the
    new field doesn't break them."""

    def test_default_construction_works(self):
        s = GateState()
        self.assertIsInstance(s.stats, GateStats)

    def test_each_state_has_own_stats(self):
        # dataclass field(default_factory=...) must NOT share state across instances
        a = GateState()
        b = GateState()
        a.stats.checks = 5
        self.assertEqual(b.stats.checks, 0)


if __name__ == "__main__":
    unittest.main()
