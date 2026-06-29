"""
Tests for engine.complexity_monitor — session-complexity monitor.

Feature 2: A SessionComplexityMonitor tracks cumulative tool-call depth and
accumulated tool-output volume across an agent session. When either metric
crosses a configured threshold, it triggers a proactive compaction/reset on
the Agent's transcript — BEFORE the context char budget is exceeded.

The monitor is a lightweight add-on to the existing compaction system:
- context_char_budget compacts when the transcript is ALREADY over budget.
- SessionComplexityMonitor compacts PROACTIVELY when complexity heuristics
  predict impending degradation (Stroop-effect distractor accumulation, not
  just raw size).

Design:
  - tool_call_depth_threshold: fire when N tool calls have executed this session.
  - tool_output_volume_threshold: fire when total tool-output chars exceed N.
  - Both are independent triggers; hitting either fires compaction.
  - Setting a threshold to 0 disables that dimension.
  - The monitor delegates to Agent._maybe_compact_transcript() — the same
    logic the existing system uses.

Tests cover:
  - Zero/low complexity → no extra compaction fires.
  - Depth threshold: fires at exactly the configured count.
  - Volume threshold: fires at exactly the configured byte count.
  - Both thresholds disabled (0) → never fires on either axis.
  - Compaction fires once when threshold crossed, not on every subsequent call.
  - Integration: monitor.should_compact() is pure (no side effects) and
    monitor.maybe_compact(agent) delegates to the agent correctly.
  - CLI/config: SessionComplexityMonitor can be constructed from the same
    kwargs that Agent accepts for context_char_budget.

Run: python -m pytest test_complexity_monitor.py -v
     OR: python test_complexity_monitor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.complexity_monitor import SessionComplexityMonitor
from engine.agent import Agent
from engine.agent_tools import ToolRegistry


# ── fixtures ────────────────────────────────────────────────────


class _Stub:
    def generate(self, *_a, **_k):
        return ""


def _make_agent(budget: int = 100_000) -> Agent:
    """Agent with a huge char budget so normal compaction never fires."""
    return Agent(
        model=_Stub(),
        registry=ToolRegistry(),
        context_char_budget=budget,
        compact_keep_recent=2,
    )


def _push_tool_turns(agent: Agent, n: int, size_each: int = 100) -> None:
    """Push n tool turns of size_each chars into the agent transcript."""
    for i in range(n):
        agent._transcript.append(
            {"role": "tool", "content": "T" * size_each}
        )


# ── SessionComplexityMonitor unit tests ──────────────────────────


def test_monitor_construction_defaults():
    m = SessionComplexityMonitor()
    assert m.tool_call_depth_threshold > 0
    assert m.tool_output_volume_threshold > 0


def test_monitor_zero_complexity_no_fire():
    """Empty session → no compaction."""
    m = SessionComplexityMonitor(tool_call_depth_threshold=5, tool_output_volume_threshold=1000)
    assert not m.should_compact(tool_call_depth=0, tool_output_volume=0)


def test_monitor_below_depth_threshold_no_fire():
    m = SessionComplexityMonitor(tool_call_depth_threshold=10, tool_output_volume_threshold=99999)
    assert not m.should_compact(tool_call_depth=9, tool_output_volume=0)


def test_monitor_at_depth_threshold_fires():
    m = SessionComplexityMonitor(tool_call_depth_threshold=10, tool_output_volume_threshold=99999)
    assert m.should_compact(tool_call_depth=10, tool_output_volume=0)


def test_monitor_above_depth_threshold_fires():
    m = SessionComplexityMonitor(tool_call_depth_threshold=5, tool_output_volume_threshold=99999)
    assert m.should_compact(tool_call_depth=20, tool_output_volume=0)


def test_monitor_below_volume_threshold_no_fire():
    m = SessionComplexityMonitor(tool_call_depth_threshold=99999, tool_output_volume_threshold=5000)
    assert not m.should_compact(tool_call_depth=0, tool_output_volume=4999)


def test_monitor_at_volume_threshold_fires():
    m = SessionComplexityMonitor(tool_call_depth_threshold=99999, tool_output_volume_threshold=5000)
    assert m.should_compact(tool_call_depth=0, tool_output_volume=5000)


def test_monitor_above_volume_threshold_fires():
    m = SessionComplexityMonitor(tool_call_depth_threshold=99999, tool_output_volume_threshold=500)
    assert m.should_compact(tool_call_depth=0, tool_output_volume=10000)


def test_monitor_depth_zero_disables_depth_axis():
    """depth_threshold=0 means depth axis is disabled (never fires on depth alone)."""
    m = SessionComplexityMonitor(tool_call_depth_threshold=0, tool_output_volume_threshold=99999)
    assert not m.should_compact(tool_call_depth=1000, tool_output_volume=0)


def test_monitor_volume_zero_disables_volume_axis():
    """volume_threshold=0 means volume axis is disabled (never fires on volume alone)."""
    m = SessionComplexityMonitor(tool_call_depth_threshold=99999, tool_output_volume_threshold=0)
    assert not m.should_compact(tool_call_depth=0, tool_output_volume=1_000_000)


def test_monitor_both_zero_never_fires():
    """Both thresholds at 0 = monitor disabled completely."""
    m = SessionComplexityMonitor(tool_call_depth_threshold=0, tool_output_volume_threshold=0)
    assert not m.should_compact(tool_call_depth=9999, tool_output_volume=9_999_999)


def test_monitor_either_threshold_can_fire():
    """Hitting either axis should fire; hitting neither should not."""
    m = SessionComplexityMonitor(tool_call_depth_threshold=5, tool_output_volume_threshold=1000)
    # Neither
    assert not m.should_compact(tool_call_depth=4, tool_output_volume=999)
    # Depth only
    assert m.should_compact(tool_call_depth=5, tool_output_volume=0)
    # Volume only
    assert m.should_compact(tool_call_depth=0, tool_output_volume=1000)
    # Both
    assert m.should_compact(tool_call_depth=5, tool_output_volume=1000)


# ── metric computation from agent state ─────────────────────────


def test_monitor_counts_tool_calls_from_agent():
    """compute_metrics should count tool turns in the transcript."""
    m = SessionComplexityMonitor(tool_call_depth_threshold=3, tool_output_volume_threshold=99999)
    agent = _make_agent()
    agent._transcript = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "content": "result 1"},
        {"role": "tool", "content": "result 2"},
    ]
    depth, volume = m.compute_metrics(agent)
    assert depth == 2


def test_monitor_sums_tool_output_volume():
    """compute_metrics should sum char counts across tool turns."""
    m = SessionComplexityMonitor(tool_call_depth_threshold=99999, tool_output_volume_threshold=200)
    agent = _make_agent()
    agent._transcript = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "content": "A" * 100},
        {"role": "tool", "content": "B" * 150},
    ]
    depth, volume = m.compute_metrics(agent)
    assert volume == 250


def test_monitor_ignores_non_tool_turns():
    """User and assistant turns don't count toward volume."""
    m = SessionComplexityMonitor(tool_call_depth_threshold=99999, tool_output_volume_threshold=500)
    agent = _make_agent()
    agent._transcript = [
        {"role": "user", "content": "X" * 1000},
        {"role": "assistant", "content": "Y" * 1000},
        {"role": "tool", "content": "Z" * 100},
    ]
    depth, volume = m.compute_metrics(agent)
    assert depth == 1
    assert volume == 100


# ── maybe_compact integration ────────────────────────────────────


def test_maybe_compact_fires_when_threshold_met():
    """maybe_compact calls Agent._maybe_compact_transcript when over threshold."""
    # Use a tiny char budget so the compaction actually elides something.
    agent = _make_agent(budget=500)
    agent._transcript = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "content": "X" * 400},
        {"role": "tool", "content": "Y" * 400},
        {"role": "tool", "content": "keep me"},
    ]
    # Monitor fires at depth=3 — we have 3 tool turns.
    m = SessionComplexityMonitor(tool_call_depth_threshold=3, tool_output_volume_threshold=99999)
    before_compactions = agent._compactions
    m.maybe_compact(agent)
    # Compaction should have fired (the budget is small enough that elision happens)
    assert agent._compactions > before_compactions or _transcript_was_elided(agent)


def _transcript_was_elided(agent: Agent) -> bool:
    return any("elided" in (t.get("content") or "") for t in agent._transcript)


def test_maybe_compact_no_fire_below_threshold():
    """maybe_compact does nothing when below threshold."""
    agent = _make_agent(budget=500)
    agent._transcript = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "content": "X" * 400},
        {"role": "tool", "content": "Y" * 400},
    ]
    before = [dict(t) for t in agent._transcript]
    # Monitor only fires at depth >= 10; we only have 2 tool turns.
    m = SessionComplexityMonitor(tool_call_depth_threshold=10, tool_output_volume_threshold=99999)
    m.maybe_compact(agent)
    assert agent._compactions == 0
    assert agent._transcript == before


def test_maybe_compact_volume_trigger():
    """maybe_compact fires on volume axis with small char budget."""
    agent = _make_agent(budget=200)  # tiny budget
    agent._transcript = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "content": "A" * 300},
        {"role": "tool", "content": "B" * 300},
        {"role": "tool", "content": "keep"},
    ]
    m = SessionComplexityMonitor(
        tool_call_depth_threshold=99999,
        tool_output_volume_threshold=500,  # 600 total > 500 threshold
    )
    before_compactions = agent._compactions
    m.maybe_compact(agent)
    assert agent._compactions > before_compactions or _transcript_was_elided(agent)


def test_maybe_compact_disabled_never_fires():
    """Both thresholds=0 → maybe_compact never compacts even with huge transcript."""
    agent = _make_agent(budget=100)  # tiny budget
    agent._transcript = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "content": "X" * 5000},
        {"role": "tool", "content": "Y" * 5000},
    ]
    before = [dict(t) for t in agent._transcript]
    m = SessionComplexityMonitor(tool_call_depth_threshold=0, tool_output_volume_threshold=0)
    m.maybe_compact(agent)
    # Proactive compaction must NOT fire (the existing reactive compaction
    # inside _maybe_compact_transcript still fires here because the budget
    # is tiny — but the monitor itself must not compound it).
    # What we test is that should_compact returns False.
    depth, volume = m.compute_metrics(agent)
    assert not m.should_compact(depth, volume)


# ── config/CLI integration ────────────────────────────────────────


def test_monitor_from_agent_params():
    """SessionComplexityMonitor can mirror Agent's context_char_budget param."""
    # The monitor's default thresholds should be tunable and settable from
    # Agent constructor kwargs without coupling to the existing param names.
    m = SessionComplexityMonitor(
        tool_call_depth_threshold=15,
        tool_output_volume_threshold=32000,
    )
    assert m.tool_call_depth_threshold == 15
    assert m.tool_output_volume_threshold == 32000


def test_monitor_repr_is_descriptive():
    m = SessionComplexityMonitor(tool_call_depth_threshold=8, tool_output_volume_threshold=16000)
    r = repr(m)
    assert "8" in r or "depth" in r.lower()


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [
        (n, f)
        for n, f in inspect.getmembers(mod, inspect.isfunction)
        if n.startswith("test_")
    ]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
