"""
Tests for engine.persist_runner — the relaunch loop.

Model-free: `run_until_done` takes the round-runner as a callable, so the whole
loop is exercised with a stub that mutates a real mission file on disk.

The behaviour that matters is knowing when to STOP. An unconditional relaunch
loop on a stuck 14B burns GPU indefinitely, so these tests pin the three exit
paths (complete / stalled / budget) and the "a note counts as progress" rule.

Run: python -m pytest test_persist_runner.py -v
     OR just: python test_persist_runner.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.mission import Mission
from engine.persist_runner import (
    Decision,
    RelaunchPolicy,
    Snapshot,
    continuation_goal,
    run_until_done,
    snapshot_of,
)


def _mission(ws: Path, titles: list[str]) -> Mission:
    m = Mission(goal="test goal")
    m.set_steps(titles)
    m.save(ws)
    return m


# ── Snapshot ────────────────────────────────────────────────────

def test_snapshot_complete():
    assert Snapshot(done=3, total=3).complete
    assert not Snapshot(done=2, total=3).complete
    assert not Snapshot(done=0, total=0).complete, "no steps is not 'complete'"


def test_a_new_note_counts_as_progress():
    """A round that only recorded a finding still moved forward."""
    before = Snapshot(done=1, total=3, notes=0)
    assert Snapshot(done=1, total=3, notes=1).advanced_past(before)
    assert Snapshot(done=2, total=3, notes=0).advanced_past(before)
    assert not Snapshot(done=1, total=3, notes=0).advanced_past(before)


# ── policy ──────────────────────────────────────────────────────

def test_policy_completes():
    d, why = RelaunchPolicy().decide(Snapshot(1, 3), Snapshot(3, 3))
    assert d is Decision.COMPLETE, why


def test_policy_continues_while_progressing():
    d, _ = RelaunchPolicy(max_rounds=5).decide(Snapshot(0, 4), Snapshot(1, 4))
    assert d is Decision.CONTINUE


def test_policy_stalls_after_two_flat_rounds():
    p = RelaunchPolicy(max_rounds=10, stall_limit=2)
    flat = Snapshot(1, 4)
    d1, _ = p.decide(flat, flat)
    assert d1 is Decision.CONTINUE, "one flat round is tolerated"
    d2, why = p.decide(flat, flat)
    assert d2 is Decision.STALLED, why


def test_progress_resets_the_stall_counter():
    p = RelaunchPolicy(max_rounds=10, stall_limit=2)
    p.decide(Snapshot(1, 5), Snapshot(1, 5))          # flat -> stalls=1
    p.decide(Snapshot(1, 5), Snapshot(2, 5))          # progress -> reset
    assert p.stalls == 0
    d, _ = p.decide(Snapshot(2, 5), Snapshot(2, 5))   # flat again -> stalls=1
    assert d is Decision.CONTINUE


def test_policy_respects_round_budget():
    p = RelaunchPolicy(max_rounds=2, stall_limit=99)
    assert p.decide(Snapshot(0, 9), Snapshot(1, 9))[0] is Decision.CONTINUE
    d, why = p.decide(Snapshot(1, 9), Snapshot(2, 9))
    assert d is Decision.BUDGET, why


def test_untracked_mission_does_not_loop():
    d, why = RelaunchPolicy().decide(Snapshot(), Snapshot())
    assert d is Decision.UNTRACKED, why


# ── driver ──────────────────────────────────────────────────────

def test_run_until_done_completes_across_rounds():
    """Stub finishes one step per round; loop should end at COMPLETE."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _mission(ws, ["one", "two", "three"])
        rounds: list[int] = []

        def run_round(goal: str, idx: int):
            rounds.append(idx)
            m = Mission.load(ws)
            nxt = m.next_pending_step()
            if nxt:
                m.mark_step_done(nxt.n if hasattr(nxt, "n") else idx)
                m.save(ws)
            return f"round{idx}"

        out = run_until_done(workspace=ws, goal="do it", run_round=run_round,
                             max_rounds=10)
        assert out["decision"] == Decision.COMPLETE.value, out
        assert out["done"] == out["total"] == 3
        assert rounds == [1, 2, 3], rounds


def test_run_until_done_gives_up_on_a_stuck_model():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _mission(ws, ["one", "two"])
        calls = []

        def run_round(goal: str, idx: int):
            calls.append(idx)          # does nothing — simulates a stuck model
            return None

        out = run_until_done(workspace=ws, goal="do it", run_round=run_round,
                             max_rounds=20, stall_limit=2)
        assert out["decision"] == Decision.STALLED.value, out
        assert len(calls) == 2, f"should abandon quickly, ran {len(calls)} rounds"


def test_run_until_done_honours_budget():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _mission(ws, [f"s{i}" for i in range(20)])
        seen = []

        def run_round(goal: str, idx: int):
            seen.append(idx)
            m = Mission.load(ws)
            nxt = m.next_pending_step()
            if nxt:
                m.mark_step_done(getattr(nxt, "n", idx))
                m.save(ws)
            return None

        out = run_until_done(workspace=ws, goal="g", run_round=run_round,
                             max_rounds=3, stall_limit=99)
        assert out["decision"] == Decision.BUDGET.value, out
        assert len(seen) == 3


def test_first_round_uses_the_original_goal():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        _mission(ws, ["one", "two"])
        goals = []

        def run_round(goal: str, idx: int):
            goals.append(goal)
            m = Mission.load(ws)
            nxt = m.next_pending_step()
            if nxt:
                m.mark_step_done(getattr(nxt, "n", idx))
                m.save(ws)
            return None

        run_until_done(workspace=ws, goal="ORIGINAL GOAL", run_round=run_round,
                       max_rounds=5)
        assert goals[0] == "ORIGINAL GOAL"
        assert len(goals) > 1
        assert "ORIGINAL GOAL" != goals[1], "later rounds should be continuation prompts"
        assert "not start over" in goals[1].lower()


def test_continuation_goal_names_the_next_step():
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        m = _mission(ws, ["alpha step", "beta step"])
        m.mark_step_done(1)
        m.save(ws)
        text = continuation_goal(ws, "orig")
        assert "beta step" in text, text
        assert "1/2" in text


def test_continuation_goal_without_mission_falls_back():
    with tempfile.TemporaryDirectory() as td:
        assert continuation_goal(Path(td), "orig goal") == "orig goal"


def test_snapshot_of_missing_mission():
    with tempfile.TemporaryDirectory() as td:
        assert snapshot_of(Path(td)) == Snapshot()


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                failures.append((name, f"{type(e).__name__}: {e}"))
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} failed")
        sys.exit(1)
    print("all passed")
