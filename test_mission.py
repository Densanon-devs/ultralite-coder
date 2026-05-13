"""Tests for engine/mission.py + the `mission` builtin tool."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from engine.mission import Mission, MissionStep, MISSION_FILENAME
from engine.agent_builtins import build_default_registry, mission_state_hint


# ── Mission dataclass: persistence + mutation ────────────────────────────────

class TestMissionPersistence:
    def test_save_creates_file(self, tmp_path):
        m = Mission(goal="do the thing")
        p = m.save(tmp_path)
        assert p == tmp_path / MISSION_FILENAME
        assert p.exists()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["goal"] == "do the thing"

    def test_load_roundtrip(self, tmp_path):
        m = Mission(goal="g")
        m.set_steps(["a", "b", "c"])
        m.mark_step_done(1, note="did a")
        m.add_note("watch out for X")
        m.set_next("do b next")
        m.save(tmp_path)
        m2 = Mission.load(tmp_path)
        assert m2 is not None
        assert m2.goal == "g"
        assert [s.title for s in m2.steps] == ["a", "b", "c"]
        assert m2.steps[0].done and m2.steps[0].note == "did a"
        assert m2.notes == ["watch out for X"]
        assert m2.next_action == "do b next"

    def test_load_returns_none_when_absent(self, tmp_path):
        assert Mission.load(tmp_path) is None

    def test_load_returns_none_on_corrupt_json(self, tmp_path):
        (tmp_path / MISSION_FILENAME).write_text("{not valid json", encoding="utf-8")
        assert Mission.load(tmp_path) is None

    def test_exists_and_delete(self, tmp_path):
        assert not Mission.exists(tmp_path)
        Mission(goal="g").save(tmp_path)
        assert Mission.exists(tmp_path)
        assert Mission.delete(tmp_path) is True
        assert not Mission.exists(tmp_path)
        assert Mission.delete(tmp_path) is False  # already gone


class TestMissionMutation:
    def test_set_steps_numbers_sequentially(self):
        m = Mission(goal="g")
        m.set_steps(["one", "two", "three"])
        assert [s.n for s in m.steps] == [1, 2, 3]

    def test_set_steps_rejects_empty(self):
        m = Mission(goal="g")
        with pytest.raises(ValueError):
            m.set_steps([])
        with pytest.raises(ValueError):
            m.set_steps(["  ", ""])

    def test_set_steps_strips_and_drops_blanks(self):
        m = Mission(goal="g")
        m.set_steps(["  a  ", "", "b", "   "])
        assert [s.title for s in m.steps] == ["a", "b"]

    def test_add_step_appends_with_next_number(self):
        m = Mission(goal="g")
        m.set_steps(["a", "b"])
        n = m.add_step("c")
        assert n == 3
        assert m.steps[-1].title == "c"

    def test_mark_step_done_sets_flag_and_note(self):
        m = Mission(goal="g")
        m.set_steps(["a", "b"])
        s = m.mark_step_done(2, note="finished b")
        assert s.done and s.note == "finished b"
        assert not m.steps[0].done

    def test_mark_step_done_unknown_raises(self):
        m = Mission(goal="g")
        m.set_steps(["a"])
        with pytest.raises(ValueError, match="no step with n=99"):
            m.mark_step_done(99)

    def test_add_note_drops_oldest_when_full(self):
        m = Mission(goal="g")
        # _MAX_NOTES is 100
        for i in range(105):
            m.add_note(f"note {i}")
        assert len(m.notes) == 100
        assert m.notes[0] == "note 5"  # 0-4 dropped
        assert m.notes[-1] == "note 104"

    def test_add_note_ignores_blank(self):
        m = Mission(goal="g")
        m.add_note("   ")
        m.add_note("")
        assert m.notes == []

    def test_progress_and_is_complete(self):
        m = Mission(goal="g")
        m.set_steps(["a", "b"])
        assert m.progress == (0, 2)
        assert not m.is_complete
        m.mark_step_done(1)
        assert m.progress == (1, 2)
        assert not m.is_complete
        m.mark_step_done(2)
        assert m.progress == (2, 2)
        assert m.is_complete

    def test_is_complete_false_with_no_steps(self):
        # An empty mission isn't "complete" — there's nothing to do yet.
        assert not Mission(goal="g").is_complete

    def test_next_pending_step(self):
        m = Mission(goal="g")
        m.set_steps(["a", "b", "c"])
        m.mark_step_done(1)
        assert m.next_pending_step().n == 2


class TestMissionRender:
    def test_summary_includes_goal_steps_notes_next(self):
        m = Mission(goal="Fix the parser")
        m.set_steps(["read code", "fix bug", "test"])
        m.mark_step_done(1, note="bug is in line 42")
        m.add_note("the test fixture is stale")
        m.set_next("apply the off-by-one fix")
        out = m.render_summary()
        assert "Fix the parser" in out
        assert "1/3" in out
        assert "[x] read code" in out
        assert "bug is in line 42" in out
        assert "[ ] fix bug" in out
        assert "the test fixture is stale" in out
        assert "apply the off-by-one fix" in out

    def test_system_prompt_variant_has_resume_instruction(self):
        m = Mission(goal="g")
        m.set_steps(["a"])
        out = m.render_summary(for_system_prompt=True)
        assert "RESUME" in out
        assert "mission(action=" in out  # tells the model how to update

    def test_summary_marks_complete(self):
        m = Mission(goal="g")
        m.set_steps(["a"])
        m.mark_step_done(1)
        assert "COMPLETE" in m.render_summary()


# ── mission_state_hint helper ────────────────────────────────────────────────

class TestMissionStateHint:
    def test_returns_empty_when_no_mission(self, tmp_path):
        assert mission_state_hint(tmp_path) == ""

    def test_returns_summary_when_mission_exists(self, tmp_path):
        m = Mission(goal="ship the feature")
        m.set_steps(["a", "b"])
        m.save(tmp_path)
        out = mission_state_hint(tmp_path)
        assert "ship the feature" in out
        assert "RESUME" in out

    def test_tolerates_corrupt_file(self, tmp_path):
        (tmp_path / MISSION_FILENAME).write_text("garbage", encoding="utf-8")
        assert mission_state_hint(tmp_path) == ""  # no crash


# ── the `mission` builtin tool ───────────────────────────────────────────────

def _registry_with_mission(workspace: Path):
    return build_default_registry(workspace, enable_mission=True)


class TestMissionTool:
    def test_tool_registered_only_when_enabled(self, tmp_path):
        reg_off = build_default_registry(tmp_path, enable_mission=False)
        reg_on = build_default_registry(tmp_path, enable_mission=True)
        assert "mission" not in {t.name for t in reg_off.enabled_tools()}
        assert "mission" in {t.name for t in reg_on.enabled_tools()}

    def test_start_creates_mission_file(self, tmp_path):
        reg = _registry_with_mission(tmp_path)
        result = reg.execute_by_name("mission", {
            "action": "start", "goal": "do X", "steps": ["s1", "s2"],
        }) if hasattr(reg, "execute_by_name") else None
        # Fall back to calling the function directly if the registry API differs
        fn = reg.get("mission").function
        out = fn(action="start", goal="do X", steps=["s1", "s2"])
        assert "Mission started" in out
        assert Mission.exists(tmp_path)
        m = Mission.load(tmp_path)
        assert m.goal == "do X" and len(m.steps) == 2

    def test_start_requires_goal(self, tmp_path):
        fn = _registry_with_mission(tmp_path).get("mission").function
        with pytest.raises(ValueError, match="requires goal"):
            fn(action="start", goal="", steps=["s1"])

    def test_actions_require_existing_mission(self, tmp_path):
        fn = _registry_with_mission(tmp_path).get("mission").function
        with pytest.raises(ValueError, match="no mission yet"):
            fn(action="status")
        with pytest.raises(ValueError, match="no mission yet"):
            fn(action="step_done", n=1)

    def test_step_done_persists(self, tmp_path):
        fn = _registry_with_mission(tmp_path).get("mission").function
        fn(action="start", goal="g", steps=["a", "b"])
        out = fn(action="step_done", n=1, note="did a")
        assert "step 1 done (1/2)" in out
        m = Mission.load(tmp_path)
        assert m.steps[0].done and m.steps[0].note == "did a"

    def test_step_done_all_marks_complete(self, tmp_path):
        fn = _registry_with_mission(tmp_path).get("mission").function
        fn(action="start", goal="g", steps=["a"])
        out = fn(action="step_done", n=1)
        assert "MISSION COMPLETE" in out

    def test_note_persists(self, tmp_path):
        fn = _registry_with_mission(tmp_path).get("mission").function
        fn(action="start", goal="g", steps=["a"])
        fn(action="note", text="gotcha: the API rate-limits at 60/min")
        m = Mission.load(tmp_path)
        assert "rate-limits at 60/min" in m.notes[0]

    def test_add_step_persists(self, tmp_path):
        fn = _registry_with_mission(tmp_path).get("mission").function
        fn(action="start", goal="g", steps=["a"])
        out = fn(action="add_step", title="b")
        assert "Added step 2" in out
        assert len(Mission.load(tmp_path).steps) == 2

    def test_next_persists(self, tmp_path):
        fn = _registry_with_mission(tmp_path).get("mission").function
        fn(action="start", goal="g", steps=["a"])
        fn(action="next", text="run the migration")
        assert Mission.load(tmp_path).next_action == "run the migration"

    def test_unknown_action_raises(self, tmp_path):
        fn = _registry_with_mission(tmp_path).get("mission").function
        fn(action="start", goal="g", steps=["a"])
        with pytest.raises(ValueError, match="unknown action"):
            fn(action="frobnicate")
