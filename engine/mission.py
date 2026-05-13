"""Durable mission state for long multi-step agent tasks.

A `Mission` is a small JSON file (`<workspace>/.ulcagent_mission.json`) that
survives across runs and context-window compaction. The agent reads it at
run start (its summary is injected into the system prompt — small enough
that compaction never elides it), updates it via the `mission` tool as it
progresses, and the file is always current. If a run dies — context limit,
crash, user interrupt — the file reflects exactly where things stand, and a
fresh run picks up from there.

This is the "externalize working memory to disk" pattern: the file is the
ground truth for progress, not the conversation transcript. The agent never
hand-edits the file — it only goes through the `mission` tool, which keeps
the JSON well-formed.

Distinct from the other ULC persistence mechanisms:
- `remember` / agent_memory.py — free-form append-only project notes
- `plan` tool — in-run-only ephemeral todo list
- `/goal` loop — persistent objective but no structured progress tracking
A Mission is structured (goal + numbered steps with status + decisions +
next-action) AND persistent AND resume-shaped.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MISSION_FILENAME = ".ulcagent_mission.json"
_SCHEMA_VERSION = 1
_MAX_STEPS = 50
_MAX_NOTES = 100
_MAX_TITLE = 200
_MAX_NOTE = 500


@dataclass
class MissionStep:
    n: int
    title: str
    done: bool = False
    note: str = ""  # brief note recorded when the step is marked done


@dataclass
class Mission:
    goal: str
    steps: list[MissionStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # discovered context / decisions / gotchas
    next_action: str = ""
    schema_version: int = _SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── persistence ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mission":
        steps = [MissionStep(**s) for s in d.get("steps", [])]
        return cls(
            goal=str(d.get("goal", "")),
            steps=steps,
            notes=[str(x) for x in d.get("notes", [])],
            next_action=str(d.get("next_action", "")),
            schema_version=int(d.get("schema_version", _SCHEMA_VERSION)),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
        )

    @classmethod
    def load(cls, workspace: Path | str) -> Optional["Mission"]:
        """Return the Mission stored in <workspace>/.ulcagent_mission.json,
        or None if absent / unreadable."""
        p = Path(workspace) / MISSION_FILENAME
        if not p.exists():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning(f"mission file at {p} unreadable ({exc}); ignoring")
            return None

    def save(self, workspace: Path | str) -> Path:
        self.updated_at = time.time()
        p = Path(workspace) / MISSION_FILENAME
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @staticmethod
    def exists(workspace: Path | str) -> bool:
        return (Path(workspace) / MISSION_FILENAME).exists()

    @staticmethod
    def delete(workspace: Path | str) -> bool:
        p = Path(workspace) / MISSION_FILENAME
        if p.exists():
            p.unlink()
            return True
        return False

    # ── mutation (used by the `mission` tool) ────────────────────────

    def set_steps(self, titles: list[str]) -> None:
        cleaned = [str(t).strip()[:_MAX_TITLE] for t in titles if str(t).strip()]
        if not cleaned:
            raise ValueError("at least one non-empty step title is required")
        self.steps = [MissionStep(n=i + 1, title=t) for i, t in enumerate(cleaned[:_MAX_STEPS])]

    def add_step(self, title: str) -> int:
        title = str(title).strip()[:_MAX_TITLE]
        if not title:
            raise ValueError("step title cannot be empty")
        if len(self.steps) >= _MAX_STEPS:
            raise ValueError(f"mission already has {_MAX_STEPS} steps (the cap)")
        n = (max((s.n for s in self.steps), default=0)) + 1
        self.steps.append(MissionStep(n=n, title=title))
        return n

    def mark_step_done(self, n: int, note: str = "") -> MissionStep:
        for s in self.steps:
            if s.n == n:
                s.done = True
                if note:
                    s.note = str(note).strip()[:_MAX_NOTE]
                return s
        raise ValueError(f"no step with n={n}; current step numbers: {[s.n for s in self.steps]}")

    def add_note(self, text: str) -> None:
        text = str(text).strip()
        if not text:
            return
        if len(self.notes) >= _MAX_NOTES:
            # drop the oldest to make room — recent decisions matter more
            self.notes.pop(0)
        self.notes.append(text[:_MAX_NOTE])

    def set_next(self, text: str) -> None:
        self.next_action = str(text).strip()[:_MAX_NOTE]

    # ── status / rendering ───────────────────────────────────────────

    @property
    def is_complete(self) -> bool:
        return bool(self.steps) and all(s.done for s in self.steps)

    @property
    def progress(self) -> tuple[int, int]:
        done = sum(1 for s in self.steps if s.done)
        return done, len(self.steps)

    def next_pending_step(self) -> Optional[MissionStep]:
        for s in self.steps:
            if not s.done:
                return s
        return None

    def render_summary(self, *, for_system_prompt: bool = False) -> str:
        """Human-readable mission state. When for_system_prompt=True, prepends
        a short instruction telling the agent it's resuming and should keep
        the mission file updated via the `mission` tool."""
        done, total = self.progress
        lines: list[str] = []
        if for_system_prompt:
            lines.append("## Current mission — RESUME IN PROGRESS")
            lines.append("")
            lines.append(
                "You are continuing a multi-step mission. Its state below is "
                "the GROUND TRUTH for progress (not this conversation). Pick up "
                "from where it stands. As you complete steps and learn things, "
                "call the `mission` tool: `mission(action=\"step_done\", n=N, "
                "note=\"...\")` after each step, `mission(action=\"note\", "
                "text=\"...\")` for decisions/gotchas, `mission(action=\"next\", "
                "text=\"...\")` to record what's next. The file persists across "
                "runs — if you don't finish, the next session continues from "
                "your updates."
            )
            lines.append("")
        lines.append(f"Goal: {self.goal}")
        lines.append(f"Progress: {done}/{total} steps done"
                     + (" — COMPLETE" if self.is_complete else ""))
        if self.steps:
            lines.append("Steps:")
            for s in self.steps:
                mark = "[x]" if s.done else "[ ]"
                suffix = f" — {s.note}" if (s.done and s.note) else ""
                lines.append(f"  {s.n}. {mark} {s.title}{suffix}")
        if self.notes:
            lines.append("Notes / decisions / gotchas:")
            for note in self.notes:
                lines.append(f"  - {note}")
        if self.next_action:
            lines.append(f"Next: {self.next_action}")
        elif not self.is_complete:
            nxt = self.next_pending_step()
            if nxt is not None:
                lines.append(f"Next: (no explicit next set) — step {nxt.n}: {nxt.title}")
        return "\n".join(lines)
