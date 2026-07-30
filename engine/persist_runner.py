"""
Relaunch driver — "keep going despite the model's shortcomings."

The 14B's real limits are per-RUN, not per-task: it fills context, it hits
max_iterations, it occasionally narrates completion and stops with work left.
A long job dies partway through and the user has to notice and re-prompt.

The durable state to survive that already exists — `engine.mission` keeps the
goal, numbered steps, notes and next-action in `.ulcagent_mission.json`, and
injects a summary into the system prompt at run start. What was missing is the
outer loop: finish a run, look at the mission, and if steps remain, start a
FRESH run (fresh context, same mission) and continue.

The interesting part is knowing when to stop. An unconditional loop on a stuck
model burns the GPU forever, so progress is measured between rounds and a
stalled mission is abandoned rather than retried. That policy is pure and lives
in RelaunchPolicy so it can be tested without loading a model.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from .mission import Mission


DEFAULT_MAX_ROUNDS = 5
DEFAULT_STALL_LIMIT = 2


class Decision(str, Enum):
    CONTINUE = "continue"            # steps remain and we're making progress
    COMPLETE = "complete"            # every step done
    BUDGET = "budget_exhausted"      # ran out of rounds
    STALLED = "stalled"              # rounds passed with no progress
    UNTRACKED = "untracked"          # no mission steps — nothing to resume from


@dataclass(frozen=True)
class Snapshot:
    """The measurable state of a mission at a round boundary."""
    done: int = 0
    total: int = 0
    notes: int = 0

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.done >= self.total

    def advanced_past(self, other: "Snapshot") -> bool:
        """Did anything actually move since *other*?

        A finished step is progress. So is a new note — a round that only
        recorded a finding still moved the mission forward and shouldn't count
        as a stall.
        """
        return self.done > other.done or self.notes > other.notes


def snapshot_of(workspace: Path | str) -> Snapshot:
    mission = Mission.load(workspace)
    if mission is None:
        return Snapshot()
    done, total = mission.progress
    return Snapshot(done=done, total=total, notes=len(getattr(mission, "notes", []) or []))


@dataclass
class RelaunchPolicy:
    """Decides whether to start another run. Pure — no I/O, no model."""
    max_rounds: int = DEFAULT_MAX_ROUNDS
    stall_limit: int = DEFAULT_STALL_LIMIT
    rounds_used: int = 0
    stalls: int = 0

    def decide(self, before: Snapshot, after: Snapshot) -> tuple[Decision, str]:
        self.rounds_used += 1

        if after.complete:
            return Decision.COMPLETE, f"all {after.total} steps done"

        if after.total == 0:
            # Nothing to resume from: the model never laid out steps, so a
            # relaunch would just re-run the same prompt from scratch.
            return Decision.UNTRACKED, "no mission steps recorded — nothing to resume"

        if after.advanced_past(before):
            self.stalls = 0
        else:
            self.stalls += 1
            if self.stalls >= self.stall_limit:
                return (Decision.STALLED,
                        f"no progress in {self.stalls} consecutive rounds "
                        f"({after.done}/{after.total} steps done)")

        if self.rounds_used >= self.max_rounds:
            return (Decision.BUDGET,
                    f"hit the {self.max_rounds}-round limit at "
                    f"{after.done}/{after.total} steps")

        return (Decision.CONTINUE,
                f"{after.done}/{after.total} steps done — continuing with fresh context")


def continuation_goal(workspace: Path | str, original_goal: str) -> str:
    """The prompt for a follow-up round.

    The mission summary is already injected into the system prompt by ulcagent,
    so this only has to point the model at the next unfinished step rather than
    restate the whole plan.
    """
    mission = Mission.load(workspace)
    if mission is None:
        return original_goal
    done, total = mission.progress
    nxt = mission.next_pending_step()
    target = nxt.title if nxt is not None else "the remaining work"
    return (
        f"Continue the mission already in progress ({done}/{total} steps done). "
        f"Do NOT start over and do not repeat completed steps. "
        f"The next unfinished step is: {target}. "
        f"Work on it now, then mark it done with the mission tool."
    )


def run_until_done(
    *,
    workspace: Path | str,
    goal: str,
    run_round: Callable[[str, int], object],
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    stall_limit: int = DEFAULT_STALL_LIMIT,
    on_event: Optional[Callable[[str], None]] = None,
) -> dict:
    """Drive `run_round` until the mission completes or the policy gives up.

    `run_round(goal_text, round_index)` should perform ONE full agent run with a
    fresh context and return whatever the caller finds useful. This module never
    touches the model itself — that keeps it unit-testable with a stub.
    """
    say = on_event or (lambda _m: None)
    policy = RelaunchPolicy(max_rounds=max_rounds, stall_limit=stall_limit)
    history: list[dict] = []
    current_goal = goal

    while True:
        before = snapshot_of(workspace)
        round_index = policy.rounds_used + 1
        say(f"round {round_index}: {before.done}/{before.total} steps done")

        result = run_round(current_goal, round_index)

        after = snapshot_of(workspace)
        decision, reason = policy.decide(before, after)
        history.append({
            "round": round_index,
            "before": (before.done, before.total),
            "after": (after.done, after.total),
            "decision": decision.value,
            "reason": reason,
            "result": result,
        })
        say(f"round {round_index}: {decision.value} — {reason}")

        if decision is not Decision.CONTINUE:
            return {
                "decision": decision.value,
                "reason": reason,
                "rounds": policy.rounds_used,
                "done": after.done,
                "total": after.total,
                "history": history,
            }

        current_goal = continuation_goal(workspace, goal)
