"""Unit tests for engine.codex_goal — the /goal persistent-objective loop.

Uses a stub agent that emits canned answers in sequence. Verifies:
  - Loop terminates on GOAL_COMPLETE sentinel
  - Loop terminates on token budget exhaustion
  - Loop honors max_loops hard ceiling
  - Continuation prompt mentions the original goal verbatim
  - Negated sentinel ("NOT GOAL_COMPLETE") does not trigger completion
  - Iteration 1 uses fresh session, all later iterations use continue_session
  - on_iteration callback fires once per iteration
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from engine.codex_goal import GoalResult, run_goal_loop
from engine.codex_goal.loop import (
    DEFAULT_MAX_ITERATIONS,
    GoalIteration,
    _estimate_tokens,
    _is_completion,
)
from engine.codex_goal.prompts import (
    ACCEPTANCE_RETRY_PROMPT,
    BUDGET_LIMIT_PROMPT,
    CONTINUATION_PROMPT,
    DONE_SENTINEL,
)


@dataclass
class _StubResult:
    final_answer: str
    iterations: int = 1
    stop_reason: str = "answered"
    wall_time: float = 0.01


@dataclass
class _StubAgent:
    """Returns canned answers in order. Records every (prompt, continue_session) call."""

    answers: list[str]
    raise_on_iteration: int | None = None
    calls: list[tuple[str, bool]] = field(default_factory=list)

    def run(self, prompt: str, continue_session: bool = False) -> _StubResult:
        self.calls.append((prompt, continue_session))
        idx = len(self.calls)
        if self.raise_on_iteration is not None and idx == self.raise_on_iteration:
            raise RuntimeError("stub explosion")
        if idx > len(self.answers):
            return _StubResult(final_answer="(no more canned answers)")
        return _StubResult(final_answer=self.answers[idx - 1])


# ── Pure helper coverage ─────────────────────────────────────────


def test_is_completion_accepts_bare_sentinel():
    assert _is_completion(DONE_SENTINEL) is True


def test_is_completion_accepts_sentinel_with_trailing_context():
    assert _is_completion(f"{DONE_SENTINEL} — all 3 files written, tests pass.") is True


def test_is_completion_rejects_negated_sentinel():
    assert _is_completion(f"This is NOT {DONE_SENTINEL} yet — still missing tests.") is False
    assert _is_completion(f"I cannot say {DONE_SENTINEL} until the migration runs.") is False


def test_is_completion_rejects_absent_sentinel():
    assert _is_completion("Done with that.") is False
    assert _is_completion("") is False


def test_estimate_tokens_minimum_one():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("x") == 1


def test_estimate_tokens_scales_roughly_linearly():
    assert _estimate_tokens("a" * 400) == 100


# ── Loop happy path ──────────────────────────────────────────────


def test_completes_on_done_sentinel():
    agent = _StubAgent(
        answers=[
            "Wrote calculator.py, started on tests next.",
            f"{DONE_SENTINEL} — calculator.py + test_calculator.py both green.",
        ]
    )
    result = run_goal_loop(agent, "Build calculator with tests")
    assert result.completed is True
    assert result.stop_reason == "completed"
    assert len(result.iterations) == 2
    assert DONE_SENTINEL in result.final_summary


def test_iteration_one_is_fresh_session_rest_continue():
    agent = _StubAgent(
        answers=[
            "Did first sub-task.",
            "Did second sub-task.",
            f"{DONE_SENTINEL}",
        ]
    )
    run_goal_loop(agent, "Persistent goal")
    # Iteration 1: fresh session
    assert agent.calls[0][1] is False
    # All subsequent iterations: continue_session=True
    for prompt, continue_session in agent.calls[1:]:
        assert continue_session is True


def test_continuation_prompt_quotes_original_goal():
    agent = _StubAgent(answers=["progress made", f"{DONE_SENTINEL}"])
    goal = "Refactor the auth module to use bcrypt"
    run_goal_loop(agent, goal)
    # Iteration 2 prompt is the continuation — must contain the exact goal text.
    cont_prompt = agent.calls[1][0]
    assert goal in cont_prompt
    assert "iteration 1" in cont_prompt.lower()


def test_on_iteration_callback_fires_per_iteration():
    seen: list[GoalIteration] = []
    agent = _StubAgent(answers=["one", "two", f"{DONE_SENTINEL}"])
    run_goal_loop(agent, "g", on_iteration=seen.append)
    assert len(seen) == 3
    assert [it.index for it in seen] == [1, 2, 3]


def test_callback_exception_does_not_break_loop():
    def boom(_it):
        raise ValueError("callback failed")

    agent = _StubAgent(answers=["progress", f"{DONE_SENTINEL}"])
    result = run_goal_loop(agent, "g", on_iteration=boom)
    assert result.completed is True


# ── Budget exhaustion ────────────────────────────────────────────


def test_budget_exhaustion_runs_budget_summary():
    # Iter 1 answer is huge (~20k tokens) — budget of 15k is crossed immediately,
    # so the very next agent.run is the budget-summary call (iter 2 in stub terms).
    huge = "x" * 80_000  # ~20k token estimate
    agent = _StubAgent(
        answers=[
            huge,
            "Budget summary: A done, B partial, C not started.",
        ]
    )
    result = run_goal_loop(agent, "Tight budget goal", token_budget=15_000)
    assert result.completed is False
    assert result.stop_reason == "budget"
    # Only one normal iteration recorded (iter 2 was the budget summary, not a goal iteration).
    assert len(result.iterations) == 1
    # The final agent.run call should have been the BUDGET_LIMIT_PROMPT.
    assert "Tight budget goal" in agent.calls[-1][0]
    assert "Token budget exhausted" in agent.calls[-1][0]
    # Budget summary text surfaces in result.
    assert "Budget summary" in result.final_summary


def test_budget_summary_failure_is_swallowed():
    # Iter 1 spends the budget, then the budget-summary call (iter 2 in agent terms) explodes.
    huge = "x" * 80_000
    agent = _StubAgent(answers=[huge], raise_on_iteration=2)
    result = run_goal_loop(agent, "g", token_budget=15_000)
    assert result.stop_reason == "budget"
    assert "[budget summary failed" in result.final_summary


# ── Hard ceilings ────────────────────────────────────────────────


def test_max_loops_caps_runaway_models():
    # Model never says GOAL_COMPLETE — we must exit on max_loops.
    agent = _StubAgent(answers=["nope"] * (DEFAULT_MAX_ITERATIONS + 5))
    result = run_goal_loop(agent, "g", max_loops=4)
    assert result.completed is False
    assert result.stop_reason == "max_loops"
    assert len(result.iterations) == 4


def test_keyboard_interrupt_surfaces_as_interrupted():
    class _Interrupting:
        def run(self, prompt, continue_session=False):
            raise KeyboardInterrupt

    result = run_goal_loop(_Interrupting(), "g")
    assert result.stop_reason == "interrupted"
    assert result.completed is False


def test_runtime_exception_surfaces_as_error():
    class _Boom:
        def run(self, prompt, continue_session=False):
            raise RuntimeError("model crashed")

    result = run_goal_loop(_Boom(), "g")
    assert result.stop_reason == "error"
    assert "model crashed" in result.final_summary


# ── Token accounting ─────────────────────────────────────────────


def test_tokens_estimate_total_sums_iteration_estimates():
    agent = _StubAgent(answers=["one short", f"{DONE_SENTINEL}"])
    result = run_goal_loop(agent, "g")
    assert result.tokens_estimate_total == sum(
        it.tokens_estimate for it in result.iterations
    )
    assert result.tokens_estimate_total > 0


# ── Prompt template smoke checks ─────────────────────────────────


def test_continuation_prompt_template_renders():
    rendered = CONTINUATION_PROMPT.format(
        goal="Do the thing", iteration=3, remaining_tokens=12345
    )
    assert "Do the thing" in rendered
    assert "12345" in rendered
    assert DONE_SENTINEL in rendered


def test_budget_limit_prompt_template_renders():
    rendered = BUDGET_LIMIT_PROMPT.format(goal="Long goal")
    assert "Long goal" in rendered
    assert "Token budget exhausted" in rendered


# ── Objective acceptance gate ────────────────────────────────────


def test_acceptance_check_none_is_unchanged_behavior():
    # Default (no acceptance_check) accepts the sentinel exactly as before.
    agent = _StubAgent(answers=["progress", f"{DONE_SENTINEL}"])
    result = run_goal_loop(agent, "g")  # acceptance_check defaults to None
    assert result.completed is True
    assert len(result.iterations) == 2


def test_acceptance_check_passing_accepts_completion():
    agent = _StubAgent(answers=[f"{DONE_SENTINEL}"])
    result = run_goal_loop(agent, "g", acceptance_check=lambda: None)
    assert result.completed is True
    assert result.stop_reason == "completed"
    assert len(result.iterations) == 1


def test_acceptance_check_rejects_then_continues_then_accepts():
    # First GOAL_COMPLETE is vetoed (check returns feedback); after one more
    # iteration the check passes and completion is accepted.
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return "tests still failing in test_calc.py" if calls["n"] == 1 else None

    agent = _StubAgent(answers=[f"{DONE_SENTINEL}", f"{DONE_SENTINEL}"])
    result = run_goal_loop(agent, "Build calc with tests", acceptance_check=check)
    assert result.completed is True
    assert len(result.iterations) == 2
    # Iteration 2's prompt is the acceptance-retry: carries the goal + feedback.
    retry_prompt = agent.calls[1][0]
    assert "tests still failing in test_calc.py" in retry_prompt
    assert "Build calc with tests" in retry_prompt
    assert "acceptance retry 1/2" in retry_prompt
    assert agent.calls[1][1] is True  # continue_session


def test_acceptance_check_respects_retry_cap():
    # Check ALWAYS rejects; with cap=1, after one veto the next sentinel is
    # accepted so a never-passing check can't trap the loop.
    agent = _StubAgent(answers=[f"{DONE_SENTINEL}"] * 5)
    result = run_goal_loop(
        agent, "g", acceptance_check=lambda: "never done", acceptance_max_retries=1
    )
    assert result.completed is True
    assert len(result.iterations) == 2  # one veto, then forced-accept
    # Exactly one acceptance-retry prompt was injected (iteration 2).
    retry_prompts = [p for p, _ in agent.calls if "acceptance retry" in p]
    assert len(retry_prompts) == 1


def test_acceptance_check_exception_treated_as_pass():
    def boom():
        raise RuntimeError("check blew up")

    agent = _StubAgent(answers=[f"{DONE_SENTINEL}"])
    result = run_goal_loop(agent, "g", acceptance_check=boom)
    assert result.completed is True  # exception → treated as no feedback → accept
    assert len(result.iterations) == 1


def test_acceptance_retry_prompt_template_renders():
    rendered = ACCEPTANCE_RETRY_PROMPT.format(
        goal="Ship it", feedback="2 of 3 steps pending", retry=1, max_retries=2
    )
    assert "Ship it" in rendered
    assert "2 of 3 steps pending" in rendered
    assert DONE_SENTINEL in rendered
    assert "acceptance retry 1/2" in rendered


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
