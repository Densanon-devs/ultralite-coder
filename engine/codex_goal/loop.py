"""Persistent-objective loop on top of `Agent.run`.

Single entry point: `run_goal_loop(agent, goal, ...)`. Uses the
existing `continue_session=True` path on `Agent.run` so the model's
transcript and tool history persist across iterations. The loop's
exit conditions are entirely prompt-mediated: the model emits
`GOAL_COMPLETE` to finish, or the loop hits a token budget and
asks for a resume summary.

This is loader-agnostic the same way `Agent` is — it accepts any
agent object that exposes `.run(goal, continue_session=...) ->
AgentResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .prompts import (
    ACCEPTANCE_RETRY_PROMPT,
    BUDGET_LIMIT_PROMPT,
    CONTINUATION_PROMPT,
    DONE_SENTINEL,
)


# Default-ish token budget: 50k tokens is roughly 6-10 iterations on
# Qwen 2.5 14B with the lean tool registry. Caller can override.
DEFAULT_TOKEN_BUDGET = 50_000

# Hard ceiling on iterations — even if the budget says we have room,
# 25 iterations on a single persistent goal is almost always a sign
# of model thrash, not progress.
DEFAULT_MAX_ITERATIONS = 25


@dataclass
class GoalIteration:
    index: int
    answer: str
    iterations_used: int
    stop_reason: str
    wall_time: float
    tokens_estimate: int


@dataclass
class GoalResult:
    goal: str
    iterations: list[GoalIteration] = field(default_factory=list)
    completed: bool = False
    stop_reason: str = "unknown"  # "completed" | "budget" | "max_loops" | "interrupted" | "error"
    final_summary: str = ""
    tokens_estimate_total: int = 0


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ~ 4 chars for English / code mix.

    Used for budget tracking. Doesn't need to be exact — the budget is
    a soft cap, not a hard contract.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _is_completion(answer: str) -> bool:
    """Did the model emit the GOAL_COMPLETE sentinel?

    We accept the sentinel anywhere in the answer (model might add
    a single line of context like `GOAL_COMPLETE — all files written,
    tests pass`). We reject if the sentinel only appears inside an
    obvious negation ("NOT GOAL_COMPLETE", "cannot say GOAL_COMPLETE").
    """
    if DONE_SENTINEL not in answer:
        return False
    # Cheap negation check — only look at the line containing the sentinel.
    for line in answer.splitlines():
        if DONE_SENTINEL in line:
            stripped = line.strip().upper()
            # Sentinel preceded by a negation word in the same line — reject.
            for neg in ("NOT ", "NEVER ", "CANNOT ", "CAN'T ", "WON'T ", "ISN'T "):
                if neg in stripped and stripped.index(neg) < stripped.index(DONE_SENTINEL):
                    return False
            return True
    return False


def run_goal_loop(
    agent: Any,
    goal: str,
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_loops: int = DEFAULT_MAX_ITERATIONS,
    on_iteration: Callable[[GoalIteration], None] | None = None,
    acceptance_check: Callable[[], str | None] | None = None,
    acceptance_max_retries: int = 2,
) -> GoalResult:
    """Run `agent` against `goal` until the model self-evaluates completion.

    Parameters
    ----------
    agent
        Any object exposing `.run(goal, continue_session: bool) -> AgentResult`.
        Typical caller passes the same `Agent` instance the REPL uses.
    goal
        The persistent objective. Will be referenced verbatim in every
        continuation prompt so the model never loses sight of it.
    token_budget
        Soft cap on total tokens the loop may consume across iterations.
        When exceeded, the loop emits a final BUDGET_LIMIT prompt and
        exits. Default is `DEFAULT_TOKEN_BUDGET` (50k).
    max_loops
        Hard cap on outer-loop iterations regardless of token budget.
        Defends against models that emit one-line answers in a tight
        ring. Default is 25.
    on_iteration
        Optional callback invoked with each `GoalIteration` as it
        completes — useful for live progress display in the CLI.
    acceptance_check
        Optional objective gate on completion. When the model emits the
        `GOAL_COMPLETE` sentinel, this callback runs; if it returns a
        non-empty string (the goal is NOT actually met — e.g. mission steps
        still pending, tests failing), that feedback is fed back into the
        session and the loop keeps going instead of accepting completion.
        Returning None/empty accepts the completion. Default None preserves
        the original sentinel-only behavior exactly. Mirrors
        `Agent.pre_finish_check` so the same gate can govern both the
        one-shot and persistent-loop completion boundaries.
    acceptance_max_retries
        Hard cap on how many times `acceptance_check` may reject a
        `GOAL_COMPLETE`. Past this, the sentinel is accepted as-is so a
        check that never passes can't trap the loop. Default 2.

    Returns
    -------
    GoalResult with the full per-iteration history, completion state,
    and exit reason.
    """
    result = GoalResult(goal=goal)
    tokens_used = 0
    iteration_index = 0
    acceptance_retries = 0

    # Iteration 1: feed the original goal as a fresh session.
    current_prompt = goal
    continue_session = False

    while True:
        iteration_index += 1
        if iteration_index > max_loops:
            result.stop_reason = "max_loops"
            break

        try:
            agent_result = agent.run(current_prompt, continue_session=continue_session)
        except KeyboardInterrupt:
            result.stop_reason = "interrupted"
            break
        except Exception as exc:  # pragma: no cover — surfaces upstream bugs
            result.stop_reason = "error"
            result.final_summary = f"[loop error: {exc}]"
            break

        answer = (agent_result.final_answer or "").strip()
        # Budget accounting: estimate tokens consumed by this iteration's
        # answer + the prompt that drove it. Tool transcripts live inside
        # agent state; we only need a coarse measure for budget gating.
        iter_tokens = _estimate_tokens(current_prompt) + _estimate_tokens(answer)
        tokens_used += iter_tokens

        gi = GoalIteration(
            index=iteration_index,
            answer=answer,
            iterations_used=getattr(agent_result, "iterations", 0),
            stop_reason=getattr(agent_result, "stop_reason", "unknown"),
            wall_time=getattr(agent_result, "wall_time", 0.0),
            tokens_estimate=iter_tokens,
        )
        result.iterations.append(gi)
        if on_iteration is not None:
            try:
                on_iteration(gi)
            except Exception:
                pass

        # Completion check: has the model self-evaluated to GOAL_COMPLETE?
        if _is_completion(answer):
            # Objective acceptance gate. The inner agent.run already gates
            # each iteration's final answer (syntax/goal-token sweeps), but
            # the *outer* loop has historically trusted the sentinel alone.
            # If an acceptance_check is wired and still has retries left, let
            # it veto a premature GOAL_COMPLETE and keep the loop running.
            feedback: str | None = None
            if (
                acceptance_check is not None
                and acceptance_retries < acceptance_max_retries
                and iteration_index < max_loops
            ):
                try:
                    feedback = acceptance_check()
                except Exception:
                    feedback = None
            if feedback:
                acceptance_retries += 1
                current_prompt = ACCEPTANCE_RETRY_PROMPT.format(
                    goal=goal,
                    feedback=feedback,
                    retry=acceptance_retries,
                    max_retries=acceptance_max_retries,
                )
                continue_session = True
                continue
            result.completed = True
            result.stop_reason = "completed"
            result.final_summary = answer
            break

        # Budget check: did this iteration push us past the soft cap?
        if tokens_used >= token_budget:
            result.stop_reason = "budget"
            try:
                final = agent.run(
                    BUDGET_LIMIT_PROMPT.format(goal=goal),
                    continue_session=True,
                )
                result.final_summary = (final.final_answer or "").strip()
            except Exception as exc:
                result.final_summary = f"[budget summary failed: {exc}]"
            break

        # Continue: feed the continuation prompt back into the same session.
        remaining = max(0, token_budget - tokens_used)
        current_prompt = CONTINUATION_PROMPT.format(
            goal=goal,
            iteration=iteration_index,
            remaining_tokens=remaining,
        )
        continue_session = True

    result.tokens_estimate_total = tokens_used
    return result


# Convenience helper for callers that just want the per-iteration answers.
def iteration_answers(result: GoalResult) -> Iterable[str]:
    return (it.answer for it in result.iterations)
