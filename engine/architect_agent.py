"""
Architect Agent — Plan-then-execute wrapper around the agentic ReAct loop.

Addresses the ceiling failures observed in the flat Agent loop (Phase 14
iter1l–iter1x) where multi-action tasks (refactor_dataclass, build_todo_cli)
fail because the 14B loses track of "which step am I on" inside a single
long transcript. The fix is to break the task into concrete atomic steps
up front, then run each step as an isolated sub-agent call with a clean
context window. The workspace is shared between steps, so later steps see
the changes earlier steps made.

Flow:
    1. PLAN — ask the model for a numbered list of concrete sub-goals
    2. For each sub-goal, run a fresh Agent instance against the shared
       workspace + shared tool registry + shared model.
    3. Return an aggregate AgentResult that looks like a single agent run
       to callers (so benchmark_agentic.py doesn't need to fork its
       reporting path).

Design choices:
    - One planning call at the start. No re-planning mid-run. (Re-planning
      is a future enhancement — starting simple.)
    - Sub-agents share the workspace via the workspace_root arg, so file
      edits from step N are visible to step N+1.
    - Each sub-agent starts with a FRESH transcript. That's the whole point
      — context isolation — so the model can't drift.
    - If a sub-agent fails (max_iterations / wall_time / error), we log it
      and continue to the next step. The final sweep still runs at the
      top level via the benchmark check.
    - Auto_verify flags and memory are threaded through unchanged.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from engine.agent import Agent, AgentEvent, AgentResult
from engine.agent_memory import AgentMemory
from engine.agent_tools import ToolCall, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


_PLAN_SYSTEM = """You are a planning assistant. Break the user's coding task into a numbered list of CONCRETE, atomic sub-steps. Each step runs as an isolated sub-agent call with a fresh context window, so fewer-but-complete steps are better than many tiny ones.

HARD RULES:
- Each step is ONE mutating file action (write_file, edit_file, run_tests, run_bash). No read-only steps.
- NEVER add "Read X", "Check Y", "Verify Z", "Understand Q", "Review", "Inspect", or "Analyze" steps. The sub-agent reads files on demand. Those steps just waste a whole sub-agent call.
- Every step starts with a concrete verb: "Write", "Add", "Replace", "Remove", "Rewrite", "Run".
- Every mutating step names a specific file.
- Maximum 6 steps total. Minimum 1 step.

DECOMPOSITION GUIDANCE:
- Simple single-action tasks ("add a docstring", "rename function X to Y", "fix the off-by-one in paginate") → 1 step.
- Multi-action changes to a SINGLE file that compose into one logical rewrite (e.g. "convert class to @dataclass: import dataclass, add @dataclass decorator, remove __init__, add type-annotated fields, keep greet()") → 1 step described as a full REWRITE of that file. Say "Rewrite person.py with ..." — the sub-agent will use write_file to replace the whole file, which avoids the state-drift problems you get when chaining several edit_file calls.
- Multi-file creation ("build a todo CLI: todo.py, storage.py, cli.py, test_todo.py") → 1 step per file (each step rewrites that file from scratch), plus an optional final "Run tests to confirm" step.
- Independent changes across different files → one step per file.

REWRITE BIAS: if you can describe the whole change as "Rewrite <file> with <full new spec>", do that in ONE step instead of multiple edit_file steps. The 14B is much more reliable when it produces the whole file at once than when it chains edit_file calls against shifting state.

OUTPUT FORMAT — numbered plain text, one step per line, nothing else. No prose before or after, no bullet points, no code fences:

1. <concrete action in file X>
2. <concrete action in file Y>
3. Run tests to confirm."""


# Patterns for steps we drop silently — pure-read or pure-verify with no
# mutating action. The flat sub-agent would spend a full iter on each of
# these, which wastes time and model attention without producing output.
_DROP_STEP_PATTERNS = (
    "read ", "review ", "inspect ", "check ", "verify that ", "verify the ",
    "verify all ", "understand ", "analyze ", "examine ", "look at ",
    "confirm that ", "confirm the ", "open ",
)


_PLAN_RE = re.compile(r"^\s*(\d+)[.)\]]\s*(.+?)\s*$", re.MULTILINE)


def parse_plan(text: str, max_steps: int = 6) -> list[str]:
    """Extract numbered steps from the planner's response. Returns a list
    of step strings in order. Robust to minor formatting variations.
    Drops pure-read/verify steps — the executor reads files on demand and
    inserting those as top-level steps wastes a sub-agent call each."""
    if not text:
        return []
    steps: list[tuple[int, str]] = []
    for match in _PLAN_RE.finditer(text):
        try:
            num = int(match.group(1))
        except ValueError:
            continue
        body = match.group(2).strip()
        if not body:
            continue
        body = body.strip("*_`")
        if body.lower().startswith(("step ", "plan", "steps", "tasks")):
            continue
        # Drop pure read/verify steps — the sub-agent reads files itself.
        lowered = body.lower().lstrip()
        if any(lowered.startswith(pat) for pat in _DROP_STEP_PATTERNS):
            continue
        steps.append((num, body))
    steps.sort(key=lambda x: x[0])
    seen: set[int] = set()
    ordered: list[str] = []
    for num, body in steps:
        if num in seen:
            continue
        seen.add(num)
        ordered.append(body)
        if len(ordered) >= max_steps:
            break
    return ordered


class ArchitectAgent:
    """Plan-then-execute wrapper. Same public surface as Agent.run() so
    callers can swap one for the other."""

    def __init__(
        self,
        model: Any,
        registry: ToolRegistry,
        system_prompt_extra: str = "",
        workspace_root: Optional[Path] = None,
        memory: Optional[AgentMemory] = None,
        auto_verify_python: bool = True,
        max_iterations_per_step: int = 6,
        max_wall_time: float = 600.0,
        max_tokens_per_turn: int = 1024,
        temperature: Optional[float] = 0.1,
        repeat_penalty: Optional[float] = 1.15,
        confirm_risky: Optional[Callable[[ToolCall], bool]] = None,
        confirm_destructive: Optional[Callable[["ToolCall", list], bool]] = None,
        confirm_supply_chain: Optional[Callable[["ToolCall", list], bool]] = None,
        trust_repo: bool = False,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
        planner_max_tokens: int = 800,
    ) -> None:
        self.model = model
        self.registry = registry
        self.system_prompt_extra = system_prompt_extra
        self.workspace_root = Path(workspace_root) if workspace_root is not None else None
        self.memory = memory
        self.auto_verify_python = auto_verify_python
        self.max_iterations_per_step = max(1, int(max_iterations_per_step))
        self.max_wall_time = float(max_wall_time)
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.confirm_risky = confirm_risky
        self.confirm_destructive = confirm_destructive
        self.confirm_supply_chain = confirm_supply_chain
        self.trust_repo = bool(trust_repo)
        self._emit = on_event or (lambda _e: None)
        self.planner_max_tokens = int(planner_max_tokens)

    def run(self, goal: str) -> AgentResult:
        start = time.monotonic()
        all_calls: list[ToolCall] = []
        all_results: list[ToolResult] = []
        transcript_merged: list[dict[str, str]] = [{"role": "user", "content": goal}]

        # Phase 1: PLAN
        plan_prompt = self._build_plan_prompt(goal)
        try:
            plan_raw = self.model.generate(
                plan_prompt,
                max_tokens=self.planner_max_tokens,
                temperature=self.temperature if self.temperature is not None else 0.1,
                repeat_penalty=self.repeat_penalty if self.repeat_penalty is not None else 1.15,
                stop=["<|im_end|>", "<|im_start|>"],
            )
        except Exception as exc:
            logger.exception("Planner model call failed; falling back to single-agent")
            return self._single_agent_fallback(goal, start)

        steps = parse_plan(plan_raw or "")
        # Log the full plan text AND the filtered step list so the benchmark
        # logs show what the architect actually decided.
        plan_preview = (plan_raw or "").strip().replace("\n", " | ")[:260]
        self._emit(AgentEvent("model_text", 0, f"PLAN: {plan_preview}"))
        self._emit(
            AgentEvent(
                "iteration",
                0,
                f"architect planned {len(steps)} step(s): "
                + " | ".join(s[:60] for s in steps),
            )
        )
        transcript_merged.append({"role": "assistant", "content": f"Plan:\n{plan_raw.strip() if plan_raw else '(empty)'}"})

        if not steps:
            # No plan recovered — fall back to flat agent.
            return self._single_agent_fallback(goal, start)

        # Phase 2: EXECUTE each step in its own sub-agent
        remaining_budget = self.max_wall_time - (time.monotonic() - start)
        per_step_wall = max(30.0, remaining_budget / max(1, len(steps)))
        last_stop_reason = "answered"
        step_summaries: list[str] = []
        for i, step in enumerate(steps, start=1):
            if time.monotonic() - start > self.max_wall_time:
                last_stop_reason = "wall_time"
                break
            # Build a prior-step summary so each sub-agent knows what's
            # already been done in the workspace. Without this, sub-agent
            # N+1 re-reads the file and repeats work sub-agent N did.
            prior_block = ""
            if step_summaries:
                prior_block = (
                    "\n\nSTEPS ALREADY COMPLETED:\n"
                    + "\n".join(f"{j}. {s}" for j, s in enumerate(step_summaries, start=1))
                    + "\n"
                )
            step_goal = (
                f"ORIGINAL TASK: {goal}\n"
                f"{prior_block}\n"
                f"YOUR SUB-STEP ({i}/{len(steps)}): {step}\n\n"
                f"IMPERATIVE: You MUST emit the write_file / edit_file / "
                f"run_tests / run_bash tool call that performs this step. "
                f"Reading files alone is NOT completion. Do ONLY this "
                f"sub-step — not the rest of the task — and then give a "
                f"one-sentence plain-text confirmation after the mutating "
                f"call lands."
            )
            worker = Agent(
                model=self.model,
                registry=self.registry,
                system_prompt_extra=self.system_prompt_extra,
                workspace_root=self.workspace_root,
                memory=self.memory,
                auto_verify_python=self.auto_verify_python,
                # Sub-agents are scoped to ONE step of the original goal;
                # the architect runs the goal-token sweep at the end over
                # the full workspace, so sub-agents must not block on it.
                enable_goal_token_sweep=False,
                # Sub-agents MUST produce at least one mutating tool call
                # before the final answer — no "read the file and declare
                # done" shortcuts.
                require_mutating_action=True,
                max_iterations=self.max_iterations_per_step,
                max_wall_time=per_step_wall,
                max_tokens_per_turn=self.max_tokens_per_turn,
                temperature=self.temperature,
                repeat_penalty=self.repeat_penalty,
                confirm_risky=self.confirm_risky,
                confirm_destructive=self.confirm_destructive,
                confirm_supply_chain=self.confirm_supply_chain,
                trust_repo=self.trust_repo,
                on_event=self._emit,
            )
            self._emit(AgentEvent("iteration", i, f"step {i}/{len(steps)}: {step[:80]}"))
            try:
                step_result = worker.run(step_goal)
            except Exception as exc:
                logger.exception("sub-agent step %d raised", i)
                last_stop_reason = "exception"
                break
            all_calls.extend(step_result.tool_calls)
            all_results.extend(step_result.tool_results)
            transcript_merged.append(
                {
                    "role": "assistant",
                    "content": f"[step {i}/{len(steps)}: {step}] -> {step_result.stop_reason}: "
                    f"{(step_result.final_answer or '').strip()[:200]}",
                }
            )
            step_summaries.append(step)
            last_stop_reason = step_result.stop_reason

        # Phase 3: final goal-token sweep — runs over the combined workspace
        # AFTER all planned steps. If the original goal mentioned a @decorator
        # or --flag that doesn't appear anywhere, dispatch one more focused
        # sub-agent to add it. This is the safety net for plans that missed
        # a requirement. Only runs once so we can't loop forever.
        final_sub_budget = self.max_wall_time - (time.monotonic() - start)
        if final_sub_budget > 20 and self.auto_verify_python:
            missing = self._final_goal_token_check(goal)
            if missing:
                fixup_goal = (
                    f"The original task was: {goal}\n\n"
                    f"The following REQUIRED tokens are still missing from "
                    f"the workspace — the task is NOT complete until they "
                    f"appear in the relevant file:\n"
                    + "\n".join(f"- {t}  ({r})" for t, r in missing)
                    + "\n\nFix ONLY the missing items. Use read_file, "
                    f"edit_file, or write_file as needed. When done, give "
                    f"a one-sentence plain-text confirmation."
                )
                fixup_wall = min(max(30.0, final_sub_budget), 180.0)
                fixup_worker = Agent(
                    model=self.model,
                    registry=self.registry,
                    system_prompt_extra=self.system_prompt_extra,
                    workspace_root=self.workspace_root,
                    memory=self.memory,
                    auto_verify_python=self.auto_verify_python,
                    enable_goal_token_sweep=False,
                    require_mutating_action=True,
                    max_iterations=max(4, self.max_iterations_per_step),
                    max_wall_time=fixup_wall,
                    max_tokens_per_turn=self.max_tokens_per_turn,
                    temperature=self.temperature,
                    repeat_penalty=self.repeat_penalty,
                    confirm_risky=self.confirm_risky,
                    confirm_destructive=self.confirm_destructive,
                    confirm_supply_chain=self.confirm_supply_chain,
                    trust_repo=self.trust_repo,
                    on_event=self._emit,
                )
                self._emit(
                    AgentEvent(
                        "iteration",
                        len(steps) + 1,
                        f"fixup: {len(missing)} missing token(s): "
                        + ", ".join(t for t, _ in missing),
                    )
                )
                try:
                    fixup_result = fixup_worker.run(fixup_goal)
                    all_calls.extend(fixup_result.tool_calls)
                    all_results.extend(fixup_result.tool_results)
                    step_summaries.append(
                        f"[fixup] added missing tokens: "
                        + ", ".join(t for t, _ in missing)
                    )
                    last_stop_reason = fixup_result.stop_reason
                except Exception:
                    logger.exception("architect fixup sub-agent raised")

        total_wall = time.monotonic() - start
        return AgentResult(
            final_answer=f"Completed {len(steps)} steps via architect.",
            iterations=len(steps),
            stop_reason=last_stop_reason,
            wall_time=total_wall,
            tool_calls=all_calls,
            tool_results=all_results,
            transcript=transcript_merged,
        )

    def _final_goal_token_check(self, goal: str) -> list[tuple[str, str]]:
        """Walk every .py file in the workspace and return goal-required
        tokens that don't appear anywhere. Uses the same extractor as
        the Agent's per-run sweep so rules stay in one place."""
        if self.workspace_root is None:
            return []
        tokens = Agent._extract_code_tokens(goal)
        if not tokens:
            return []
        try:
            ws = self.workspace_root.resolve()
            if not ws.is_dir():
                return []
        except OSError:
            return []
        combined_parts: list[str] = []
        try:
            for child in sorted(ws.rglob("*.py")):
                try:
                    combined_parts.append(child.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
        except OSError:
            return []
        combined = "\n".join(combined_parts)
        missing: list[tuple[str, str]] = []
        for token, reason in tokens:
            if token not in combined:
                missing.append((token, reason))
        return missing

    def _build_plan_prompt(self, goal: str) -> str:
        return (
            f"<|im_start|>system\n{_PLAN_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{goal}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _single_agent_fallback(self, goal: str, start_time: float) -> AgentResult:
        """When planning fails or returns no steps, run the flat agent once."""
        worker = Agent(
            model=self.model,
            registry=self.registry,
            system_prompt_extra=self.system_prompt_extra,
            workspace_root=self.workspace_root,
            memory=self.memory,
            auto_verify_python=self.auto_verify_python,
            max_iterations=max(10, self.max_iterations_per_step * 2),
            max_wall_time=self.max_wall_time - (time.monotonic() - start_time),
            max_tokens_per_turn=self.max_tokens_per_turn,
            temperature=self.temperature,
            repeat_penalty=self.repeat_penalty,
            confirm_risky=self.confirm_risky,
            confirm_destructive=self.confirm_destructive,
            confirm_supply_chain=self.confirm_supply_chain,
            trust_repo=self.trust_repo,
            on_event=self._emit,
        )
        return worker.run(goal)


# ── Smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    from engine.agent_builtins import build_default_registry

    class StubModel:
        def __init__(self, plan_text: str, step_responses: list[list[str]]) -> None:
            self.plan_text = plan_text
            self.step_responses = list(step_responses)
            self.calls = 0

        def generate(self, prompt: str, **kwargs: Any) -> str:
            # First call is the plan; subsequent calls are per-step sub-agent turns.
            if self.calls == 0 and _PLAN_SYSTEM.split("\n", 1)[0] in prompt:
                self.calls += 1
                return self.plan_text
            # Find the next queued step response (drained FIFO)
            for queue in self.step_responses:
                if queue:
                    self.calls += 1
                    return queue.pop(0)
            return "done"

    # Test 1: parse_plan happy path
    plan_text = "1. Add `import json` to foo.py.\n2. Add save(obj) to foo.py.\n3. Run tests."
    steps = parse_plan(plan_text)
    assert steps == [
        "Add `import json` to foo.py.",
        "Add save(obj) to foo.py.",
        "Run tests.",
    ], steps

    # Test 2: parse_plan tolerates prose around the list
    plan_text2 = (
        "Here's the plan:\n"
        "1. First step.\n"
        "2. Second step.\n"
        "\n"
        "That's the plan."
    )
    assert parse_plan(plan_text2) == ["First step.", "Second step."], parse_plan(plan_text2)

    # Test 3: end-to-end with stub model — plan 2 steps, each step resolves in 1 turn
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "target.py").write_text("# starting point\n", encoding="utf-8")
        reg = build_default_registry(ws)
        plan = "1. Add an import line to target.py.\n2. Add a function definition to target.py."
        step1_responses = [
            '<tool_call>{"name": "edit_file", "arguments": {"path": "target.py", "old_string": "", "new_string": "import os\\n"}}</tool_call>',
            "Import added.",
        ]
        step2_responses = [
            '<tool_call>{"name": "edit_file", "arguments": {"path": "target.py", "old_string": "import os", "new_string": "import os\\n\\ndef hello(): return 42"}}</tool_call>',
            "Function added.",
        ]
        model = StubModel(plan, [step1_responses, step2_responses])
        arch = ArchitectAgent(
            model=model,
            registry=reg,
            workspace_root=ws,
            max_iterations_per_step=5,
        )
        result = arch.run("Add an import and a function to target.py.")
        assert result.iterations == 2, result.iterations
        assert len(result.tool_calls) >= 2, result.tool_calls
        final_text = (ws / "target.py").read_text(encoding="utf-8")
        assert "import os" in final_text, final_text
        assert "def hello" in final_text, final_text

    print("OK: engine/architect_agent.py smoke test passed")
