"""Iterative scaffold-then-fill driver for single-file builds.

Designed to work around the 14B's per-tool-call content size ceiling. When
asked to build a large file in one shot, the 14B's JSON tool calls break:
multi-line content + quote-escape + regex backslashes consistently produce
parse errors that even self_heal can't recover from past ~50-80 LOC.

Decomposition that sidesteps the ceiling:

  Phase 1 — SCAFFOLD: model emits a SKELETON file (<60 LOC), each function
            body just `# TODO[N]: description` + `pass`. Small content =
            JSON parses cleanly.
  Phase 2 — FILL: for each TODO marker, a fresh sub-agent reads the file,
            replaces ONLY that TODO line via edit_file (anchored on the
            unique numbered marker). One function body per call = small
            content, no transcript bloat between fills.
  Phase 3 — VERIFY: final compile + check no TODO markers remain.

Sibling to engine/architect_agent.py (multi-file decomposition). Mirrors
the plan-then-execute shape but on a different axis. Same public surface
as Agent.run() so callers can swap.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from engine.agent import Agent, AgentEvent, AgentResult, ToolCall, ToolResult
from engine.agent_memory import AgentMemory
from engine.agent_tools import ToolRegistry

logger = logging.getLogger(__name__)


# Regex matching numbered TODOs the scaffold layer emits.
# Examples:
#   # TODO[1]: parse_iwlist output into list of Cell dicts
#   # TODO[12]: render the executive summary
TODO_RE = re.compile(r"^(\s*)#\s*TODO\[(\d+)\]\s*:\s*(.*?)\s*$", re.MULTILINE)


SCAFFOLD_HINT = """\
## Scaffold-then-fill mode

You are building this file in TWO phases. THIS IS PHASE 1 (SCAFFOLD).

Produce a SKELETON of the target file. Rules:

1. **Total file size: < 60 lines.** This is non-negotiable; large content
   blocks break the JSON tool-call layer.
2. Define every function signature with proper type hints + a docstring,
   but replace each function BODY with exactly two lines:

       # TODO[N]: <one-line description of what this function should do>
       pass

   Number TODOs sequentially starting from 1.
3. Top of file: imports, module docstring, any constants the structure
   demands (port lists, regex patterns, etc — keep small).
4. Include an `if __name__ == "__main__":` block if the goal asks for
   one, but its body is also a single TODO[N].
5. DO NOT implement function bodies in this phase. Resist the temptation
   to fill in "trivial" ones. Phase 2 fills everything.
6. Make EXACTLY ONE write_file call with the scaffold. Then declare done.

After Phase 1 you'll see the scaffolded file. Phase 2 fills each TODO[N]
in turn via fresh sub-agents that each get a small focused task.
"""


def _fill_hint(file_path: str, todo_num: int, todo_text: str,
               context_lines: str) -> str:
    """Build the system_prompt_extra for a per-TODO fill sub-agent."""
    return (
        f"## Scaffold-then-fill mode — PHASE 2 (FILL TODO[{todo_num}])\n\n"
        f"The file `{file_path}` has the marker:\n\n"
        f"    # TODO[{todo_num}]: {todo_text}\n"
        f"    pass\n\n"
        f"Context around the TODO (do not re-read the whole file unless\n"
        f"essential):\n\n"
        f"```python\n{context_lines}\n```\n\n"
        "Your task: replace ONLY the TODO line and its following `pass`\n"
        "with the function implementation. Use edit_file with:\n\n"
        f'  old_string=\'    # TODO[{todo_num}]: {todo_text}\\n    pass\'\n'
        f"  new_string=<the actual implementation lines>\n\n"
        "Rules:\n"
        f"- Modify ONLY this TODO[{todo_num}]. Other TODOs are filled by\n"
        "  their own sub-agents.\n"
        "- Keep the function signature exactly as scaffolded.\n"
        "- Use the array-of-strings form for write_file/edit_file content\n"
        "  if your new_string spans more than ~10 lines.\n"
        "- After the edit, declare done. Do NOT continue to other TODOs."
    )


def _scaffold_goal_wrapper(original_goal: str) -> str:
    """Decorate the user's goal with scaffold-phase framing."""
    return (
        f"PHASE 1: SCAFFOLD ONLY. Build the SKELETON for this goal:\n\n"
        f"{original_goal}\n\n"
        f"Remember: small skeleton with # TODO[N] markers, < 60 lines, "
        f"single write_file call. Phase 2 (separate sub-agents) fills the "
        f"TODOs."
    )


def _list_user_files(workspace: Path) -> set[Path]:
    """Snapshot the workspace's .py / .md files (top-level + 1 level deep)."""
    out: set[Path] = set()
    if not workspace.exists():
        return out
    for entry in workspace.iterdir():
        if entry.is_file() and entry.suffix in (".py", ".md", ".txt", ".json", ".yaml", ".yml"):
            out.add(entry.resolve())
        elif entry.is_dir() and not entry.name.startswith(".") and entry.name not in ("__pycache__",):
            for sub in entry.iterdir():
                if sub.is_file() and sub.suffix in (".py", ".md", ".txt"):
                    out.add(sub.resolve())
    return out


def _find_next_todo(file_path: Path) -> Optional[tuple[int, str, str]]:
    """Read file, return (todo_num, todo_text, context_lines) for the first
    remaining TODO marker, or None if none left."""
    if not file_path.exists():
        return None
    text = file_path.read_text(encoding="utf-8", errors="replace")
    m = TODO_RE.search(text)
    if not m:
        return None
    todo_num = int(m.group(2))
    todo_text = m.group(3).strip()
    # Pull ~5 lines of context before + after the TODO for the fill sub-agent
    lines = text.splitlines()
    todo_line_idx = text[:m.start()].count("\n")
    ctx_start = max(0, todo_line_idx - 5)
    ctx_end = min(len(lines), todo_line_idx + 6)
    context = "\n".join(lines[ctx_start:ctx_end])
    return todo_num, todo_text, context


def _count_remaining_todos(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return len(TODO_RE.findall(text))


class IterativeScaffoldDriver:
    """Scaffold-then-fill driver. Same .run() surface as Agent."""

    def __init__(
        self,
        model: Any,
        registry: ToolRegistry,
        system_prompt_extra: str = "",
        workspace_root: Optional[Path] = None,
        memory: Optional[AgentMemory] = None,
        max_scaffold_iterations: int = 8,
        max_fill_iterations_per_todo: int = 6,
        max_total_fills: int = 25,
        max_wall_time: float = 900.0,
        max_tokens_per_turn: int = 1024,
        temperature: Optional[float] = 0.1,
        repeat_penalty: Optional[float] = 1.15,
        confirm_risky: Optional[Callable[[ToolCall], bool]] = None,
        augment_for_goal: Optional[Callable[[str], str]] = None,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.system_prompt_extra = system_prompt_extra
        self.workspace_root = Path(workspace_root) if workspace_root is not None else Path.cwd()
        self.memory = memory
        self.max_scaffold_iterations = max(1, int(max_scaffold_iterations))
        self.max_fill_iterations_per_todo = max(1, int(max_fill_iterations_per_todo))
        self.max_total_fills = max(1, int(max_total_fills))
        self.max_wall_time = float(max_wall_time)
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.confirm_risky = confirm_risky
        self.augment_for_goal = augment_for_goal
        self._emit = on_event or (lambda _e: None)

    def _build_subagent(self, max_iters: int, system_extra: str,
                        wall_budget: float) -> Agent:
        """Construct a fresh per-phase Agent. Shares model+registry but has
        a clean transcript and its own iteration budget."""
        return Agent(
            model=self.model,
            registry=self.registry,
            system_prompt_extra=(
                (self.system_prompt_extra + "\n\n" + system_extra).strip()
                if self.system_prompt_extra else system_extra
            ),
            workspace_root=self.workspace_root,
            memory=None,  # No cross-session memory in the sub-agent — it's stateless
            auto_verify_python=True,
            enable_goal_token_sweep=False,  # not useful for tiny per-TODO goals
            max_iterations=max_iters,
            max_wall_time=max(30.0, wall_budget),
            max_tokens_per_turn=self.max_tokens_per_turn,
            temperature=self.temperature,
            repeat_penalty=self.repeat_penalty,
            confirm_risky=self.confirm_risky or (lambda _: True),
            augment_for_goal=self.augment_for_goal,
            enable_self_heal=True,
            on_event=self._emit,
        )

    def run(self, goal: str, continue_session: bool = False) -> AgentResult:
        """Run the scaffold-then-fill loop. Returns a synthesized AgentResult.

        continue_session is accepted for API compatibility with Agent.run
        but is ignored — each iterative-scaffold run is fresh.
        """
        start = time.monotonic()
        all_calls: list[ToolCall] = []
        all_results: list[ToolResult] = []
        merged_transcript: list[dict[str, str]] = [{"role": "user", "content": goal}]
        stop_reason = "completed"

        # ── Snapshot pre-existing files so we can identify the scaffolded one
        before = _list_user_files(self.workspace_root)

        # ── Phase 1: SCAFFOLD ──────────────────────────────────────────────
        self._emit(AgentEvent("iteration", 0,
                              "iterative_scaffold: PHASE 1 (skeleton)"))
        scaffold_agent = self._build_subagent(
            max_iters=self.max_scaffold_iterations,
            system_extra=SCAFFOLD_HINT,
            wall_budget=self.max_wall_time * 0.25,
        )
        try:
            scaffold_result = scaffold_agent.run(_scaffold_goal_wrapper(goal))
        except Exception:
            logger.exception("scaffold phase raised")
            stop_reason = "scaffold_error"
            scaffold_result = None

        if scaffold_result is not None:
            all_calls.extend(scaffold_result.tool_calls)
            all_results.extend(scaffold_result.tool_results)
            merged_transcript.append({
                "role": "assistant",
                "content": "Scaffold: " + (scaffold_result.final_answer or "")[:200],
            })

        # ── Discover the scaffolded file ──────────────────────────────────
        after = _list_user_files(self.workspace_root)
        new_files = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
        scaffold_file = None
        for candidate in new_files:
            if candidate.suffix == ".py" and _count_remaining_todos(candidate) > 0:
                scaffold_file = candidate
                break
        if scaffold_file is None:
            # No identifiable scaffold — fall through. Return whatever the
            # scaffold agent produced.
            stop_reason = "no_scaffold"
            return AgentResult(
                final_answer=(scaffold_result.final_answer if scaffold_result else
                              "scaffold phase failed to create a TODO-marked file"),
                iterations=getattr(scaffold_result, "iterations", 0),
                tool_calls=all_calls,
                tool_results=all_results,
                stop_reason=stop_reason,
                transcript=merged_transcript,
            )

        self._emit(AgentEvent("iteration", 0,
                              f"iterative_scaffold: scaffold at {scaffold_file.name}, "
                              f"{_count_remaining_todos(scaffold_file)} TODOs"))

        # ── Phase 2: FILL each TODO in its own sub-agent ──────────────────
        fills_completed = 0
        previous_todo_num = None
        same_todo_streak = 0
        for _ in range(self.max_total_fills):
            elapsed = time.monotonic() - start
            if elapsed >= self.max_wall_time:
                stop_reason = "wall_time"
                break

            todo = _find_next_todo(scaffold_file)
            if todo is None:
                break

            todo_num, todo_text, context = todo

            # Same-TODO streak detection — if the previous fill didn't make
            # progress (TODO still there with same number), give up after 2
            # consecutive failures on the same one.
            if previous_todo_num == todo_num:
                same_todo_streak += 1
                if same_todo_streak >= 2:
                    self._emit(AgentEvent("iteration", fills_completed,
                                          f"iterative_scaffold: stuck on TODO[{todo_num}], abandoning"))
                    stop_reason = "stuck_on_todo"
                    break
            else:
                same_todo_streak = 0
            previous_todo_num = todo_num

            remaining_budget = self.max_wall_time - elapsed
            fill_wall = max(30.0, remaining_budget / max(1, _count_remaining_todos(scaffold_file)))

            self._emit(AgentEvent("iteration", fills_completed,
                                  f"iterative_scaffold: PHASE 2 fill TODO[{todo_num}]: {todo_text[:60]}"))

            fill_agent = self._build_subagent(
                max_iters=self.max_fill_iterations_per_todo,
                system_extra=_fill_hint(
                    str(scaffold_file.relative_to(self.workspace_root)),
                    todo_num, todo_text, context,
                ),
                wall_budget=fill_wall,
            )

            fill_goal = (
                f"Replace TODO[{todo_num}] in `{scaffold_file.relative_to(self.workspace_root)}` "
                f"with the implementation. Original task context:\n\n{goal}"
            )

            try:
                fill_result = fill_agent.run(fill_goal)
            except Exception:
                logger.exception(f"fill TODO[{todo_num}] raised")
                stop_reason = "fill_error"
                break

            all_calls.extend(fill_result.tool_calls)
            all_results.extend(fill_result.tool_results)
            fills_completed += 1

        # ── Phase 3: VERIFY ───────────────────────────────────────────────
        remaining = _count_remaining_todos(scaffold_file)
        compile_ok = True
        compile_reason = ""
        try:
            text = scaffold_file.read_text(encoding="utf-8", errors="replace")
            compile(text, str(scaffold_file), "exec")
        except SyntaxError as exc:
            compile_ok = False
            compile_reason = f"SyntaxError: {exc.msg} at line {exc.lineno}"

        if remaining > 0 and stop_reason == "completed":
            stop_reason = "todos_remaining"
        if not compile_ok and stop_reason == "completed":
            stop_reason = "syntax_error"

        final_answer = (
            f"iterative_scaffold: completed {fills_completed} fills. "
            f"{remaining} TODOs remain. "
            f"Compile: {'OK' if compile_ok else compile_reason}."
        )

        return AgentResult(
            final_answer=final_answer,
            iterations=fills_completed,
            tool_calls=all_calls,
            tool_results=all_results,
            stop_reason=stop_reason,
            transcript=merged_transcript,
        )
