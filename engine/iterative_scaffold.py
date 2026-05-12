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


# Regex matching numbered TODO markers the scaffold layer emits. Lenient
# about separators because the 14B drifts: it may write `# TODO[1]:`,
# `# TODO1:`, `# TODO 1:`, `# TODO-1:`, or `# TODO #1:`. Requires at least
# one digit so it never matches a plain `# TODO: do later`.
#   groups: (indent, number, description)
TODO_RE = re.compile(
    r"^(\s*)#\s*TODO\s*[\[\-#]?\s*(\d+)\s*\]?\s*[:.]?\s*(.*?)\s*$",
    re.MULTILINE,
)
# Quick presence check (used by callers / verifiers that don't need groups)
TODO_PRESENT_RE = re.compile(r"#\s*TODO\s*[\[\-#]?\s*\d+", re.IGNORECASE)


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

   Number TODOs sequentially starting from 1. Use the EXACT format
   `# TODO[1]:`, `# TODO[2]:` — square brackets, then a colon.
3. Top of file: imports, module docstring, any constants the structure
   demands (port lists, etc — keep small). Avoid regex literals here;
   those go in Phase 2 fills where the content is small enough to escape
   correctly.
4. Include an `if __name__ == "__main__":` block if the goal asks for
   one, but its body is also a single TODO[N].
5. DO NOT implement function bodies in this phase. Resist the temptation
   to fill in "trivial" ones. Phase 2 fills everything. (And so a body
   should never call a helper — there's nothing to call yet; if the
   design needs a helper, give it its own `def` + `# TODO[N]:`.)
6. Make EXACTLY ONE write_file call with the scaffold (array form, content
   as a list of lines). Then declare done.

After Phase 1 you'll see the scaffolded file. Phase 2 fills each TODO[N]
in turn via fresh sub-agents that each get a small focused task.
"""


def _fill_hint(file_path: str, todo_num: int, todo_text: str,
               todo_raw_line: str, context_lines: str) -> str:
    """Build the system_prompt_extra for a per-TODO fill sub-agent.

    todo_raw_line is the EXACT line as it appears in the file (leading
    indentation preserved) so the fill agent's edit_file anchor matches
    even if the scaffold drifted, AND so the indentation requirement is
    concrete in the example.
    """
    raw = todo_raw_line.rstrip()
    indent = raw[:len(raw) - len(raw.lstrip())]
    n = len(indent)
    pad = " " * n
    return (
        f"## Scaffold-then-fill mode — PHASE 2 (FILL ONE TODO)\n\n"
        f"The file `{file_path}` has this block ({n}-space indented — it's "
        f"a function body):\n\n"
        f"{raw}\n"
        f"{pad}pass\n\n"
        f"Context around the TODO (don't re-read the whole file unless "
        f"essential):\n\n```python\n{context_lines}\n```\n\n"
        "Replace ONLY that TODO line and its following `pass` with the "
        "function implementation. ONE edit_file call:\n\n"
        f'  old_string = "{raw}\\n{pad}pass"   '
        f"(include the {n} leading spaces, exactly as shown above)\n"
        "  new_string = the body as a STRING (NOT an array) — use \\n "
        f"between lines, and start EVERY line with at least {n} spaces of "
        "indentation (deeper nesting gets more). Example for a loop body "
        f"at {n}-space indent:\n"
        f'    "{pad}result = []\\n{pad}for x in items:\\n{pad}{pad}'
        f'result.append(x)\\n{pad}return result"\n\n'
        "Rules:\n"
        f"- Modify ONLY this TODO[{todo_num}]. Other TODOs are filled by "
        "their own sub-agents.\n"
        "- Keep the function signature exactly as scaffolded.\n"
        "- Don't call helper functions that aren't already defined in the "
        "file — inline what you need.\n"
        "- After the edit, declare done. Do NOT continue to other TODOs."
    )


def _scaffold_goal_wrapper(original_goal: str) -> str:
    """Decorate the user's goal with scaffold-phase framing."""
    return (
        f"PHASE 1: SCAFFOLD ONLY. Build the SKELETON for this goal:\n\n"
        f"{original_goal}\n\n"
        f"Remember: small skeleton with # TODO[N] markers, < 60 lines, "
        f"single write_file call (array form). Phase 2 (separate sub-agents) "
        f"fills the TODOs."
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


def _find_next_todo(file_path: Path) -> Optional[tuple[int, str, str, str]]:
    """Read file, return (todo_num, todo_text, raw_line, context_lines) for
    the first remaining TODO marker, or None if none left.

    raw_line is the exact line as it appears in the file — needed so the
    fill sub-agent's edit_file anchor matches whatever format drift the
    scaffold model introduced.
    """
    if not file_path.exists():
        return None
    text = file_path.read_text(encoding="utf-8", errors="replace")
    m = TODO_RE.search(text)
    if not m:
        return None
    todo_num = int(m.group(2))
    todo_text = m.group(3).strip()
    lines = text.splitlines()
    todo_line_idx = text[:m.start()].count("\n")
    raw_line = lines[todo_line_idx] if todo_line_idx < len(lines) else m.group(0)
    ctx_start = max(0, todo_line_idx - 5)
    ctx_end = min(len(lines), todo_line_idx + 6)
    context = "\n".join(lines[ctx_start:ctx_end])
    return todo_num, todo_text, raw_line, context


def _count_remaining_todos(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return len(TODO_RE.findall(text))


def _file_is_complete(file_path: Path) -> tuple[bool, str]:
    """A scaffolded file is 'complete' when it compiles AND has no TODO
    markers left. Returns (ok, reason)."""
    if not file_path.exists():
        return False, "file missing"
    text = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        compile(text, str(file_path), "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} at line {exc.lineno}"
    n = _count_remaining_todos(file_path)
    if n > 0:
        return False, f"{n} TODO marker(s) remaining"
    return True, "compiles, no TODOs"


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
        """Run the build-then-fill loop. Returns a synthesized AgentResult.

        continue_session is accepted for API compatibility with Agent.run
        but is ignored — each iterative-scaffold run is fresh.
        """
        start = time.monotonic()
        all_calls: list[ToolCall] = []
        all_results: list[ToolResult] = []
        merged_transcript: list[dict[str, str]] = [{"role": "user", "content": goal}]

        def _result(final_answer: str, stop_reason: str,
                    iterations: int) -> AgentResult:
            """Build an AgentResult with wall_time always populated."""
            return AgentResult(
                final_answer=final_answer,
                iterations=iterations,
                stop_reason=stop_reason,
                wall_time=time.monotonic() - start,
                tool_calls=all_calls,
                tool_results=all_results,
                transcript=merged_transcript,
            )

        before = _list_user_files(self.workspace_root)

        # ── Phase 1: BUILD (whole file if small, skeleton if large) ───────
        self._emit(AgentEvent("iteration", 0,
                              "iterative_scaffold: PHASE 1 (build)"))
        build_agent = self._build_subagent(
            max_iters=self.max_scaffold_iterations,
            system_extra=SCAFFOLD_HINT,
            wall_budget=self.max_wall_time * 0.45,  # more budget — Option A may do it all
        )
        try:
            build_result = build_agent.run(_scaffold_goal_wrapper(goal))
        except Exception:
            logger.exception("build phase raised")
            build_result = None

        if build_result is not None:
            all_calls.extend(build_result.tool_calls)
            all_results.extend(build_result.tool_results)
            merged_transcript.append({
                "role": "assistant",
                "content": "Build: " + (build_result.final_answer or "")[:200],
            })

        # ── Identify the file Phase 1 produced ────────────────────────────
        after = _list_user_files(self.workspace_root)
        new_py = sorted(
            (p for p in (after - before) if p.suffix == ".py"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        target_file = new_py[0] if new_py else None

        if target_file is None:
            return _result(
                build_result.final_answer if build_result else
                "build phase produced no .py file",
                "no_output", getattr(build_result, "iterations", 0),
            )

        # ── Did Phase 1 already finish it? (Option A path) ────────────────
        complete, why = _file_is_complete(target_file)
        if complete:
            self._emit(AgentEvent("iteration", 0,
                                  f"iterative_scaffold: Phase 1 wrote {target_file.name} "
                                  f"complete in one pass ({why})"))
            return _result(
                f"iterative_scaffold: {target_file.name} built in one pass — {why}.",
                "completed", getattr(build_result, "iterations", 0),
            )

        # ── Phase 1 produced a skeleton — proceed to fills ────────────────
        n_todos = _count_remaining_todos(target_file)
        if n_todos == 0:
            # No TODOs but not complete = syntax error in the skeleton.
            return _result(
                f"iterative_scaffold: skeleton {target_file.name} has no TODOs "
                f"but doesn't compile — {why}.",
                "skeleton_broken", getattr(build_result, "iterations", 0),
            )

        self._emit(AgentEvent("iteration", 0,
                              f"iterative_scaffold: skeleton {target_file.name}, {n_todos} TODOs"))

        # (No Phase 1.5 forward-reference fix step — it disrupted the fill
        # flow more than it helped in soak v2/v6. The SCAFFOLD_HINT now tells
        # the model not to call undefined helpers in skeleton bodies; if a
        # forward-ref slips through, the per-TODO fill agent's auto_verify
        # will surface it and the fill agent can add the stub itself.)

        # ── Phase 2: fill each TODO in its own fresh sub-agent ────────────
        fills_completed = 0
        prev_todo_num = None
        same_streak = 0
        rel = target_file.relative_to(self.workspace_root)
        for _ in range(self.max_total_fills):
            if time.monotonic() - start >= self.max_wall_time:
                return _result(
                    f"iterative_scaffold: hit wall time after {fills_completed} fills, "
                    f"{_count_remaining_todos(target_file)} TODOs remain.",
                    "wall_time", fills_completed,
                )
            todo = _find_next_todo(target_file)
            if todo is None:
                break
            todo_num, todo_text, raw_line, context = todo

            if prev_todo_num == todo_num:
                same_streak += 1
                if same_streak >= 2:
                    self._emit(AgentEvent("iteration", fills_completed,
                                          f"iterative_scaffold: stuck on TODO[{todo_num}], abandoning"))
                    return _result(
                        f"iterative_scaffold: stuck on TODO[{todo_num}] ({todo_text[:60]}); "
                        f"{fills_completed} fills done, {_count_remaining_todos(target_file)} remain.",
                        "stuck_on_todo", fills_completed,
                    )
            else:
                same_streak = 0
            prev_todo_num = todo_num

            remaining_n = max(1, _count_remaining_todos(target_file))
            fill_wall = max(30.0, (self.max_wall_time - (time.monotonic() - start)) / remaining_n)

            self._emit(AgentEvent("iteration", fills_completed,
                                  f"iterative_scaffold: PHASE 2 fill TODO[{todo_num}]: {todo_text[:50]}"))
            fill_agent = self._build_subagent(
                max_iters=self.max_fill_iterations_per_todo,
                system_extra=_fill_hint(str(rel), todo_num, todo_text, raw_line, context),
                wall_budget=fill_wall,
            )
            try:
                fr = fill_agent.run(
                    f"Fill TODO[{todo_num}] in `{rel}`. Original task:\n\n{goal}"
                )
                all_calls.extend(fr.tool_calls)
                all_results.extend(fr.tool_results)
            except Exception:
                logger.exception(f"fill TODO[{todo_num}] raised")
                return _result(
                    f"iterative_scaffold: fill TODO[{todo_num}] crashed; "
                    f"{fills_completed} fills done.",
                    "fill_error", fills_completed,
                )
            fills_completed += 1

        # ── Phase 3: final verify ─────────────────────────────────────────
        complete, why = _file_is_complete(target_file)
        if complete:
            return _result(
                f"iterative_scaffold: {target_file.name} complete after {fills_completed} fills.",
                "completed", fills_completed,
            )
        return _result(
            f"iterative_scaffold: {fills_completed} fills done but {target_file.name} "
            f"not complete — {why}.",
            "incomplete", fills_completed,
        )
