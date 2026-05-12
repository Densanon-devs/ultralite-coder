"""
Phase 14 Agent — ReAct loop on top of engine.agent_tools.ToolRegistry.

The agent runs a multi-turn loop:
    plan -> tool_call -> observe -> repeat -> final_answer

Each iteration:
1. Build a ChatML prompt from the running transcript (system block with the
   Hermes tool schemas + user goal + assistant/tool history so far).
2. Generate one assistant turn from the model (capped at max_tokens_per_turn).
3. Parse <tool_call> tags. No tool calls -> that turn IS the final answer; stop.
4. Otherwise execute every parsed call (with optional risky-confirm hook),
   append the tool results as a "tool"-role message, and loop.

Budgets:
- max_iterations      — hard cap on tool-call rounds
- max_wall_time       — wall-clock budget across the whole run
- max_tokens_per_turn — cap on each model.generate() call

The Agent class is loader-agnostic — it only needs an object exposing
`.generate(prompt, max_tokens=..., stop=...) -> str`. That makes BaseModel-backed
runs and stub-backed unit tests both straightforward.

Zero servers, pure in-process.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from engine.agent_memory import AgentMemory
from engine.agent_tools import ToolCall, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Tools that are safe to execute concurrently in a thread pool. Must be
# read-only w.r.t. the workspace (no file writes, no shell commands) so
# that dispatch order doesn't matter. Write-touching tools (write_file,
# edit_file, run_bash) stay serial — the model's intended order must be
# preserved for them.
_PARALLELIZABLE_TOOLS = frozenset({"read_file", "list_dir", "glob", "grep"})


# ── Types ───────────────────────────────────────────────────────


class _ModelLike(Protocol):
    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str: ...


@dataclass
class AgentEvent:
    """Telemetry event surfaced via the on_event callback. CLI/REPL renders these."""

    type: str  # "iteration" | "model_text" | "tool_call" | "tool_result" | "final" | "stopped"
    iteration: int
    payload: Any = None


@dataclass
class AgentResult:
    final_answer: str
    iterations: int
    stop_reason: str  # "answered" | "max_iterations" | "wall_time" | "model_error"
    wall_time: float
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)
    compactions: int = 0  # how many context-compaction passes fired during this run
    self_heals: int = 0   # how many live diagnose-and-repair injections fired


# ── System prompt ───────────────────────────────────────────────

_DEFAULT_SYSTEM = """You are Ultralight Coder, an autonomous local coding agent.

Your job: complete the user's goal by calling tools, then giving a short final answer.

HOW TO CALL A TOOL — you MUST use this exact format:

<tool_call>
{"name": "read_file", "arguments": {"path": "calculator.py"}}
</tool_call>

Do NOT wrap the JSON in ```json code fences.
Do NOT write "Here is the tool call:" — just emit the <tool_call> block.
Do NOT write the JSON without the <tool_call>...</tool_call> tags.
Every tool call MUST be wrapped in literal <tool_call> and </tool_call> XML tags.

JSON STRING ESCAPING — VERY IMPORTANT:
Inside JSON string values you MUST use double quotes only and escape inner double quotes as \\". Never wrap a string in Python-style single quotes like '...' or f"..." — JSON doesn't allow that.

Correct Python docstring edit (note the escaped \\" around the triple-quoted string):
<tool_call>
{"name": "edit_file", "arguments": {"path": "x.py", "old_string": "def foo():", "new_string": "def foo():\\n    \\"\\"\\"Short description.\\"\\"\\""}}
</tool_call>

Correct multi-line write_file (use \\n for newlines inside the content string):
<tool_call>
{"name": "write_file", "arguments": {"path": "x.py", "content": "def hello():\\n    return 'hi'\\n"}}
</tool_call>

RECOMMENDED for files with many lines or embedded quotes/f-strings — use the array form for `content` where each element is one line of the file (no \\n needed, no escape hell):
<tool_call>
{"name": "write_file", "arguments": {"path": "x.py", "content": ["import os", "", "def hello(name):", "    return f\"Hi, {name}\""]}}
</tool_call>
Each array element becomes one line of the file. Strongly prefer this form for any file with 5+ lines or any embedded `"` characters — it eliminates almost every JSON-escape failure.

Newlines inside a JSON string can be written either as literal newlines OR as \\n. Both work. But QUOTES must always be escaped.

HOW TO FINISH:
When the goal is complete, write a short plain-text final answer with NO tool_call tags in it. That plain-text message is how the user knows you are done.

WORKFLOW:
1. Inspect what you need (read_file, list_dir, glob, grep) before editing.
2. Make minimal targeted changes (edit_file for small edits, write_file for new files).
3. To run tests, ALWAYS prefer `run_tests` (supports pytest, unittest, npm, go, cargo) over `run_bash`. `run_bash` is a risky tool that prompts the user; `run_tests` is a safe dedicated tool and is the right choice for any test-running step.
4. After write_file or edit_file on a .py file you will see an `auto_verify`
   observation. If it reports a SyntaxError, fix the file before doing anything else.
5. **Create all requested files BEFORE running tests or the CLI.** If the user asks for files A, B, C, D, make sure every one exists before moving on to verification. Don't skip a file.
6. When the whole goal is complete, write the final plain-text answer.

CRITICAL RULES:
- **READING IS NOT THE GOAL.** If the user asks you to "add", "change", "rename", "fix", "create", "remove", or "build" something, you MUST emit an edit_file or write_file call before giving a final answer. Reading files is preparation — actually making the change is completion. A final answer after only read_file calls is WRONG unless the user specifically asked you to "read", "show", "explain", "find", "describe", or "list" something.
- **DO NOT REPEAT A FAILING CALL.** If a tool returned an error, try a DIFFERENT approach on the next turn — different tool, different arguments, or re-read the current file state first. Emitting the same call with the same arguments 3 times triggers a `stuck_repeat` error.
- **`stuck_repeat` IS NOT A REASON TO QUIT.** If you see a `stuck_repeat` result, the task is STILL doable — you just need a different approach. Pivot: read the file again to see its current state, switch from write_file to edit_file with a short unique `old_string`, or use grep to locate a specific anchor point. NEVER respond to `stuck_repeat` with "I can't complete the task" — the task is completable, you just need to try something you haven't tried.
- **FOR RENAMES, USE GREP FIRST.** Rename tasks REQUIRE running `grep` over the whole project (`path="."`, pattern=the old name) to locate EVERY occurrence before making any edits. An incomplete rename that misses a call site will leave the code broken. After grepping, edit every file that matched.
- **edit_file old_string: use a SHORT unique substring OR empty for prepend.** Pick a SHORT, specific substring from the file — usually a unique code fragment like `pages.append([])` or `def do_thing(x):`. Keep it short and strip any leading/trailing whitespace; whitespace in old_string has to match the file EXACTLY (tabs vs spaces) and guessing wrong is the most common cause of `old_string not found`. **SPECIAL CASE — empty `old_string` means PREPEND.** When you need to add a line at the very top of a file (typically a new `import` or `from X import Y` statement), pass `"old_string": ""` and put the full line (with trailing `\n`) in `new_string`. Example: `<tool_call>{"name": "edit_file", "arguments": {"path": "person.py", "old_string": "", "new_string": "from dataclasses import dataclass\\n"}}</tool_call>` — this places the import at the very top of person.py, BEFORE any existing decorators/classes/code. This is the canonical way to add a missing import when auto_verify reports "references undefined names".
- **WHEN EDITING A DECORATED CLASS OR FUNCTION, INCLUDE THE DECORATOR IN old_string.** If you want to replace `@dataclass\nclass Person:` with something, your `old_string` MUST start with `@dataclass\n` — not with `class Person:`. Otherwise the decorator becomes an orphan on its own line when the class line is replaced, and Python will say "invalid syntax" on the line after the decorator.
- **IF edit_file RETURNS `old_string not found`, THE FILE WAS NOT MODIFIED.** You MUST immediately re-emit edit_file with a different, SHORTER old_string — strip any whitespace, drop the newline character, try just the unique code fragment alone. Do NOT claim the edit succeeded, do NOT run tests, do NOT move on. Running tests after a failed edit will produce the same failure because the file is unchanged. An edit is confirmed ONLY when the result says `Replaced N occurrence(s)`.
- **FOR RENAME/REPLACE-ALL TASKS, PROCESS EVERY GREP MATCH.** After running grep for the old identifier, count the matches in the grep output. You must make ONE edit_file call for EACH unique match location (including call sites, not just definitions). A task that says "rename X to Y everywhere" is only done when grepping for X returns zero matches.
- **FOR CIRCULAR IMPORTS, USE LAZY IMPORTS INSIDE THE FUNCTION THAT NEEDS THEM.** The canonical fix for `from b import b_thing` at the top of a.py (when b.py also imports from a.py) is: (1) DELETE the top-level `from b import b_thing` line from a.py, (2) ADD `from b import b_thing` INSIDE the body of the function that calls `b_thing()`. The import now runs at call time, after both modules have finished loading, breaking the cycle. Do NOT just delete the import — the function still needs the name. Do NOT move functions between files — just move the import into the function body.
- **`read_file` OUTPUT IS LINE-NUMBER PREFIXED — STRIP THE PREFIX BEFORE USING CONTENT IN edit_file.** Every line of read_file output starts with a right-aligned line number + tab (e.g. `     9\targs = parser.parse_args()`). Those `     9\t` characters are NOT part of the file on disk. When you copy a line into an `old_string`, DROP the leading number and tab — use only the actual code text after the tab (e.g. `    args = parser.parse_args()`, keeping the real indentation). Leaving the `     9\t` prefix in your old_string will always fail with "old_string not found".
- **IF auto_verify REPORTS "references undefined names", FIX THAT FILE NEXT.** The name you just wrote uses is not imported or defined. Add the missing `from X import Y` statement at the top of the file (or move it inside the function for lazy imports) BEFORE doing anything else. Do not run tests — the NameError will reproduce at runtime. Fix the import first.
- **NEVER DELETE EXISTING CODE UNLESS THE TASK EXPLICITLY SAYS SO.** If the user says "add a function to X", ADD, do not replace. Use edit_file with a specific `old_string` anchor (e.g. the last line of the existing content) and put the new content in `new_string` AFTER a newline. Do NOT choose `old_string` that includes existing functions you are not asked to modify — replacing `def existing_func():\n    ...` with `def new_func():\n    ...` silently deletes existing_func. Always ADD alongside, never overwrite what was there before unless the task says "replace" or "rewrite".
- **AVOID TRIPLE-QUOTED PYTHON STRINGS INSIDE write_file CONTENT.** When writing a Python module that needs a docstring, prefer a single-line docstring: put the opening three double-quotes, the text, and the closing three double-quotes all on ONE line. Multi-line triple-quoted strings inside a JSON string value are error-prone because it is easy to miscount the quotes. If you must embed a multi-line docstring, count the opening and closing quote characters carefully: 3 to open, 3 to close, same number on both sides.
- **F-STRING NESTED QUOTES TRAP (Python 3.10/3.11).** Inside a single-quoted f-string you CANNOT put another single-quoted string, and inside a double-quoted f-string you CANNOT put another double-quoted string. `f'{x if y else 'default'}'` is a SYNTAX ERROR in Python 3.10 — the inner `'` closes the outer `'`. (Python 3.12 fixed this, but the benchmark and most real projects still target 3.10/3.11.) The two safe patterns are: (a) use DIFFERENT quote styles for the f-string and the inner literals: `f"{x if y else 'default'}"` — outer `"`, inner `'`; (b) compute the value BEFORE the f-string: `label = 'Done' if done else 'Todo'` then `f'id:{i} {label}'`. Same rule for conditional expressions with string literals, dict lookups like `{d['key']}`, and `.get(...)` calls — swap quote styles or extract the expression first. When writing f-strings that contain string literals inside `{}`, default to DOUBLE-QUOTED f-strings with SINGLE-QUOTED inner literals.
- **FIX SYNTAX ERRORS BEFORE DOING ANYTHING ELSE.** If `auto_verify` reports `SyntaxError in X.py`, that file is broken and importing it will crash. Fix X.py IMMEDIATELY on the very next turn — do not move to another file, do not run tests, do not declare success. Running tests against a module with a syntax error always fails and never reveals anything new. If multiple files have syntax errors, fix them one at a time, starting with the one that was most recently touched.

Other rules:
- Always read a file before editing it.
- Prefer edit_file over write_file for small changes — it preserves the rest of the file.
- Keep tool arguments compact.
- If a tool returns an error, read the error carefully and try a different approach.
- **Check every run_bash and run_tests result before declaring success.** If stderr contains a Traceback, SyntaxError, AssertionError, or any non-empty error text, the step FAILED — fix it before moving on. Do not say "it works" when stderr shows an error.
- If you see a `parse_error` tool result, one of your earlier tool calls was malformed JSON and did NOT execute. Re-emit the call with valid JSON (remember: JSON strings use regular `"`, never Python `f"..."` or `'...'` — and inner `"` must be escaped as `\\"`).
- Be terse between tool calls — one or two sentences of reasoning at most.
- NEVER write `<tool_response>` in your output. Tool responses are produced
  by the runtime AFTER your tool calls execute. You only emit `<tool_call>`
  blocks to REQUEST execution, then stop and wait for the next turn.
- NEVER write `<|im_start|>` or `<|im_end|>` tags. Those are role boundaries
  handled by the runtime, not by you.
- Do not pretend a tool ran by describing its output — actually emit the
  `<tool_call>` block and wait for the real observation on the next turn."""


# ── Agent ───────────────────────────────────────────────────────


class Agent:
    """
    ReAct-style agent loop. Stateless across run() calls — each run() rebuilds
    the transcript from scratch. (Cross-session memory is Phase 14a step 5.)
    """

    def __init__(
        self,
        model: _ModelLike,
        registry: ToolRegistry,
        system_prompt_extra: str = "",
        workspace_root: Optional[Path] = None,
        memory: Optional[AgentMemory] = None,
        auto_verify_python: bool = True,
        enable_goal_token_sweep: bool = True,
        require_mutating_action: bool = False,
        # Live self_heal diagnose-and-repair injection on consecutive
        # same-class tool failures. Default-OFF as of 2026-05-05 — the
        # A/B (standard 14-task --repeat 2 + stress fixture --repeat 3,
        # all on Qwen 2.5 Coder 14B) recorded 0 firings across 31 task
        # runs because the upstream layers (parser repair, structured
        # tool-error feedback, stuck_repeat, pre_finish_check, auto-apply
        # heuristics) already prevent same-class 2-streaks. Layer kept
        # reachable via this opt-in for stress scenarios + smaller models
        # with weaker tool-call discipline. See
        # session_2026-05-05_self_heal_ab_finding.md.
        enable_self_heal: bool = False,
        suppress_think: bool = False,
        max_iterations: int = 10,
        max_wall_time: float = 300.0,
        max_tokens_per_turn: int = 1024,
        # temp 0.1 is tight enough to produce valid JSON reliably but
        # repeat_penalty 1.15 is required to break the "defaultdict\n ..."
        # degenerate loop that temp 0.1 alone falls into. Discovered in
        # Phase 14 stress test v5→v6.
        temperature: Optional[float] = 0.1,
        repeat_penalty: Optional[float] = 1.15,
        # Context compaction. When the rendered prompt approaches the
        # model's context ceiling, the runtime elides old tool_response
        # bodies (file reads, long bash output) so the window stays
        # within budget. `context_char_budget` is a rough char cap — set
        # to ~3.2 * n_ctx (coarse tokens-to-chars). `compact_keep_recent`
        # is how many most-recent transcript turns stay untouched; older
        # tool outputs get replaced with a one-line elision marker.
        # Setting `context_char_budget=0` disables compaction entirely.
        context_char_budget: int = 52000,   # ~16k tokens * 3.25 chars/token
        compact_keep_recent: int = 6,
        confirm_risky: Optional[Callable[[ToolCall], bool]] = None,
        # Destructive-shell-command gate. Independent of confirm_risky.
        # Runs only on run_bash calls whose command matches the
        # destructive_command_gate denylist (rm -rf, ~/ expansion, format,
        # dd to /dev, git reset --hard, DROP TABLE, etc.). The callback
        # MUST NOT honor --yes / auto-approve — destructive prompts are
        # mandatory regardless of the global permission mode. Default
        # None: matched commands are refused outright (fail safe). See
        # engine/destructive_command_gate.py + the May 2026 r/ClaudeAI
        # `rm -rf ~/` incident for rationale.
        confirm_destructive: Optional[Callable[["ToolCall", list], bool]] = None,
        # Pre-finish check: called when the model declares "done" (no tool
        # calls). Returns None to accept the answer, or a non-empty string
        # describing what's still wrong — which gets injected as a user
        # message so the model can self-correct. Max 2 retries to avoid
        # infinite loops. Used by the benchmark to wire task.check().
        pre_finish_check: Optional[Callable[[], Optional[str]]] = None,
        pre_finish_max_retries: int = 2,
        # Per-goal augmentor injection. Called once at the start of each
        # run() (when not continue_session) with the goal text. The string
        # it returns is appended to the system prompt for that run only,
        # AFTER `system_prompt_extra` and BEFORE the tools block. Used by
        # ulcagent to wire AugmentorRouter retrieval into the agent path —
        # surfaces relevant YAML examples (security/, web_research_to_file/,
        # etc.) when the goal matches keyword/embedding gates. Default None
        # is a no-op (no augmentor injection, original Phase 13 behavior).
        # Keyword gating happens inside the callback (in AugmentorRouter),
        # so the Phase 13 "augmentor drags 14B off-canon for plain Python"
        # finding is preserved — empty return on non-matching goals.
        augment_for_goal: Optional[Callable[[str], str]] = None,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.system_prompt_extra = system_prompt_extra
        self.workspace_root = Path(workspace_root) if workspace_root is not None else None
        self.memory = memory
        self.auto_verify_python = auto_verify_python
        self.enable_goal_token_sweep = bool(enable_goal_token_sweep)
        self.require_mutating_action = bool(require_mutating_action)
        self.enable_self_heal = bool(enable_self_heal)
        self.suppress_think = bool(suppress_think)
        self.max_iterations = max(1, int(max_iterations))
        self.max_wall_time = float(max_wall_time)
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.confirm_risky = confirm_risky
        self.confirm_destructive = confirm_destructive
        self.pre_finish_check = pre_finish_check
        self.pre_finish_max_retries = max(0, int(pre_finish_max_retries))
        self.augment_for_goal = augment_for_goal
        self.context_char_budget = int(context_char_budget)
        self.compact_keep_recent = max(2, int(compact_keep_recent))
        self._compactions: int = 0  # how many compaction passes fired this run
        self._emit = on_event or (lambda _e: None)

        self._transcript: list[dict[str, str]] = []
        self._tool_calls: list[ToolCall] = []
        self._tool_results: list[ToolResult] = []
        self._memory_block: str = ""
        # Per-goal augmentor block, populated by self.augment_for_goal at
        # run() start. Empty string when augmentor returned nothing or
        # callback isn't wired. Read by _system_prompt() each iteration.
        self._goal_augmentor_block: str = ""
        # Live diagnose-and-repair: per-iteration failure-class history
        # used by the self_heal injector. Keyed by classify_failure()
        # output (or None on success). See engine/self_heal.py.
        self._failure_streak: list = []
        self._self_heal_fired: int = 0

    # ── prompt assembly ──

    def _system_prompt(self) -> str:
        parts = [_DEFAULT_SYSTEM.strip()]
        if self.system_prompt_extra.strip():
            parts.append(self.system_prompt_extra.strip())
        if self._goal_augmentor_block.strip():
            parts.append(self._goal_augmentor_block.strip())
        if self._memory_block:
            parts.append(self._memory_block)
        tool_block = self.registry.hermes_system_block()
        if tool_block:
            parts.append(tool_block)
        return "\n\n".join(parts)

    def _maybe_compact_transcript(self) -> None:
        """Keep the transcript within the configured context budget.

        Compaction strategy:
        - Never touch the initial user turn (the goal).
        - Never touch the last `compact_keep_recent` turns — the model
          needs recent context to act coherently.
        - For older tool turns (`role == "tool"`), replace the payload
          with a one-line marker noting how many chars were elided. Tool
          results are the bulk of the transcript growth (file reads,
          bash output) — eliding them alone is usually enough.
        - If that's still over budget, also elide older assistant turns'
          bodies (keep only the tool_call lines) — much rarer case.

        Each compaction pass emits a `compacted` event so the bench can
        see it happen. Multiple passes can fire across a run.
        """
        if self.context_char_budget <= 0 or not self._transcript:
            return
        total = sum(len(t.get("content") or "") for t in self._transcript)
        if total <= self.context_char_budget:
            return

        ELIDE_MARK = (
            "[tool output elided — {n} chars dropped by context compaction. "
            "If you need this information, re-invoke the tool.]"
        )
        keep_from = max(1, len(self._transcript) - self.compact_keep_recent)
        compacted_any = False
        elided_chars = 0

        # Pass 1: elide old tool bodies (the usual bulk).
        for idx in range(1, keep_from):  # 0 is the goal; never touch
            turn = self._transcript[idx]
            if turn.get("role") != "tool":
                continue
            body = turn.get("content") or ""
            if len(body) <= 200:
                continue
            marker = ELIDE_MARK.format(n=len(body))
            turn["content"] = marker
            elided_chars += len(body) - len(marker)
            compacted_any = True

        # Pass 2: if still over budget (or pass 1 found nothing to do), also
        # drop prose from older assistant turns, preserving the canonical
        # tool_call blocks so the call/result pairing stays intact.
        total_after = sum(len(t.get("content") or "") for t in self._transcript)
        if total_after > self.context_char_budget:
            for idx in range(1, keep_from):
                turn = self._transcript[idx]
                if turn.get("role") != "assistant":
                    continue
                body = turn.get("content") or ""
                if len(body) <= 200:
                    continue
                tool_calls = re.findall(
                    r"<tool_call>\s*\{.*?\}\s*</tool_call>", body, flags=re.DOTALL,
                )
                new_body = (
                    "\n".join(tool_calls) if tool_calls
                    else f"[earlier assistant message elided — {len(body)} chars]"
                )
                if new_body != body:
                    turn["content"] = new_body
                    elided_chars += len(body) - len(new_body)
                    compacted_any = True

        if not compacted_any:
            return

        self._compactions += 1
        total_final = sum(len(t.get("content") or "") for t in self._transcript)
        self._emit(
            AgentEvent(
                "compacted",
                self._compactions,
                payload={
                    "elided_chars": elided_chars,
                    "total_before": total,
                    "total_after": total_final,
                    "budget": self.context_char_budget,
                },
            )
        )

    def _build_prompt(self) -> str:
        """ChatML prompt = system + transcript so far + open assistant header.

        With suppress_think=True, we pre-open-and-close an empty <think> block
        at the start of the assistant turn. R1-distill-style reasoning models
        are trained to ALWAYS emit a <think>...</think> pair at the start of
        each response; pre-closing it tells the model "reasoning is done,
        emit the answer now" and skips the 2-4k token reasoning tax per turn.
        Known hack from the R1 community — suppresses reasoning for tasks
        where tool format discipline matters more than chain-of-thought."""
        chunks = [f"<|im_start|>system\n{self._system_prompt()}<|im_end|>"]
        for turn in self._transcript:
            chunks.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>")
        assistant_header = "<|im_start|>assistant\n"
        if self.suppress_think:
            assistant_header += "<think>\n\n</think>\n\n"
        chunks.append(assistant_header)
        return "\n".join(chunks)

    @staticmethod
    def _format_tool_responses(results: list[ToolResult]) -> str:
        """Hermes-format tool response blocks, one per executed call."""
        return "\n".join(
            f"<tool_response>\n{r.format_for_model()}\n</tool_response>" for r in results
        )

    @staticmethod
    def _canonical_assistant_turn(raw_response: str, calls: list[ToolCall]) -> str:
        """
        Rebuild the assistant turn for the transcript using ONLY the parsed
        tool calls, serialized in canonical Hermes format. Raw prose and any
        truncated/garbled JSON from the model's actual output are dropped.

        Why: Qwen 2.5 14B frequently emits tool calls in ```json fences with
        long multi-step plans. When max_tokens cuts the output mid-JSON, the
        raw text contains a truncated string that poisons the next turn's
        context — the model sees a broken conversation ("assistant started a
        tool call but didn't finish it, yet here are tool responses") and
        gets confused. By storing only the CLEAN canonical form of what we
        actually executed, the model sees a well-formed history every turn.

        If there are no parsed tool calls, this returns the raw response
        with any <think>...</think> reasoning blocks stripped — the model's
        private reasoning is not part of the final answer and shouldn't
        pollute the transcript for subsequent turns.
        """
        if not calls:
            stripped = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
            if "<think>" in stripped:
                stripped = stripped[: stripped.find("<think>")]
            return stripped.strip()
        parts = []
        for c in calls:
            call_json = json.dumps(
                {"name": c.name, "arguments": c.arguments}, ensure_ascii=False
            )
            parts.append(f"<tool_call>\n{call_json}\n</tool_call>")
        return "\n".join(parts)

    # ── execution ──

    def _maybe_auto_verify(self, call: ToolCall, result: ToolResult) -> Optional[ToolResult]:
        """
        Auto-syntax-check files after write_file/edit_file. Dispatches by
        extension:
            .py              -> compile() via stdlib (always available)
            .json            -> json.loads() via stdlib (always available)
            .js/.ts/.jsx/.tsx -> `node --check` if node is installed
            .rs              -> `rustc --edition=2021 --emit=metadata` if rustc is installed
            .go              -> `go vet` on the file's directory if go is installed

        External compiler checks (node/rustc/go) skip CLEANLY if the binary
        isn't on PATH — Phase 14 privacy rule + "it just works" UX means
        we never fail a write just because the user doesn't have the
        toolchain installed.

        Synthetic result with name='auto_verify' is returned and threaded
        into the same observation block as the original tool result so
        the model sees both.
        """
        if not self.auto_verify_python or not result.success:
            return None
        if call.name not in ("write_file", "edit_file"):
            return None
        path_arg = call.arguments.get("path")
        if not isinstance(path_arg, str):
            return None

        p = Path(path_arg)
        if not p.is_absolute() and self.workspace_root is not None:
            p = (self.workspace_root / p).resolve()

        if not p.exists():
            return None

        ext = p.suffix.lower()
        if ext == ".py":
            return self._verify_python(p)
        if ext == ".json":
            return self._verify_json(p)
        if ext in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
            return self._verify_node(p)
        if ext == ".rs":
            return self._verify_rust(p)
        if ext == ".go":
            return self._verify_go(p)
        if ext in (".sh", ".bash"):
            return self._verify_bash(p)
        if ext in (".yml", ".yaml"):
            return self._verify_yaml(p)
        return None

    @staticmethod
    def _verify_python(p: Path) -> ToolResult:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            compile(text, str(p), "exec")
        except SyntaxError as exc:
            hint = Agent._syntax_error_hint(text, exc, p.name)
            return ToolResult(
                name="auto_verify",
                success=False,
                error=f"SyntaxError in {p.name} line {exc.lineno}: {exc.msg}{hint}",
            )
        except OSError as exc:
            return ToolResult(name="auto_verify", success=False, error=f"verify failed: {exc}")
        # Syntax is fine — also run a static undefined-name check. Catches
        # the "model writes storage.py using Todo without importing it"
        # pattern that pytest would otherwise catch one iteration later.
        undefined = Agent._static_undefined_names(text)
        if undefined:
            names_list = ", ".join(sorted(undefined)[:6])
            more = f" (+{len(undefined) - 6} more)" if len(undefined) > 6 else ""
            hints = Agent._import_hints_for_undefined(undefined)
            # Workspace-aware: for names not in the stdlib hint table,
            # scan sibling .py files for a matching top-level class/def
            # and suggest `from <module> import <Name>`. Catches the
            # build_todo_cli pattern where storage.py uses `Todo` but
            # forgot `from todo import Todo`.
            if not hints:
                ws_hints = Agent._workspace_import_hints_for_undefined(undefined, p)
                if ws_hints:
                    hints = ws_hints
            hint_block = ""
            if hints:
                hint_block = (
                    f" Add this line at the top of {p.name}: `{hints[0]}`. "
                    f"This is ONE step — continue with the rest of the "
                    f"task after fixing the import."
                )
            return ToolResult(
                name="auto_verify",
                success=False,
                error=(
                    f"syntax OK but {p.name} references undefined names: "
                    f"{names_list}{more}. You MUST add the missing imports "
                    f"to {p.name} ITSELF, not to __init__.py or any other "
                    f"file — Python imports are per-file and are NOT shared "
                    f"across modules.{hint_block}"
                ),
            )
        return ToolResult(name="auto_verify", success=True, content=f"syntax OK ({p.name})")

    @staticmethod
    def _syntax_error_hint(source: str, exc: SyntaxError, fname: str) -> str:
        """Return a surgical fix hint for the most common SyntaxError patterns
        the 14B emits. Returns an empty string when no specific hint applies,
        so the caller can just append it verbatim to the base error message."""
        msg = (exc.msg or "").lower()
        lineno = exc.lineno or 0
        lines = source.splitlines()
        if 0 < lineno <= len(lines):
            broken_line = lines[lineno - 1]
        else:
            broken_line = ""
        # Pattern 1: f-string nested single quotes (Python 3.10/3.11 only).
        # `print(f'{x if y else 'default'}')` → the inner `'` terminates the
        # outer `'`, SyntaxError. Fix: swap the outer quote style and use
        # SINGLE quotes for the inner literals. Full programmatic rewrite
        # is impossible in general (regex can't tell which `'` closes the
        # f-string — that's the very bug), so we give a detailed fix rule
        # and the broken line's text so the model can correctly rewrite it.
        if "f-string" in msg:
            if "f'" in broken_line or "F'" in broken_line:
                return (
                    f". NESTED-QUOTE TRAP (Python 3.10/3.11): line {lineno} "
                    f"has a single-quoted f-string that contains `'` inside "
                    f"its `{{ }}` expression. Python 3.10 can't parse nested "
                    f"single quotes. Broken line is: `{broken_line.strip()}` "
                    f"— rewrite it as a DOUBLE-quoted f-string with the "
                    f"inner literals still in SINGLE quotes. Pattern: "
                    f"`print(f'{{x if y else 'X'}}')` → "
                    f"`print(f\"{{x if y else 'X'}}\")` (outer `'` swapped "
                    f"to `\"`; inner `'X'` kept as is). Emit a single "
                    f"edit_file call whose old_string is the broken line "
                    f"and new_string is the rewritten line."
                )
            return (
                f". Line {lineno}: `{broken_line.strip()}`. The most "
                f"common cause is nested quotes inside an f-string — if "
                f"the f-string uses `'`, switch it to `\"` (and vice versa)."
            )
        # Pattern 2: decorator-followed-by-non-def. `@dataclass\nfrom X import Y`
        # is a SyntaxError because a decorator must precede a def/class.
        if "invalid syntax" in msg and lineno >= 2 and len(lines) >= lineno:
            prior = lines[lineno - 2]
            if prior.lstrip().startswith("@") and not broken_line.lstrip().startswith(("def ", "class ", "async def ")):
                return (
                    f". DECORATOR ORPHAN: line {lineno - 1} is a decorator "
                    f"`{prior.strip()}` but line {lineno} is `{broken_line.strip()}` "
                    f"which is not a def/class. A decorator must be immediately "
                    f"followed by the def/class it decorates. Move the decorator "
                    f"down so it sits directly above the class/def, OR remove "
                    f"the line between them."
                )
        return ""

    @staticmethod
    def _swap_fstring_quotes(line: str) -> str:
        """Attempt to convert `f'...'` to `f\"...\"` on a single line,
        preserving content. Only swaps if the line contains exactly one
        `f'...'` or `F'...'` f-string. Returns empty string if ambiguous."""
        import re as _re
        # Find f'...' or F'...' spans. Be conservative: only one per line.
        pattern = _re.compile(r"""([fF])(')((?:[^'\\]|\\.)*)(')""")
        matches = list(pattern.finditer(line))
        if len(matches) != 1:
            return ""
        m = matches[0]
        inner = m.group(3)
        # The inner body uses `'` for literals — swap those to nothing special.
        # We need the inner to NOT contain unescaped `"` (else double-quoting
        # would break it). If it does, give up.
        if '"' in inner:
            return ""
        # Build f"..." with the same inner body. Inner `'` characters become
        # literal (no escape needed inside double quotes).
        rebuilt = line[: m.start()] + f'{m.group(1)}"{inner}"' + line[m.end() :]
        return rebuilt

    # Well-known stdlib names -> recommended import line. Order matters for
    # ambiguity: pick the single most likely module. Extend as failure modes
    # surface in benchmarks.
    _STDLIB_IMPORT_HINTS: dict[str, list[str]] = {
        "dataclass": ["from dataclasses import dataclass"],
        "field": ["from dataclasses import field"],
        "asdict": ["from dataclasses import asdict"],
        "astuple": ["from dataclasses import astuple"],
        "fields": ["from dataclasses import fields"],
        "Optional": ["from typing import Optional"],
        "List": ["from typing import List"],
        "Dict": ["from typing import Dict"],
        "Tuple": ["from typing import Tuple"],
        "Any": ["from typing import Any"],
        "Union": ["from typing import Union"],
        "Callable": ["from typing import Callable"],
        "Iterable": ["from typing import Iterable"],
        "Iterator": ["from typing import Iterator"],
        "Generator": ["from typing import Generator"],
        "Sequence": ["from typing import Sequence"],
        "Mapping": ["from typing import Mapping"],
        "Path": ["from pathlib import Path"],
        "defaultdict": ["from collections import defaultdict"],
        "deque": ["from collections import deque"],
        "Counter": ["from collections import Counter"],
        "OrderedDict": ["from collections import OrderedDict"],
        "namedtuple": ["from collections import namedtuple"],
        "datetime": ["from datetime import datetime", "import datetime"],
        "timedelta": ["from datetime import timedelta"],
        "date": ["from datetime import date"],
        "time": ["from datetime import time", "import time"],
        "json": ["import json"],
        "os": ["import os"],
        "sys": ["import sys"],
        "re": ["import re"],
        "math": ["import math"],
        "random": ["import random"],
        "argparse": ["import argparse"],
        "logging": ["import logging"],
        "subprocess": ["import subprocess"],
        "shutil": ["import shutil"],
        "tempfile": ["import tempfile"],
        "contextmanager": ["from contextlib import contextmanager"],
        "suppress": ["from contextlib import suppress"],
        "wraps": ["from functools import wraps"],
        "partial": ["from functools import partial"],
        "reduce": ["from functools import reduce"],
        "cache": ["from functools import cache"],
        "lru_cache": ["from functools import lru_cache"],
        "ABC": ["from abc import ABC"],
        "abstractmethod": ["from abc import abstractmethod"],
        "Enum": ["from enum import Enum"],
        "auto": ["from enum import auto"],
        "pytest": ["import pytest"],
        "mock": ["from unittest import mock"],
        "MagicMock": ["from unittest.mock import MagicMock"],
        "patch": ["from unittest.mock import patch"],
    }

    @staticmethod
    def _import_hints_for_undefined(undefined: set[str]) -> list[str]:
        """Return a list of suggested `import ...` lines for each undefined
        name we recognize, in a stable order. Empty list means no hint."""
        hints: list[str] = []
        for name in sorted(undefined):
            suggestion = Agent._STDLIB_IMPORT_HINTS.get(name)
            if suggestion:
                hints.extend(suggestion)
        return hints

    @staticmethod
    def _workspace_import_hints_for_undefined(
        undefined: set[str], broken_file: Path
    ) -> list[str]:
        """Scan sibling .py files in the workspace for top-level class/def
        names that match the undefined references. Returns suggested
        `from <module> import <Name>` lines. Catches project-local
        imports the model forgot — specifically the build_todo_cli
        pattern where `storage.py` uses `Todo` but didn't import it
        from `todo.py`."""
        import ast as _ast
        import re as _re
        try:
            ws_dir = broken_file.parent.resolve()
        except OSError:
            return []
        if not ws_dir.is_dir():
            return []
        hints: list[str] = []
        try:
            siblings = [
                p for p in ws_dir.iterdir()
                if p.is_file() and p.suffix == ".py" and p.resolve() != broken_file.resolve()
            ]
        except OSError:
            return []
        # Map: undefined_name -> list of candidate sibling modules that
        # define it at top level. Pick the first match deterministically.
        candidates: dict[str, list[str]] = {}
        for sib in siblings:
            try:
                src = sib.read_text(encoding="utf-8", errors="replace")
                tree = _ast.parse(src)
            except (OSError, SyntaxError):
                continue
            for node in tree.body:
                if isinstance(
                    node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)
                ):
                    if node.name in undefined:
                        candidates.setdefault(node.name, []).append(sib.stem)
                elif isinstance(node, _ast.Assign):
                    for target in node.targets:
                        if isinstance(target, _ast.Name) and target.id in undefined:
                            candidates.setdefault(target.id, []).append(sib.stem)
        for name in sorted(candidates.keys()):
            module = candidates[name][0]
            # Avoid generating `from <broken_stem> import ...` — can't self-import
            if module == broken_file.stem:
                continue
            # Skip module names that aren't valid Python identifiers
            if not _re.match(r"^[A-Za-z_][A-Za-z_0-9]*$", module):
                continue
            hints.append(f"from {module} import {name}")
        return hints

    @staticmethod
    def _static_undefined_names(source: str) -> set[str]:
        """Find Name references used in function bodies that are NOT:
        - imported at module level
        - defined at module level (class/function/assignment)
        - builtin
        - a parameter / local in the function scope

        Returns a set of undefined name strings. Empty set = clean. This is
        a minimal pyflakes-equivalent — catches the most common 14B failure
        mode (uses a cross-file type without importing it) without adding a
        real dependency. Conservative: false-negatives are better than
        false-positives so we don't annoy the model with spurious errors.
        """
        import ast
        import builtins

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()

        # Step 1: collect all names defined at module level
        module_defined: set[str] = set()

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    # `import foo as bar` -> bar, `import foo` -> foo
                    # `from x import y` -> y, `from x import y as z` -> z
                    # `from x import *` -> can't track, be permissive
                    if alias.name == "*":
                        # star import: assume everything is imported
                        return set()
                    module_defined.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                module_defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_defined.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                module_defined.add(elt.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    module_defined.add(node.target.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    module_defined.add(node.target.id)
            elif isinstance(node, ast.If):
                # Conditional imports / definitions in if-blocks — be permissive
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        for alias in sub.names:
                            if alias.name == "*":
                                return set()
                            module_defined.add(alias.asname or alias.name.split(".")[0])
                    elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        module_defined.add(sub.name)
                    elif isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            if isinstance(target, ast.Name):
                                module_defined.add(target.id)
            elif isinstance(node, ast.Try):
                # try/except imports too
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        for alias in sub.names:
                            if alias.name == "*":
                                return set()
                            module_defined.add(alias.asname or alias.name.split(".")[0])

        # Always available
        module_defined |= set(dir(builtins))
        module_defined |= {"__name__", "__file__", "__doc__", "__package__", "__loader__", "__spec__", "__builtins__"}

        # Step 2: walk every function body, collect Name references that
        # aren't defined in the function's local scope. Flag if they're
        # also not in module_defined.
        undefined: set[str] = set()

        def walk_function(fn_node: ast.AST, outer_locals: set[str]) -> None:
            locals_here: set[str] = set(outer_locals)
            # Function parameters
            args_node = getattr(fn_node, "args", None)
            if args_node is not None:
                for arg in list(args_node.args) + list(args_node.kwonlyargs) + list(args_node.posonlyargs or []):
                    locals_here.add(arg.arg)
                if args_node.vararg is not None:
                    locals_here.add(args_node.vararg.arg)
                if args_node.kwarg is not None:
                    locals_here.add(args_node.kwarg.arg)
            # Walk body, collecting assignments and references
            for sub in ast.walk(fn_node):
                if sub is fn_node:
                    continue
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        for n in _collect_name_targets(target):
                            locals_here.add(n)
                elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                    if isinstance(sub.target, ast.Name):
                        locals_here.add(sub.target.id)
                elif isinstance(sub, ast.For):
                    for n in _collect_name_targets(sub.target):
                        locals_here.add(n)
                elif isinstance(sub, ast.With):
                    for item in sub.items:
                        if item.optional_vars is not None:
                            for n in _collect_name_targets(item.optional_vars):
                                locals_here.add(n)
                elif isinstance(sub, ast.ExceptHandler):
                    if sub.name:
                        locals_here.add(sub.name)
                elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    for gen in sub.generators:
                        for n in _collect_name_targets(gen.target):
                            locals_here.add(n)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    # Lazy imports inside function body — common pattern for
                    # breaking circular imports. Add the imported names to
                    # the function's local scope.
                    for alias in sub.names:
                        if alias.name == "*":
                            # Permissive on star imports
                            locals_here.add("__STAR_IMPORT__")
                            continue
                        locals_here.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                    for name in sub.names:
                        locals_here.add(name)
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # nested function: bind its name locally
                    if hasattr(sub, "name"):
                        locals_here.add(sub.name)
                elif isinstance(sub, ast.Lambda):
                    # Lambda parameters are local to the lambda body. We do a
                    # flat walk (not scope-aware), so the safe approximation is
                    # to add the lambda's params to the enclosing function's
                    # locals — slightly over-permissive (a name shadowed only
                    # inside the lambda won't be flagged elsewhere) but it kills
                    # the false positive on `key=lambda x: x['foo']`, which
                    # otherwise flags `x` as undefined and derails the model
                    # into "fixing" a non-bug. False negatives over false
                    # positives, per this checker's design.
                    largs = getattr(sub, "args", None)
                    if largs is not None:
                        for a in (list(largs.args) + list(largs.kwonlyargs)
                                  + list(largs.posonlyargs or [])):
                            locals_here.add(a.arg)
                        if largs.vararg is not None:
                            locals_here.add(largs.vararg.arg)
                        if largs.kwarg is not None:
                            locals_here.add(largs.kwarg.arg)
            # If function body has any star import, skip undefined-name
            # reporting for this function — we can't know what names exist.
            if "__STAR_IMPORT__" in locals_here:
                return
            # Skip functions containing match statements — case patterns
            # bind names in complex ways (MatchMapping, MatchAs, MatchSequence)
            # that would need a full pattern walker. False negatives here are
            # fine; false positives would annoy the model.
            for sub in ast.walk(fn_node):
                if hasattr(ast, "Match") and isinstance(sub, ast.Match):
                    return
            # Second pass: find Name loads not in locals or module scope
            for sub in ast.walk(fn_node):
                if sub is fn_node:
                    continue
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id not in locals_here and sub.id not in module_defined:
                        undefined.add(sub.id)

        def _collect_name_targets(node: ast.AST) -> list[str]:
            names: list[str] = []
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    names.extend(_collect_name_targets(elt))
            elif isinstance(node, ast.Starred):
                names.extend(_collect_name_targets(node.value))
            return names

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk_function(node, set())
            elif isinstance(node, ast.ClassDef):
                # Class body doesn't create a usable scope for methods, but
                # method bodies are handled separately via the walk above.
                pass
            # Decorators on any def/class are evaluated at module load time,
            # so their Name references must exist at module scope. This
            # catches the `@dataclass` with no `from dataclasses import
            # dataclass` pattern observed in Phase 14 iter1m refactor_dataclass
            # failure.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in getattr(node, "decorator_list", []) or []:
                    for sub in ast.walk(dec):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                            if sub.id not in module_defined:
                                undefined.add(sub.id)
                        elif isinstance(sub, ast.Attribute):
                            # `@x.y.z` — only the leftmost name (`x`) must exist
                            cur = sub
                            while isinstance(cur, ast.Attribute):
                                cur = cur.value
                            if isinstance(cur, ast.Name) and cur.id not in module_defined:
                                undefined.add(cur.id)

        return undefined

    @staticmethod
    def _verify_json(p: Path) -> ToolResult:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            json.loads(text)
            return ToolResult(name="auto_verify", success=True, content=f"valid JSON ({p.name})")
        except json.JSONDecodeError as exc:
            return ToolResult(
                name="auto_verify",
                success=False,
                error=f"Invalid JSON in {p.name} line {exc.lineno}: {exc.msg}",
            )
        except OSError as exc:
            return ToolResult(name="auto_verify", success=False, error=f"verify failed: {exc}")

    @staticmethod
    def _verify_node(p: Path) -> Optional[ToolResult]:
        import shutil
        import subprocess
        node = shutil.which("node")
        if node is None:
            return None
        try:
            result = subprocess.run(
                [node, "--check", str(p)],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return ToolResult(
                name="auto_verify", success=False, error=f"node --check error: {exc}"
            )
        if result.returncode == 0:
            return ToolResult(
                name="auto_verify", success=True, content=f"syntax OK ({p.name}, node --check)"
            )
        err = (result.stderr or result.stdout or "").strip()
        # Trim to first few lines for the model
        first_lines = "\n".join(err.splitlines()[:5])
        return ToolResult(
            name="auto_verify",
            success=False,
            error=f"node --check failed on {p.name}:\n{first_lines}",
        )

    @staticmethod
    def _verify_rust(p: Path) -> Optional[ToolResult]:
        import shutil
        import subprocess
        rustc = shutil.which("rustc")
        if rustc is None:
            return None
        # Single-file parse check. `--emit=metadata -o -` lets rustc parse
        # the file without linking or needing a Cargo manifest. We don't
        # capture the emitted output; we only care about exit code + stderr.
        try:
            result = subprocess.run(
                [
                    rustc,
                    "--edition=2021",
                    "--emit=metadata",
                    "-o",
                    "-",
                    str(p),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return ToolResult(
                name="auto_verify", success=False, error=f"rustc error: {exc}"
            )
        if result.returncode == 0:
            return ToolResult(
                name="auto_verify", success=True, content=f"syntax OK ({p.name}, rustc)"
            )
        err = (result.stderr or "").strip()
        first_lines = "\n".join(err.splitlines()[:8])
        return ToolResult(
            name="auto_verify",
            success=False,
            error=f"rustc check failed on {p.name}:\n{first_lines}",
        )

    @staticmethod
    def _verify_go(p: Path) -> Optional[ToolResult]:
        import shutil
        import subprocess
        go = shutil.which("go")
        if go is None:
            return None
        # `go vet` on a single file works best when invoked in the file's
        # directory. We pass the relative filename so go vet can resolve
        # the package context.
        try:
            result = subprocess.run(
                [go, "vet", p.name],
                capture_output=True,
                text=True,
                timeout=20,
                cwd=str(p.parent),
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return ToolResult(
                name="auto_verify", success=False, error=f"go vet error: {exc}"
            )
        if result.returncode == 0:
            return ToolResult(
                name="auto_verify", success=True, content=f"syntax OK ({p.name}, go vet)"
            )
        err = (result.stderr or "").strip()
        first_lines = "\n".join(err.splitlines()[:8])
        return ToolResult(
            name="auto_verify",
            success=False,
            error=f"go vet failed on {p.name}:\n{first_lines}",
        )

    @staticmethod
    def _verify_bash(p: Path) -> Optional[ToolResult]:
        import shutil
        import subprocess
        bash = shutil.which("bash")
        if bash is None:
            return None
        try:
            result = subprocess.run(
                [bash, "-n", str(p)],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return ToolResult(name="auto_verify", success=False, error=f"bash -n error: {exc}")
        if result.returncode == 0:
            return ToolResult(
                name="auto_verify", success=True, content=f"syntax OK ({p.name}, bash -n)"
            )
        err = (result.stderr or result.stdout or "").strip()
        first_lines = "\n".join(err.splitlines()[:5])
        return ToolResult(
            name="auto_verify",
            success=False,
            error=f"bash -n failed on {p.name}:\n{first_lines}",
        )

    @staticmethod
    def _verify_yaml(p: Path) -> Optional[ToolResult]:
        try:
            import yaml  # type: ignore
        except ImportError:
            return None
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            list(yaml.safe_load_all(text))
        except yaml.YAMLError as exc:
            msg = getattr(exc, "problem", None) or str(exc)
            # Auto-fix: the 14B sometimes emits \t for YAML indentation.
            # Tabs are illegal in YAML. Replace with 2 spaces and re-verify.
            if "\\t" in msg or "tab" in msg.lower():
                fixed = text.replace("\t", "  ")
                try:
                    list(yaml.safe_load_all(fixed))
                except yaml.YAMLError:
                    pass  # tab fix didn't help — fall through to error
                else:
                    p.write_text(fixed, encoding="utf-8")
                    return ToolResult(
                        name="auto_verify",
                        success=True,
                        content=f"valid YAML ({p.name}) — auto-fixed tabs to spaces",
                    )
            mark = getattr(exc, "problem_mark", None)
            loc = f" line {mark.line + 1}" if mark is not None else ""
            return ToolResult(
                name="auto_verify",
                success=False,
                error=f"Invalid YAML in {p.name}{loc}: {msg}",
            )
        except OSError as exc:
            return ToolResult(name="auto_verify", success=False, error=f"verify failed: {exc}")
        return ToolResult(name="auto_verify", success=True, content=f"valid YAML ({p.name})")

    def _sweep_goal_required_tokens(self) -> list[tuple[str, str]]:
        """Scan the user goal for literal code tokens (backtick-quoted
        identifiers, @decorators, def/class keywords) and check that each
        one appears in the workspace files touched by this run. Returns
        a list of (token, reason) for tokens that are missing. Empty list
        means every required token is present. Task-agnostic — works on
        any goal that explicitly names code tokens."""
        if self.workspace_root is None or not self._transcript:
            return []
        goal = ""
        for turn in self._transcript:
            if turn.get("role") == "user":
                goal = str(turn.get("content") or "")
                break
        if not goal:
            return []
        try:
            ws = self.workspace_root.resolve()
            if not ws.is_dir():
                return []
        except OSError:
            return []
        tokens = self._extract_code_tokens(goal)
        absent_tokens = self._extract_rename_targets(goal)
        if not tokens and not absent_tokens:
            return []
        # Collect the content of every .py file touched during this run.
        touched_src = []
        for call in self._tool_calls:
            if call.name not in {"write_file", "edit_file"}:
                continue
            args = call.arguments if isinstance(call.arguments, dict) else {}
            raw_path = args.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                full = (self.workspace_root / raw_path).resolve()
                full.relative_to(ws)
            except (OSError, ValueError):
                continue
            if not full.exists() or not full.is_file():
                continue
            try:
                touched_src.append(full.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        if not touched_src:
            return []
        combined = "\n".join(touched_src)
        missing: list[tuple[str, str]] = []
        for token, reason in tokens:
            present = token in combined
            # For CLI flags (--X), tighten the check: the token must appear
            # inside an `add_argument(` call, not just anywhere in the file.
            # Without this, a model that wrote `# pass --verbose to enable`
            # as a comment would satisfy the sweep without actually
            # registering the flag with argparse. Phase 14 add_cli_flag
            # failure mode observed across iter1l → iter2v.
            if present and token.startswith("--"):
                import re as _re
                flag_in_add_arg = _re.compile(
                    r"add_argument\([\s\S]{0,40}?['\"]"
                    + _re.escape(token)
                    + r"['\"]"
                )
                if not flag_in_add_arg.search(combined):
                    present = False
                    reason = (
                        f"{reason} — `{token}` appears somewhere in the "
                        f"workspace but NOT inside an `add_argument(...)` "
                        f"call, so argparse won't recognize it. Register "
                        f"the flag with `parser.add_argument('{token}', "
                        f"action='store_true')` (or the action the goal "
                        f"specifies) BEFORE `parser.parse_args()`"
                    )
            if not present:
                missing.append((token, f"{reason} — `{token}` MUST appear in the relevant file before the task can be marked complete"))
        # Rename targets must be ABSENT. Use a word-boundary-aware check —
        # `do_thing` must match as a whole identifier, not as a substring
        # inside e.g. `my_do_thing_wrapper`.
        import re as _re
        for token, reason in absent_tokens:
            pat = _re.compile(r"\b" + _re.escape(token) + r"\b")
            if pat.search(combined):
                missing.append(
                    (
                        token,
                        f"{reason} — but `{token}` STILL APPEARS in the workspace. "
                        f"Use grep to find every remaining `{token}` and edit_file to "
                        f"rewrite each call site. The rename is only complete when "
                        f"every reference is updated, not just the definition.",
                    )
                )
        return missing

    @staticmethod
    def _extract_code_tokens(goal: str) -> list[tuple[str, str]]:
        """Pull literal code tokens from the goal that a completed task must
        have produced. Returns [(token, reason)]. Conservative — only the
        patterns below, which are always additive (the goal mentions the
        token AND the token must survive in the result):

        - `@decorator` names ("convert to a @dataclass" → @dataclass)
        - `--flag-name` CLI args ("add a --verbose argparse flag" → --verbose)

        Deliberately skips plain backticked identifiers because "rename X to Y"
        mentions BOTH X and Y but X should be absent post-rename — that case
        is handled by `_extract_rename_targets` via a directional pattern."""
        import re as _re
        required: list[tuple[str, str]] = []
        seen: set[str] = set()
        for m in _re.finditer(r"@([A-Za-z_][A-Za-z_0-9]*)", goal):
            tok = f"@{m.group(1)}"
            if tok in seen:
                continue
            seen.add(tok)
            required.append((tok, f"goal mentions decorator {tok}"))
        for m in _re.finditer(r"(--[A-Za-z][A-Za-z0-9\-]*)", goal):
            tok = m.group(1)
            if tok in seen:
                continue
            seen.add(tok)
            required.append((tok, f"goal mentions CLI flag {tok}"))
        return required

    @staticmethod
    def _extract_rename_targets(goal: str) -> list[tuple[str, str]]:
        """Detect rename-pattern goals and return identifiers that must be
        ABSENT from the workspace after the edit. Returns [(token, reason)].
        Matches phrases like:
            rename `do_thing` to `compute`
            rename the function `do_thing` to `compute`
            rename do_thing to compute
        and only emits the FROM identifier as a required-absent token —
        the TO identifier is already handled by positive-presence checks
        when explicitly listed in the goal."""
        import re as _re
        patterns = [
            _re.compile(
                r"rename\s+(?:the\s+(?:function|method|class|variable)\s+)?"
                r"`?([A-Za-z_][A-Za-z_0-9]*)`?\s+to\s+`?([A-Za-z_][A-Za-z_0-9]*)`?",
                _re.IGNORECASE,
            ),
        ]
        absent: list[tuple[str, str]] = []
        seen: set[str] = set()
        for pat in patterns:
            for m in pat.finditer(goal):
                old_name = m.group(1)
                new_name = m.group(2)
                if old_name == new_name or old_name in seen:
                    continue
                # Skip very short names (single letters) to avoid hitting
                # parameter-renames where they don't appear in code scope.
                if len(old_name) < 3:
                    continue
                seen.add(old_name)
                absent.append(
                    (old_name, f"goal says to rename `{old_name}` → `{new_name}`, so `{old_name}` must be gone")
                )
        return absent

    def _sweep_workspace_syntax_errors(self) -> list[tuple[str, str]]:
        """Scan .py files that were written or edited during this run for
        SyntaxErrors. Returns a list of (filename, error_message). Empty list
        means all touched files compile cleanly. Scoped to the current run
        so pre-existing broken files in the workspace don't spuriously
        reject final answers."""
        if self.workspace_root is None:
            return []
        try:
            ws = self.workspace_root.resolve()
            if not ws.is_dir():
                return []
        except OSError:
            return []
        touched: list[Path] = []
        seen_paths: set[str] = set()
        for call in self._tool_calls:
            if call.name not in {"write_file", "edit_file"}:
                continue
            args = call.arguments if isinstance(call.arguments, dict) else {}
            raw_path = args.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                full = (self.workspace_root / raw_path).resolve()
                full.relative_to(ws)
            except (OSError, ValueError):
                continue
            if full.suffix != ".py":
                continue
            key = str(full)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            touched.append(full)
        broken: list[tuple[str, str]] = []
        for full in touched:
            if not full.exists() or not full.is_file():
                continue
            try:
                src = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            try:
                compile(src, str(full), "exec")
            except SyntaxError as exc:
                broken.append((full.name, f"SyntaxError line {exc.lineno}: {exc.msg}"))
                continue
            undefined = Agent._static_undefined_names(src)
            if undefined:
                names_list = ", ".join(sorted(undefined)[:6])
                more = f" (+{len(undefined) - 6} more)" if len(undefined) > 6 else ""
                hints = Agent._import_hints_for_undefined(undefined)
                if not hints:
                    hints = Agent._workspace_import_hints_for_undefined(undefined, full)
                hint_block = ""
                if hints:
                    hint_block = (
                        f" Add this line at the top: `{hints[0]}`. Emit "
                        f"EXACTLY this call: "
                        f'<tool_call>{{"name": "edit_file", "arguments": '
                        f'{{"path": "{full.name}", "old_string": "", '
                        f'"new_string": "{hints[0]}\\n"}}}}</tool_call>'
                    )
                broken.append(
                    (
                        full.name,
                        f"references undefined names: {names_list}{more} "
                        f"(imports must be added to {full.name} itself, not "
                        f"another file).{hint_block}",
                    )
                )
        return broken

    def _auto_strip_init_from_dataclass(self) -> list[str]:
        """When a touched file has `@dataclass` above a class AND still
        has the class's original `def __init__(self, ...)`, remove the
        __init__ method (plus its docstring and body) and replace it
        with type-annotated fields derived from the __init__ params.
        Returns files changed. Narrow heuristic — only fires when the
        existing __init__ body is a sequence of `self.<name> = <name>`
        assignments, which is exactly the refactor_dataclass pattern.
        Does not touch classes without @dataclass or __init__ methods
        that do more than trivial attribute assignment."""
        if self.workspace_root is None:
            return []
        try:
            ws = self.workspace_root.resolve()
            if not ws.is_dir():
                return []
        except OSError:
            return []
        import ast as _ast
        changed: list[str] = []
        touched: list[Path] = []
        seen_paths: set[str] = set()
        for call in self._tool_calls:
            if call.name not in {"write_file", "edit_file"}:
                continue
            args = call.arguments if isinstance(call.arguments, dict) else {}
            raw_path = args.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                full = (self.workspace_root / raw_path).resolve()
                full.relative_to(ws)
            except (OSError, ValueError):
                continue
            if full.suffix != ".py":
                continue
            key = str(full)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            touched.append(full)
        for full in touched:
            if not full.exists() or not full.is_file():
                continue
            try:
                src = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "@dataclass" not in src or "def __init__" not in src:
                continue
            try:
                tree = _ast.parse(src)
            except SyntaxError:
                continue
            # Find a class that has @dataclass decorator AND a def __init__
            # with trivial self.x = x body.
            lines = src.splitlines(keepends=True)
            new_src = None
            for node in tree.body:
                if not isinstance(node, _ast.ClassDef):
                    continue
                has_dataclass = any(
                    (isinstance(d, _ast.Name) and d.id == "dataclass")
                    or (
                        isinstance(d, _ast.Attribute)
                        and isinstance(d.value, _ast.Name)
                        and d.attr == "dataclass"
                    )
                    for d in node.decorator_list
                )
                if not has_dataclass:
                    continue
                init_fn = None
                for item in node.body:
                    if (
                        isinstance(item, _ast.FunctionDef)
                        and item.name == "__init__"
                    ):
                        init_fn = item
                        break
                if init_fn is None:
                    continue
                # Extract (name, annotation-or-None) from init params.
                params = []
                args_node = init_fn.args
                for arg in args_node.args[1:]:  # skip `self`
                    annotation = None
                    if arg.annotation is not None:
                        try:
                            annotation = _ast.unparse(arg.annotation)
                        except Exception:
                            annotation = None
                    params.append((arg.arg, annotation))
                # Verify the body is just self.X = X assignments (safe to remove).
                body_ok = True
                assigned = set()
                for stmt in init_fn.body:
                    if (
                        isinstance(stmt, _ast.Assign)
                        and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], _ast.Attribute)
                        and isinstance(stmt.targets[0].value, _ast.Name)
                        and stmt.targets[0].value.id == "self"
                        and isinstance(stmt.value, _ast.Name)
                        and stmt.targets[0].attr == stmt.value.id
                    ):
                        assigned.add(stmt.targets[0].attr)
                    elif isinstance(stmt, _ast.Pass):
                        pass
                    else:
                        body_ok = False
                        break
                if not body_ok:
                    continue
                # Build the replacement: one type-annotated field per param.
                class_indent = ""
                # Find the indent of the init def line.
                init_line = lines[init_fn.lineno - 1]
                field_indent = init_line[: len(init_line) - len(init_line.lstrip())]
                field_lines = []
                for name, ann in params:
                    if ann is None:
                        # Default to str if no annotation was provided — matches
                        # the most common refactor_dataclass pattern.
                        ann = "str"
                    field_lines.append(f"{field_indent}{name}: {ann}\n")
                # Drop lines from init def's start through end_lineno.
                start_idx = init_fn.lineno - 1
                end_idx = (
                    init_fn.end_lineno - 1
                    if init_fn.end_lineno is not None
                    else start_idx
                )
                # Also consume a trailing blank line after __init__ if present.
                consume_blank = (
                    end_idx + 1 < len(lines) and lines[end_idx + 1].strip() == ""
                )
                tail_end = end_idx + (2 if consume_blank else 1)
                new_lines = lines[:start_idx] + field_lines + lines[tail_end:]
                new_src = "".join(new_lines)
                break
            if new_src is not None:
                try:
                    full.write_text(new_src, encoding="utf-8")
                    changed.append(full.name)
                except OSError:
                    continue
        return changed

    def _auto_apply_decorator(self, decorator: str) -> list[str]:
        """Inject `@decorator` above the first class definition in every
        touched .py file that references the class (but doesn't already
        have the decorator above it). Returns files changed. Fires after
        the @decorator goal-token sweep has surfaced the same decorator
        2+ times — last-resort fixer for `refactor_dataclass` where the
        model adds the `from dataclasses import dataclass` but forgets
        to actually apply `@dataclass` to `class Person:`."""
        if self.workspace_root is None:
            return []
        try:
            ws = self.workspace_root.resolve()
            if not ws.is_dir():
                return []
        except OSError:
            return []
        import re as _re
        dec_name = decorator.lstrip("@")
        decorator_line = f"@{dec_name}"
        # Find any top-level `class Foo:` or `class Foo(Bar):` line. Also
        # require that the decorator isn't already on the line directly
        # above it.
        class_pat = _re.compile(r"^class\s+[A-Za-z_][A-Za-z_0-9]*", _re.MULTILINE)
        changed: list[str] = []
        touched: list[Path] = []
        seen_paths: set[str] = set()
        for call in self._tool_calls:
            if call.name not in {"write_file", "edit_file"}:
                continue
            args = call.arguments if isinstance(call.arguments, dict) else {}
            raw_path = args.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                full = (self.workspace_root / raw_path).resolve()
                full.relative_to(ws)
            except (OSError, ValueError):
                continue
            if full.suffix != ".py":
                continue
            key = str(full)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            touched.append(full)
        for full in touched:
            if not full.exists() or not full.is_file():
                continue
            try:
                src = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Skip if decorator already applied anywhere in this file
            if decorator_line in src:
                continue
            m = class_pat.search(src)
            if not m:
                continue
            # Look at the preceding non-empty line — if it's already a
            # decorator we're still adding ours above it (multi-decorator
            # stacks are valid); but if it IS our decorator, skip.
            before = src[: m.start()]
            # Insert `@decorator\n` directly above the class.
            new_src = before + decorator_line + "\n" + src[m.start() :]
            try:
                full.write_text(new_src, encoding="utf-8")
                changed.append(full.name)
            except OSError:
                continue
        return changed

    def _auto_apply_argparse_flag(self, flag: str, action: str = "store_true") -> list[str]:
        """Inject `parser.add_argument(<flag>, action=<action>)` into every
        touched .py file that has `parser.parse_args()` but no existing
        registration of `<flag>`. Returns the list of files changed.
        Fires after the flag-absent goal-token sweep has surfaced the
        same flag 2+ times — last-resort fixer for the "model wrote
        `if args.verbose` but forgot `parser.add_argument('--verbose')`"
        pattern that otherwise fails `add_cli_flag` every run."""
        if self.workspace_root is None:
            return []
        try:
            ws = self.workspace_root.resolve()
            if not ws.is_dir():
                return []
        except OSError:
            return []
        import re as _re
        flag_pat = _re.compile(
            r"add_argument\([\s\S]{0,40}?['\"]" + _re.escape(flag) + r"['\"]"
        )
        # Match `parser.parse_args()` with optional `args = ` prefix, and
        # capture the indentation so the inserted line matches.
        parse_args_pat = _re.compile(
            r"^(\s*)(?:args\s*=\s*)?parser\.parse_args\(\)", _re.MULTILINE
        )
        changed: list[str] = []
        touched: list[Path] = []
        seen_paths: set[str] = set()
        for call in self._tool_calls:
            if call.name not in {"write_file", "edit_file"}:
                continue
            args = call.arguments if isinstance(call.arguments, dict) else {}
            raw_path = args.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                full = (self.workspace_root / raw_path).resolve()
                full.relative_to(ws)
            except (OSError, ValueError):
                continue
            if full.suffix != ".py":
                continue
            key = str(full)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            touched.append(full)
        for full in touched:
            if not full.exists() or not full.is_file():
                continue
            try:
                src = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if flag_pat.search(src):
                # Already registered, skip
                continue
            m = parse_args_pat.search(src)
            if not m:
                continue
            indent = m.group(1)
            new_line = f"{indent}parser.add_argument('{flag}', action='{action}')\n"
            new_src = src[: m.start()] + new_line + src[m.start() :]
            try:
                full.write_text(new_src, encoding="utf-8")
                changed.append(full.name)
            except OSError:
                continue
        return changed

    def _auto_apply_rename_workspace(self, old_name: str, new_name: str) -> list[str]:
        """Rewrite every word-boundary occurrence of `old_name` to `new_name`
        across .py files the model touched this run. Returns a list of
        filenames that changed. Empty list means the rename was a no-op.
        Fires after the rename-absent sweep has surfaced the same (old,
        new) pair 2+ times — the last-resort fixer for when the model
        renames the definition but misses a call site."""
        if self.workspace_root is None:
            return []
        try:
            ws = self.workspace_root.resolve()
            if not ws.is_dir():
                return []
        except OSError:
            return []
        # Collect every .py file touched during this run.
        import re as _re
        touched: list[Path] = []
        seen_paths: set[str] = set()
        for call in self._tool_calls:
            if call.name not in {"write_file", "edit_file"}:
                continue
            args = call.arguments if isinstance(call.arguments, dict) else {}
            raw_path = args.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                full = (self.workspace_root / raw_path).resolve()
                full.relative_to(ws)
            except (OSError, ValueError):
                continue
            if full.suffix != ".py":
                continue
            key = str(full)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            touched.append(full)
        pattern = _re.compile(r"\b" + _re.escape(old_name) + r"\b")
        changed: list[str] = []
        for full in touched:
            if not full.exists() or not full.is_file():
                continue
            try:
                src = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            new_src = pattern.sub(new_name, src)
            if new_src != src:
                try:
                    full.write_text(new_src, encoding="utf-8")
                    changed.append(full.name)
                except OSError:
                    continue
        return changed

    def _auto_apply_import_prepend(self, filename: str, import_line: str) -> bool:
        """Directly prepend `import_line` to the workspace file `filename`.
        Returns True on success, False on any failure. Only fires after the
        sweep has surfaced the same hint 2+ times — last-resort action when
        the model repeatedly ignores a concrete fix suggestion.

        Cycle-aware: if adding `from X import Y` at the top would create a
        circular import (the target module X itself imports from `filename`),
        do a LAZY import inside the function that uses Y instead. Fix for
        Phase 14 fix_import_cycle: a.py ↔ b.py mutual imports."""
        if self.workspace_root is None:
            return False
        try:
            full = (self.workspace_root / filename).resolve()
            ws = self.workspace_root.resolve()
            try:
                full.relative_to(ws)
            except ValueError:
                return False
            if not full.exists() or not full.is_file():
                return False
            existing = full.read_text(encoding="utf-8", errors="replace")
            if import_line in existing:
                return True
            # Parse the import to extract "from <module> import <name>"
            import re as _re
            m = _re.match(
                r"from\s+([A-Za-z_][A-Za-z_0-9]*)\s+import\s+([A-Za-z_][A-Za-z_0-9]*)",
                import_line.strip(),
            )
            if m:
                other_module = m.group(1)
                imported_name = m.group(2)
                other_file = (ws / f"{other_module}.py")
                if other_file.exists() and other_file.is_file():
                    try:
                        other_src = other_file.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        other_src = ""
                    this_module = full.stem
                    # Does the other module import from this one? Check
                    # both `from <this> import X` (top-level) and
                    # `import <this>`. Either pattern means a cycle.
                    cycle_pat = _re.compile(
                        r"^\s*(?:from\s+" + _re.escape(this_module) + r"\s+import\s|import\s+" + _re.escape(this_module) + r"\b)",
                        _re.MULTILINE,
                    )
                    if cycle_pat.search(other_src):
                        if self._lazy_import_inside_user_function(
                            full, existing, import_line.strip(), imported_name
                        ):
                            return True
            # Default path: prepend at top (no cycle detected).
            new_src = import_line.rstrip("\n") + "\n" + existing
            full.write_text(new_src, encoding="utf-8")
            return True
        except OSError:
            return False

    @staticmethod
    def _lazy_import_inside_user_function(
        full: Path, existing: str, import_line: str, imported_name: str
    ) -> bool:
        """Find a function in `existing` whose body references `imported_name`
        and insert `import_line` as the first statement inside that function.
        Matches by word-boundary. Returns True on success."""
        import re as _re
        # Match top-level `def NAME(...):` followed by body. Grab the header
        # line and its trailing indent so we can insert the lazy import at
        # the right depth.
        def_pat = _re.compile(
            r"^(def\s+([A-Za-z_][A-Za-z_0-9]*)\s*\([^)]*\)\s*(?:->[^:]+)?:\s*\n)"
            r"(?P<indent>[ \t]+)",
            _re.MULTILINE,
        )
        ref_pat = _re.compile(r"\b" + _re.escape(imported_name) + r"\b")
        for m in def_pat.finditer(existing):
            header_end = m.end(1)
            indent = m.group("indent")
            # Find the function body's end (next top-level def/class or EOF)
            remainder = existing[header_end:]
            end_of_body_match = _re.search(
                r"^(?:def |class |async def )", remainder, _re.MULTILINE
            )
            body = (
                remainder[: end_of_body_match.start()]
                if end_of_body_match
                else remainder
            )
            if ref_pat.search(body):
                # Insert the lazy import at the function body's first line.
                new_src = (
                    existing[:header_end]
                    + f"{indent}{import_line}\n"
                    + existing[header_end:]
                )
                try:
                    full.write_text(new_src, encoding="utf-8")
                    return True
                except OSError:
                    return False
        return False

    def _safe_file_snapshot(self, path: Any, max_lines: int = 40, max_chars: int = 2000) -> str:
        """Read the workspace file at `path` and return its raw contents (no
        line numbers, no trimming), capped to `max_lines` / `max_chars`.
        Returns empty string on any failure. Used by stuck_repeat to give
        the model fresh file context after it's exhausted its retries."""
        if not isinstance(path, str) or not path:
            return ""
        if self.workspace_root is None:
            return ""
        try:
            full = (self.workspace_root / path).resolve()
            ws_root = self.workspace_root.resolve()
            try:
                full.relative_to(ws_root)
            except ValueError:
                return ""
            if not full.exists() or not full.is_file():
                return ""
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines truncated)"]
        snapshot = "\n".join(lines)
        if len(snapshot) > max_chars:
            snapshot = snapshot[:max_chars] + "... (truncated)"
        return snapshot

    def _is_stuck_repeat(self, call: ToolCall) -> bool:
        """
        True if this call would be the 3rd consecutive identical call.
        Triggers when the model emits write_file(calculator.py, <broken>)
        three times in a row without changing approach — observed in the
        Phase 14 bench1 run where extend_calculator looped on an unterminated
        triple-quoted string literal.
        """
        if len(self._tool_calls) < 2:
            return False
        last = self._tool_calls[-1]
        prev = self._tool_calls[-2]
        if last.name != call.name or prev.name != call.name:
            return False
        if last.arguments != call.arguments or prev.arguments != call.arguments:
            return False
        return True

    def _execute_call(self, call: ToolCall) -> ToolResult:
        schema = self.registry.get(call.name)
        if schema is not None and schema.risky and self.confirm_risky is not None:
            try:
                approved = bool(self.confirm_risky(call))
            except Exception as exc:
                logger.exception("confirm_risky callback raised; treating as denied")
                approved = False
            if not approved:
                return ToolResult(
                    name=call.name,
                    success=False,
                    error="User denied execution of risky tool",
                )

        # Destructive-shell-command gate. Runs AFTER confirm_risky so the
        # standard y/N still fires first, but BEFORE registry.execute so a
        # matched destructive command never reaches subprocess.run. Applies
        # only to run_bash (the only tool dispatching arbitrary shell
        # strings). Unlike confirm_risky, this gate cannot be bypassed by
        # an auto-approve flag — destructive prompts are mandatory.
        if call.name == "run_bash":
            from engine.destructive_command_gate import check_command
            command_arg = call.arguments.get("command", "")
            destructive_matches = check_command(command_arg)
            if destructive_matches:
                if self.confirm_destructive is None:
                    matched_ids = ", ".join(m.pattern_id for m in destructive_matches)
                    return ToolResult(
                        name=call.name,
                        success=False,
                        error=(
                            "Destructive shell command refused (no confirm_destructive "
                            f"callback wired). Matched patterns: {matched_ids}. "
                            "Rewrite as a non-destructive operation (e.g. move to a "
                            ".trash/ directory instead of rm -rf)."
                        ),
                    )
                try:
                    approved = bool(self.confirm_destructive(call, destructive_matches))
                except Exception:
                    logger.exception(
                        "confirm_destructive callback raised; treating as denied"
                    )
                    approved = False
                if not approved:
                    matched_ids = ", ".join(m.pattern_id for m in destructive_matches)
                    return ToolResult(
                        name=call.name,
                        success=False,
                        error=(
                            f"User denied destructive command (matched: {matched_ids})"
                        ),
                    )
        return self.registry.execute(call)

    def run(self, goal: str, continue_session: bool = False) -> AgentResult:
        if continue_session and self._transcript:
            # Append to existing conversation — session memory
            self._transcript.append({"role": "user", "content": goal})
        else:
            self._transcript = [{"role": "user", "content": goal}]
            self._tool_calls = []
            self._tool_results = []
            self._memory_block = ""
            self._failure_streak = []
            self._self_heal_fired = 0
            # Per-goal augmentor injection: ask the optional callback for
            # a system-prompt block tailored to this goal. Empty string =
            # no injection (e.g. plain Python codegen with no keyword
            # match — preserves the Phase 13 baseline). Errors in the
            # callback never block the run.
            self._goal_augmentor_block = ""
            if self.augment_for_goal is not None:
                try:
                    block = self.augment_for_goal(goal) or ""
                    if block.strip():
                        self._goal_augmentor_block = block.strip()
                except Exception as exc:
                    logger.warning(f"augment_for_goal callback raised: {exc!r}")

        if not continue_session and self.memory is not None:
            notes = self.memory.load()
            if notes:
                self._memory_block = (
                    "# Notes from previous sessions on this project\n"
                    "(Use the remember tool to add a note for next time.)\n"
                    + notes
                )

        start = time.monotonic()
        stop_reason = "max_iterations"
        final_answer = ""
        iteration = 0
        last_swept_response = ""
        sweep_repeat_count = 0
        pre_finish_retries = 0
        # Track how many times the pre-finish sweep has surfaced the same
        # (file, undefined_name) pair. After 2 surfacings the model has
        # clearly failed to heed the hint — we auto-apply the suggested
        # import by prepending it to the file ourselves. Breaks the
        # "sweep fires N times, model ignores, hit max_iterations" loop
        # observed in Phase 14 iter2n build_todo_cli (14 identical sweep
        # errors for missing `import json` in cli.py).
        sweep_miss_counter: dict[tuple[str, str], int] = {}

        for iteration in range(1, self.max_iterations + 1):
            elapsed = time.monotonic() - start
            if elapsed > self.max_wall_time:
                stop_reason = "wall_time"
                iteration -= 1  # we never actually started this iteration
                break

            self._emit(AgentEvent("iteration", iteration))

            # Compact the transcript before building the next prompt.
            # Fires only when the rendered size exceeds context_char_budget;
            # cheap to call every iteration because the common case is a
            # single char-count summation and an early return.
            self._maybe_compact_transcript()
            prompt = self._build_prompt()
            try:
                # Stop at turn boundaries AND at any attempt by the model to
                # emit runtime-owned markers (`<tool_response>`, new role
                # headers). Prevents the "hallucinate the rest of the trace"
                # failure mode observed on Qwen 2.5 14B in Phase 14 testing.
                #
                # Agent mode uses a slightly higher temperature than benchmark
                # mode (0.3 vs 0.1) plus repeat_penalty=1.1. At temp 0.1 the
                # 14B fell into a "defaultdict\n defaultdict\n ..." degenerate
                # loop on the TODO CLI stress test — the penalty + variety
                # broke that out without hurting code quality noticeably.
                gen_kwargs = {
                    "max_tokens": self.max_tokens_per_turn,
                    "stop": ["<|im_end|>", "<|im_start|>", "<tool_response>"],
                }
                if self.temperature is not None:
                    gen_kwargs["temperature"] = self.temperature
                if self.repeat_penalty is not None:
                    gen_kwargs["repeat_penalty"] = self.repeat_penalty
                response = self.model.generate(prompt, **gen_kwargs)
            except Exception as exc:
                logger.exception("Model generation failed during agent loop")
                stop_reason = "model_error"
                final_answer = f"[model error: {exc}]"
                break

            response = (response or "").strip()
            self._emit(AgentEvent("model_text", iteration, response))

            calls, parse_errors = self.registry.parse_with_errors(response)

            # Mid-thought truncation detector for R1-distill-style reasoning
            # models: if the response contains `<think>` with no matching
            # `</think>`, the model's max_tokens budget was consumed entirely
            # by the reasoning block and the tool call never landed. Treating
            # this as a final answer wastes the task — iterate once more
            # with a synthetic "you were cut off, wrap up your reasoning
            # briefly and emit the tool call now" observation.
            if (
                not calls
                and not parse_errors
                and iteration < self.max_iterations
                and "<think>" in response
                and response.rfind("<think>") > response.rfind("</think>")
            ):
                self._transcript.append(
                    {
                        "role": "assistant",
                        "content": self._canonical_assistant_turn(response, []),
                    }
                )
                truncation_result = ToolResult(
                    name="truncated_reasoning",
                    success=False,
                    error=(
                        "Your previous turn ran out of tokens mid-reasoning "
                        "(<think> block was not closed). Skip the reasoning "
                        "this turn — immediately emit the single tool call "
                        "that performs the next concrete step, wrapped in "
                        "`<tool_call>...</tool_call>` tags or as bare JSON. "
                        "Do not write another <think> block."
                    ),
                )
                self._tool_results.append(truncation_result)
                self._emit(AgentEvent("tool_result", iteration, truncation_result))
                self._transcript.append(
                    {"role": "tool", "content": self._format_tool_responses([truncation_result])}
                )
                continue

            if not calls and not parse_errors:
                # Mutation gate: in architect sub-agent mode (or any caller
                # that sets require_mutating_action=True), reject a final
                # answer that hasn't produced a single write_file / edit_file
                # / run_tests / run_bash call. The model sometimes reads the
                # file and declares "task done" without writing — this is
                # especially common when the sub-step goal mentions code in
                # a fenced block and the model mistakes the plan text for
                # the actual file contents.
                _MUTATING = {"write_file", "edit_file", "run_tests", "run_bash"}
                has_mutation = any(c.name in _MUTATING for c in self._tool_calls)
                if (
                    self.require_mutating_action
                    and not has_mutation
                    and iteration < self.max_iterations
                ):
                    self._transcript.append(
                        {
                            "role": "assistant",
                            "content": self._canonical_assistant_turn(response, []),
                        }
                    )
                    gate_result = ToolResult(
                        name="mutation_gate",
                        success=False,
                        error=(
                            "You have not yet emitted any mutating tool "
                            "call (write_file, edit_file, run_tests, "
                            "run_bash). Reading files alone does NOT "
                            "complete a step that requires a change. "
                            "Emit the appropriate write_file or edit_file "
                            "call to actually make the change, then give "
                            "the final answer."
                        ),
                    )
                    self._tool_results.append(gate_result)
                    self._emit(AgentEvent("tool_result", iteration, gate_result))
                    self._transcript.append(
                        {"role": "tool", "content": self._format_tool_responses([gate_result])}
                    )
                    continue

                # Final-answer turn candidate: before accepting it, sweep the
                # workspace for .py files with lingering SyntaxErrors. Phase
                # 14 iter1m build_todo_cli failure: the model wrote cli.py
                # with an f-string nested-quote bug in iter 1, got distracted
                # fixing storage.py in iters 2–4, then said "tool" and gave
                # up with cli.py still broken. If we catch that here, we can
                # inject a synthetic observation naming the broken file and
                # keep the loop running for one more turn.
                broken = (
                    self._sweep_workspace_syntax_errors()
                    if self.auto_verify_python
                    else []
                )
                # Also check for goal-required tokens that didn't land in
                # the workspace (task-agnostic completeness check). Skipped
                # under architect mode where each sub-agent is only tasked
                # with ONE step of the original goal — checking the full
                # goal's tokens at sub-step granularity always fails.
                if self.auto_verify_python and self.enable_goal_token_sweep:
                    missing_tokens = self._sweep_goal_required_tokens()
                    for token, reason in missing_tokens:
                        broken.append(("(task)", reason))
                # Abandon-detection: if the sweep has fired and the model's
                # response is identical (or effectively the same) to the
                # last response we swept, the model is in a degenerate
                # state where it repeats "I can't complete the task" while
                # the sweep keeps forcing another iteration. Burning 15
                # iters × 10s each on "I can't" is worse than just accepting
                # the failure. Break out after 2 identical consecutive
                # sweep responses.
                response_key = " ".join((response or "").strip().lower().split())[:200]
                if broken:
                    if response_key and response_key == last_swept_response:
                        sweep_repeat_count += 1
                    else:
                        sweep_repeat_count = 1
                        last_swept_response = response_key
                    if sweep_repeat_count >= 2:
                        # Accept the answer and stop — the loop is degenerate.
                        self._transcript.append({"role": "assistant", "content": response})
                        final_answer = response
                        stop_reason = "answered"
                        self._emit(AgentEvent("final", iteration, final_answer))
                        break
                else:
                    sweep_repeat_count = 0
                    last_swept_response = ""
                if broken and iteration < self.max_iterations:
                    self._transcript.append(
                        {
                            "role": "assistant",
                            "content": self._canonical_assistant_turn(response, []),
                        }
                    )
                    # Auto-apply: if the sweep has surfaced the SAME
                    # (file, missing-import) pair 2+ times, the model is
                    # ignoring the hint. Apply the suggested import line
                    # ourselves and inject a synthetic observation saying
                    # we did it. Parses the suggested `import ...` line
                    # out of the sweep error via a regex marker.
                    import_hint_re = re.compile(
                        r"Add (?:this line at the top(?: of [^:]+)?): `([^`]+)`"
                    )
                    # Auto-apply for rename-absent sweep too. When the goal
                    # says "rename X to Y" and X is still in the workspace
                    # after 2+ sweeps, do a word-boundary regex replace
                    # across every touched .py file. Catches the case where
                    # the model renames the definition but misses a call
                    # site (iter2p+2r rename_function failure).
                    rename_hint_re = re.compile(
                        r"goal says to rename `([A-Za-z_][A-Za-z_0-9]*)` "
                        r"(?:→|->) `([A-Za-z_][A-Za-z_0-9]*)`"
                    )
                    # Auto-apply for CLI flag missing from `add_argument`.
                    # Matches both the in-file sentence (`--X` appears but
                    # not in an add_argument call) and the goal-token
                    # missing sentence. Action defaults to store_true —
                    # correct for boolean flags, which is the bulk of the
                    # real pattern.
                    flag_hint_re = re.compile(
                        r"`(--[A-Za-z][A-Za-z0-9\-]*)`"
                    )
                    sweep_results = []
                    for fname, err in broken:
                        applied_fix = None
                        applied_msg = ""
                        m = import_hint_re.search(err)
                        if m and fname != "(task)":
                            key = (fname, m.group(1))
                            sweep_miss_counter[key] = sweep_miss_counter.get(key, 0) + 1
                            if sweep_miss_counter[key] >= 2 and self.workspace_root is not None:
                                applied_fix = self._auto_apply_import_prepend(fname, m.group(1))
                                if applied_fix:
                                    applied_msg = (
                                        f"Auto-applied missing import to "
                                        f"{fname}: prepended `{m.group(1)}`. "
                                        f"The file now compiles. You can "
                                        f"continue with the remaining task "
                                        f"steps."
                                    )
                        rm = rename_hint_re.search(err) if not applied_fix else None
                        if rm and fname == "(task)":
                            old_name = rm.group(1)
                            new_name = rm.group(2)
                            key = ("rename", old_name, new_name)
                            sweep_miss_counter[key] = sweep_miss_counter.get(key, 0) + 1
                            if sweep_miss_counter[key] >= 2 and self.workspace_root is not None:
                                changed = self._auto_apply_rename_workspace(old_name, new_name)
                                if changed:
                                    applied_fix = True
                                    applied_msg = (
                                        f"Auto-applied rename `{old_name}` → "
                                        f"`{new_name}` across {len(changed)} "
                                        f"file(s): {', '.join(changed)}. "
                                        f"All call sites now use the new name."
                                    )
                        # CLI flag missing from add_argument — only on
                        # task-level sweep, and only when the error
                        # message actually mentions a `--X` flag.
                        if not applied_fix and fname == "(task)":
                            fm = flag_hint_re.search(err)
                            if fm and "add_argument" in err:
                                flag = fm.group(1)
                                key = ("flag", flag)
                                sweep_miss_counter[key] = sweep_miss_counter.get(key, 0) + 1
                                if sweep_miss_counter[key] >= 2 and self.workspace_root is not None:
                                    changed = self._auto_apply_argparse_flag(flag)
                                    if changed:
                                        applied_fix = True
                                        applied_msg = (
                                            f"Auto-applied `parser.add_argument"
                                            f"('{flag}', action='store_true')` "
                                            f"to {', '.join(changed)}. "
                                            f"argparse now recognizes {flag}."
                                        )
                        # @decorator missing above a class — task-level
                        # sweep for goals like "Convert the Person class
                        # to a @dataclass" where the model adds the import
                        # but forgets to actually apply the decorator.
                        if not applied_fix and fname == "(task)":
                            dec_match = re.search(
                                r"decorator (@[A-Za-z_][A-Za-z_0-9]*)",
                                err,
                            )
                            if dec_match:
                                dec = dec_match.group(1)
                                key = ("decorator", dec)
                                sweep_miss_counter[key] = sweep_miss_counter.get(key, 0) + 1
                                if sweep_miss_counter[key] >= 2 and self.workspace_root is not None:
                                    changed = self._auto_apply_decorator(dec)
                                    if changed:
                                        applied_fix = True
                                        applied_msg = (
                                            f"Auto-applied decorator `{dec}` "
                                            f"above the class in "
                                            f"{', '.join(changed)}."
                                        )
                                        # If the decorator was @dataclass,
                                        # also strip any lingering
                                        # `def __init__(self, x, y)` with
                                        # trivial `self.x = x` body — the
                                        # full refactor_dataclass bundle.
                                        if dec == "@dataclass":
                                            stripped = self._auto_strip_init_from_dataclass()
                                            if stripped:
                                                applied_msg += (
                                                    f" Also stripped now-"
                                                    f"redundant `def __init__` "
                                                    f"from {', '.join(stripped)} "
                                                    f"and replaced it with "
                                                    f"type-annotated fields."
                                                )
                        if applied_fix:
                            sr = ToolResult(
                                name="auto_apply",
                                success=True,
                                content=applied_msg,
                            )
                        elif fname == "(task)":
                            sr = ToolResult(
                                name="task_incomplete",
                                success=False,
                                error=f"Cannot finish: {err}",
                            )
                        else:
                            sr = ToolResult(
                                name="auto_verify",
                                success=False,
                                error=(
                                    f"Cannot finish — {fname} still has a "
                                    f"syntax error that must be fixed first: "
                                    f"{err}. Re-read {fname}, identify the "
                                    f"broken line, and fix it before giving "
                                    f"a final answer."
                                ),
                            )
                        sweep_results.append(sr)
                        self._tool_results.append(sr)
                        self._emit(AgentEvent("tool_result", iteration, sr))
                    self._transcript.append(
                        {"role": "tool", "content": self._format_tool_responses(sweep_results)}
                    )
                    continue

                # Final answer turn — but first, run pre_finish_check if
                # configured. This lets the benchmark wire task.check() so the
                # model gets feedback on partial work before the run ends.
                if (
                    self.pre_finish_check is not None
                    and pre_finish_retries < self.pre_finish_max_retries
                    and iteration < self.max_iterations
                ):
                    try:
                        check_feedback = self.pre_finish_check()
                    except Exception:
                        check_feedback = None
                    if check_feedback:
                        pre_finish_retries += 1
                        self._transcript.append(
                            {"role": "assistant", "content": response}
                        )
                        self._transcript.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Task NOT complete — the check found: {check_feedback}\n"
                                    f"Fix the remaining issue and try again. "
                                    f"(auto-retry {pre_finish_retries}/{self.pre_finish_max_retries})"
                                ),
                            }
                        )
                        self._emit(AgentEvent(
                            "pre_finish_retry", iteration,
                            {"retry": pre_finish_retries, "feedback": check_feedback[:200]},
                        ))
                        continue

                # Accept the final answer
                self._transcript.append({"role": "assistant", "content": response})
                final_answer = response
                stop_reason = "answered"
                self._emit(AgentEvent("final", iteration, final_answer))
                break

            # Tool-call turn: store the CANONICAL form in the transcript so
            # the model's next turn sees a clean well-formed history with
            # no truncated JSON from max_tokens cutoffs. (See docstring on
            # `_canonical_assistant_turn` for why this matters.)
            self._transcript.append(
                {
                    "role": "assistant",
                    "content": self._canonical_assistant_turn(response, calls),
                }
            )

            results: list[ToolResult] = []

            # Split calls into (stuck_repeats) and (real calls to execute).
            # stuck_repeat check must stay serial because it depends on the
            # running history of previously-executed calls.
            real_calls: list[ToolCall] = []
            for call in calls:
                if self._is_stuck_repeat(call):
                    # Tailor the suggestion to the tool that's stuck.
                    if call.name == "write_file":
                        suggestion = (
                            "write_file keeps producing the same bad result. "
                            "Likely causes: (a) JSON escape errors in the "
                            "`content` string — try the ARRAY form for content "
                            "(each element = one line, no \\n escaping needed); "
                            "(b) the file already exists and needs edit_file "
                            "for a targeted change; (c) your content has an "
                            "f-string or triple-quoted string that's broken — "
                            "simplify it."
                        )
                    elif call.name == "edit_file":
                        # Surface the target file's current contents inline so
                        # the model has fresh context to construct a new
                        # old_string. Phase 14 iter1m add_cli_flag failure:
                        # model pasted `9\t` line-number prefix + missing a
                        # middle line of the file, re-emitted 4 times, gave
                        # up — never saw the real file content after the first
                        # read. Including it here turns "try again" into
                        # "here is exactly what the file looks like now".
                        snapshot = self._safe_file_snapshot(
                            call.arguments.get("path") if isinstance(call.arguments, dict) else None
                        )
                        suggestion = (
                            "edit_file keeps failing with the same arguments. "
                            "Likely causes: (a) old_string doesn't match the "
                            "file (pick a SHORT unique substring); (b) you "
                            "included a line-number prefix like `    9\\t` "
                            "from read_file output — strip those before "
                            "pasting; (c) you assumed the file has a line "
                            "that isn't actually there (no continuation "
                            "between your two lines). Use a single short "
                            "unique line as old_string, NOT a multi-line "
                            "span."
                        )
                        if snapshot:
                            suggestion += (
                                f"\n\nCURRENT FILE CONTENTS (raw, no line "
                                f"numbers — copy a short unique substring "
                                f"from here):\n```\n{snapshot}\n```"
                            )
                    else:
                        suggestion = (
                            "Try a different tool or arguments. Read the "
                            "current file state before retrying."
                        )
                    stuck = ToolResult(
                        name="stuck_repeat",
                        success=False,
                        error=(
                            f"You just called {call.name} with the exact same "
                            f"arguments 3 times in a row. The previous attempts "
                            f"produced the same error. CHANGE YOUR APPROACH: "
                            f"{suggestion} Do not emit this exact call again."
                        ),
                    )
                    results.append(stuck)
                    self._tool_results.append(stuck)
                    self._emit(AgentEvent("tool_result", iteration, stuck))
                    continue
                real_calls.append(call)
                self._tool_calls.append(call)

            # Decide whether to parallelize. Only a batch of >=2 read-only
            # tool calls is safe to run concurrently. If ANY call in the
            # batch mutates state (write_file / edit_file / run_bash /
            # run_tests / remember) we stay serial to preserve model-
            # intended ordering.
            can_parallelize = (
                len(real_calls) >= 2
                and all(c.name in _PARALLELIZABLE_TOOLS for c in real_calls)
            )

            # Emit all tool_call events upfront so the user sees what's
            # about to execute before results stream in.
            for call in real_calls:
                self._emit(AgentEvent("tool_call", iteration, call))

            if can_parallelize:
                # ThreadPoolExecutor.map preserves input order in output.
                with ThreadPoolExecutor(
                    max_workers=min(len(real_calls), 4)
                ) as pool:
                    call_results = list(pool.map(self._execute_call, real_calls))
            else:
                call_results = [self._execute_call(c) for c in real_calls]

            # Emit results in call order + auto-verify after each.
            for call, result in zip(real_calls, call_results):
                results.append(result)
                self._tool_results.append(result)
                self._emit(AgentEvent("tool_result", iteration, result))

                verify = self._maybe_auto_verify(call, result)
                if verify is not None:
                    results.append(verify)
                    self._tool_results.append(verify)
                    self._emit(AgentEvent("tool_result", iteration, verify))

            # Surface parser errors as synthetic tool results so the model
            # sees "your malformed call didn't run" and can retry with
            # corrected JSON. Without this, a busted tool call is silently
            # dropped and the model assumes it succeeded.
            for err in parse_errors:
                synthetic = ToolResult(name="parse_error", success=False, error=err)
                results.append(synthetic)
                self._tool_results.append(synthetic)
                self._emit(AgentEvent("tool_result", iteration, synthetic))

            # Live self-heal step (arXiv 2604.27096 pattern): classify the
            # iteration's outcome into a failure class, and if the same
            # class has now fired twice consecutively, inject a synthetic
            # `self_heal_diagnose` ToolResult that asks the model to pause
            # and write a 2-3 line repair plan before its next tool call.
            # Composes with the existing stuck_repeat guard (which fires
            # only on identical-arg 3-peats); self_heal fires earlier on
            # any same-class 2-streak.
            #
            # Skipped entirely when enable_self_heal=False so the layer
            # can be A/B-measured against the historical baseline.
            if self.enable_self_heal:
                from engine import self_heal as _sh
                iter_classes = [
                    _sh.classify_failure(r) for r in results
                ]
                # An iteration's "class" is the most-severe non-None tag in
                # its results. If everything's clean, append None so the
                # streak resets correctly.
                iter_class = next((c for c in iter_classes if c is not None), None)
                self._failure_streak.append(iter_class)
                if iter_class is not None and _sh.should_inject_diagnose(self._failure_streak):
                    # Count how long the matching streak is so the prompt can
                    # mention "you have hit this N times".
                    streak_len = 1
                    for prev in reversed(self._failure_streak[:-1]):
                        if prev == iter_class:
                            streak_len += 1
                        else:
                            break
                    diagnose = ToolResult(
                        name="self_heal_diagnose",
                        success=False,
                        error=_sh.diagnose_message(iter_class, attempts=streak_len),
                    )
                    results.append(diagnose)
                    self._tool_results.append(diagnose)
                    self._self_heal_fired += 1
                    self._emit(AgentEvent("tool_result", iteration, diagnose))

            self._transcript.append(
                {"role": "tool", "content": self._format_tool_responses(results)}
            )
        else:
            stop_reason = "max_iterations"

        wall = time.monotonic() - start
        self._emit(AgentEvent("stopped", iteration, stop_reason))

        return AgentResult(
            final_answer=final_answer,
            iterations=iteration,
            stop_reason=stop_reason,
            wall_time=wall,
            tool_calls=list(self._tool_calls),
            tool_results=list(self._tool_results),
            transcript=list(self._transcript),
            compactions=self._compactions,
            self_heals=self._self_heal_fired,
        )


# ── Smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    from engine.agent_builtins import build_default_registry
    from engine.agent_memory import AgentMemory

    class StubModel:
        """
        Loader-free model stub. Returns canned responses round-by-round so we
        can exercise the loop without a real GGUF. Captures every prompt for
        assertion.
        """

        def __init__(self, responses: list[str]) -> None:
            self.responses = list(responses)
            self.prompts: list[str] = []
            self.calls = 0

        def generate(self, prompt: str, max_tokens: int, stop: list[str], **kwargs) -> str:
            self.prompts.append(prompt)
            if self.calls >= len(self.responses):
                # Default to a final-answer response so runaway tests still terminate.
                self.calls += 1
                return "(stub exhausted)"
            r = self.responses[self.calls]
            self.calls += 1
            return r

    # ── Test 1: single-shot final answer (no tool calls) ──
    with tempfile.TemporaryDirectory() as tmp:
        reg = build_default_registry(tmp)
        model = StubModel(["The answer is 42."])
        agent = Agent(model, reg, max_iterations=5)
        result = agent.run("What is the answer?")
        assert result.stop_reason == "answered", result.stop_reason
        assert result.final_answer == "The answer is 42."
        assert result.iterations == 1
        assert result.tool_calls == []
        # System prompt must include the Hermes tool block
        assert "<tools>" in model.prompts[0]
        assert '"name": "read_file"' in model.prompts[0]
        # ChatML scaffolding present
        assert "<|im_start|>system" in model.prompts[0]
        assert "<|im_start|>assistant" in model.prompts[0]

    # ── Test 2: read -> edit -> answer happy path ──
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "greet.py"
        target.write_text("print('hello')\n", encoding="utf-8")

        reg = build_default_registry(tmp)
        responses = [
            'I will read it first.\n<tool_call>{"name": "read_file", "arguments": {"path": "greet.py"}}</tool_call>',
            'Now edit.\n<tool_call>{"name": "edit_file", "arguments": {"path": "greet.py", "old_string": "hello", "new_string": "world"}}</tool_call>',
            "Done. greet.py now prints 'world'.",
        ]
        model = StubModel(responses)
        events: list[AgentEvent] = []
        agent = Agent(
            model, reg, workspace_root=Path(tmp), max_iterations=5, on_event=events.append
        )
        result = agent.run("Change greet.py to say world.")

        assert result.stop_reason == "answered", result.stop_reason
        assert result.iterations == 3
        assert result.final_answer.startswith("Done.")
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[1].name == "edit_file"
        # tool_results includes the auto_verify after edit_file
        result_names = [r.name for r in result.tool_results]
        assert result_names == ["read_file", "edit_file", "auto_verify"], result_names
        assert all(r.success for r in result.tool_results), [r.error for r in result.tool_results]
        # File was actually edited
        assert target.read_text(encoding="utf-8") == "print('world')\n"
        # Second prompt must contain the tool turn (model sees the read_file output)
        assert "<|im_start|>tool" in model.prompts[1]
        assert "<tool_response>" in model.prompts[1]
        # Event types fired in order
        types = [e.type for e in events]
        assert "iteration" in types and "tool_call" in types and "tool_result" in types
        assert types[-1] == "stopped"
        assert types[-2] == "final"

    # ── Test 3: max_iterations cutoff ──
    with tempfile.TemporaryDirectory() as tmp:
        reg = build_default_registry(tmp)
        # Model never stops calling tools. Cycle through distinct paths so
        # the stuck_repeat guard doesn't block consecutive identical calls.
        spams = [
            '<tool_call>{"name": "list_dir", "arguments": {"path": "."}}</tool_call>',
            '<tool_call>{"name": "list_dir", "arguments": {"path": "./a"}}</tool_call>',
            '<tool_call>{"name": "list_dir", "arguments": {"path": "./b"}}</tool_call>',
        ] * 10
        model = StubModel(spams)
        agent = Agent(model, reg, max_iterations=3)
        result = agent.run("Loop forever.")
        assert result.stop_reason == "max_iterations", result.stop_reason
        assert result.iterations == 3
        assert len(result.tool_calls) == 3
        assert result.final_answer == ""  # never produced one

    # ── Test 4: risky tool denial ──
    with tempfile.TemporaryDirectory() as tmp:
        reg = build_default_registry(tmp)
        responses = [
            '<tool_call>{"name": "run_bash", "arguments": {"command": "rm -rf /"}}</tool_call>',
            "Aborted as requested.",
        ]
        model = StubModel(responses)
        denied: list[ToolCall] = []

        def deny(call: ToolCall) -> bool:
            denied.append(call)
            return False

        agent = Agent(model, reg, max_iterations=5, confirm_risky=deny)
        result = agent.run("Wreck the disk.")
        assert result.stop_reason == "answered", result.stop_reason
        assert result.iterations == 2
        assert len(denied) == 1
        # The denied result must be in the loop's history
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        assert "denied" in (result.tool_results[0].error or "").lower()
        # Model's second prompt should have seen the denial in the tool turn
        assert "denied" in model.prompts[1].lower()

    # ── Test 5: unknown tool name surfaces as a recoverable error ──
    with tempfile.TemporaryDirectory() as tmp:
        reg = build_default_registry(tmp)
        responses = [
            '<tool_call>{"name": "fly_to_mars", "arguments": {}}</tool_call>',
            "OK, that tool does not exist. Final answer.",
        ]
        model = StubModel(responses)
        agent = Agent(model, reg, max_iterations=5)
        result = agent.run("Try a bogus tool.")
        assert result.stop_reason == "answered", result.stop_reason
        assert result.iterations == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        assert "Unknown tool" in (result.tool_results[0].error or "")
        # Model saw the error in turn 2 and could choose to recover
        assert "Unknown tool" in model.prompts[1]

    # ── Test 6: model error surfaced cleanly ──
    with tempfile.TemporaryDirectory() as tmp:
        reg = build_default_registry(tmp)

        class BoomModel:
            def generate(self, prompt: str, max_tokens: int, stop: list[str], **kwargs) -> str:
                raise RuntimeError("backend exploded")

        agent = Agent(BoomModel(), reg, max_iterations=5)
        result = agent.run("Anything.")
        assert result.stop_reason == "model_error", result.stop_reason
        assert "backend exploded" in result.final_answer

    # ── Test 7: auto-verify catches a SyntaxError in a generated .py file ──
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        reg = build_default_registry(ws)
        responses = [
            # First write: broken syntax
            '<tool_call>{"name": "write_file", "arguments": {"path": "broken.py", "content": "def foo(:\\n    pass\\n"}}</tool_call>',
            # Second write: fixed syntax (model corrects after seeing auto_verify error)
            '<tool_call>{"name": "write_file", "arguments": {"path": "broken.py", "content": "def foo():\\n    pass\\n"}}</tool_call>',
            "Fixed the syntax error.",
        ]
        model = StubModel(responses)
        agent = Agent(model, reg, workspace_root=ws, max_iterations=5)
        result = agent.run("Create broken.py with a function.")

        assert result.stop_reason == "answered", result.stop_reason
        # tool_results contains: write_file(ok), auto_verify(fail), write_file(ok), auto_verify(ok)
        kinds = [(r.name, r.success) for r in result.tool_results]
        assert kinds == [
            ("write_file", True),
            ("auto_verify", False),
            ("write_file", True),
            ("auto_verify", True),
        ], kinds
        # The model's second prompt must have included the auto_verify error
        assert "auto_verify" in model.prompts[1]
        assert "SyntaxError" in model.prompts[1]
        # Final file is valid Python
        compile((ws / "broken.py").read_text(encoding="utf-8"), "broken.py", "exec")

    # ── Test 8: agent memory loaded into system prompt and remember tool works ──
    with tempfile.TemporaryDirectory() as tmp:
        with tempfile.TemporaryDirectory() as mem_root:
            ws = Path(tmp)
            mem = AgentMemory(workspace=ws, root=Path(mem_root))
            mem.remember("auth uses JWT not session cookies")
            mem.remember("tests live in tests/ not test/")

            reg = build_default_registry(ws, memory=mem)
            assert "remember" in reg.status()["tools"]

            responses = [
                '<tool_call>{"name": "remember", "arguments": {"note": "Phase 14 agent ran here on 2026-04-14"}}</tool_call>',
                "Saved a note. All done.",
            ]
            model = StubModel(responses)
            agent = Agent(model, reg, workspace_root=ws, memory=mem, max_iterations=5)
            result = agent.run("Save a note about today.")

            assert result.stop_reason == "answered", result.stop_reason
            # Memory block was injected into the system prompt
            assert "auth uses JWT" in model.prompts[0]
            assert "tests live in tests/" in model.prompts[0]
            assert "Notes from previous sessions" in model.prompts[0]
            # The model successfully called remember and the note was persisted
            assert any(r.name == "remember" and r.success for r in result.tool_results)
            assert "Phase 14 agent ran here" in mem.load()

    # ── Test 9: auto-verify is OPT-OUT-able and not triggered on non-py files ──
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        reg = build_default_registry(ws)
        responses = [
            # Broken Python BUT auto_verify_python=False
            '<tool_call>{"name": "write_file", "arguments": {"path": "still_broken.py", "content": "def foo(:\\n"}}</tool_call>',
            "Wrote it.",
        ]
        model = StubModel(responses)
        agent = Agent(model, reg, workspace_root=ws, auto_verify_python=False, max_iterations=5)
        result = agent.run("Write broken python.")
        assert result.stop_reason == "answered"
        kinds = [r.name for r in result.tool_results]
        assert kinds == ["write_file"], kinds  # No auto_verify

        # Non-py file: verification skipped even with auto-verify ON
        reg2 = build_default_registry(ws)
        responses2 = [
            '<tool_call>{"name": "write_file", "arguments": {"path": "notes.md", "content": "hello"}}</tool_call>',
            "Wrote markdown.",
        ]
        model2 = StubModel(responses2)
        agent2 = Agent(model2, reg2, workspace_root=ws, auto_verify_python=True, max_iterations=5)
        result2 = agent2.run("Write notes.")
        assert [r.name for r in result2.tool_results] == ["write_file"]

    # ── Test 10: stuck-repeat detection blocks 3rd consecutive identical call ──
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "oops.py").write_text("broken\n", encoding="utf-8")
        reg = build_default_registry(ws)
        # Model tries the exact same edit_file 4 times, then gives up
        identical_call = (
            '<tool_call>{"name": "edit_file", "arguments": '
            '{"path": "oops.py", "old_string": "broken", "new_string": "fixed"}}'
            '</tool_call>'
        )
        responses = [
            identical_call,
            identical_call,
            identical_call,  # 3rd identical — should be blocked
            identical_call,  # 4th — should also be blocked
            "I give up, the file cannot be fixed.",
        ]
        model = StubModel(responses)
        agent = Agent(model, reg, workspace_root=ws, max_iterations=10)
        result = agent.run("Fix the file.")

        # First 2 execute (the edit succeeds on the 1st call, the 2nd finds
        # old_string="broken" gone and errors). The 3rd and 4th are blocked
        # by stuck_repeat BEFORE execution.
        names = [r.name for r in result.tool_results]
        assert names.count("stuck_repeat") >= 2, f"expected >=2 stuck_repeat, got {names}"
        # The model sees stuck_repeat in its context
        stuck_idx = names.index("stuck_repeat")
        # Model's prompt at the iteration AFTER the first stuck_repeat must include it
        stuck_seen_in_prompt = any("stuck_repeat" in p for p in model.prompts[2:])
        assert stuck_seen_in_prompt, "stuck_repeat should appear in later prompts"

    # ── Test 11: parallel dispatch on multi-read-only batch ──
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        for name in ("a.py", "b.py", "c.py"):
            (ws / name).write_text(f"# {name}\n", encoding="utf-8")
        reg = build_default_registry(ws)
        # Model emits 3 read_file calls in one turn
        batch = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>\n'
            '<tool_call>{"name": "read_file", "arguments": {"path": "b.py"}}</tool_call>\n'
            '<tool_call>{"name": "read_file", "arguments": {"path": "c.py"}}</tool_call>'
        )
        responses = [batch, "Read all three."]
        model = StubModel(responses)
        agent = Agent(model, reg, workspace_root=ws, max_iterations=5)
        result = agent.run("Read a.py, b.py, c.py.")
        assert result.stop_reason == "answered", result.stop_reason
        assert len(result.tool_calls) == 3
        # All 3 reads should have completed and be in order
        names = [r.name for r in result.tool_results]
        assert names == ["read_file", "read_file", "read_file"], names
        # Content preserved in order — first read should contain a.py, etc.
        assert "a.py" in str(result.tool_results[0].content)
        assert "b.py" in str(result.tool_results[1].content)
        assert "c.py" in str(result.tool_results[2].content)

    # ── Test 12: mixed batch (write + reads) falls back to serial ──
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "a.py").write_text("x = 1\n", encoding="utf-8")
        reg = build_default_registry(ws)
        # Write + read in same batch — must stay serial so the read sees
        # the written content.
        batch = (
            '<tool_call>{"name": "write_file", "arguments": {"path": "b.py", "content": "y = 2\\n"}}</tool_call>\n'
            '<tool_call>{"name": "read_file", "arguments": {"path": "b.py"}}</tool_call>'
        )
        responses = [batch, "Done."]
        model = StubModel(responses)
        agent = Agent(model, reg, workspace_root=ws, max_iterations=5)
        result = agent.run("Write b.py then read it.")
        assert result.stop_reason == "answered"
        # Both tool calls executed, both succeeded, read sees written content
        assert result.tool_results[0].name == "write_file"
        assert result.tool_results[0].success
        # index 1 is auto_verify (syntax OK b.py)
        # index 2 is read_file
        read_idx = next(
            i for i, r in enumerate(result.tool_results) if r.name == "read_file"
        )
        assert "y = 2" in str(result.tool_results[read_idx].content)

    # ── Test 13: JSON auto-verify catches invalid JSON writes ──
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        reg = build_default_registry(ws)
        # Model writes invalid JSON, sees auto_verify error, fixes it
        responses = [
            '<tool_call>{"name": "write_file", "arguments": {"path": "data.json", "content": "{broken}"}}</tool_call>',
            '<tool_call>{"name": "write_file", "arguments": {"path": "data.json", "content": "{\\"ok\\": true}"}}</tool_call>',
            "Fixed the JSON.",
        ]
        model = StubModel(responses)
        agent = Agent(model, reg, workspace_root=ws, max_iterations=5)
        result = agent.run("Create data.json.")
        assert result.stop_reason == "answered", result.stop_reason
        # tool_results: write_file, auto_verify(fail), write_file, auto_verify(ok)
        kinds = [(r.name, r.success) for r in result.tool_results]
        assert kinds == [
            ("write_file", True),
            ("auto_verify", False),
            ("write_file", True),
            ("auto_verify", True),
        ], kinds
        assert "JSON" in (result.tool_results[1].error or "")
        # File is now valid JSON
        json.loads((ws / "data.json").read_text(encoding="utf-8"))

    # ── Test 14: non-Python, non-JSON extensions with no external
    # toolchain just return None (no verify result). Using a .md file as
    # a stand-in because its extension isn't in the dispatch table. ──
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        reg = build_default_registry(ws)
        responses = [
            '<tool_call>{"name": "write_file", "arguments": {"path": "notes.md", "content": "# hello"}}</tool_call>',
            "Done.",
        ]
        model = StubModel(responses)
        agent = Agent(model, reg, workspace_root=ws, max_iterations=5)
        result = agent.run("Write notes.md.")
        # Just write_file; no auto_verify should fire for .md
        assert [r.name for r in result.tool_results] == ["write_file"]

    # ── Test 15: .js/.ts files trigger node --check if node is installed,
    # otherwise skip cleanly (return None). Either outcome is acceptable
    # because node may or may not be on PATH in the test environment. ──
    import shutil as _shutil
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        reg = build_default_registry(ws)
        responses = [
            '<tool_call>{"name": "write_file", "arguments": {"path": "app.js", "content": "const x = 1; console.log(x);"}}</tool_call>',
            "Done.",
        ]
        model = StubModel(responses)
        agent = Agent(model, reg, workspace_root=ws, max_iterations=5)
        result = agent.run("Write app.js.")
        names = [r.name for r in result.tool_results]
        if _shutil.which("node"):
            # Node present: auto_verify should have fired successfully
            assert "auto_verify" in names
            verify = next(r for r in result.tool_results if r.name == "auto_verify")
            assert verify.success, f"expected node --check OK, got {verify.error}"
        else:
            # Node absent: skip cleanly, no verify result
            assert "auto_verify" not in names

    print("OK: engine/agent.py ReAct loop smoke test passed (15/15)")
