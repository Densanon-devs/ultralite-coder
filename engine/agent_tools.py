"""
Phase 14 Agent Tool System — Qwen 2.5 Native Tool Calling

Hermes-style format emitted by Qwen 2.5 (the format the model was trained on):

    <tool_call>
    {"name": "write_file", "arguments": {"path": "foo.py", "content": "..."}}
    </tool_call>

Tools are registered with a JSON-Schema parameter description, injected into
the system prompt in the Hermes <tools>...</tools> block, and invoked via
parsed <tool_call> tags emitted by the model.

This module is intentionally kept in ultralight-coder/engine/ and NOT promoted
to densanon-core. Phase 14 agent tooling must not ripple to PIE or npc-engine,
and the legacy densanon.core.tools.ToolRegistry (utility tools: calculate,
run_python, format_json, etc.) stays the shared surface those projects depend on.

Zero servers, pure in-process, local stdlib + subprocess only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _dump_parse_failure(tier: str, body: str) -> None:
    """When ULTRALITE_PARSE_FAILURE_DUMP is set, write the full failing body to
    that directory so we can inspect it offline. Truncated warnings hide the
    actual broken character — this dump captures everything."""
    dump_dir = os.environ.get("ULTRALITE_PARSE_FAILURE_DUMP")
    if not dump_dir:
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        h = hashlib.md5(body.encode("utf-8", errors="replace")).hexdigest()[:10]
        path = os.path.join(dump_dir, f"{tier}_{h}.txt")
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(body)
    except OSError:
        pass


# ── Data types ──────────────────────────────────────────────────


@dataclass
class ToolSchema:
    """JSON-Schema description of a tool, emitted to the model in Hermes format."""

    name: str
    description: str
    # JSON Schema object, e.g.
    #   {"type": "object",
    #    "properties": {"path": {"type": "string", "description": "..."}},
    #    "required": ["path"]}
    parameters: dict
    function: Callable[..., Any]
    category: str = "general"
    enabled: bool = True
    # True = the agent loop must prompt the user to confirm before executing
    # (e.g. run_bash outside cwd, destructive git ops).
    risky: bool = False

    def to_hermes_dict(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A parsed <tool_call> block from model output."""

    name: str
    arguments: dict
    raw: str  # the full "<tool_call>...</tool_call>" text


@dataclass
class ToolResult:
    """Result of executing a ToolCall. Fed back to the model as the next observation."""

    name: str
    success: bool
    content: Any = None
    error: Optional[str] = None

    def format_for_model(self) -> str:
        """Serialize for the next turn's tool message. Compact JSON."""
        if self.success:
            body: dict[str, Any] = {"name": self.name, "success": True}
            if isinstance(self.content, (str, int, float, bool, list, dict, type(None))):
                body["content"] = self.content
            else:
                body["content"] = str(self.content)
            return json.dumps(body, ensure_ascii=False)
        return json.dumps(
            {"name": self.name, "success": False, "error": self.error or "unknown error"},
            ensure_ascii=False,
        )


# ── Parser ──────────────────────────────────────────────────────

# Matches one <tool_call>...</tool_call> block containing a JSON object.
# DOTALL so that multi-line JSON (common for write_file content arguments)
# is captured. Non-greedy body so multiple calls in one response each parse
# independently.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# R1-distill reasoning-block matcher. Non-greedy so multiple blocks parse
# independently; DOTALL so the reasoning can span newlines.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Fallback matcher for ```json ... ``` fenced tool calls. Some 14B-class
# models (Qwen 2.5 in particular, when the chat template isn't applied
# verbatim) emit tool calls as fenced JSON instead of Hermes tags. We
# accept these IF the object has both "name" and "arguments" keys —
# otherwise it's probably just example JSON in a final answer.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# strict=False allows literal control chars (\n, \t) inside JSON string values.
# Small models commonly emit raw newlines inside `write_file` content args
# instead of escaping them as \\n, and rejecting that would force the model
# into a retry loop on every multi-line write.
_LENIENT_JSON = json.JSONDecoder(strict=False)

# Valid JSON escape characters per RFC 8259. Anything else after a \ in a
# JSON string value is an error. The 14B repeatedly invents `\@` for Python
# decorators and other non-standard escapes; this regex strips the backslash
# from invalid ones so `\@dataclass` becomes `@dataclass`.
_INVALID_ESCAPE_RE = re.compile(r'\\([^"\\/bfnrtu])')


def _repair_json_body(body: str) -> str:
    """
    Fix common JSON-invalid escape sequences the model emits inside string
    values. Specifically, strips the backslash from `\\X` where X is not a
    valid JSON escape character — so `\\@dataclass` becomes `@dataclass`.
    Safe because valid escapes (`\\"`, `\\\\`, `\\n`, etc.) are preserved.
    """
    return _INVALID_ESCAPE_RE.sub(r"\1", body)


def _decode_with_repair(body: str) -> Optional[dict]:
    """
    Try to decode *body* as lenient JSON. Repair tiers on failure:

    1. Strict full-string decode (works for clean single objects). This is
       the baseline path — unchanged from the pre-Phase-14 parser behavior.
    2. Invalid-escape repair — strips backslashes from non-standard escapes
       (`\\@dataclass`, `\\d`, etc.) and retries.
    3. Extra-trailing-brace recovery — when strict decode fails with "Extra
       data" AND the trailing content is ONLY closing braces/whitespace (the
       model's `}}}` habit on write_file calls with nested content), accept
       the raw_decode prefix. Rejects multi-object fences so the model
       re-emits them as separate fences.
    4. Python literal fallback — `ast.literal_eval` parses mixed single/
       double quoted strings (Python-style). The 14B sometimes starts a
       content array with `"..."` strings then switches to `'...'` mid-array,
       which JSON rejects. `ast.literal_eval` handles both, and since it
       only evaluates constants, it is safe to run on untrusted input.

    Returns the parsed object on success, None on failure.
    """
    try:
        return _LENIENT_JSON.decode(body)
    except json.JSONDecodeError as exc:
        err_msg = str(exc)
    # Tier 2: escape repair
    if "escape" in err_msg:
        try:
            return _LENIENT_JSON.decode(_repair_json_body(body))
        except json.JSONDecodeError:
            pass
    # Tier 3: extra-trailing-brace recovery. Only fires on "Extra data"
    # errors to avoid shadowing other decode failures.
    if "Extra data" in err_msg:
        obj = _try_extra_brace_recovery(body)
        if obj is not None:
            return obj
        # Also retry with escape repair applied
        repaired = _repair_json_body(body)
        if repaired != body:
            obj = _try_extra_brace_recovery(repaired)
            if obj is not None:
                return obj
    # Tier 4: Truncated content-array recovery. When max_tokens cuts the
    # model's output mid-string inside a content array like
    #   {"name":"write_file","arguments":{"path":"x.py","content":["line1","line2","li
    # the parser sees "Unterminated string". Fix: find the last complete
    # array element, close the array + arguments + outer object, and parse
    # the prefix. The model gets a shorter file written (some lines lost)
    # but the tool call EXECUTES instead of being silently dropped.
    if "Unterminated" in err_msg or "Expecting" in err_msg:
        recovered = _try_truncated_array_recovery(body)
        if recovered is not None:
            return recovered

    # Tier 5: Python literal fallback — handles mixed `'...'` + `"..."`
    # strings the model sometimes emits inside content arrays. Safe because
    # `ast.literal_eval` only accepts literal constants (strings, numbers,
    # lists, dicts, tuples, None, True, False) — no function calls, no
    # imports, no attribute access.
    try:
        import ast as _ast
        obj = _ast.literal_eval(body)
    except (ValueError, SyntaxError, MemoryError, TypeError):
        pass
    else:
        if isinstance(obj, dict):
            return obj
    # Also try the literal eval on the repaired body in case both escape
    # errors AND mixed quotes are present simultaneously.
    repaired = _repair_json_body(body)
    if repaired != body:
        try:
            import ast as _ast
            obj = _ast.literal_eval(repaired)
        except (ValueError, SyntaxError, MemoryError, TypeError):
            pass
        else:
            if isinstance(obj, dict):
                return obj
    return None


def _try_truncated_array_recovery(body: str) -> Optional[dict]:
    """Recover from max_tokens truncation inside a content array.

    Pattern: the body is a JSON object with a `"content": [...]` field whose
    last string element is cut off mid-character. We find the last complete
    `","` boundary (end of a string element), chop there, close the brackets.

    Returns the parsed object on success, None on failure.
    """
    content_idx = body.find('"content"')
    if content_idx == -1:
        return None
    bracket_idx = body.find("[", content_idx)
    if bracket_idx == -1:
        return None
    # Walk backwards from the end looking for the last '","' pattern which
    # marks the boundary between two complete array elements.
    last_comma = body.rfind('",')
    if last_comma == -1 or last_comma <= bracket_idx:
        # Only one element or none — try closing after the first complete string
        last_quote = body.rfind('"', bracket_idx + 1)
        if last_quote <= bracket_idx:
            return None
        last_comma = last_quote
    # Chop after the complete element (include the closing `"`) and close out.
    prefix = body[: last_comma + 1]  # up to and including the `"`
    # Count how many `{` vs `}` to figure out how many closing braces needed.
    open_braces = prefix.count("{") - prefix.count("}")
    open_brackets = prefix.count("[") - prefix.count("]")
    suffix = "]" * max(open_brackets, 0) + "}" * max(open_braces, 0)
    candidate = prefix + suffix
    try:
        obj = _LENIENT_JSON.decode(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Also try with escape repair
    try:
        obj = _LENIENT_JSON.decode(_repair_json_body(candidate))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _try_extra_brace_recovery(body: str) -> Optional[dict]:
    """Use raw_decode to parse the first JSON object in *body*. If the
    trailing content is only closing braces/brackets/whitespace (the model's
    extra-brace habit), return the parsed object. Otherwise return None so
    the caller knows the body has ambiguous trailing content (e.g. another
    JSON object) and should fall through to the multi-call scanner."""
    try:
        obj, end = _LENIENT_JSON.raw_decode(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    trailing = body[end:].strip()
    if trailing and not all(c in "}]" for c in trailing):
        return None
    return obj


def _is_tool_call_obj(obj: Any) -> bool:
    """True if *obj* is a dict with string `name` and dict `arguments`."""
    if not isinstance(obj, dict):
        return False
    if "name" not in obj or "arguments" not in obj:
        return False
    name = obj.get("name")
    args = obj.get("arguments")
    if args is None:
        args = {}
    return isinstance(name, str) and isinstance(args, dict)


def _scan_bare_json_calls(text: str) -> tuple[list[tuple[dict, str]], list[str]]:
    """
    Scan *text* for top-level JSON objects via `raw_decode`. Returns a tuple
    `(found, errors)`:
    - `found` — every object that looks like a tool call (has `name` +
      `arguments` keys) with its raw source text
    - `errors` — human-readable messages for text regions that CLEARLY
      looked like tool-call attempts (contained both `"name"` and
      `"arguments"` substrings and started with `{`) but failed to parse.
      Used by the Agent loop to tell the model "your bare JSON tool call
      was malformed, retry."

    Common failure mode this catches: model wraps a Python string literal
    in single quotes inside a JSON value (e.g. new_string gets a Python
    triple-quoted docstring wrapped with single quotes). JSON requires
    double quotes for strings, and the surfaced error lets the model see
    that and correct on the next turn.
    """
    found: list[tuple[dict, str]] = []
    errors: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        idx = text.find("{", i)
        if idx == -1:
            break
        try:
            obj, end = _LENIENT_JSON.raw_decode(text, idx)
        except json.JSONDecodeError as exc:
            # Try repairing invalid JSON escapes (model commonly writes
            # \@dataclass, \d, etc.) and retrying raw_decode on the repaired
            # tail.
            tail = text[idx:]
            repaired = _repair_json_body(tail)
            if repaired != tail:
                try:
                    obj, rel_end = _LENIENT_JSON.raw_decode(repaired, 0)
                    if _is_tool_call_obj(obj):
                        found.append((obj, tail[:min(rel_end, len(tail))]))
                    # Advance past the region we just consumed in the ORIGINAL
                    # text. rel_end is in the repaired string (which may be
                    # shorter), so we move forward at least that many chars.
                    i = idx + max(rel_end, 1)
                    continue
                except json.JSONDecodeError:
                    pass
            # Tier 4 fallback for bare tier: ast.literal_eval on the tail,
            # which handles Python-style single-quoted strings that the
            # 14B emits inside `new_string` values (`'print(...)' with \\'
            # escapes). Scan outward to find a matching closing brace.
            py_obj, py_end = _try_python_literal_eval(text, idx)
            if py_obj is not None and _is_tool_call_obj(py_obj):
                found.append((py_obj, text[idx:py_end]))
                i = py_end
                continue
            # Probe: does this `{...}` region look like a tool call attempt?
            probe = text[idx : idx + 800]
            if '"name"' in probe and '"arguments"' in probe:
                _dump_parse_failure("bare", text[idx : idx + 8000])
                msg = (
                    f"bare JSON tool call failed to parse: {exc}. "
                    f"JSON strings must use DOUBLE quotes and inner \" must be "
                    f"escaped as \\\". Never use Python-style '...' or f\"...\" "
                    f"around string values. Also never use \\@, \\d, or other "
                    f"non-standard escapes — only \\\", \\\\, \\/, \\b, \\f, "
                    f"\\n, \\r, \\t, \\uXXXX are valid JSON escapes. "
                    f"Starts: {probe[:160]!r}"
                )
                logger.warning(msg)
                errors.append(msg)
            i = idx + 1
            continue
        if _is_tool_call_obj(obj):
            found.append((obj, text[idx:end]))
        i = end if end > idx else idx + 1
    return found, errors


def _try_python_literal_eval(text: str, start: int) -> tuple[Optional[dict], int]:
    """Starting at `text[start]` (which should be `{`), scan forward to find
    a matching closing brace and try `ast.literal_eval` on the substring.
    Returns `(parsed_dict, end_index)` on success, or `(None, start)` on
    failure. Handles mixed single/double quoted strings the way Python
    parses them, tolerating cases JSON rejects."""
    import ast as _ast
    if start >= len(text) or text[start] != "{":
        return None, start
    # Bracket-matching scanner that respects string literals and escapes.
    depth = 0
    i = start
    n = len(text)
    in_str: Optional[str] = None  # None, or '"', "'"
    while i < n:
        c = text[i]
        if in_str is not None:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == in_str:
                in_str = None
        else:
            if c == '"' or c == "'":
                in_str = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    body = text[start : i + 1]
                    try:
                        obj = _ast.literal_eval(body)
                    except (ValueError, SyntaxError, MemoryError, TypeError):
                        return None, start
                    if isinstance(obj, dict):
                        return obj, i + 1
                    return None, start
        i += 1
    return None, start


def parse_tool_calls_with_errors(text: str) -> tuple[list[ToolCall], list[str]]:
    """
    Extract tool calls from model output. Three tiers of fallback, each only
    applied if the prior tier found nothing — so a model that emits correct
    Hermes tags wins on tier 1 and the fallbacks never fire.

    Tier 1 (preferred): <tool_call>...</tool_call> Hermes tags, exactly what
        Qwen 2.5 was trained on when the chat template is applied.
    Tier 2: ```json fenced JSON — observed from Qwen 2.5 14B when the chat
        template isn't applied and the model falls back to Markdown habits.
    Tier 3: bare JSON objects with `name` + `arguments` keys — last resort
        when the model drops both tags AND fences. Requires both keys to
        avoid false-positiving on plain JSON documentation in a final answer.

    Returns `(calls, errors)`. Errors are strings describing fence/bare blocks
    that looked like tool calls but failed to parse — they are surfaced to the
    model as synthetic tool results by the Agent loop so the model knows its
    malformed call didn't run and can retry. (Without this, a busted call to
    `edit_file` with Python f-string syntax inside a JSON value would be
    silently dropped and the model would assume it succeeded.)

    Tolerant to whitespace, multiple calls in one response, and literal
    newlines/tabs inside JSON string values.
    """
    calls: list[ToolCall] = []
    errors: list[str] = []

    # R1-distill / reasoning-model <think>...</think> blocks: strip them
    # before parsing so tool calls emitted AFTER the reasoning block are
    # found, and tool-call-shaped text WITHIN the think block is NOT
    # executed (it's model-private reasoning, not a real tool request).
    # The open tag may appear without a close tag when the model is
    # truncated mid-thought, so we also tolerate an unclosed <think> at
    # the end by stripping everything from an unmatched open to EOS.
    text = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in text:
        idx = text.find("<think>")
        text = text[:idx]

    # Tier 1: Hermes tags
    tier1_found = False
    for match in _TOOL_CALL_RE.finditer(text):
        tier1_found = True
        raw = match.group(0)
        body = match.group(1).strip()
        parsed = _decode_with_repair(body)
        if parsed is None:
            msg = f"<tool_call> JSON malformed (body starts: {body[:80]!r})"
            logger.warning(msg)
            errors.append(msg)
            continue
        if not _is_tool_call_obj(parsed):
            msg = f"<tool_call> missing `name` or `arguments`: {str(parsed)[:160]}"
            logger.warning(msg)
            errors.append(msg)
            continue
        calls.append(
            ToolCall(name=parsed["name"], arguments=parsed.get("arguments") or {}, raw=raw)
        )
    if tier1_found:
        # Only the Hermes tier ran — tiers 2/3 are FALLBACKS and should not
        # fire if the model used the correct format at all.
        return calls, errors

    # Tier 2: ```json fences. ONE fence = ONE call, not N. Multi-call
    # plans the model writes inside a single fence are step-plans, not
    # batches — running them all greedily corrupts file state when an
    # intermediate edit changes the file in a way the next edit's
    # `old_string` no longer matches. The agent loop runs ONE call per
    # fence and lets the model react to its result before the next call.
    # raw_decode tolerates trailing junk (extra `}}}` is a common 14B
    # habit on write_file calls with multi-line content).
    for match in _JSON_FENCE_RE.finditer(text):
        raw = match.group(0)
        body = match.group(1).strip()
        parsed = _decode_with_repair(body)
        if parsed is not None and _is_tool_call_obj(parsed):
            calls.append(
                ToolCall(name=parsed["name"], arguments=parsed.get("arguments") or {}, raw=raw)
            )
            continue
        # Multi-call fence: model wrote >1 tool_call JSON objects inside a
        # single ```json fence. The single-object decoders above can't handle
        # "Extra data" from multiple objects, so fall back to the bare scanner
        # on just the fence body. Recovers all N calls rather than dropping
        # the whole fence. Phase 14 iter1t regression — build_todo_cli's 4
        # file writes were landing in one fence and the first-object recovery
        # was dropping 3 of 4.
        scanned, _scan_errors = _scan_bare_json_calls(body)
        if scanned:
            for obj, obj_raw in scanned:
                calls.append(
                    ToolCall(name=obj["name"], arguments=obj.get("arguments") or {}, raw=obj_raw)
                )
            continue
        # Genuinely unparseable. Only surface as an error if it LOOKS like a
        # tool call attempt (has both "name" and "arguments" substrings).
        if '"name"' in body and '"arguments"' in body:
            _dump_parse_failure("fence", body)
            msg = (
                f"```json tool-call failed to parse. "
                f"Remember: JSON string values must start with a regular \" — "
                f"never Python-style f\"...\" or '...' — and inner quotes "
                f"must be escaped as \\\". Also never use JSON escapes like "
                f"\\@ or \\d — only \\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, "
                f"\\uXXXX are valid. Body starts: {body[:120]!r}"
            )
            logger.warning(msg)
            errors.append(msg)
    if calls or errors:
        return calls, errors

    # Tier 3: bare JSON
    bare_found, bare_errors = _scan_bare_json_calls(text)
    for obj, raw in bare_found:
        calls.append(
            ToolCall(name=obj["name"], arguments=obj.get("arguments") or {}, raw=raw)
        )
    errors.extend(bare_errors)

    return calls, errors


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Back-compat wrapper: returns just the list of successfully parsed calls."""
    calls, _errors = parse_tool_calls_with_errors(text)
    return calls


def has_tool_calls(text: str) -> bool:
    return bool(_TOOL_CALL_RE.search(text))


def strip_tool_calls(text: str) -> str:
    return _TOOL_CALL_RE.sub("", text).strip()


# ── Registry ────────────────────────────────────────────────────


class ToolRegistry:
    """
    Registry of agent tools.

    Responsibilities:
    - Tool registration (register / unregister / get)
    - Hermes-format system-prompt block generation
    - Tool-call parsing (delegates to parse_tool_calls)
    - Tool execution with structured error surfacing
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSchema] = {}
        # G-STEP gate (arXiv 2605.00136). Default-off — set via
        # configure_gate(...) when the Agent wants the gate engaged.
        # State is per-registry instance; the Agent constructs one
        # registry per task run.
        self._gate_config = None  # type: ignore[assignment]
        self._gate_state = None  # type: ignore[assignment]
        # PostToolUse-style sanitizer (Task 5, 2026-05-19). Default-on,
        # narrow scope (lone surrogate strip + control-char escape +
        # 30k-char soft cap). Per-registry instance; opt out by setting
        # self._sanitizer_config.enabled = False.
        from engine.post_tool_sanitize import SanitizerConfig
        self._sanitizer_config = SanitizerConfig()

    def configure_gate(self, config, goal_text: str = "") -> None:
        """Engage the G-STEP confidence gate for this registry.

        Pass a populated ``ToolGateConfig``; the gate stays a no-op
        until ``config.enabled`` is True. ``goal_text`` seeds the
        anchor corpus so the very first tool call can match against
        the goal (otherwise the corpus would be empty until the first
        observation arrives).
        """
        from engine.tool_gate import GateState
        self._gate_config = config
        self._gate_state = GateState()
        if goal_text:
            self._gate_state.record_observation(goal_text)

    # ── registration ──
    def register(self, schema: ToolSchema) -> None:
        self._tools[schema.name] = schema
        logger.debug("Registered agent tool: %s (%s)", schema.name, schema.category)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolSchema]:
        return self._tools.get(name)

    def enabled_tools(self) -> list[ToolSchema]:
        return [t for t in self._tools.values() if t.enabled]

    # ── prompt generation ──
    def hermes_system_block(self) -> str:
        """
        Build the exact Qwen 2.5 / Hermes tool-use system-prompt block.

        Qwen 2.5's chat template emits tool calls in this format natively —
        matching the format the model was trained on is what unlocks reliable
        tool calling from the 14B without custom parsing gymnastics.
        """
        tools = self.enabled_tools()
        if not tools:
            return ""
        tool_lines = [json.dumps(t.to_hermes_dict(), ensure_ascii=False) for t in tools]
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            + "\n".join(tool_lines)
            + "\n</tools>\n\n"
            "For each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    # ── parsing / execution ──
    def parse(self, text: str) -> list[ToolCall]:
        return parse_tool_calls(text)

    def parse_with_errors(self, text: str) -> tuple[list[ToolCall], list[str]]:
        """Parse *text* and also return a list of human-readable parse errors."""
        return parse_tool_calls_with_errors(text)

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                name=call.name,
                success=False,
                error=f"Unknown tool: {call.name!r}. Available: {sorted(self._tools)}",
            )
        if not tool.enabled:
            return ToolResult(
                name=call.name, success=False, error=f"Tool {call.name!r} is disabled"
            )
        # G-STEP confidence gate (arXiv 2605.00136). Suppress tool
        # calls whose expected execution gain is unlikely to clear the
        # protocol-overhead tax — argument has no lexical anchor in the
        # goal or prior observations, or the agent is in a doom loop
        # of consecutive exploration calls. Opt-in: the gate is a
        # no-op unless the registry is given a populated GateConfig.
        if self._gate_config is not None and self._gate_config.enabled:
            from engine.tool_gate import check as _gate_check
            decision = _gate_check(
                call.name, call.arguments, self._gate_state, self._gate_config
            )
            if not decision.allow:
                return ToolResult(
                    name=call.name,
                    success=False,
                    error=decision.rejection_message(call.name),
                )
            self._gate_state.record_call(call.name)
        # Pre-dispatch JSON Schema validation (arXiv 2604.26091 typed
        # controls). Reject malformed argument shapes BEFORE running the
        # function, so the model gets a clean structured rejection
        # instead of a Python TypeError stack frame. The function never
        # executes on a rejected call — important for any tool that
        # mutates state.
        from engine.tool_validator import format_rejection, validate_arguments
        schema_errors = validate_arguments(call.arguments, tool.parameters)
        if schema_errors:
            return ToolResult(
                name=call.name,
                success=False,
                error=format_rejection(call.name, schema_errors),
            )
        try:
            result = tool.function(**call.arguments)
            # PostToolUse sanitization (Task 5, 2026-05-19). Cleans lone
            # surrogates + control chars and soft-truncates oversized
            # content before the result gets serialized for the next
            # model turn. Idempotent and skipped for tools in the
            # opt-out set (e.g. auto_verify). See engine/post_tool_sanitize.py.
            if self._sanitizer_config is not None and self._sanitizer_config.enabled:
                from engine.post_tool_sanitize import sanitize_tool_result
                result = sanitize_tool_result(
                    call.name, result, self._sanitizer_config
                )
            return ToolResult(name=call.name, success=True, content=result)
        except TypeError as exc:
            # Signature mismatch the validator missed (e.g. unknown kwarg
            # passed through additionalProperties=True). Surface cleanly
            # so the model can retry with the correct shape.
            return ToolResult(
                name=call.name, success=False, error=f"Bad arguments: {exc}"
            )
        except (FileNotFoundError, ValueError) as exc:
            # Expected validation errors — the model should see them and retry,
            # not traceback-log them as bugs.
            logger.debug("Agent tool %s returned validation error: %s", call.name, exc)
            return ToolResult(name=call.name, success=False, error=str(exc))
        except Exception as exc:
            logger.exception("Agent tool %s raised", call.name)
            return ToolResult(name=call.name, success=False, error=str(exc))

    def execute_text(self, text: str) -> tuple[list[ToolResult], str]:
        """Parse *text*, execute every call, return (results, text-with-calls-stripped)."""
        calls = self.parse(text)
        results = [self.execute(c) for c in calls]
        # Feed successful tool results back into the G-STEP anchor
        # corpus so subsequent calls can lexically match against
        # observed paths/symbols.
        if self._gate_state is not None:
            for r in results:
                if r.success and r.content:
                    self._gate_state.record_observation(str(r.content))
        return results, strip_tool_calls(text)

    # ── introspection ──
    def status(self) -> dict:
        enabled = self.enabled_tools()
        return {
            "total": len(self._tools),
            "enabled": len(enabled),
            "tools": [t.name for t in enabled],
        }


# ── Smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Parser handles the exact shape Qwen 2.5 emits, including multi-line
    # arguments and multiple calls in one response.
    sample = """\
Let me read that file first.

<tool_call>
{"name": "read_file", "arguments": {"path": "README.md"}}
</tool_call>

And also grep for TODO markers:

<tool_call>
{"name": "grep", "arguments": {"pattern": "TODO", "path": "."}}
</tool_call>

Then I'll be ready.
"""
    calls = parse_tool_calls(sample)
    assert len(calls) == 2, f"expected 2 calls, got {len(calls)}"
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "README.md"}
    assert calls[1].name == "grep"
    assert calls[1].arguments == {"pattern": "TODO", "path": "."}
    assert has_tool_calls(sample)
    stripped = strip_tool_calls(sample)
    assert "<tool_call>" not in stripped
    assert "Let me read that file first." in stripped

    # Registry round-trip
    reg = ToolRegistry()
    assert reg.status() == {"total": 0, "enabled": 0, "tools": []}

    def _echo(message: str) -> str:
        return f"echo: {message}"

    reg.register(
        ToolSchema(
            name="echo",
            description="Echo a message back. Test tool.",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string", "description": "text to echo"}},
                "required": ["message"],
            },
            function=_echo,
            category="test",
        )
    )
    assert reg.status()["total"] == 1
    assert "echo" in reg.hermes_system_block()
    assert '"type": "function"' in reg.hermes_system_block()

    # Round-trip: model emits a call, we parse + execute it
    model_output = (
        '<tool_call>{"name": "echo", "arguments": {"message": "hi"}}</tool_call>'
    )
    results, _cleaned = reg.execute_text(model_output)
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].content == "echo: hi"
    assert json.loads(results[0].format_for_model())["content"] == "echo: hi"

    # Error path: unknown tool
    bad = '<tool_call>{"name": "nope", "arguments": {}}</tool_call>'
    results, _ = reg.execute_text(bad)
    assert len(results) == 1
    assert results[0].success is False
    assert "Unknown tool" in (results[0].error or "")

    # Error path: bad arguments
    bad_args = '<tool_call>{"name": "echo", "arguments": {"wrong": "x"}}</tool_call>'
    results, _ = reg.execute_text(bad_args)
    assert results[0].success is False
    assert "Bad arguments" in (results[0].error or "")

    # Error path: malformed JSON body is skipped, not raised
    malformed = "<tool_call>{bad json}</tool_call>"
    assert parse_tool_calls(malformed) == []

    # Robustness: literal newlines inside string values are tolerated
    # (small models emit this for multi-line write_file content)
    multiline = (
        '<tool_call>\n'
        '{"name": "write_file", "arguments": {"path": "x.py", "content": "def foo():\n    return 1\n"}}\n'
        '</tool_call>'
    )
    parsed = parse_tool_calls(multiline)
    assert len(parsed) == 1, f"expected 1 call, got {len(parsed)}"
    assert parsed[0].name == "write_file"
    assert parsed[0].arguments["path"] == "x.py"
    assert "def foo():" in parsed[0].arguments["content"]
    assert "return 1" in parsed[0].arguments["content"]

    # Robustness: ```json fenced tool calls are accepted as a fallback
    # (observed Qwen 2.5 14B emit this instead of Hermes tags in stress test)
    fenced = '''Here's the read.
```json
{"name": "read_file", "arguments": {"path": "calculator.py"}}
```
'''
    parsed = parse_tool_calls(fenced)
    assert len(parsed) == 1, f"expected fenced json parse, got {len(parsed)}"
    assert parsed[0].name == "read_file"
    assert parsed[0].arguments == {"path": "calculator.py"}

    # Fenced JSON without name+arguments is NOT treated as a tool call
    not_a_call = '```json\n{"status": "done", "result": 42}\n```'
    assert parse_tool_calls(not_a_call) == []

    # If Hermes tags are present, fenced JSON is ignored (no double-counting)
    mixed = (
        '```json\n{"name":"ignored","arguments":{}}\n```\n'
        '<tool_call>{"name":"real","arguments":{"x":1}}</tool_call>'
    )
    parsed = parse_tool_calls(mixed)
    assert len(parsed) == 1
    assert parsed[0].name == "real"

    # Tier 3: bare JSON tool calls (no tags, no fences) — observed from
    # Qwen 2.5 14B in the Phase 14 stress test
    bare = 'I will read the file.\n{"name": "read_file", "arguments": {"path": "calculator.py"}}\nDone.'
    parsed = parse_tool_calls(bare)
    assert len(parsed) == 1, f"expected 1 bare call, got {len(parsed)}"
    assert parsed[0].name == "read_file"
    assert parsed[0].arguments == {"path": "calculator.py"}

    # Bare JSON without name+arguments is NOT accepted as a tool call
    bare_summary = 'Here is the result: {"status": "complete", "count": 5}'
    assert parse_tool_calls(bare_summary) == []

    # Two bare tool calls in one response
    two_bare = (
        '{"name": "read_file", "arguments": {"path": "a.py"}}\n'
        'and then\n'
        '{"name": "grep", "arguments": {"pattern": "TODO", "path": "."}}'
    )
    parsed = parse_tool_calls(two_bare)
    assert len(parsed) == 2
    assert parsed[0].name == "read_file"
    assert parsed[1].name == "grep"

    # Invalid JSON escape repair — model emits \@dataclass which is not
    # a valid JSON escape. Parser should auto-repair and accept.
    # (Real failure mode from Phase 14 bench_full build_todo_cli attempt.)
    with_bad_escape = (
        '```json\n'
        '{"name": "write_file", "arguments": {"path": "todo.py", '
        '"content": "from dataclasses import dataclass\\n\\@dataclass\\nclass Todo:\\n    id: int"}}\n'
        '```'
    )
    parsed = parse_tool_calls(with_bad_escape)
    assert len(parsed) == 1, f"expected repair to succeed, got {len(parsed)}"
    assert parsed[0].name == "write_file"
    assert "@dataclass" in parsed[0].arguments["content"]
    assert "\\@" not in parsed[0].arguments["content"]  # backslash stripped

    # Bare JSON with Python-style single-quoted inner string (real failure
    # mode from Phase 14 iter1s build_todo_cli: model emits a new_string
    # wrapped in '...' with embedded \' escapes). As of iter1s, the bare
    # tier has an ast.literal_eval fallback that parses this as a Python
    # dict literal, so the call is recovered instead of surfacing a parse
    # error. This is the intended upgrade — accepting what the model
    # actually emits instead of forcing it to re-emit.
    bare_py_quoted = (
        'I will add a docstring.  '
        '{"name": "edit_file", "arguments": {"path": "geometry.py", '
        '"old_string": "def area_of_circle(radius):", '
        '"new_string": \'"""Calculate area of a circle."""\'}}'
    )
    calls, errors = parse_tool_calls_with_errors(bare_py_quoted)
    assert len(calls) == 1, f"expected literal_eval fallback to recover call, got {calls}"
    assert calls[0].name == "edit_file"
    assert calls[0].arguments["path"] == "geometry.py"
    assert '"""Calculate area of a circle."""' == calls[0].arguments["new_string"]
    assert errors == [], f"no errors expected with successful fallback, got {errors}"

    # Parse errors: malformed JSON inside a fence that clearly LOOKS like
    # a tool call (has "name" and "arguments") must surface as an error
    malformed_fence = (
        '```json\n'
        '{"name": "edit_file", "arguments": {"new_string": f"oops python fstring"}}\n'
        '```'
    )
    calls, errors = parse_tool_calls_with_errors(malformed_fence)
    assert calls == [], calls
    assert len(errors) == 1, errors
    assert "failed to parse" in errors[0].lower() or "malformed" in errors[0].lower()

    # Mixed: one valid fenced call + one malformed one → 1 call + 1 error
    mixed_good_bad = (
        '```json\n{"name": "run_tests", "arguments": {"runner": "pytest"}}\n```\n'
        '```json\n{"name": "edit_file", "arguments": {"new": f"bad"}}\n```'
    )
    calls, errors = parse_tool_calls_with_errors(mixed_good_bad)
    assert len(calls) == 1
    assert calls[0].name == "run_tests"
    assert len(errors) == 1

    # Non-tool-call JSON fences (no "name"/"arguments") don't become errors
    doc_fence = '```json\n{"status": "done", "count": 42}\n```'
    calls, errors = parse_tool_calls_with_errors(doc_fence)
    assert calls == []
    assert errors == []

    # R1-distill <think>...</think> reasoning block is stripped before
    # tool-call parsing — the model's private reasoning should not trigger
    # tool execution, and tool calls emitted AFTER the block must still be
    # found.
    r1_response = (
        "<think>\n"
        "Let me plan this. First I need to read foo.py.\n"
        "Then I'll call edit_file on foo.py to add the decorator.\n"
        "I should NOT execute the tool_call markup inside this thinking.\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "IGNORE_ME.py"}}</tool_call>\n'
        "</think>\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "foo.py"}}</tool_call>'
    )
    calls = parse_tool_calls(r1_response)
    assert len(calls) == 1, f"expected 1 post-think call, got {calls}"
    assert calls[0].arguments["path"] == "foo.py", calls[0].arguments
    # Multiple <think> blocks + tool call after the last one
    r1_multi = (
        "<think>first plan</think>some prose<think>second plan</think>\n"
        '<tool_call>{"name": "grep", "arguments": {"pattern": "TODO", "path": "."}}</tool_call>'
    )
    calls = parse_tool_calls(r1_multi)
    assert len(calls) == 1 and calls[0].name == "grep", calls
    # Unclosed <think> at end (model truncated mid-thought) — everything from
    # the open tag to EOS is dropped.
    r1_unclosed = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "real.py"}}</tool_call>\n'
        "<think>I was in the middle of thinking when the context ran out"
    )
    calls = parse_tool_calls(r1_unclosed)
    assert len(calls) == 1 and calls[0].arguments["path"] == "real.py", calls

    print("OK: engine/agent_tools.py parser + registry smoke test passed")
