"""Self-healing diagnose-and-repair pass for the agent loop.

Adapted from arXiv 2604.27096 ("Think it, Run it: autonomous ML pipeline
generation via self-healing multi-agent AI"). Pattern: when a tool call
fails twice in a row with similar shape, the agent is asked to step out
of execute mode and write a brief diagnosis + concrete repair plan
before its next attempt — instead of letting the failure thrash the
loop until `stuck_repeat` fires on the third identical call.

The classifier is intentionally narrow: it only fires on clear,
recurring failure shapes (Traceback, SyntaxError, old_string not found,
NameError, CWD failures, parse errors). If two consecutive results
share the same shape, the agent gets a synthetic `self_heal_diagnose`
observation telling it to slow down and diagnose. Static prompt — no
LLM call inside the classifier itself, so it costs nothing.

Composes with the existing 12-detector harvest pipeline in
`failure_flagger.py` (post-mortem). This module is the *live* version:
during the loop, before the third attempt, prompt the model to repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Failure classes ──────────────────────────────────────────────
#
# Coarse, pattern-matched. Each maps to a tailored diagnose prompt so
# the model gets a concrete repair direction, not a generic "try again".
SYNTAX_ERROR = "syntax_error"
TRACEBACK = "traceback"
STALE_ANCHOR = "stale_anchor"          # edit_file: old_string not found
NAME_NOT_DEFINED = "name_not_defined"  # NameError / undefined names
MISSING_IMPORT = "missing_import"      # auto_verify: references undefined names
CWD_FAILURE = "cwd_failure"
PARSE_ERROR = "parse_error"            # tool-call JSON parse failed
TOOL_REJECTED = "tool_rejected"        # generic tool error
# Task 8, 2026-05-19: finer-grained Python failure modes lifted out of
# the TRACEBACK catch-all. Each maps to a tailored repair hint so the
# model's next attempt is class-targeted, not generic.
IMPORT_ERROR = "import_error"          # ImportError / ModuleNotFoundError
ASSERTION_FAILURE = "assertion_failure"  # pytest AssertionError / assert X
TYPE_ERROR = "type_error"              # TypeError (wrong arg types / nones)
ATTRIBUTE_ERROR = "attribute_error"    # AttributeError ('X' has no attribute 'Y')
KEY_INDEX_ERROR = "key_index_error"    # KeyError / IndexError on lookup
UNKNOWN = None                         # not a recognized failure shape


@dataclass
class _FailureProbe:
    """Just enough of a ToolResult to classify. Decoupled from the full
    `engine.agent.ToolResult` dataclass so unit tests stay light."""
    name: str
    success: bool
    error: str | None
    output: str | None


def _shape(result) -> _FailureProbe:
    """Extract classifier-relevant fields from a ToolResult-shaped object."""
    return _FailureProbe(
        name=getattr(result, "name", "") or "",
        success=bool(getattr(result, "success", True)),
        error=getattr(result, "error", None),
        output=getattr(result, "output", None),
    )


def classify_failure(result) -> Optional[str]:
    """Return a failure class for the given ToolResult-shaped object, or None.

    Order matters — more specific classes win over generic ones.
    """
    probe = _shape(result)

    # An auto_verify success can still flag undefined names — surface as
    # missing_import even though success=True.
    output_low = (probe.output or "").lower()
    if "references undefined names" in output_low:
        return MISSING_IMPORT

    if probe.success:
        return None

    err = (probe.error or "").strip()
    err_low = err.lower()
    if not err:
        return TOOL_REJECTED

    # Specific shapes first.
    if probe.name == "parse_error" or "tool call" in err_low and "json" in err_low:
        return PARSE_ERROR
    if "syntaxerror" in err_low or "syntax error" in err_low:
        return SYNTAX_ERROR
    if "old_string not found" in err_low or "old_string did not match" in err_low:
        return STALE_ANCHOR
    # ImportError / ModuleNotFoundError — check BEFORE NameError because
    # "ModuleNotFoundError: No module named 'X'" contains "not" + a name
    # but the right repair is "install/import", not "fix the symbol name".
    if (
        "modulenotfounderror" in err_low
        or "importerror" in err_low
        or "no module named" in err_low
    ):
        return IMPORT_ERROR
    if "nameerror" in err_low or "is not defined" in err_low:
        return NAME_NOT_DEFINED
    # AttributeError — check before TypeError because "AttributeError"
    # responses sometimes also embed the offending type name.
    if "attributeerror" in err_low or "has no attribute" in err_low:
        return ATTRIBUTE_ERROR
    if "typeerror" in err_low:
        return TYPE_ERROR
    if "keyerror" in err_low or "indexerror" in err_low:
        return KEY_INDEX_ERROR
    # AssertionError comes from pytest failures. Surface specifically so
    # the repair prompt focuses on the failed expectation, not the
    # general "something raised" hint.
    if "assertionerror" in err_low or "assert " in err_low:
        return ASSERTION_FAILURE
    if "no such file or directory" in err_low or "[winerror 2]" in err_low:
        return CWD_FAILURE
    if "traceback" in err_low or 'file "' in err_low and ", line " in err_low:
        return TRACEBACK

    return TOOL_REJECTED


# ── Streak tracking ──────────────────────────────────────────────


def should_inject_diagnose(streak: list[Optional[str]], min_streak: int = 2) -> bool:
    """Should the next prompt include a diagnose preamble?

    We fire when the LAST two non-None entries in `streak` are the SAME
    failure class. The min_streak parameter lets callers tune for
    chattier-or-stricter behavior; default 2 = after two consecutive
    same-class failures, before the model's next attempt.

    Streak is the running history of `classify_failure` returns from
    the agent loop, oldest-first. None values represent successful
    iterations and reset the run.
    """
    if min_streak < 2:
        return False
    # Walk the tail backwards collecting the most-recent run of identical
    # non-None classes. Stop as soon as we hit a None or a different class.
    run = 0
    last: Optional[str] = None
    for cls in reversed(streak):
        if cls is None:
            break
        if last is None:
            last = cls
            run = 1
        elif cls == last:
            run += 1
        else:
            break
    return last is not None and run >= min_streak


# ── Diagnose prompt construction ─────────────────────────────────

_PER_CLASS_HINT = {
    SYNTAX_ERROR: (
        "The last two attempts wrote code that fails to parse. Read the "
        "file fresh, locate the exact column the parser is complaining "
        "about, and emit a SHORT targeted edit_file fixing only that "
        "syntactic problem. Do NOT rewrite the whole file."
    ),
    TRACEBACK: (
        "The last two attempts produced runtime tracebacks. State which "
        "line in which file is the actual cause (not where the error "
        "surfaced — the cause). Then make a minimal edit_file fix at "
        "the cause line. Do not run tests again until the fix is in."
    ),
    STALE_ANCHOR: (
        "edit_file's old_string did not match twice in a row. The file "
        "likely changed since you last read it OR you copied a "
        "line-number prefix from read_file output. Re-read the target "
        "file fresh, then pick a SHORT unique fragment (~1 line, no "
        "leading whitespace prefix) as your new old_string. If the "
        "fragment looks ambiguous, pick a longer one with surrounding "
        "context."
    ),
    NAME_NOT_DEFINED: (
        "NameError twice in a row — the symbol you're using doesn't "
        "exist in scope. Either: (a) add the missing import at the "
        "top of the file (use edit_file with empty old_string to "
        "PREPEND); or (b) the function/class name has a typo — re-read "
        "the file and grep for the real name."
    ),
    MISSING_IMPORT: (
        "auto_verify keeps reporting undefined names. Add the missing "
        "import using edit_file with empty old_string and the import "
        "line as new_string — that prepends to the top of the file. "
        "Then re-run the failing step."
    ),
    CWD_FAILURE: (
        "A path didn't exist twice in a row. You're probably running "
        "from the wrong directory or the path you're computing has a "
        "typo. List the workspace root with list_dir to confirm what's "
        "actually there before retrying."
    ),
    PARSE_ERROR: (
        "Your tool calls are not parsing as JSON. Two common causes: "
        "(a) you used Python single quotes inside the JSON value — JSON "
        "requires double quotes; (b) you have unescaped double quotes "
        "inside a string. For multi-line content use the ARRAY form "
        "(content as a list of strings, one per line) — it removes "
        "almost all escape errors."
    ),
    TOOL_REJECTED: (
        "Two consecutive tool errors. State in one sentence what's "
        "failing and propose a DIFFERENT approach — different tool, "
        "different arguments, or read fresh state before retrying."
    ),
    IMPORT_ERROR: (
        "ImportError twice in a row — Python can't find the module. "
        "Two paths: (a) the module exists in the workspace but the import "
        "is wrong — grep for the actual file name and adjust the import "
        "(`from foo` -> `from src.foo`, etc.); (b) the module is a third-"
        "party package that isn't installed and CAN'T be installed here — "
        "rewrite the code using stdlib or workspace-local modules instead. "
        "Do not retry the same import path."
    ),
    ASSERTION_FAILURE: (
        "Two consecutive pytest AssertionError failures — the code "
        "computes a value the test rejects. Read the failing test FIRST: "
        "find the exact assertion line and the expected value. Then re-read "
        "the function under test and identify the one line whose output "
        "diverges from expected. Make a MINIMAL edit at that line — do "
        "not rewrite the function. If the test itself looks wrong, say so "
        "in the diagnose step and stop; don't silently change the test."
    ),
    TYPE_ERROR: (
        "TypeError twice in a row — wrong argument shape. Read the call "
        "site AND the function signature. Common causes: (a) None where a "
        "value is expected (add a guard or fix the producer); (b) a list "
        "where a single item is expected (or vice versa — index in or "
        "wrap in a list); (c) wrong arity (missing positional / extra "
        "kwarg). State which of these applies in one line, then make the "
        "smallest edit that fixes it."
    ),
    ATTRIBUTE_ERROR: (
        "AttributeError twice in a row — `obj.X` doesn't exist. Two paths: "
        "(a) you have the wrong object — grep for the class definition and "
        "find the real attribute name (commonly snake/camel case mismatch, "
        "or a renamed field); (b) the object is None at the call site — "
        "trace back to where it was set and fix the producer. Don't "
        "blindly add hasattr() guards — find the real cause."
    ),
    KEY_INDEX_ERROR: (
        "KeyError / IndexError twice in a row — looking up a key or index "
        "that isn't there. State in one line whether the lookup is "
        "(a) a typo (fix the key), (b) a path the producer doesn't always "
        "fill (use .get() with a default, or guard upstream), or (c) an "
        "off-by-one on indices (recompute the bounds). Pick one and make "
        "the smallest edit."
    ),
}


def diagnose_message(failure_class: str, *, attempts: int = 2) -> str:
    """Construct the body of the synthetic `self_heal_diagnose` observation.

    Returns the text the agent will see embedded in its tool-response
    block. Front-loaded with the explicit instruction to PAUSE because
    14B models otherwise skim past advisory text.
    """
    hint = _PER_CLASS_HINT.get(failure_class, _PER_CLASS_HINT[TOOL_REJECTED])
    return (
        f"PAUSE — diagnose-and-repair step. Failure class: {failure_class}. "
        f"You have hit this same failure shape {attempts} times in a row.\n\n"
        "Before your next tool call, write 2-3 short lines:\n"
        "1. What is actually failing (one sentence).\n"
        "2. The single most likely root cause.\n"
        "3. Your concrete next action (must differ from the last two attempts).\n\n"
        f"Class-specific guidance: {hint}\n\n"
        "Then make ONE tool call that implements step 3. Do not retry the "
        "same call with the same arguments — that is what triggered this "
        "step in the first place."
    )
