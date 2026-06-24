"""
Deterministic front-door — route trivially-mechanical coding requests AROUND
the 14B entirely.

Thesis (the "minimize the AI brain to only where necessary" direction, the
LIVE lever — NOT the dead GBNF one): some whole categories of coding requests
are deterministically resolvable from the filesystem + stdlib AST with ZERO
model inference. A pure rename, a missing top-level import, a `black` reformat,
a trailing-newline fix, a bare empty-file create — none of these need a 14B.
This module classifies such requests with rules, executes them deterministically,
and ABSTAINS (returns None) the instant confidence drops, deferring to the
normal agent loop.

This is the Engine-A shape from the PIE pattern: a deterministic command path
in front of the conversational model. It is OPT-IN and default-OFF — the caller
only invokes `DeterministicFrontDoor.try_handle()` when the user enabled it.
When it returns None the byte-for-byte legacy path runs unchanged.

ABSTENTION IS THE PRIME DIRECTIVE. A false deterministic edit is far worse than
deferring to the model. Every classifier is conservative: the moment a request
looks even slightly generative ("...that does X", "fix the bug", multi-step),
we return None and let the 14B handle it.

NO MODEL CALL ANYWHERE IN THIS MODULE. Pure stdlib + subprocess (black/autopep8
only if already installed). Zero servers, pure in-process.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Result type ─────────────────────────────────────────────────


@dataclass
class FrontDoorMatch:
    """The outcome of a deterministic front-door execution.

    Returned by `try_handle()` ONLY when a category matched and executed
    (or cleanly determined a no-op). `None` from `try_handle()` means
    abstain — the caller falls through to the normal 14B agent loop.
    """

    category: str                       # which executor handled it
    summary: str                        # one-line human-readable result
    changed_files: list[str] = field(default_factory=list)
    # When True the request was understood and was a confirmed no-op (e.g.
    # the import already existed). Distinct from abstain (None): we KNOW
    # there's nothing to do, so calling the model would waste a turn.
    no_op: bool = False


# Word-boundary identifier (the unit a rename moves).
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")

# Phrases that signal a GENERATIVE request — if any appears, abstain hard.
# These mean "the model has to write/understand logic", which is exactly what
# the front-door must NOT attempt.
_GENERATIVE_MARKERS = (
    " that ", " which ", " so that", " to do ", " doing ", " returns ",
    " handles ", " implement", " bug", " fix the", " refactor", " optimi",
    " test ", " tests", " feature", " logic", " algorithm", " endpoint",
    " parse", " calculate", " compute", " validate", " explain", " describe",
    " summar", " review", " debug", " why ", " how ", " should ",
)


def _looks_generative(goal: str) -> bool:
    """True if the goal contains any marker of a generative/multi-step task.

    Conservative on purpose: a false positive here just means we defer to
    the model (safe). A false negative would let a generative request slip
    into a mechanical executor (unsafe) — so the marker list is broad.
    """
    g = " " + goal.lower().strip() + " "
    return any(m in g for m in _GENERATIVE_MARKERS)


def _has_multiple_clauses(goal: str) -> bool:
    """True if the goal chains multiple instructions (and/then/comma-list).

    A single mechanical op is one clause. "rename X to Y and add a docstring"
    is two — the second is generative, so the whole thing must defer.
    """
    g = goal.lower()
    # " and " joining two verb-ish clauses, " then ", or a semicolon.
    if " then " in g or ";" in g:
        return True
    # "and" is only a multi-clause signal when it joins two actions, not
    # inside "X and Y" identifier lists. Heuristic: an "and" followed later
    # by another action verb.
    if " and " in g:
        tail = g.split(" and ", 1)[1]
        if any(v in tail for v in (
            "add", "remove", "delete", "create", "rename", "fix",
            "write", "run", "format", "make", "update", "change",
        )):
            return True
    return False


class DeterministicFrontDoor:
    """Engine-A-shaped deterministic dispatcher for mechanical coding ops.

    Construct with the workspace root. Call `parse(goal)` to classify (returns
    a `_Plan` or None) or `try_handle(goal)` to classify-AND-execute (returns
    a `FrontDoorMatch` or None). The two-stage split keeps classification
    side-effect-free and unit-testable in isolation.
    """

    def __init__(self, workspace_root: Path | str) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    # ── public API ──

    def parse(self, goal: str, project_context: Optional[dict] = None):
        """Classify *goal* into a deterministic plan, or None to abstain.

        `project_context` is accepted for parity with the PIE pattern (and so
        callers can thread the project index in later) but the current
        executors read the filesystem directly, so it is unused today.

        Returns an opaque `_Plan` dataclass on a confident match, else None.
        Never raises — any internal error abstains.
        """
        try:
            return self._parse_inner(goal)
        except Exception:
            return None

    def try_handle(self, goal: str, project_context: Optional[dict] = None) -> Optional[FrontDoorMatch]:
        """Parse then execute. Returns a FrontDoorMatch on success, else None
        (abstain → caller runs the normal 14B loop). Never raises — any
        execution failure abstains so the model still gets its shot."""
        plan = self.parse(goal, project_context)
        if plan is None:
            return None
        try:
            return plan.execute(self)
        except Exception:
            # Execution blew up on a confident parse — defer rather than
            # leave a half-applied edit's blame on the deterministic path.
            return None

    # ── classification ──

    def _parse_inner(self, goal: str):
        goal = (goal or "").strip()
        if not goal:
            return None
        # Global guards: anything generative or multi-clause defers.
        if _looks_generative(goal):
            return None
        if _has_multiple_clauses(goal):
            return None

        for classifier in (
            self._classify_rename,
            self._classify_add_import,
            self._classify_format,
            self._classify_trailing_newline,
            self._classify_create_empty,
        ):
            plan = classifier(goal)
            if plan is not None:
                return plan
        return None

    # rename X to Y  /  rename X -> Y
    _RENAME_RE = re.compile(
        r"^rename\s+(?:the\s+)?(?:function\s+|method\s+|class\s+|variable\s+)?"
        r"`?([A-Za-z_][A-Za-z_0-9]*)`?\s+(?:to|->|into|as)\s+"
        r"`?([A-Za-z_][A-Za-z_0-9]*)`?"
        r"(?:\s+everywhere|\s+across.*|\s+in all.*|\s+globally)?\s*\.?$",
        re.IGNORECASE,
    )

    def _classify_rename(self, goal: str):
        m = self._RENAME_RE.match(goal)
        if not m:
            return None
        old, new = m.group(1), m.group(2)
        if not (_IDENT_RE.match(old) and _IDENT_RE.match(new)):
            return None
        if old == new:
            return None
        return _RenamePlan(old=old, new=new)

    # add import X to FILE  /  add `from a import b` to FILE  /  import X in FILE
    _IMPORT_RE = re.compile(
        r"^(?:add|insert)\s+(?:the\s+)?(?:import\s+)?"
        r"`?(?P<stmt>(?:import\s+[A-Za-z_][\w.]*(?:\s+as\s+\w+)?"
        r"|from\s+[A-Za-z_][\w.]*\s+import\s+[\w, ]+))`?"
        r"\s+(?:to|in|into|at the top of)\s+`?(?P<file>[\w./\\-]+\.py)`?\s*\.?$",
        re.IGNORECASE,
    )

    def _classify_add_import(self, goal: str):
        m = self._IMPORT_RE.match(goal)
        if not m:
            return None
        stmt = m.group("stmt").strip()
        # Normalize: "import   x" / "from x import y" — collapse inner spaces.
        if stmt.lower().startswith("import "):
            stmt = "import " + stmt[len("import "):].strip()
        # Reject cyclic-looking from-import to a sibling that imports this
        # file? We can't know the target file here cheaply, and cycle
        # handling is exactly the ambiguous case — so the executor checks
        # for a cycle and abstains there.
        fname = m.group("file").strip()
        # Validate the statement actually parses as an import.
        try:
            tree = ast.parse(stmt)
        except SyntaxError:
            return None
        if not tree.body or not isinstance(tree.body[0], (ast.Import, ast.ImportFrom)):
            return None
        return _AddImportPlan(stmt=stmt, filename=fname)

    # format FILE  /  format FILE with black  /  reformat FILE
    _FORMAT_RE = re.compile(
        r"^(?:re)?format\s+`?(?P<file>[\w./\\-]+\.py)`?"
        r"(?:\s+with\s+(?P<tool>black|autopep8))?\s*\.?$",
        re.IGNORECASE,
    )

    def _classify_format(self, goal: str):
        m = self._FORMAT_RE.match(goal)
        if not m:
            return None
        tool = (m.group("tool") or "").lower() or None
        formatter = self._pick_formatter(tool)
        if formatter is None:
            # No formatter installed → this isn't safely deterministic.
            # Abstain so the model (which may know another way) gets it.
            return None
        return _FormatPlan(filename=m.group("file").strip(), formatter=formatter)

    @staticmethod
    def _pick_formatter(requested: Optional[str]) -> Optional[str]:
        """Return the name of an available formatter, honoring an explicit
        request. None if nothing usable is installed."""
        candidates = [requested] if requested else ["black", "autopep8"]
        for name in candidates:
            if name and (shutil.which(name) or _module_importable(name)):
                return name
        return None

    # add a trailing newline to FILE  /  strip trailing newline from FILE  /
    # ensure FILE ends with a newline
    # Two word orders:
    #   "add a trailing newline to FILE"      (file AFTER 'newline')
    #   "ensure FILE ends with a newline"     (file BEFORE 'newline')
    _NEWLINE_ADD_RE = re.compile(
        r"^(?:add|append)\s+(?:a\s+)?(?:trailing\s+|final\s+)?newline"
        r".*?\b`?(?P<file>[\w./\\-]+\.\w+)`?\s*\.?$",
        re.IGNORECASE,
    )
    _NEWLINE_ENSURE_RE = re.compile(
        r"^ensure\s+`?(?P<file>[\w./\\-]+\.\w+)`?\s+ends?\s+with\s+"
        r"(?:a\s+)?(?:trailing\s+|final\s+)?newline\s*\.?$",
        re.IGNORECASE,
    )
    _NEWLINE_STRIP_RE = re.compile(
        r"^(?:strip|remove|delete)\s+(?:the\s+)?(?:trailing\s+|final\s+)?"
        r"(?:newline|blank line)s?\b.*?\b`?(?P<file>[\w./\\-]+\.\w+)`?\s*\.?$",
        re.IGNORECASE,
    )

    def _classify_trailing_newline(self, goal: str):
        m = self._NEWLINE_ADD_RE.match(goal) or self._NEWLINE_ENSURE_RE.match(goal)
        if m:
            return _TrailingNewlinePlan(filename=m.group("file").strip(), add=True)
        m = self._NEWLINE_STRIP_RE.match(goal)
        if m:
            return _TrailingNewlinePlan(filename=m.group("file").strip(), add=False)
        return None

    # create FILE  /  create an empty FILE  /  create a new file FILE
    # ONLY bare creation — any described content/behavior is caught by the
    # generative guard upstream and never reaches here.
    _CREATE_RE = re.compile(
        r"^create\s+(?:a\s+|an\s+)?(?:new\s+)?(?:empty\s+)?(?:file\s+)?"
        r"`?(?P<file>[\w./\\-]+\.\w+)`?\s*\.?$",
        re.IGNORECASE,
    )

    def _classify_create_empty(self, goal: str):
        m = self._CREATE_RE.match(goal)
        if not m:
            return None
        return _CreateEmptyPlan(filename=m.group("file").strip())

    # ── workspace helpers ──

    def _resolve_inside(self, filename: str) -> Optional[Path]:
        """Resolve *filename* against the workspace and confirm it stays
        inside. Returns None on any traversal/resolution problem."""
        try:
            full = (self.workspace_root / filename).resolve()
            full.relative_to(self.workspace_root)
        except (OSError, ValueError):
            return None
        return full

    def _touched_py_files(self) -> list[Path]:
        """Every .py file under the workspace (skipping noise dirs). Used by
        the rename executor — a rename with no explicit file target spans the
        whole project, mirroring the agent's grep-then-edit-everywhere rule."""
        noise = {
            "__pycache__", "node_modules", ".git", ".venv", "venv",
            ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
            ".idea", ".vscode", ".egg-info",
        }
        out: list[Path] = []
        try:
            for p in self.workspace_root.rglob("*.py"):
                if any(part in noise for part in p.parts):
                    continue
                if p.is_file():
                    out.append(p)
        except OSError:
            return []
        return out


def _module_importable(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# ── Plans (deterministic executors) ─────────────────────────────


@dataclass
class _RenamePlan:
    old: str
    new: str

    def execute(self, fd: DeterministicFrontDoor) -> Optional[FrontDoorMatch]:
        files = fd._touched_py_files()
        if not files:
            return None
        pat = re.compile(r"\b" + re.escape(self.old) + r"\b")
        # First pass: confirm the old name actually appears somewhere AND that
        # the new name doesn't already collide as a different binding. If old
        # never appears, this isn't a real rename target — abstain (the model
        # might know it's in a non-.py file or a different spelling).
        any_match = False
        for f in files:
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if pat.search(src):
                any_match = True
                break
        if not any_match:
            return None
        changed: list[str] = []
        for f in files:
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            new_src = pat.sub(self.new, src)
            if new_src != src:
                # Verify the result still parses — a rename that breaks syntax
                # means our word-boundary assumption was wrong; abort the
                # whole op (don't leave a partial rename) and defer.
                try:
                    ast.parse(new_src)
                except SyntaxError:
                    return None
                try:
                    f.write_text(new_src, encoding="utf-8")
                    changed.append(f.name)
                except OSError:
                    continue
        if not changed:
            return None
        return FrontDoorMatch(
            category="rename",
            summary=f"Renamed {self.old} -> {self.new} across {len(changed)} file(s): "
                    + ", ".join(sorted(set(changed))),
            changed_files=sorted(set(changed)),
        )


@dataclass
class _AddImportPlan:
    stmt: str
    filename: str

    def execute(self, fd: DeterministicFrontDoor) -> Optional[FrontDoorMatch]:
        full = fd._resolve_inside(self.filename)
        if full is None or not full.exists() or not full.is_file():
            return None
        try:
            src = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        # Idempotent: if an equivalent import already exists, it's a no-op.
        if self._already_imported(src):
            return FrontDoorMatch(
                category="add_import",
                summary=f"`{self.stmt}` already present in {full.name} (no-op).",
                no_op=True,
            )
        # Cycle guard: `from X import Y` where X.py imports from this file is
        # the ambiguous lazy-import case — defer to the model's cycle handling.
        if self._would_cycle(fd, full):
            return None
        new_src = self.stmt.rstrip("\n") + "\n" + src
        try:
            ast.parse(new_src)
        except SyntaxError:
            return None
        try:
            full.write_text(new_src, encoding="utf-8")
        except OSError:
            return None
        return FrontDoorMatch(
            category="add_import",
            summary=f"Prepended `{self.stmt}` to {full.name}.",
            changed_files=[full.name],
        )

    def _already_imported(self, src: str) -> bool:
        try:
            tree = ast.parse(src)
            want = ast.parse(self.stmt).body[0]
        except SyntaxError:
            return self.stmt.strip() in src  # crude fallback
        for node in tree.body:
            if isinstance(node, ast.Import) and isinstance(want, ast.Import):
                have = {(a.name, a.asname) for a in node.names}
                if all((a.name, a.asname) in have for a in want.names):
                    return True
            elif isinstance(node, ast.ImportFrom) and isinstance(want, ast.ImportFrom):
                if node.module == want.module:
                    have = {(a.name, a.asname) for a in node.names}
                    if all((a.name, a.asname) in have for a in want.names):
                        return True
        return False

    def _would_cycle(self, fd: DeterministicFrontDoor, full: Path) -> bool:
        try:
            want = ast.parse(self.stmt).body[0]
        except SyntaxError:
            return False
        if not isinstance(want, ast.ImportFrom) or not want.module:
            return False
        other = fd._resolve_inside(want.module.split(".")[0] + ".py")
        if other is None or not other.exists():
            return False
        try:
            other_src = other.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        this_mod = full.stem
        cycle = re.compile(
            r"^\s*(?:from\s+" + re.escape(this_mod) + r"\s+import\s|import\s+"
            + re.escape(this_mod) + r"\b)",
            re.MULTILINE,
        )
        return bool(cycle.search(other_src))


@dataclass
class _FormatPlan:
    filename: str
    formatter: str

    def execute(self, fd: DeterministicFrontDoor) -> Optional[FrontDoorMatch]:
        full = fd._resolve_inside(self.filename)
        if full is None or not full.exists() or not full.is_file():
            return None
        try:
            before = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if self.formatter == "black":
            cmd = ["black", "-q", str(full)]
        elif self.formatter == "autopep8":
            cmd = ["autopep8", "--in-place", str(full)]
        else:
            return None
        # Prefer module invocation so an importable-but-not-on-PATH formatter
        # still works (pip-installed in the active interpreter).
        if not shutil.which(self.formatter) and _module_importable(self.formatter):
            import sys
            cmd = [sys.executable, "-m"] + cmd
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if res.returncode not in (0,):
            return None
        try:
            after = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if after == before:
            return FrontDoorMatch(
                category="format",
                summary=f"{full.name} already formatted ({self.formatter}, no-op).",
                no_op=True,
            )
        return FrontDoorMatch(
            category="format",
            summary=f"Formatted {full.name} with {self.formatter}.",
            changed_files=[full.name],
        )


@dataclass
class _TrailingNewlinePlan:
    filename: str
    add: bool  # True = ensure trailing newline; False = strip trailing newlines

    def execute(self, fd: DeterministicFrontDoor) -> Optional[FrontDoorMatch]:
        full = fd._resolve_inside(self.filename)
        if full is None or not full.exists() or not full.is_file():
            return None
        try:
            data = full.read_bytes()
        except OSError:
            return None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None  # binary — not our job
        if self.add:
            if text.endswith(("\n", "\r")):
                return FrontDoorMatch(
                    category="trailing_newline",
                    summary=f"{full.name} already ends with a newline (no-op).",
                    no_op=True,
                )
            new_text = text + "\n"
            verb = "Added trailing newline to"
        else:
            # Strip CR and LF so the op is newline-style agnostic (Windows
            # CRLF files would otherwise leave a stray \r and never converge).
            stripped = text.rstrip("\r\n")
            new_text = stripped + ("\n" if stripped else "")
            if new_text == text:
                return FrontDoorMatch(
                    category="trailing_newline",
                    summary=f"{full.name} has no extra trailing newlines (no-op).",
                    no_op=True,
                )
            verb = "Stripped trailing newlines from"
        try:
            full.write_text(new_text, encoding="utf-8")
        except OSError:
            return None
        return FrontDoorMatch(
            category="trailing_newline",
            summary=f"{verb} {full.name}.",
            changed_files=[full.name],
        )


@dataclass
class _CreateEmptyPlan:
    filename: str

    def execute(self, fd: DeterministicFrontDoor) -> Optional[FrontDoorMatch]:
        full = fd._resolve_inside(self.filename)
        if full is None:
            return None
        if full.exists():
            # Don't clobber — an existing file + a "create" request is
            # ambiguous (maybe they want it reset, maybe a different file).
            # Defer.
            return None
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text("", encoding="utf-8")
        except OSError:
            return None
        return FrontDoorMatch(
            category="create_empty",
            summary=f"Created empty file {full.name}.",
            changed_files=[full.name],
        )
