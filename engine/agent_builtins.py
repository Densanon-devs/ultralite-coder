"""
Phase 14 Agent Builtin Tools.

Implements the 8 core tools the Phase 14 agent loop uses to do real project
work: read_file, write_file, edit_file, list_dir, glob, grep, run_bash,
run_tests. Each is registered into a ToolRegistry via build_default_registry.

Design notes:

- All paths resolve relative to the workspace_root passed to the factory.
  Absolute paths outside the workspace are still allowed (the agent loop
  decides whether to confirm), but relative paths anchor inside it.
- grep prefers `rg` (ripgrep) via subprocess and falls back to a stdlib
  re + os.walk scanner. The fallback is slow on large trees; users who
  care about agent speed should install ripgrep.
- Tool return values are either strings or small dicts — structured enough
  to serialize compactly into the next model turn without blowing context.
- Output size caps are enforced per-tool so a greedy grep or cat -n can't
  fill the context window in one call.

Zero servers, pure in-process, local stdlib + subprocess only.
"""

from __future__ import annotations

import difflib
import fnmatch
import logging
import os
import re
import shutil


def _diff_preview(before: str, after: str, path: str, max_lines: int = 20) -> str:
    """Return a compact unified-diff summary, capped at max_lines. Empty
    string means no diff (content unchanged). Used by write_file/edit_file
    return values so the model sees exactly what changed."""
    if before == after:
        return ""
    diff_iter = difflib.unified_diff(
        before.splitlines(keepends=False),
        after.splitlines(keepends=False),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=2,
        lineterm="",
    )
    lines = list(diff_iter)
    if not lines:
        return ""
    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        lines = lines[:max_lines] + [f"... ({omitted} more diff lines elided)"]
    return "\n".join(lines)
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from engine.agent_memory import AgentMemory
from engine.agent_tools import ToolRegistry, ToolSchema

logger = logging.getLogger(__name__)

# Output caps — keep tool results small enough to feed back into the model
# context without blowing the budget.
_MAX_READ_LINES = 2000
_MAX_READ_LINE_CHARS = 2000
_MAX_LIST_ENTRIES = 500
_MAX_GLOB_RESULTS = 200
_MAX_GREP_MATCHES = 100
_BASH_OUTPUT_CAP = 16000
_TEST_OUTPUT_CAP = 12000
_BASH_TIMEOUT_DEFAULT = 30
_TEST_TIMEOUT_DEFAULT = 300

# Directories we never descend into — standard noise.
_NOISE_DIRS = frozenset({
    "__pycache__", "node_modules", ".git", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".tox", "dist", "build", ".idea", ".vscode",
})

# Detected once at import. grep() uses subprocess rg if set, stdlib re otherwise.
_RG_BIN: Optional[str] = shutil.which("rg")


# ── Workspace guard ─────────────────────────────────────────────


@dataclass
class Workspace:
    """
    Path anchor for agent tools. Resolves relative paths against root.
    """

    root: Path

    def resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        return p.resolve()

    def is_inside(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.root)
            return True
        except ValueError:
            return False


# ── File tools ──────────────────────────────────────────────────


def _read_file(ws: Workspace, path: str, offset: int = 0, limit: int = _MAX_READ_LINES) -> str:
    p = ws.resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)

    offset = max(0, int(offset))
    limit = max(1, min(int(limit), _MAX_READ_LINES))
    selected = lines[offset : offset + limit]

    numbered: list[str] = []
    for i, line in enumerate(selected, start=offset + 1):
        if len(line) > _MAX_READ_LINE_CHARS:
            line = line[:_MAX_READ_LINE_CHARS] + f"  ... [line truncated, {len(line)} chars]"
        numbered.append(f"{i:6d}\t{line}")

    out = "\n".join(numbered)
    if offset + limit < total:
        out += (
            f"\n... [truncated, {total} lines total, "
            f"showing {offset + 1}-{offset + len(selected)}]"
        )
    elif not selected:
        out = f"(empty file: {p})"
    return out


def _coerce_text(val) -> str:
    """Polymorphic content coercion shared by write_file and edit_file.

    - None -> ""
    - list of strings -> joined with "\\n" (the array-form escape hatch
      that sidesteps JSON quote-escape hell on multi-line content)
    - anything else -> str(val)

    Used for write_file's `content` and edit_file's `new_string`/`old_string`
    so the model's "use array form" reflex (taught by self_heal's PARSE_ERROR
    hint) works consistently across both tools.
    """
    if val is None:
        return ""
    if isinstance(val, list):
        lines = [str(x) if x is not None else "" for x in val]
        s = "\n".join(lines)
        if lines and lines[-1] != "":
            if s and not s.endswith("\n"):
                s += "\n"
        return s
    return str(val)


def _write_file(ws: Workspace, path: str, content=None, new_string=None, **_ignored) -> str:
    """Write the whole file.

    Polymorphic `content`:
    - str: written as-is (the normal case)
    - list of strings: joined with "\\n" — escape hatch for long files with
      embedded quotes.

    Tolerance for model confusion:
    - `new_string` kwarg is accepted and, if present, concatenated with
      `content`. The 14B sometimes treats write_file like edit_file and
      emits both `content` (import header) and `new_string` (rest of
      file). Merging them matches the model's intent instead of erroring
      out (Phase 14 iter3e/3f build_todo_cli).
    - `**_ignored` catches any other unexpected kwargs the model invents
      (common for `write_file` + append-related ones) so dispatch doesn't
      hard-error on the registry boundary.
    """
    p = ws.resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    content_str = _coerce_text(content)
    extra = _coerce_text(new_string)
    if extra:
        # Merge content + new_string — the model's mental model of "content
        # goes first, new_string goes after" is the most common intent.
        sep = "" if (not content_str) or content_str.endswith("\n") else "\n"
        content_str = content_str + sep + extra

    if not content_str.endswith("\n") and content_str:
        content_str += "\n"
    before = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    p.write_text(content_str, encoding="utf-8")
    # Preserve the legacy "Wrote N chars to X" prefix — the 14B was trained
    # on it and swaps to a different (worse) codegen strategy when it sees
    # anything else. Only append a diff tail on overwrites with changes.
    msg = f"Wrote {len(content_str)} chars to {p}"
    if before:
        diff = _diff_preview(before, content_str, path)
        if diff:
            msg += f"\n\n{diff}"
    return msg


def _edit_file(
    ws: Workspace,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    # `new_string` must be a plain string (use \\n for newlines), NOT an
    # array. Unlike write_file's `content` — where each array element is a
    # full FILE line — an edit_file `new_string` replaces a fragment that's
    # usually indented, and the model reliably forgets to put the leading
    # indentation in array elements (it thinks "lines of the body", not
    # "lines of the file"), producing unindented code where indented code
    # belongs → IndentationError. So if the model passes an array, raise a
    # clear error directing it to use a string. (Tried accepting arrays
    # here 2026-05-11; it broke all 4 iterative-scaffold tasks with
    # indentation errors. Reverted.)
    if isinstance(new_string, list) or isinstance(old_string, list):
        raise ValueError(
            "edit_file's old_string/new_string must be plain strings (use "
            "\\n for newlines), not arrays. The array-of-lines form is only "
            "for write_file's `content`. For edit_file, write the replacement "
            "as a single string with the correct indentation baked in — e.g. "
            'new_string="    x = 1\\n    return x" (note the 4-space indent on '
            "each line if you're inside a function body)."
        )
    if old_string is None:
        old_string = ""
    if new_string is None:
        new_string = ""

    p = ws.resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = p.read_text(encoding="utf-8", errors="replace")
    if old_string == "":
        # Hard reject empty old_string on structured formats — append/prepend
        # always breaks JSON/YAML/TOML structure. Tell the model to use
        # write_file with the full new document instead. This catches a
        # real Phase 14 failure mode where the model tried to "insert" a
        # top-level field into package.json and got an append-after-closing-
        # brace that broke parsing.
        ext = p.suffix.lower()
        if ext in (".json", ".yaml", ".yml", ".toml"):
            raise ValueError(
                f"edit_file with empty old_string does not work on "
                f"structured-data files (.json/.yaml/.yml/.toml) — "
                f"append/prepend would break the document structure. "
                f"To modify {p.name}, either: (a) read the file, construct "
                f"the full updated document, and call write_file to replace "
                f"it; or (b) pass a non-empty old_string that uniquely "
                f"anchors the insertion point (e.g. the line you want the "
                f"new field to follow)."
            )
        # Empty old_string = INSERT, with position inferred from content:
        #   - Starts with `import`, `from X import`, `#!` shebang, or `"""`
        #     docstring → PREPEND (top-of-file, what imports need)
        #   - Otherwise → APPEND at end of file (what function/class
        #     definitions need — e.g. "add multiply(a,b) and divide(a,b)
        #     to calculator.py" wants append, not prepend)
        # Also reject as full-file-rewrite if content is very large —
        # full rewrites should use write_file. Observed in Phase 14
        # iter2t add_cli_flag where a 20-line app.py prepend duplicated
        # the file's argparse setup.
        newline_count = new_string.count("\n")
        body = new_string.lstrip()
        starts_with_import = any(
            body.startswith(prefix)
            for prefix in ("import ", "from ", "#!", '"""', "'''")
        )
        # Detect "pure import block" — every non-blank line is an import/
        # from statement or a shebang. This is the ONLY content that may
        # be prepended to the top. Functions, classes, and conditionals
        # are NOT import-like, even if they happen to start with `from X
        # import Y` on line 1.
        def _is_pure_imports(text: str) -> bool:
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith(("import ", "from ")) or s.startswith("#!"):
                    continue
                return False
            return True
        is_import_only = starts_with_import and _is_pure_imports(new_string)
        # Multi-construct detector: content that has imports AND a def/class
        # or a main guard (`if __name__`) is a full-file rewrite, not a
        # small append. PREPEND would duplicate the existing code — so
        # auto-route to REPLACE-THE-WHOLE-FILE instead (same effect as
        # write_file). This makes the tool forgiving to a common model
        # mistake: "the file needs to be rewritten, but let me use
        # edit_file with empty old_string as a shortcut." We honor the
        # intent (rewrite) without breaking file state.
        # Match only TOP-LEVEL (unindented) imports + defs — an indented
        # `from b import x` INSIDE a function body is not a top-level
        # import and doesn't count as a multi-construct file.
        has_import_stmt = bool(
            re.search(r"^(?:import |from \S+ import )", new_string, re.MULTILINE)
        )
        has_def_or_class = bool(
            re.search(r"^(?:def |class |async def )", new_string, re.MULTILINE)
        )
        has_main_guard = 'if __name__' in new_string
        if (has_import_stmt and has_def_or_class) or has_main_guard:
            chunk = new_string if new_string.endswith("\n") else new_string + "\n"
            p.write_text(chunk, encoding="utf-8")
            return (
                f"Rewrote {p} ({len(chunk)} chars) — empty old_string + "
                f"multi-construct content interpreted as full-file replace. "
                f"(Next time, use write_file directly for rewrites — "
                f"edit_file's prepend semantics would have duplicated the "
                f"existing code.)"
            )
        # Size thresholds — generous enough for a full small function
        # (add_docstring / extend_calculator append a few functions at
        # ~15 lines / ~500 chars), but firm enough to reject long blobs.
        max_lines = 3 if is_import_only else 20
        max_chars = 200 if is_import_only else 700
        if newline_count > max_lines or len(new_string) > max_chars:
            raise ValueError(
                f"edit_file with empty old_string is for appending a "
                f"small block (single function, up to a few imports, a "
                f"single decorator). Your new_string is {len(new_string)} "
                f"chars / {newline_count + 1} lines — that exceeds the "
                f"{'import' if is_import_only else 'code'} append limit "
                f"({max_chars} chars / {max_lines + 1} lines). Use "
                f"write_file to replace the whole file, OR pass a "
                f"non-empty old_string that uniquely anchors the insertion "
                f"point inside {path}."
            )
        chunk = new_string if new_string.endswith("\n") else new_string + "\n"
        if is_import_only:
            # Cycle detection: if this is `from X import Y` AND X.py
            # itself imports from the current file, prepending Y would
            # recreate the circular import at top level. Use a lazy
            # import inside the function that references Y instead.
            cycle_m = re.match(
                r"\s*from\s+([A-Za-z_][A-Za-z_0-9]*)\s+import\s+"
                r"([A-Za-z_][A-Za-z_0-9]*)",
                new_string,
            )
            if cycle_m:
                other_module = cycle_m.group(1)
                imported_name = cycle_m.group(2)
                other_file = p.parent / f"{other_module}.py"
                this_module = p.stem
                if other_file.exists() and other_file.is_file():
                    try:
                        other_src = other_file.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        other_src = ""
                    cycle_pat = re.compile(
                        r"^\s*(?:from\s+" + re.escape(this_module) + r"\s+import\s|import\s+" + re.escape(this_module) + r"\b)",
                        re.MULTILINE,
                    )
                    if cycle_pat.search(other_src):
                        # Find a function in THIS file that references
                        # imported_name and insert the import inside it.
                        def_pat = re.compile(
                            r"^(def\s+[A-Za-z_][A-Za-z_0-9]*\s*\([^)]*\)\s*(?:->[^:]+)?:\s*\n)"
                            r"(?P<indent>[ \t]+)",
                            re.MULTILINE,
                        )
                        ref_pat = re.compile(r"\b" + re.escape(imported_name) + r"\b")
                        lazy_line = new_string.strip()
                        for m2 in def_pat.finditer(text):
                            header_end = m2.end(1)
                            indent = m2.group("indent")
                            remainder = text[header_end:]
                            end_of_body = re.search(
                                r"^(?:def |class |async def )", remainder, re.MULTILINE
                            )
                            body = (
                                remainder[: end_of_body.start()]
                                if end_of_body
                                else remainder
                            )
                            if ref_pat.search(body):
                                new_src = (
                                    text[:header_end]
                                    + f"{indent}{lazy_line}\n"
                                    + text[header_end:]
                                )
                                p.write_text(new_src, encoding="utf-8")
                                return (
                                    f"Inserted lazy import `{lazy_line}` "
                                    f"inside the function in {p} that uses "
                                    f"{imported_name} — empty old_string + "
                                    f"cyclic import detected (top-level "
                                    f"prepend would have recreated the cycle)."
                                )
            p.write_text(chunk + text, encoding="utf-8")
            return (
                f"Prepended {len(chunk)} chars to {p} (empty old_string + "
                f"import-only content interpreted as prepend-to-file)."
            )
        # Append — ensure the existing file ends with a newline before
        # concatenating so we don't merge the last existing line with the
        # first inserted line.
        sep = "" if text.endswith("\n") or not text else "\n"
        # Insert a blank separator line between existing code and the
        # appended block, for readability. Only if the existing file isn't
        # empty.
        gap = "\n" if text.rstrip() else ""
        p.write_text(text + sep + gap + chunk, encoding="utf-8")
        return (
            f"Appended {len(chunk)} chars to {p} (empty old_string + "
            f"non-import content interpreted as append-to-end)."
        )
    count = text.count(old_string)

    # Fuzzy-match fallback: if exact old_string doesn't match but a
    # whitespace-normalized version matches UNIQUELY, do the replacement.
    # Phase 14 bench1 v7 showed the 14B has a hard generative bias toward
    # `\t` for Python indentation in JSON args even when the file uses
    # spaces — hinting at the fix does NOT help, the model just re-emits
    # the same broken old_string. Tool-side forgiveness turns this into a
    # successful edit with a clear success message so the model learns
    # the loop completed.
    if count == 0:
        candidates = []  # list of (label, normalized_old_string)
        stripped = old_string.strip()
        if stripped and stripped != old_string:
            candidates.append(("stripped whitespace", stripped))
        tab_to_spaces = old_string.replace("\t", "    ")
        if tab_to_spaces != old_string:
            candidates.append(("tabs -> 4 spaces", tab_to_spaces))
        spaces_to_tab = old_string.replace("    ", "\t")
        if spaces_to_tab != old_string:
            candidates.append(("4 spaces -> tabs", spaces_to_tab))
        # Combined: strip AND tab conversion
        stripped_tab = tab_to_spaces.strip()
        if stripped_tab and stripped_tab not in {old_string, stripped, tab_to_spaces}:
            candidates.append(("stripped + tabs -> spaces", stripped_tab))
        # Line-number prefix stripping: read_file output is line-number
        # prefixed ("     1\tcode", "    42\tcode"). The 14B sometimes
        # copies these prefixes into old_string verbatim. Strip them before
        # matching. Pattern: optional whitespace + digit(s) + tab.
        import re as _re
        line_prefix_pat = _re.compile(r"^\s*\d+\t", _re.MULTILINE)
        no_line_nums = line_prefix_pat.sub("", old_string)
        if no_line_nums and no_line_nums != old_string:
            candidates.append(("line numbers stripped", no_line_nums))
            no_line_nums_stripped = no_line_nums.strip()
            if no_line_nums_stripped and no_line_nums_stripped != no_line_nums:
                candidates.append(("line numbers + whitespace stripped", no_line_nums_stripped))

        for label, fuzzy in candidates:
            fuzzy_count = text.count(fuzzy)
            if fuzzy_count == 1 and not replace_all:
                new_text = text.replace(fuzzy, new_string, 1)
                p.write_text(new_text, encoding="utf-8")
                return (
                    f"Replaced 1 occurrence(s) in {p} "
                    f"(fuzzy match: {label})"
                )
            if fuzzy_count > 0 and replace_all:
                new_text = text.replace(fuzzy, new_string)
                p.write_text(new_text, encoding="utf-8")
                return (
                    f"Replaced {fuzzy_count} occurrence(s) in {p} "
                    f"(fuzzy match: {label})"
                )

        # No fuzzy match either. Give a hint if one of the normalized
        # versions appears in the file at all (even if not unique) — still
        # more actionable than a bare error.
        hint = ""
        for label, fuzzy in candidates:
            if fuzzy and fuzzy in text:
                hint = (
                    f" HINT: `{fuzzy!r}` ({label}) IS in the file. "
                    f"Retry with that exact value as old_string."
                )
                break
        raise ValueError(f"old_string not found in {path}.{hint}")

    if count > 1 and not replace_all:
        # If the match count is huge, old_string is almost certainly too
        # short (1-2 chars matching many places) — don't waste tokens on
        # line-by-line locations, just redirect to write_file or a longer
        # old_string.
        if count > 5:
            raise ValueError(
                f"old_string matches {count} times in {path} — that's far "
                f"too many. Your old_string is too short to identify a "
                f"unique location. Either (a) use a longer old_string that "
                f"includes several unique lines of surrounding context, or "
                f"(b) use write_file to rewrite the whole file if the "
                f"changes are extensive."
            )
        # 2–5 matches: show line numbers + 1-line context for each.
        locations = []
        search_from = 0
        for _ in range(count):
            idx = text.find(old_string, search_from)
            if idx < 0:
                break
            line_num = text.count("\n", 0, idx) + 1
            line_start = text.rfind("\n", 0, idx) + 1
            line_end = text.find("\n", idx)
            if line_end < 0:
                line_end = len(text)
            line_text = text[line_start:line_end].strip()
            if len(line_text) > 80:
                line_text = line_text[:77] + "..."
            locations.append(f"line {line_num}: {line_text}")
            search_from = idx + max(len(old_string), 1)
        loc_summary = "; ".join(locations)
        raise ValueError(
            f"old_string matches {count} times in {path} at [{loc_summary}]. "
            f"Options: (a) pass replace_all=true to update every match, "
            f"(b) call edit_file {count} times with distinct old_strings "
            f"using more surrounding context, or (c) re-read the file to "
            f"decide which matches need updating."
        )

    if replace_all:
        new_text = text.replace(old_string, new_string)
    else:
        new_text = text.replace(old_string, new_string, 1)
    p.write_text(new_text, encoding="utf-8")
    diff = _diff_preview(text, new_text, path)
    tail = f"\n\n{diff}" if diff else ""
    return f"Replaced {count if replace_all else 1} occurrence(s) in {p}{tail}"


def _list_dir(ws: Workspace, path: str = ".", depth: int = 1, show_hidden: bool = False) -> str:
    p = ws.resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not p.is_dir():
        raise ValueError(f"Not a directory: {path}")

    depth = max(1, min(int(depth), 5))
    entries: list[str] = [f"{p}/"]
    count = 0

    def walk(dirpath: Path, current_depth: int, indent: str) -> None:
        nonlocal count
        if count >= _MAX_LIST_ENTRIES:
            return
        try:
            children = sorted(dirpath.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return
        for child in children:
            if count >= _MAX_LIST_ENTRIES:
                entries.append(f"{indent}... [truncated at {_MAX_LIST_ENTRIES} entries]")
                return
            if not show_hidden and child.name.startswith("."):
                continue
            if child.name in _NOISE_DIRS:
                continue
            marker = "/" if child.is_dir() else ""
            entries.append(f"{indent}{child.name}{marker}")
            count += 1
            if child.is_dir() and current_depth < depth:
                walk(child, current_depth + 1, indent + "  ")

    walk(p, 1, "  ")
    return "\n".join(entries)


def _glob(ws: Workspace, pattern: str, path: str = ".") -> str:
    root = ws.resolve(path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    matches: list[Path] = []
    try:
        for match in root.rglob(pattern):
            # Skip noise directories
            if any(part in _NOISE_DIRS for part in match.parts):
                continue
            matches.append(match)
            if len(matches) >= _MAX_GLOB_RESULTS * 2:
                break
    except OSError as exc:
        raise RuntimeError(f"glob failed: {exc}")

    # Sort by mtime (newest first); fall back to path for deterministic ordering.
    def sort_key(item: Path) -> tuple:
        try:
            return (-item.stat().st_mtime, str(item))
        except OSError:
            return (0, str(item))

    matches.sort(key=sort_key)
    matches = matches[:_MAX_GLOB_RESULTS]

    if not matches:
        return f"No files matched {pattern!r} under {root}"

    lines = [str(m) for m in matches]
    if len(matches) == _MAX_GLOB_RESULTS:
        lines.append(f"... [truncated to first {_MAX_GLOB_RESULTS} results by mtime]")
    return "\n".join(lines)


def _grep(
    ws: Workspace,
    pattern: str,
    path: str = ".",
    glob: Optional[str] = None,
    case_insensitive: bool = False,
    context: int = 0,
) -> str:
    root = ws.resolve(path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Fast path: ripgrep subprocess.
    if _RG_BIN:
        cmd = [_RG_BIN, "--line-number", "--no-heading", "--color", "never"]
        if case_insensitive:
            cmd.append("-i")
        if context > 0:
            cmd.extend(["-C", str(int(context))])
        if glob:
            cmd.extend(["-g", glob])
        cmd.extend(["--", pattern, str(root)])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return "[grep timed out after 60s]"
        # rg returns 1 on "no matches" — not an error for us.
        if result.returncode not in (0, 1):
            return f"[grep error: {result.stderr.strip() or 'rc=' + str(result.returncode)}]"
        lines = result.stdout.splitlines()
        if not lines:
            return f"No matches for {pattern!r} under {root}"
        truncated = lines[:_MAX_GREP_MATCHES]
        out = "\n".join(truncated)
        if len(lines) > _MAX_GREP_MATCHES:
            out += f"\n... [truncated to first {_MAX_GREP_MATCHES} matches of {len(lines)}]"
        return out

    # Stdlib fallback — slower but works without ripgrep installed.
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regex {pattern!r}: {exc}")

    matches: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _NOISE_DIRS]
        for fname in filenames:
            if glob and not fnmatch.fnmatch(fname, glob):
                continue
            fpath = Path(dirpath) / fname
            try:
                with fpath.open("r", encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, start=1):
                        if regex.search(line):
                            snippet = line.rstrip("\n")
                            if len(snippet) > 500:
                                snippet = snippet[:500] + "..."
                            matches.append(f"{fpath}:{lineno}:{snippet}")
                            if len(matches) >= _MAX_GREP_MATCHES:
                                matches.append(
                                    f"... [truncated to first {_MAX_GREP_MATCHES} matches]"
                                )
                                return "\n".join(matches)
            except (OSError, UnicodeDecodeError):
                continue

    if not matches:
        return f"No matches for {pattern!r} under {root}"
    return "\n".join(matches)


# ── Shell / test tools ──────────────────────────────────────────


def _run_bash(
    ws: Workspace,
    command: str,
    timeout: int = _BASH_TIMEOUT_DEFAULT,
    cwd: Optional[str] = None,
) -> dict:
    work_cwd = ws.resolve(cwd) if cwd else ws.root
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1, min(int(timeout), 600)),
            cwd=str(work_cwd),
        )
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "error": f"timed out after {timeout}s",
        }

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if len(stdout) > _BASH_OUTPUT_CAP:
        stdout = stdout[:_BASH_OUTPUT_CAP] + f"\n... [stdout truncated, {len(stdout)} chars total]"
    if len(stderr) > _BASH_OUTPUT_CAP:
        stderr = stderr[:_BASH_OUTPUT_CAP] + f"\n... [stderr truncated, {len(stderr)} chars total]"
    return {"stdout": stdout, "stderr": stderr, "exit_code": result.returncode}


def _run_tests(
    ws: Workspace,
    path: str = ".",
    runner: str = "pytest",
    timeout: int = _TEST_TIMEOUT_DEFAULT,
) -> dict:
    target = ws.resolve(path)
    if runner == "pytest":
        cmd = ["pytest", "-x", "--tb=short", "-q", str(target)]
    elif runner == "unittest":
        cmd = ["python", "-m", "unittest", "discover", "-s", str(target)]
    elif runner == "npm":
        cmd = ["npm", "test"]
    elif runner == "go":
        cmd = ["go", "test", "./..."]
    elif runner == "cargo":
        cmd = ["cargo", "test"]
    else:
        raise ValueError(f"Unknown test runner {runner!r}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(10, min(int(timeout), 1800)),
            cwd=str(ws.root),
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "exit_code": -1, "error": f"tests timed out after {timeout}s"}
    except FileNotFoundError:
        return {"passed": False, "exit_code": -1, "error": f"runner not found on PATH: {cmd[0]}"}

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    # Keep the tail — test failures appear at the bottom.
    if len(stdout) > _TEST_OUTPUT_CAP:
        stdout = (
            f"... [stdout truncated, keeping last {_TEST_OUTPUT_CAP} chars]\n"
            + stdout[-_TEST_OUTPUT_CAP:]
        )
    return {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr[-4000:] if stderr else "",
    }


# ── Factory ─────────────────────────────────────────────────────


_WEB_RESEARCH_PATTERN_HINT = """\
## Pattern: web research → write file (CRITICAL when --web is enabled)

When a goal asks you to research something AND write a file, the sequence is:

    web_search  →  fetch_url (optional)  →  write_file  →  final answer

The fetched body is INPUT to your write, not the deliverable itself. Common
failure: model fetches a docs page, summarizes the answer in prose inside
the final_answer, and never calls write_file. The user looks for the file
on disk — prose in the final answer does not count.

Rule: if the goal contains "write FILENAME", "save to FILENAME", or
"create FILENAME", your turn after the last fetch_url MUST be a write_file
tool_call, NOT a final answer with a code block. Markdown deliverables are
extra-tempting to leak into prose — write the .md file via write_file too.
"""


def web_research_pattern_hint() -> str:
    """Return the web-research-to-file pattern as a system_prompt_extra block.

    Callers append this to their workspace hint when enable_web=True so the
    model has the search→fetch→write_file pattern in context. Counters the
    narration_without_action failure mode that fetch_url tends to amplify
    (the fetched body primes the model into summarize-mode, which competes
    with the file-write deliverable).
    """
    return _WEB_RESEARCH_PATTERN_HINT


def build_default_registry(
    workspace_root: Path | str,
    memory: Optional[AgentMemory] = None,
    ask_user_fn: Optional[Callable] = None,
    extended_tools: bool = False,
    mcp_servers: Optional[list[str]] = None,
    mcp_tool_pack: Optional[list[str]] = None,
    enable_web: bool = False,
) -> ToolRegistry:
    """
    Build a ToolRegistry pre-populated with the 8 core Phase 14 agent tools,
    anchored at *workspace_root*. All tool paths resolve relative to this
    root unless they are absolute.

    If *memory* is provided, also registers a `remember` tool that lets the
    model save short cross-session notes to the per-project memory store.

    `mcp_servers` (optional, default None): list of MCP server names or
    config paths. Empty/None = no MCP tools (the default and the only
    code path exercised today). When non-empty, mounts tools from each
    server alongside the builtins. **Currently scaffolded but not wired** —
    see `engine/mcp_adapter.py`. Passing a non-empty list today raises
    NotImplementedError so the gap is loud, not silent.

    `mcp_tool_pack` (optional): whitelist of MCP tool names to actually
    expose. Important because the 14B models in this repo regress when
    the registry crosses ~10 tools (see user-memory
    `feedback_tool_count_regression.md`). Pass e.g.
    `["search_cards", "analyze_deck"]` to mount only those.
    """
    ws = Workspace(root=Path(workspace_root).resolve())
    reg = ToolRegistry()

    reg.register(ToolSchema(
        name="read_file",
        description="Read a text file with line numbers. Use offset/limit for large files.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (absolute or relative to workspace root)"},
                "offset": {"type": "integer", "description": "0-indexed starting line", "default": 0},
                "limit": {"type": "integer", "description": f"Max lines to return (<= {_MAX_READ_LINES})", "default": _MAX_READ_LINES},
            },
            "required": ["path"],
        },
        function=lambda path, offset=0, limit=_MAX_READ_LINES: _read_file(ws, path, offset, limit),
        category="file",
    ))

    reg.register(ToolSchema(
        name="write_file",
        description=(
            "Create or overwrite a file with the given content. Creates parent "
            "directories as needed. content can be EITHER a single string (with "
            "\\n for newlines), OR an array of strings where each element is one "
            "line of the file (no \\n needed, escaping inner quotes is easier). "
            "Prefer the array form for multi-line files with embedded quotes or "
            "f-strings — it sidesteps the JSON escape hell that breaks large "
            "write_file calls."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {
                    "description": (
                        "Full file content. Either a single string (use \\n "
                        "for newlines) OR an array of strings (one line per "
                        "element, joined with \\n). The array form is RECOMMENDED "
                        "for multi-line files with embedded quotes or f-strings."
                    ),
                },
            },
            "required": ["path", "content"],
        },
        function=lambda path, content=None, new_string=None, **kw: _write_file(
            ws, path, content=content, new_string=new_string, **kw
        ),
        category="file",
    ))

    reg.register(ToolSchema(
        name="edit_file",
        description=(
            "Replace old_string with new_string in a file. old_string must match exactly. "
            "Fails if old_string appears multiple times unless replace_all=true. "
            "new_string and old_string are PLAIN STRINGS (use \\n for newlines) — "
            "NOT arrays. (The array-of-lines form is only for write_file's content.) "
            "When replacing an indented block, bake the indentation into new_string: "
            'e.g. new_string="    x = 1\\n    return x" for code inside a function.'
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "old_string": {"type": "string", "description": "Exact text to find"},
                "new_string": {"type": "string", "description": "Replacement text (string, \\n for newlines, indentation baked in)"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence", "default": False},
            },
            "required": ["path", "old_string", "new_string"],
        },
        function=lambda path, old_string, new_string, replace_all=False: _edit_file(
            ws, path, old_string, new_string, replace_all
        ),
        category="file",
    ))

    reg.register(ToolSchema(
        name="list_dir",
        description=(
            "List the contents of a directory. Ignores noise dirs (__pycache__, node_modules, "
            ".git, .venv, dist, build, etc.)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path", "default": "."},
                "depth": {"type": "integer", "description": "Recursion depth (1-5)", "default": 1},
                "show_hidden": {"type": "boolean", "description": "Include dotfiles", "default": False},
            },
        },
        function=lambda path=".", depth=1, show_hidden=False: _list_dir(ws, path, depth, show_hidden),
        category="fs",
    ))

    reg.register(ToolSchema(
        name="glob",
        description=(
            "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/test_*.py'). "
            "Returns paths sorted by mtime (newest first)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "path": {"type": "string", "description": "Root directory", "default": "."},
            },
            "required": ["pattern"],
        },
        function=lambda pattern, path=".": _glob(ws, pattern, path),
        category="fs",
    ))

    reg.register(ToolSchema(
        name="grep",
        description=(
            "Search file contents for a regex pattern. Uses ripgrep if installed, falls back "
            "to Python re + os.walk. Returns 'path:line:match' lines."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {"type": "string", "description": "Root directory", "default": "."},
                "glob": {"type": "string", "description": "Optional file glob filter (e.g. '*.py')"},
                "case_insensitive": {"type": "boolean", "default": False},
                "context": {"type": "integer", "description": "Lines of context around each match", "default": 0},
            },
            "required": ["pattern"],
        },
        function=lambda pattern, path=".", glob=None, case_insensitive=False, context=0: _grep(
            ws, pattern, path, glob, case_insensitive, context
        ),
        category="search",
    ))

    reg.register(ToolSchema(
        name="run_bash",
        description=(
            "Execute a shell command. Returns {stdout, stderr, exit_code}. Runs in the "
            "workspace root by default. Marked risky — the agent loop may prompt to confirm."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (1-600)", "default": _BASH_TIMEOUT_DEFAULT},
                "cwd": {"type": "string", "description": "Working directory (relative to workspace root)"},
            },
            "required": ["command"],
        },
        function=lambda command, timeout=_BASH_TIMEOUT_DEFAULT, cwd=None: _run_bash(
            ws, command, timeout, cwd
        ),
        category="shell",
        risky=True,
    ))

    # Plan/todo state — per-registry (= per agent.run()) ephemeral list.
    # The model uses this to track multi-step goals across iterations.
    # Each item is {"id": int, "title": str, "done": bool}.
    plan_state: list[dict] = []

    def _plan(action: str, items: Any = None, id: Any = None) -> str:  # noqa: A002
        action = (action or "").strip().lower()
        if action == "list":
            if not plan_state:
                return "Plan is empty. Use action=\"set\" with items=[...] to start."
            lines = []
            for it in plan_state:
                mark = "[x]" if it["done"] else "[ ]"
                lines.append(f"{it['id']}. {mark} {it['title']}")
            done = sum(1 for it in plan_state if it["done"])
            return "\n".join(lines) + f"\n\n({done}/{len(plan_state)} done)"
        if action == "set":
            if not isinstance(items, list) or not items:
                raise ValueError("action='set' requires items=[list of strings]")
            titles = [str(x).strip() for x in items if str(x).strip()]
            if not titles:
                raise ValueError("items was empty after stripping")
            plan_state.clear()
            for i, title in enumerate(titles, 1):
                plan_state.append({"id": i, "title": title[:200], "done": False})
            return f"Plan set with {len(plan_state)} items. Use action=\"done\" with id=N as you complete each."
        if action == "done":
            try:
                idn = int(id) if id is not None else None
            except (TypeError, ValueError):
                raise ValueError(f"action='done' requires id=<int>; got {id!r}")
            if idn is None:
                raise ValueError("action='done' requires id=<int>")
            for it in plan_state:
                if it["id"] == idn:
                    if it["done"]:
                        return f"Item {idn} already marked done: {it['title']}"
                    it["done"] = True
                    remaining = sum(1 for x in plan_state if not x["done"])
                    return f"Marked {idn} done: {it['title']}. {remaining} step(s) remaining."
            raise ValueError(f"no plan item with id={idn}; current ids: {[it['id'] for it in plan_state]}")
        if action == "reset":
            n = len(plan_state)
            plan_state.clear()
            return f"Plan reset ({n} items cleared)."
        raise ValueError(
            f"unknown action={action!r}. Valid: 'set' (with items=[...]), "
            f"'done' (with id=N), 'list', 'reset'."
        )

    reg.register(ToolSchema(
        name="plan",
        description=(
            "Track a multi-step plan across iterations. Call once at the start "
            "of a complex goal with action=\"set\" and items=[\"step 1\", \"step 2\", ...]. "
            "As you complete each step, call action=\"done\" with id=N. Use "
            "action=\"list\" any time to see progress. State persists within one "
            "agent run only — no files are written. Use this for goals that "
            "require 4+ distinct subtasks so you don't lose track."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "done", "list", "reset"],
                    "description": "What to do with the plan",
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only for action='set': the list of step titles",
                },
                "id": {
                    "type": "integer",
                    "description": "Only for action='done': which step id to mark complete",
                },
            },
            "required": ["action"],
        },
        function=lambda action, items=None, id=None: _plan(action, items, id),
        category="planning",
    ))

    if ask_user_fn is not None:
        reg.register(ToolSchema(
            name="ask_user",
            description=(
                "Ask the human user a clarifying question and wait for their "
                "answer. Use when the goal is ambiguous or you need a decision "
                "before proceeding. Returns the user's typed response."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user",
                    },
                },
                "required": ["question"],
            },
            function=lambda question: ask_user_fn(str(question)),
            category="interactive",
        ))

    if memory is not None:
        reg.register(ToolSchema(
            name="remember",
            description=(
                "Save a short note to per-project memory so future sessions on this "
                "workspace can recall it. Use sparingly — only for facts the model "
                "would benefit from knowing next session (e.g. 'auth uses JWT, not "
                "sessions', 'tests live in /tests, not /test')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "Single short note (<= 500 chars)"},
                },
                "required": ["note"],
            },
            function=lambda note: memory.remember(note),
            category="memory",
        ))

    reg.register(ToolSchema(
        name="run_tests",
        description=(
            "Run the project's test suite. Supports pytest, unittest, npm, go, cargo. "
            "Returns {passed, exit_code, stdout, stderr}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Test path or directory", "default": "."},
                "runner": {
                    "type": "string",
                    "enum": ["pytest", "unittest", "npm", "go", "cargo"],
                    "default": "pytest",
                },
                "timeout": {"type": "integer", "description": "Timeout in seconds (10-1800)", "default": _TEST_TIMEOUT_DEFAULT},
            },
        },
        function=lambda path=".", runner="pytest", timeout=_TEST_TIMEOUT_DEFAULT: _run_tests(
            ws, path, runner, timeout
        ),
        category="test",
    ))

    # ── insert_at_line: line-number based insertion ──
    # Sidesteps the "can't find a unique anchor" problem on large HTML/config
    # files. The model reads the file (with line numbers), picks a line N,
    # and inserts text BEFORE that line. Supports multi-line text.

    def _insert_at_line(path: str, line: int, text: str) -> str:
        p = ws.resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        line = int(line)
        if line < 1 or line > len(lines) + 1:
            raise ValueError(f"line must be 1..{len(lines) + 1}; got {line}")
        insert_text = str(text) if isinstance(text, str) else "\n".join(str(x) for x in text)
        if not insert_text.endswith("\n"):
            insert_text += "\n"
        before = "".join(lines)
        lines.insert(line - 1, insert_text)
        after = "".join(lines)
        p.write_text(after, encoding="utf-8")
        diff = _diff_preview(before, after, path, max_lines=15)
        tail = f"\n\n{diff}" if diff else ""
        return f"Inserted {len(insert_text)} chars before line {line} in {p}{tail}"

    reg.register(ToolSchema(
        name="insert_at_line",
        description=(
            "Insert text BEFORE a specific line number in a file. Useful when "
            "edit_file can't find a unique anchor (e.g. many </div> tags in HTML). "
            "Read the file first to find the right line number, then call this."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "line": {"type": "integer", "description": "1-indexed line number to insert BEFORE"},
                "text": {"description": "Text to insert (string or array of strings)"},
            },
            "required": ["path", "line", "text"],
        },
        function=_insert_at_line,
        category="file",
    ))

    # ── Web tools (opt-in via enable_web) ──
    # Orthogonal axis to extended_tools — web is "online access," extended is
    # "local advanced." Both web tools are flagged risky=True so the existing
    # confirm_risky agent hook prompts the user before each call. Default
    # daily-driver mode (enable_web=False) preserves the zero-server invariant
    # documented in CLAUDE.md.
    if enable_web:
        from engine.web_tools import (
            DEFAULT_MAX_BYTES,
            DEFAULT_N_RESULTS,
            DEFAULT_TIMEOUT,
            fetch_url as _fetch_url,
            web_search as _web_search,
        )

        reg.register(ToolSchema(
            name="web_search",
            description=(
                "Search the public web via DuckDuckGo and return ranked results "
                "(title, URL, snippet). Use to discover URLs/docs you don't "
                "already know. Then use fetch_url on a specific result. "
                "Safe-search is OFF by default for broader technical results; "
                "pass safe_search=true if you want filtering. "
                "If the user's goal asks you to write a file (e.g. 'write X.md', "
                "'save to Y.txt', 'create Z.py'), you MUST call write_file with "
                "your synthesized result before giving the final answer — do not "
                "summarize the answer in prose only. "
                "Marked risky — the agent loop may prompt to confirm."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n_results": {
                        "type": "integer",
                        "description": f"Max results (1-15, default {DEFAULT_N_RESULTS})",
                        "default": DEFAULT_N_RESULTS,
                    },
                    "safe_search": {
                        "type": "boolean",
                        "description": "Enable DDG safe-search filtering (default false, broader results)",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
            function=lambda query, n_results=DEFAULT_N_RESULTS, safe_search=False: _web_search(
                query, n_results=n_results, timeout=DEFAULT_TIMEOUT, safe_search=safe_search
            ),
            category="web",
            risky=True,
        ))

        reg.register(ToolSchema(
            name="fetch_url",
            description=(
                "Fetch an http/https URL and return the response body as text. "
                "HTML is converted to readable text by default; pass raw=true "
                "for the raw body. Body is size-capped at 30 KB (~9k tokens) "
                "to fit in context — bump max_bytes only if the page is known "
                "to be small. Blocks file://, localhost, and private IP ranges. "
                "If the user's goal asks you to write a file with the "
                "information you fetched, you MUST call write_file before "
                "giving the final answer — prose summaries don't count. "
                "Marked risky."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http:// or https:// URL"},
                    "raw": {
                        "type": "boolean",
                        "description": "Return raw body without HTML stripping",
                        "default": False,
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": f"Response cap in bytes (default {DEFAULT_MAX_BYTES})",
                        "default": DEFAULT_MAX_BYTES,
                    },
                },
                "required": ["url"],
            },
            function=lambda url, raw=False, max_bytes=DEFAULT_MAX_BYTES: _fetch_url(
                url, timeout=DEFAULT_TIMEOUT, max_bytes=max_bytes, raw=raw
            ),
            category="web",
            risky=True,
        ))

    # ── Extended tools (opt-in) ──
    # These add ~3000 tokens to the system prompt. On a 16k context model,
    # that's 15-20% of the budget. Benchmarks show the 14B regresses from
    # 97.6% to 85.7% when all 21 tools are registered — the model gets
    # confused by the extra schemas. Only enable for interactive use where
    # the user needs refactoring/git/navigation tools.
    if not extended_tools:
        # MCP-mounted tools fire regardless of extended_tools — they're a
        # separate axis (external server vs. local builtin). Empty list
        # is a no-op so this is safe to call unconditionally.
        if mcp_servers:
            from engine.mcp_adapter import register_mcp_tools
            register_mcp_tools(reg, mcp_servers, tool_pack=mcp_tool_pack)
        return reg

    # ── rename_symbol: atomic multi-file rename ──

    def _rename_symbol(old_name: str, new_name: str, glob_pattern: str = "**/*") -> str:
        old_name, new_name = str(old_name).strip(), str(new_name).strip()
        if not old_name or not new_name:
            raise ValueError("old_name and new_name must be non-empty")
        root = ws.root
        changed_files: list[str] = []
        # Use Path.match() which handles ** correctly (unlike fnmatch)
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix in (".pyc", ".pyo"):
                continue
            if not p.match(glob_pattern):
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if old_name not in text:
                continue
            new_text = text.replace(old_name, new_name)
            p.write_text(new_text, encoding="utf-8")
            count = text.count(old_name)
            changed_files.append(f"{rel} ({count} replacements)")
        if not changed_files:
            return f"No files matching '{glob_pattern}' contain '{old_name}'"
        summary = "\n".join(f"  {f}" for f in changed_files)
        return f"Renamed '{old_name}' -> '{new_name}' in {len(changed_files)} file(s):\n{summary}"

    reg.register(ToolSchema(
        name="rename_symbol",
        description=(
            "Atomically rename a symbol (function, class, variable name) across "
            "all files matching a glob pattern. Replaces every occurrence of "
            "old_name with new_name in every matching file. Use for multi-file "
            "refactoring — much safer than calling edit_file N times."
        ),
        parameters={
            "type": "object",
            "properties": {
                "old_name": {"type": "string", "description": "Current symbol name"},
                "new_name": {"type": "string", "description": "New symbol name"},
                "glob_pattern": {"type": "string", "description": "File glob (default '**/*')", "default": "**/*"},
            },
            "required": ["old_name", "new_name"],
        },
        function=lambda old_name, new_name, glob_pattern="**/*": _rename_symbol(old_name, new_name, glob_pattern),
        category="refactor",
    ))

    # ── read_function: AST-based extraction ──

    def _read_function(path: str, name: str) -> str:
        p = ws.resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if p.suffix == ".py":
            import ast as _ast
            try:
                tree = _ast.parse(text)
            except SyntaxError:
                pass  # fall through to regex
            else:
                for node in _ast.walk(tree):
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                        if node.name == name:
                            start = node.lineno
                            end = node.end_lineno or start
                            block = lines[start - 1 : end]
                            numbered = "\n".join(
                                f"{start + i:4d}\t{ln}" for i, ln in enumerate(block)
                            )
                            return f"{path}:{start}-{end} ({end - start + 1} lines):\n{numbered}"
        # Regex fallback for any language
        import re as _re
        pattern = _re.compile(
            rf"^(\s*(?:def|class|function|async\s+function|async\s+def|export\s+function|export\s+default\s+function)\s+{_re.escape(name)}\b)",
            _re.MULTILINE,
        )
        m = pattern.search(text)
        if m:
            start_line = text[: m.start()].count("\n")
            # Find end: next top-level def/class or EOF
            indent = len(m.group(1)) - len(m.group(1).lstrip())
            end_line = start_line + 1
            for i in range(start_line + 1, len(lines)):
                stripped = lines[i]
                if stripped.strip() and not stripped.startswith(" " * (indent + 1)) and not stripped.startswith("\t"):
                    if _re.match(r"\s*(def |class |function |async |export )", stripped):
                        break
                end_line = i + 1
            block = lines[start_line:end_line]
            numbered = "\n".join(f"{start_line + 1 + i:4d}\t{ln}" for i, ln in enumerate(block))
            return f"{path}:{start_line + 1}-{end_line} ({len(block)} lines):\n{numbered}"
        return f"'{name}' not found in {path}"

    reg.register(ToolSchema(
        name="read_function",
        description=(
            "Extract a single function or class from a file by name, with line "
            "numbers. Uses Python AST for .py files, regex for others. Returns "
            "ONLY the relevant block — much cheaper than reading a 500-line file "
            "when you only need one function."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "name": {"type": "string", "description": "Function or class name to extract"},
            },
            "required": ["path", "name"],
        },
        function=lambda path, name: _read_function(path, name),
        category="file",
    ))

    # ── checkpoint / restore: workspace snapshots ──

    _checkpoints: dict[str, Path] = {}
    _MAX_CHECKPOINTS = 3

    def _checkpoint(name: str) -> str:
        name = str(name).strip()
        if not name:
            raise ValueError("checkpoint name must be non-empty")
        if len(_checkpoints) >= _MAX_CHECKPOINTS and name not in _checkpoints:
            raise ValueError(
                f"Max {_MAX_CHECKPOINTS} checkpoints. Existing: {list(_checkpoints.keys())}. "
                f"Restore or overwrite one first."
            )
        snap_dir = ws.root.parent / f".checkpoint_{name}"
        if snap_dir.exists():
            shutil.rmtree(snap_dir)
        shutil.copytree(ws.root, snap_dir, dirs_exist_ok=False)
        _checkpoints[name] = snap_dir
        file_count = sum(1 for _ in snap_dir.rglob("*") if _.is_file())
        return f"Checkpoint '{name}' saved ({file_count} files). Use restore(name='{name}') to roll back."

    def _restore(name: str) -> str:
        name = str(name).strip()
        snap_dir = _checkpoints.get(name)
        if snap_dir is None or not snap_dir.exists():
            available = list(_checkpoints.keys())
            raise ValueError(f"No checkpoint '{name}'. Available: {available}")
        # Clear workspace and copy snapshot back
        for item in ws.root.iterdir():
            if item.name.startswith(".checkpoint_"):
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        for item in snap_dir.iterdir():
            dest = ws.root / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        file_count = sum(1 for _ in ws.root.rglob("*") if _.is_file())
        return f"Restored checkpoint '{name}' ({file_count} files). Workspace rolled back."

    reg.register(ToolSchema(
        name="checkpoint",
        description=(
            "Save a named snapshot of the entire workspace. Use before risky "
            "changes so you can restore() if something breaks. Max 3 checkpoints."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Snapshot name (e.g. 'before_refactor')"},
            },
            "required": ["name"],
        },
        function=lambda name: _checkpoint(name),
        category="workspace",
    ))

    reg.register(ToolSchema(
        name="restore",
        description=(
            "Restore the workspace to a previously saved checkpoint. All current "
            "files are replaced with the snapshot's contents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Checkpoint name to restore"},
            },
            "required": ["name"],
        },
        function=lambda name: _restore(name),
        category="workspace",
    ))

    # ── git tools: safe version control ──

    def _git_cmd(args: list[str], check: bool = True) -> str:
        import subprocess
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=30, cwd=str(ws.root),
        )
        if check and result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"git {' '.join(args)} failed: {err}")
        return (result.stdout or "").strip()

    def _git_status() -> str:
        return _git_cmd(["status", "--short"]) or "(clean — no changes)"

    def _git_diff(staged: bool = False) -> str:
        args = ["diff"]
        if staged:
            args.append("--staged")
        diff = _git_cmd(args, check=False)
        if not diff:
            return "(no changes)" if not staged else "(no staged changes)"
        lines = diff.splitlines()
        if len(lines) > 100:
            return "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
        return diff

    def _git_commit(message: str) -> str:
        message = str(message).strip()
        if not message:
            raise ValueError("commit message must be non-empty")
        _git_cmd(["add", "-A"])
        return _git_cmd(["commit", "-m", message])

    for git_name, git_desc, git_params, git_fn in [
        ("git_status", "Show git status (short format). Safe, read-only.",
         {"type": "object", "properties": {}}, lambda: _git_status()),
        ("git_diff", "Show git diff (unstaged by default, staged=true for staged). Safe, read-only.",
         {"type": "object", "properties": {"staged": {"type": "boolean", "default": False}}},
         lambda staged=False: _git_diff(staged)),
        ("git_commit", "Stage all changes and commit with a message. Does NOT push.",
         {"type": "object", "properties": {"message": {"type": "string", "description": "Commit message"}}, "required": ["message"]},
         lambda message: _git_commit(message)),
    ]:
        reg.register(ToolSchema(
            name=git_name, description=git_desc, parameters=git_params,
            function=git_fn, category="git",
        ))

    # ── add_import: AST-aware import insertion ──

    def _add_import(path: str, statement: str) -> str:
        p = ws.resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        statement = str(statement).strip()
        if not statement.startswith(("import ", "from ")):
            raise ValueError(f"statement must start with 'import' or 'from': got {statement!r}")
        text = p.read_text(encoding="utf-8", errors="replace")
        # Deduplicate
        for line in text.splitlines():
            if line.strip() == statement:
                return f"'{statement}' already exists in {path} — no change"
        # Find insertion point: after last existing import, before first code
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(("import ", "from ")) or s.startswith("#") or not s:
                insert_at = i + 1
            elif s and not s.startswith(("import ", "from ", "#", '"""', "'''")):
                break
        lines.insert(insert_at, statement + "\n")
        p.write_text("".join(lines), encoding="utf-8")
        return f"Added '{statement}' at line {insert_at + 1} in {path}"

    reg.register(ToolSchema(
        name="add_import",
        description=(
            "Add an import statement to a Python file at the correct location "
            "(after existing imports, before code). Deduplicates — won't add if "
            "already present. Safer than edit_file for import management."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Python file path"},
                "statement": {"type": "string", "description": "Import statement (e.g. 'from os import path')"},
            },
            "required": ["path", "statement"],
        },
        function=lambda path, statement: _add_import(path, statement),
        category="refactor",
    ))

    # ── find_definition / find_usages: code navigation ──

    def _find_definition(symbol: str, glob_pattern: str = "**/*.py") -> str:
        import re as _re
        symbol = str(symbol).strip()
        pattern = _re.compile(
            rf"^\s*(?:def|class|async\s+def)\s+{_re.escape(symbol)}\b"
            rf"|^\s*{_re.escape(symbol)}\s*=",
            _re.MULTILINE,
        )
        results = []
        for p in sorted(ws.root.rglob("*")):
            if not p.is_file():
                continue
            if not p.match(glob_pattern):
                continue
            rel = str(p.relative_to(ws.root)).replace("\\", "/")
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in pattern.finditer(text):
                lineno = text[:m.start()].count("\n") + 1
                line_text = text.splitlines()[lineno - 1].strip()
                results.append(f"{rel}:{lineno}: {line_text}")
        if not results:
            return f"No definition found for '{symbol}'"
        return "\n".join(results)

    def _find_usages(symbol: str, glob_pattern: str = "**/*.py") -> str:
        import re as _re
        symbol = str(symbol).strip()
        usage_pat = _re.compile(rf"\b{_re.escape(symbol)}\b")
        def_pat = _re.compile(
            rf"^\s*(?:def|class|async\s+def)\s+{_re.escape(symbol)}\b"
        )
        results = []
        for p in sorted(ws.root.rglob("*")):
            if not p.is_file():
                continue
            if not p.match(glob_pattern):
                continue
            rel = str(p.relative_to(ws.root)).replace("\\", "/")
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines):
                if usage_pat.search(line) and not def_pat.match(line):
                    results.append(f"{rel}:{i + 1}: {line.strip()}")
        if not results:
            return f"No usages found for '{symbol}'"
        if len(results) > 50:
            return "\n".join(results[:50]) + f"\n... ({len(results) - 50} more)"
        return "\n".join(results)

    reg.register(ToolSchema(
        name="find_definition",
        description=(
            "Find where a symbol (function, class, variable) is defined. "
            "Returns file:line:context for each definition found. Smarter than "
            "grep — only matches def/class/assignment patterns, not usages."
        ),
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name to find"},
                "glob_pattern": {"type": "string", "default": "**/*.py"},
            },
            "required": ["symbol"],
        },
        function=lambda symbol, glob_pattern="**/*.py": _find_definition(symbol, glob_pattern),
        category="search",
    ))

    reg.register(ToolSchema(
        name="find_usages",
        description=(
            "Find all usages of a symbol EXCLUDING its definition. Returns "
            "file:line:context. Use after find_definition to see everywhere a "
            "symbol is referenced before renaming or removing it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name"},
                "glob_pattern": {"type": "string", "default": "**/*.py"},
            },
            "required": ["symbol"],
        },
        function=lambda symbol, glob_pattern="**/*.py": _find_usages(symbol, glob_pattern),
        category="search",
    ))

    # ── apply_patch: unified diff application ──

    def _apply_patch(patch: str) -> str:
        """Apply a unified diff patch to workspace files."""
        patch = str(patch).strip()
        if not patch:
            raise ValueError("patch must be non-empty")
        # Parse unified diff header lines to find target files + hunks
        current_file = None
        changed_files = []
        lines = patch.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("+++ b/") or line.startswith("+++ "):
                fname = line.split("+++ ")[-1].lstrip("b/").strip()
                current_file = ws.resolve(fname)
                if not current_file.exists():
                    raise FileNotFoundError(f"Patch target not found: {fname}")
                i += 1
                continue
            if line.startswith("@@ ") and current_file is not None:
                # Parse hunk header: @@ -start,count +start,count @@
                import re as _re
                m = _re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if not m:
                    i += 1
                    continue
                orig_start = int(m.group(1)) - 1
                file_lines = current_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                # Collect hunk lines
                i += 1
                removes = []
                adds = []
                context_offset = 0
                while i < len(lines) and not lines[i].startswith(("@@ ", "--- ", "+++ ", "diff ")):
                    hl = lines[i]
                    if hl.startswith("-"):
                        removes.append(hl[1:])
                    elif hl.startswith("+"):
                        adds.append(hl[1:])
                    else:
                        context_offset += 1
                    i += 1
                # Apply: find the remove block in file_lines near orig_start
                found = False
                for offset in range(max(len(file_lines), 20)):
                    for sign in (0, -1, 1):
                        pos = orig_start + offset * sign
                        if pos < 0 or pos + len(removes) > len(file_lines):
                            continue
                        chunk = [l.rstrip("\n") for l in file_lines[pos:pos + len(removes)]]
                        if chunk == [r.rstrip("\n") for r in removes]:
                            new_lines = file_lines[:pos] + [a + "\n" for a in adds] + file_lines[pos + len(removes):]
                            current_file.write_text("".join(new_lines), encoding="utf-8")
                            changed_files.append(str(current_file.relative_to(ws.root)))
                            found = True
                            break
                    if found:
                        break
                if not found and removes:
                    raise ValueError(f"Could not find hunk context in {current_file.name} near line {orig_start + 1}")
                continue
            i += 1
        if not changed_files:
            return "Patch applied no changes (empty or no matching hunks)"
        return f"Patch applied to {len(set(changed_files))} file(s): {', '.join(set(changed_files))}"

    reg.register(ToolSchema(
        name="apply_patch",
        description=(
            "Apply a unified diff patch to workspace files. The patch format is "
            "standard unified diff (--- a/file, +++ b/file, @@ hunks). Use when "
            "you want to express a complex multi-hunk change as a single patch "
            "instead of multiple edit_file calls."
        ),
        parameters={
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Unified diff text"},
            },
            "required": ["patch"],
        },
        function=lambda patch: _apply_patch(patch),
        category="file",
    ))

    # Same MCP hook the lean-mode return runs. Non-empty mcp_servers will
    # raise NotImplementedError today (see engine/mcp_adapter.py), and
    # the empty/None case is a clean no-op.
    if mcp_servers:
        from engine.mcp_adapter import register_mcp_tools
        register_mcp_tools(reg, mcp_servers, tool_pack=mcp_tool_pack)

    return reg


# ── Smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (root / "b.py").write_text("# TODO: implement\n", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "c.py").write_text("print('hello')\n# TODO: remove print\n", encoding="utf-8")

        reg = build_default_registry(root)
        status = reg.status()
        assert status["total"] == 8, status
        assert set(status["tools"]) == {
            "read_file", "write_file", "edit_file", "list_dir",
            "glob", "grep", "run_bash", "run_tests",
        }

        def call(payload: str):
            parsed = reg.parse(f"<tool_call>{payload}</tool_call>")
            assert len(parsed) == 1, f"parse failed for {payload}"
            return reg.execute(parsed[0])

        # read_file
        r = call('{"name":"read_file","arguments":{"path":"a.py"}}')
        assert r.success, r.error
        assert "def add" in r.content
        assert r.content.startswith("     1\t")  # line numbers

        # write_file
        r = call('{"name":"write_file","arguments":{"path":"new.py","content":"x = 1\\n"}}')
        assert r.success, r.error
        assert (root / "new.py").read_text() == "x = 1\n"

        # edit_file — happy path
        r = call('{"name":"edit_file","arguments":{"path":"a.py","old_string":"return a + b","new_string":"return a - b"}}')
        assert r.success, r.error
        assert "return a - b" in (root / "a.py").read_text()

        # edit_file — old_string missing
        r = call('{"name":"edit_file","arguments":{"path":"a.py","old_string":"nope","new_string":"zzz"}}')
        assert not r.success
        assert "not found" in (r.error or "")

        # edit_file — fuzzy match: model gave `\treturn 42` but file has
        # `    return 42` (4 spaces). Tool auto-strips and replaces.
        (root / "indented.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
        r = call('{"name":"edit_file","arguments":{"path":"indented.py","old_string":"\\treturn 42","new_string":"return 99"}}')
        assert r.success, f"expected fuzzy match success, got {r.error}"
        assert "fuzzy" in r.content.lower()
        final = (root / "indented.py").read_text(encoding="utf-8")
        assert "return 99" in final
        assert "return 42" not in final

        # edit_file — fuzzy match: model gave `\tx = 1` but file has spaces
        (root / "spaces.py").write_text("def foo():\n    x = 1\n    return x\n", encoding="utf-8")
        r = call('{"name":"edit_file","arguments":{"path":"spaces.py","old_string":"\\tx = 1","new_string":"x = 2"}}')
        assert r.success, f"expected fuzzy match success, got {r.error}"
        assert "x = 2" in (root / "spaces.py").read_text(encoding="utf-8")

        # edit_file — still fails cleanly when nothing matches at all
        (root / "clean.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        r = call('{"name":"edit_file","arguments":{"path":"clean.py","old_string":"completely_absent_zzz","new_string":"x"}}')
        assert not r.success
        assert "not found" in (r.error or "")

        # list_dir
        r = call('{"name":"list_dir","arguments":{"path":".","depth":2}}')
        assert r.success, r.error
        assert "a.py" in r.content
        assert "sub/" in r.content
        assert "c.py" in r.content

        # glob
        r = call('{"name":"glob","arguments":{"pattern":"**/*.py"}}')
        assert r.success, r.error
        assert "a.py" in r.content
        assert "c.py" in r.content

        # grep — find TODOs (uses ripgrep if available)
        r = call('{"name":"grep","arguments":{"pattern":"TODO","path":"."}}')
        assert r.success, r.error
        assert "TODO" in r.content
        assert "b.py" in r.content
        assert "c.py" in r.content

        # grep — no matches
        r = call('{"name":"grep","arguments":{"pattern":"completely_absent_pattern_xyz"}}')
        assert r.success, r.error
        assert "No matches" in r.content

        # run_bash
        r = call('{"name":"run_bash","arguments":{"command":"echo hello_from_bash"}}')
        assert r.success, r.error
        assert "hello_from_bash" in r.content["stdout"]
        assert r.content["exit_code"] == 0

        # run_bash — timeout handling (use a shorter-than-command sleep)
        r = call('{"name":"run_bash","arguments":{"command":"python -c \\"import time; time.sleep(5)\\"","timeout":1}}')
        assert r.success, r.error
        # Timed-out calls still return success=True at the tool layer;
        # the dict content carries exit_code=-1 + error message.
        assert r.content["exit_code"] == -1
        assert "timed out" in r.content.get("error", "")

        # Memory tool is opt-in
        with tempfile.TemporaryDirectory() as mem_root:
            mem = AgentMemory(workspace=root, root=Path(mem_root))
            reg_with_mem = build_default_registry(root, memory=mem)
            assert reg_with_mem.status()["total"] == 9
            assert "remember" in reg_with_mem.status()["tools"]

            calls = reg_with_mem.parse(
                '<tool_call>{"name":"remember","arguments":{"note":"this is a test note"}}</tool_call>'
            )
            r = reg_with_mem.execute(calls[0])
            assert r.success, r.error
            assert "Saved" in r.content
            assert "test note" in mem.load()

        print("OK: engine/agent_builtins.py smoke test passed")
        print(f"    ripgrep path: {_RG_BIN or '(not found, using stdlib fallback)'}")
