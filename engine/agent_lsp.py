"""
Code-intelligence tools for the agent (modular, optional).

Backed by `jedi` (pure-Python, in-process — preserves the zero-servers/100%-
privacy invariant from feedback_local_ai_privacy.md). NOT a Language Server
Protocol process. The file is named agent_lsp.py because it fills the same
*role* as an LSP would for the agent.

This module is OPT-IN — `register_lsp_tools` is only called when the caller
wants the extra tools. To remove jedi entirely:
  1. Delete this file.
  2. Remove the jedi line from requirements.txt.
  3. Delete the `--lsp` flag handling in ulcagent.py / agent_builtins.py.
The rest of the agent has no jedi dependency.

Tools registered (4):
  goto_definition  — symbol -> file:line of its definition
  find_references  — symbol -> file:line of every usage
  get_diagnostics  — file -> list of static-analysis problems
  get_completions  — file:line:col -> ranked completion candidates

Each tool catches `ImportError` at registration time and degrades gracefully
if jedi is not installed (the registration call simply no-ops).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import jedi as _jedi
    _JEDI_AVAILABLE = True
except ImportError:
    _jedi = None
    _JEDI_AVAILABLE = False


def is_available() -> bool:
    """True if jedi is importable. Call before register_lsp_tools to decide
    whether to advertise the tools at all."""
    return _JEDI_AVAILABLE


def _format_location(name) -> str:
    """jedi Name -> 'path:line:col  description' string."""
    path = name.module_path
    line = name.line or 0
    col = (name.column or 0) + 1
    desc = (name.description or "").strip()
    return f"{path}:{line}:{col}  {desc}"


# ── Tool implementations ────────────────────────────────────────────


def _goto_definition(workspace_root: Path, path: str, symbol: str) -> str:
    if not _JEDI_AVAILABLE:
        raise RuntimeError("jedi not installed — install with: pip install jedi")
    p = (workspace_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    source = p.read_text(encoding="utf-8", errors="replace")
    # Find symbol's first occurrence — caller can re-call with a more specific
    # location if there are multiple matches and the first isn't the right one.
    lines = source.splitlines()
    target_line = None
    target_col = None
    for i, line in enumerate(lines, start=1):
        idx = line.find(symbol)
        if idx >= 0:
            # Make sure it's a whole-word match
            before = line[idx - 1] if idx > 0 else ""
            after = line[idx + len(symbol)] if idx + len(symbol) < len(line) else ""
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                target_line = i
                target_col = idx + 1
                break
    if target_line is None:
        return f"symbol {symbol!r} not found in {path}"

    script = _jedi.Script(source, path=str(p), project=_jedi.Project(str(workspace_root)))
    defs = script.goto(line=target_line, column=target_col, follow_imports=True)
    if not defs:
        return f"no definition found for {symbol!r} (referenced at {path}:{target_line})"
    out = [f"Definitions for {symbol!r} (referenced at {path}:{target_line}):"]
    for d in defs[:10]:
        out.append(f"  {_format_location(d)}")
    if len(defs) > 10:
        out.append(f"  ... +{len(defs) - 10} more")
    return "\n".join(out)


def _find_references(workspace_root: Path, path: str, symbol: str) -> str:
    if not _JEDI_AVAILABLE:
        raise RuntimeError("jedi not installed — install with: pip install jedi")
    p = (workspace_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    source = p.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    target_line = None
    target_col = None
    for i, line in enumerate(lines, start=1):
        idx = line.find(symbol)
        if idx >= 0:
            before = line[idx - 1] if idx > 0 else ""
            after = line[idx + len(symbol)] if idx + len(symbol) < len(line) else ""
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                target_line = i
                target_col = idx + 1
                break
    if target_line is None:
        return f"symbol {symbol!r} not found in {path}"

    script = _jedi.Script(source, path=str(p), project=_jedi.Project(str(workspace_root)))
    refs = script.get_references(line=target_line, column=target_col, include_builtins=False)
    if not refs:
        return f"no references found for {symbol!r}"
    out = [f"References to {symbol!r} ({len(refs)} total):"]
    for r in refs[:50]:
        out.append(f"  {_format_location(r)}")
    if len(refs) > 50:
        out.append(f"  ... +{len(refs) - 50} more")
    return "\n".join(out)


def _get_diagnostics(workspace_root: Path, path: str) -> str:
    """Static-analysis diagnostics for a Python file. Uses py_compile +
    AST analysis — jedi's syntax error reporting is limited, so we layer
    AST on top to catch undefined names, etc."""
    p = (workspace_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix != ".py":
        return f"diagnostics only available for .py files; got {p.suffix}"
    source = p.read_text(encoding="utf-8", errors="replace")

    issues: list[str] = []

    # SyntaxError check
    import ast
    try:
        tree = ast.parse(source, filename=str(p))
    except SyntaxError as e:
        return f"{path}:{e.lineno or 0}:{e.offset or 0}  SyntaxError: {e.msg}"

    # Defined names (names in this module's top-level scope)
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])

    # jedi Name resolution for unresolved references
    if _JEDI_AVAILABLE:
        try:
            script = _jedi.Script(source, path=str(p), project=_jedi.Project(str(workspace_root)))
            for name in script.get_names(all_scopes=False, definitions=False, references=True):
                # Heuristic: if the name appears nowhere as a definition AND
                # jedi can't resolve it, flag it.
                if name.name in defined or name.name in dir(__builtins__):
                    continue
                try:
                    if not name.goto():
                        issues.append(f"{path}:{name.line}:{name.column + 1}  Possibly undefined: {name.name!r}")
                except Exception:
                    pass
        except Exception:
            # jedi parse failures shouldn't kill diagnostics — AST parse already passed
            pass

    if not issues:
        return f"{path}: no issues found"
    return f"{path}: {len(issues)} issue(s)\n" + "\n".join(f"  {i}" for i in issues[:20])


def _get_completions(workspace_root: Path, path: str, line: int, column: int) -> str:
    if not _JEDI_AVAILABLE:
        raise RuntimeError("jedi not installed — install with: pip install jedi")
    p = (workspace_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    source = p.read_text(encoding="utf-8", errors="replace")
    script = _jedi.Script(source, path=str(p), project=_jedi.Project(str(workspace_root)))
    completions = script.complete(line=line, column=column)
    if not completions:
        return f"no completions at {path}:{line}:{column}"
    out = [f"Completions at {path}:{line}:{column}:"]
    for c in completions[:20]:
        out.append(f"  {c.name}  {(c.description or '').strip()[:60]}")
    if len(completions) > 20:
        out.append(f"  ... +{len(completions) - 20} more")
    return "\n".join(out)


# ── Registration ────────────────────────────────────────────────────


# Curated default — `--lsp` (no value) registers ONLY these two. The
# 14B regresses past ~10 tools per feedback_tool_count_regression.md;
# 9 builtin + 4 LSP = 13 (past the cliff). Capping the default at 2
# means the registry sits at 11 — closer to the proven safe count, and
# the omitted tools (get_diagnostics, get_completions) overlap with
# existing capabilities (auto_verify already runs after every write;
# completions are mid-edit affordances the agent rarely uses). Users
# can opt back into the full set with `--lsp=all`.
DEFAULT_LSP_TOOLS = ("goto_definition", "find_references")
ALL_LSP_TOOLS = ("goto_definition", "find_references",
                 "get_diagnostics", "get_completions")


def register_lsp_tools(registry, workspace_root: Path,
                       tool_pack: "list[str] | None" = None) -> int:
    """Register code-intelligence tools on `registry`. Returns the count
    actually registered.

    `tool_pack` (optional):
      - None or [] → register the curated DEFAULT_LSP_TOOLS (2 tools).
        This is what `ulcagent --lsp` (bare flag) selects.
      - ['all'] → register every tool in ALL_LSP_TOOLS (4 tools).
        Selected by `ulcagent --lsp=all`. Past the 14B regression cliff;
        only safe with extended_tools=False or models with looser
        tool-count tolerance.
      - explicit list of names → register only those (validated against
        ALL_LSP_TOOLS; unknown names raise ValueError so typos surface
        loudly).

    Returns 0 if jedi isn't installed (graceful degradation).
    """
    from engine.agent_tools import ToolSchema  # local import — avoid eager dep at module load

    if not _JEDI_AVAILABLE:
        return 0

    # Resolve the tool pack:
    if not tool_pack:
        wanted = set(DEFAULT_LSP_TOOLS)
    elif list(tool_pack) == ["all"]:
        wanted = set(ALL_LSP_TOOLS)
    else:
        wanted = {t.strip() for t in tool_pack if t.strip()}
        unknown = wanted - set(ALL_LSP_TOOLS)
        if unknown:
            raise ValueError(
                f"Unknown LSP tool name(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(ALL_LSP_TOOLS)} (or 'all')."
            )

    workspace_root = Path(workspace_root).resolve()
    schemas = {
        "goto_definition": ToolSchema(
            name="goto_definition",
            description=(
                "Jump to the definition of a Python symbol. Pass the file you're "
                "looking AT (where the symbol is referenced) and the symbol name. "
                "Returns file:line:col of the definition site, even if it's in a "
                "different file. Use this BEFORE editing a function whose body you "
                "haven't seen — much more reliable than grep for cross-file work."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File where the symbol is referenced"},
                    "symbol": {"type": "string", "description": "Symbol name (function, class, variable)"},
                },
                "required": ["path", "symbol"],
            },
            function=lambda path, symbol: _goto_definition(workspace_root, path, symbol),
            category="code_intel",
        ),
        "find_references": ToolSchema(
            name="find_references",
            description=(
                "List every place a Python symbol is used across the project. "
                "Pass the file you're looking AT and the symbol name. Returns "
                "file:line:col for each reference. Use this BEFORE renaming or "
                "removing a symbol — catches call sites grep can miss (e.g. "
                "import aliases, getattr access)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File where the symbol is referenced or defined"},
                    "symbol": {"type": "string", "description": "Symbol name"},
                },
                "required": ["path", "symbol"],
            },
            function=lambda path, symbol: _find_references(workspace_root, path, symbol),
            category="code_intel",
        ),
        "get_diagnostics": ToolSchema(
            name="get_diagnostics",
            description=(
                "Run static analysis on a Python file: SyntaxError, possibly-undefined "
                "names. Faster and more thorough than running the file. Use after a "
                "non-trivial edit to verify the file is well-formed before running tests."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Python file path"},
                },
                "required": ["path"],
            },
            function=lambda path: _get_diagnostics(workspace_root, path),
            category="code_intel",
        ),
        "get_completions": ToolSchema(
            name="get_completions",
            description=(
                "List completions at a specific cursor position in a Python file. "
                "Useful for discovering API surface — e.g. 'what methods does this "
                "object have' — without reading the whole module."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Python file path"},
                    "line": {"type": "integer", "description": "1-indexed line number"},
                    "column": {"type": "integer", "description": "0-indexed column"},
                },
                "required": ["path", "line", "column"],
            },
            function=lambda path, line, column: _get_completions(workspace_root, path, line, column),
            category="code_intel",
        ),
    }

    registered = 0
    for name in ALL_LSP_TOOLS:  # iterate in stable order
        if name in wanted:
            registry.register(schemas[name])
            registered += 1
    return registered
