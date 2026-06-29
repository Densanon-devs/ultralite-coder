"""
AGENTS.md Generator — Feature 1 of Phase 14+

Inspects a project directory and emits a well-formed AGENTS.md file that
follows the emerging AGENTS.md convention (OpenAI spec, 60k+ repos, Linux
Foundation AAIF). The file tells AI coding agents:

  - What language(s) the project uses
  - How to run tests
  - Key directories and entry points
  - Code conventions and what to avoid

Design constraints:
- Markdown only — no external deps, stdlib only.
- No GGUF inference — purely static inspection.
- Does NOT change existing tool return-string prefixes.
- Safe to overwrite on repeated calls (idempotent structure).
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

# ── Language detection ───────────────────────────────────────────

# Mapping: file extension → canonical language label
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sh": "bash",
    ".bash": "bash",
    ".sql": "sql",
    ".php": "php",
    ".lua": "lua",
}

# Directories to skip when walking
_SKIP_DIRS = frozenset({
    "__pycache__", "node_modules", ".git", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
    ".idea", ".vscode", "coverage", ".coverage",
})


def _detect_language_breakdown(project_dir: str | Path) -> dict[str, int]:
    """Walk the project and return a Counter of {language: file_count}.

    Only counts source files — skips test files for the language survey
    so the primary language isn't skewed by a large test suite in a
    different language (e.g. a Python project with a JS test harness).
    """
    counts: Counter[str] = Counter()
    root = Path(project_dir).resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune noise dirs in-place
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            lang = _EXT_TO_LANG.get(ext)
            if lang:
                counts[lang] += 1

    return dict(counts)


def _primary_language(breakdown: dict[str, int]) -> str:
    """Return the language with the most files, or 'unknown'."""
    if not breakdown:
        return "unknown"
    return max(breakdown, key=lambda k: breakdown[k])


# ── Project structure detection ──────────────────────────────────


def _detect_test_dirs(root: Path) -> list[str]:
    """Return relative paths to test directories (common conventions)."""
    candidates = ["tests", "test", "spec", "__tests__", "test_suite"]
    found = []
    for name in candidates:
        p = root / name
        if p.is_dir():
            found.append(name)
    return found


def _detect_test_command(root: Path, primary_lang: str) -> str:
    """Infer the most likely test command for this project."""
    # Python
    if primary_lang == "python":
        if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
            return "python -m pytest"
        if (root / "setup.py").exists() or (root / "tox.ini").exists():
            return "python -m pytest"
        return "python -m pytest"

    # JavaScript / TypeScript
    if primary_lang in ("javascript", "typescript"):
        pkg = root / "package.json"
        if pkg.exists():
            try:
                data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
                scripts = data.get("scripts", {})
                if "test" in scripts:
                    cmd = scripts["test"]
                    # Return the npm run test shorthand
                    return "npm test"
                if "jest" in scripts or (root / "jest.config.js").exists():
                    return "npx jest"
                if "vitest" in scripts:
                    return "npx vitest"
            except (json.JSONDecodeError, OSError):
                pass
        return "npm test"

    # Go
    if primary_lang == "go":
        return "go test ./..."

    # Rust
    if primary_lang == "rust":
        return "cargo test"

    # Ruby
    if primary_lang == "ruby":
        if (root / "Gemfile").exists():
            return "bundle exec rspec"
        return "ruby -Itest test/**/*_test.rb"

    # Java
    if primary_lang == "java":
        if (root / "pom.xml").exists():
            return "mvn test"
        if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
            return "gradle test"
        return "mvn test"

    # C#
    if primary_lang == "csharp":
        return "dotnet test"

    # Kotlin
    if primary_lang == "kotlin":
        if (root / "build.gradle.kts").exists():
            return "gradle test"
        return "gradle test"

    return f"<run tests for {primary_lang}>"


def _detect_build_command(root: Path, primary_lang: str) -> Optional[str]:
    """Return a build command if one is applicable."""
    if primary_lang == "typescript":
        if (root / "tsconfig.json").exists():
            return "npx tsc"
    if primary_lang == "go":
        return "go build ./..."
    if primary_lang == "rust":
        return "cargo build"
    if primary_lang == "java":
        if (root / "pom.xml").exists():
            return "mvn package -DskipTests"
        if (root / "build.gradle").exists():
            return "gradle build"
    if primary_lang == "csharp":
        return "dotnet build"
    if primary_lang == "kotlin":
        return "gradle assemble"
    return None


def _detect_lint_command(root: Path, primary_lang: str) -> Optional[str]:
    """Return a lint/format command if common config files exist."""
    if primary_lang == "python":
        if (root / ".ruff.toml").exists() or (root / "ruff.toml").exists():
            return "ruff check ."
        if (root / ".flake8").exists() or (root / "setup.cfg").exists():
            return "flake8 ."
        # Check pyproject.toml for ruff/flake8
        pp = root / "pyproject.toml"
        if pp.exists():
            try:
                content = pp.read_text(encoding="utf-8", errors="replace")
                if "ruff" in content:
                    return "ruff check ."
                if "flake8" in content:
                    return "flake8 ."
            except OSError:
                pass
        return None
    if primary_lang in ("javascript", "typescript"):
        if (root / ".eslintrc.js").exists() or (root / ".eslintrc.json").exists():
            return "npx eslint ."
        if (root / ".eslintrc.cjs").exists() or (root / "eslint.config.js").exists():
            return "npx eslint ."
        pkg = root / "package.json"
        if pkg.exists():
            try:
                data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
                scripts = data.get("scripts", {})
                if "lint" in scripts:
                    return "npm run lint"
            except (json.JSONDecodeError, OSError):
                pass
    if primary_lang == "rust":
        return "cargo clippy"
    if primary_lang == "go":
        return "golint ./..."
    return None


def _detect_entry_points(root: Path, primary_lang: str) -> list[str]:
    """Return likely entry-point files."""
    candidates: list[str] = []

    if primary_lang == "python":
        for name in ["main.py", "app.py", "cli.py", "server.py", "__main__.py", "run.py"]:
            if (root / name).exists():
                candidates.append(name)
    elif primary_lang in ("javascript", "typescript"):
        for name in ["index.js", "index.ts", "src/index.js", "src/index.ts",
                     "app.js", "app.ts", "server.js"]:
            if (root / name).exists():
                candidates.append(name)
    elif primary_lang == "go":
        for name in ["main.go", "cmd/main.go"]:
            if (root / name).exists():
                candidates.append(name)
    elif primary_lang == "rust":
        if (root / "src" / "main.rs").exists():
            candidates.append("src/main.rs")
        elif (root / "src" / "lib.rs").exists():
            candidates.append("src/lib.rs")

    return candidates


def _detect_key_dirs(root: Path) -> list[str]:
    """Return notable subdirectories (non-noise, non-hidden)."""
    _IGNORE = frozenset({
        "__pycache__", "node_modules", ".git", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
        ".idea", ".vscode", ".github",
    })
    found = []
    try:
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and child.name not in _IGNORE:
                found.append(child.name)
    except OSError:
        pass
    return found[:12]  # cap at 12 dirs


# ── Conventions per language ──────────────────────────────────────


_CONVENTIONS: dict[str, list[str]] = {
    "python": [
        "Use 4-space indentation (PEP 8).",
        "Prefer f-strings over `.format()` or `%` for string formatting.",
        "Add type hints to all new functions and methods.",
        "Write docstrings for public functions, classes, and modules.",
        "Do not import `*` from any module.",
        "Keep functions short and single-purpose.",
    ],
    "javascript": [
        "Use `const` by default; only use `let` when reassignment is needed.",
        "Use arrow functions for callbacks and short expressions.",
        "Prefer async/await over raw `.then()` chains.",
        "Use strict equality (`===`) instead of loose equality (`==`).",
        "Add JSDoc comments to exported functions.",
    ],
    "typescript": [
        "Avoid `any`; use explicit types or `unknown`.",
        "Use `const` by default; only use `let` when reassignment is needed.",
        "Prefer `interface` over `type` for object shapes.",
        "Enable `strict` mode in tsconfig.",
        "Add JSDoc comments to exported functions.",
    ],
    "go": [
        "Follow `gofmt` formatting — run `go fmt ./...` before committing.",
        "Handle every `error` return value; never ignore with `_`.",
        "Use short variable names for short-lived variables (e.g., `i`, `n`, `err`).",
        "Exported identifiers must have GoDoc comments.",
        "Prefer table-driven tests.",
    ],
    "rust": [
        "Follow `rustfmt` formatting — run `cargo fmt` before committing.",
        "Prefer `Result<T, E>` over `unwrap()`/`expect()` in library code.",
        "Use `clippy` to catch common mistakes (`cargo clippy`).",
        "Document public items with `///` doc comments.",
        "Prefer iterators over index loops.",
    ],
    "ruby": [
        "Follow Ruby style guide (2-space indentation, snake_case).",
        "Use frozen string literals (`# frozen_string_literal: true`).",
        "Prefer `do...end` for multi-line blocks, `{ }` for single-line.",
        "Write RSpec examples that are small and independent.",
    ],
    "java": [
        "Follow Google Java Style Guide.",
        "Use 4-space indentation.",
        "Keep methods under 30 lines where possible.",
        "Write Javadoc for public APIs.",
        "Prefer composition over inheritance.",
    ],
    "csharp": [
        "Follow Microsoft C# coding conventions.",
        "Use `var` when the type is obvious from the right-hand side.",
        "Use async/await for I/O-bound operations.",
        "Add XML doc comments to public APIs.",
        "Prefer `using` statements to manage IDisposable resources.",
    ],
}

_WHAT_TO_AVOID: dict[str, list[str]] = {
    "python": [
        "Do not use mutable default arguments (e.g., `def f(x=[])`).",
        "Do not catch bare `except:` — always catch a specific exception type.",
        "Do not use `print()` for structured logging — use the `logging` module.",
        "Do not commit `*.pyc` files or `__pycache__/` directories.",
    ],
    "javascript": [
        "Do not use `var` — use `const` or `let`.",
        "Do not rely on implicit type coercion.",
        "Do not push secrets or API keys; use environment variables.",
        "Do not leave `console.log()` calls in production code.",
    ],
    "typescript": [
        "Do not use `any` — prefer `unknown` with type guards.",
        "Do not suppress TypeScript errors with `// @ts-ignore` without a comment.",
        "Do not push `dist/` or compiled output to version control.",
    ],
    "go": [
        "Do not ignore error returns.",
        "Do not use `panic()` in library code.",
        "Do not commit vendor/ if go modules are used.",
    ],
    "rust": [
        "Do not call `.unwrap()` or `.expect()` in library code without justification.",
        "Do not ignore `#[must_use]` return values.",
        "Do not commit `target/` directory.",
    ],
}


def _conventions_block(lang: str) -> str:
    lines = _CONVENTIONS.get(lang, [
        "Follow the existing code style in this project.",
        "Keep functions short and single-purpose.",
        "Write tests for new functionality.",
    ])
    avoid = _WHAT_TO_AVOID.get(lang, [])
    out = "### Code Conventions\n\n"
    out += "\n".join(f"- {line}" for line in lines)
    if avoid:
        out += "\n\n### What to Avoid\n\n"
        out += "\n".join(f"- {line}" for line in avoid)
    return out


# ── AGENTS.md assembly ───────────────────────────────────────────


def generate_agents_md(project_dir: str | Path) -> Path:
    """Inspect *project_dir* and write an AGENTS.md at its root.

    Returns the Path to the written file.

    The file is always named AGENTS.md and lives at the project root.
    Calling this multiple times is safe (overwrites with fresh content).
    """
    root = Path(project_dir).resolve()
    breakdown = _detect_language_breakdown(root)
    lang = _primary_language(breakdown)
    test_dirs = _detect_test_dirs(root)
    test_cmd = _detect_test_command(root, lang)
    build_cmd = _detect_build_command(root, lang)
    lint_cmd = _detect_lint_command(root, lang)
    entry_points = _detect_entry_points(root, lang)
    key_dirs = _detect_key_dirs(root)

    project_name = root.name

    sections: list[str] = []

    # ── Title ──
    sections.append(f"# AGENTS.md — {project_name}\n")
    sections.append(
        "This file tells AI coding agents (Claude Code, Codex, Cursor, etc.) "
        "how to work in this project.\n"
    )

    # ── Language ──
    if breakdown:
        lang_summary = ", ".join(
            f"{k} ({v} file{'s' if v != 1 else ''})"
            for k, v in sorted(breakdown.items(), key=lambda x: -x[1])
        )
        sections.append(f"## Language\n\nPrimary: **{lang}**\n\nAll: {lang_summary}\n")
    else:
        sections.append(f"## Language\n\nPrimary: **{lang}**\n")

    # ── Project structure ──
    structure_lines = []
    for d in key_dirs:
        p = root / d
        if p.is_dir():
            structure_lines.append(f"- `{d}/`")
    if entry_points:
        for ep in entry_points:
            structure_lines.append(f"- `{ep}` — main entry point")
    if structure_lines:
        sections.append(
            "## Project Structure\n\n"
            + "\n".join(structure_lines)
            + "\n"
        )

    # ── Commands ──
    cmd_lines = [f"- **Test:** `{test_cmd}`"]
    if test_dirs:
        cmd_lines[0] += f"  (tests in `{'`, `'.join(test_dirs)}/`)"
    if build_cmd:
        cmd_lines.append(f"- **Build:** `{build_cmd}`")
    if lint_cmd:
        cmd_lines.append(f"- **Lint:** `{lint_cmd}`")

    sections.append(
        "## Commands\n\n"
        "Run these commands from the project root:\n\n"
        + "\n".join(cmd_lines)
        + "\n"
    )

    # ── Conventions ──
    sections.append(f"## Conventions\n\n{_conventions_block(lang)}\n")

    # ── General agent guidelines ──
    sections.append(
        "## Agent Guidelines\n\n"
        "- Always read a file before editing it.\n"
        "- Make minimal targeted changes; prefer `edit_file` over full rewrites.\n"
        "- Run the test command after any code change.\n"
        "- Do not modify `AGENTS.md` unless instructed by the user.\n"
        "- If a task is ambiguous, ask the user for clarification before proceeding.\n"
    )

    content = "\n".join(sections)
    out = root / "AGENTS.md"
    out.write_text(content, encoding="utf-8")
    return out
