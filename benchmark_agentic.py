"""
Phase 14 agentic benchmark (Step 8).

Runs the Ultralight Coder agent on a suite of multi-step real-project tasks
and scores it. Unlike benchmark_realworld.py which measures single-query
codegen accuracy, this measures end-to-end "can the agent actually DO things"
readiness — the number that tells us if the agent is a viable daily driver.

Each task is self-contained:
- `setup(workspace)` — creates the starting files the agent will work with
- `goal` — the natural-language instruction given to the agent
- `check(workspace, result)` — validates the final workspace state AND
  optionally the AgentResult (iterations, tool calls, etc.)

Tasks are ordered easy → hard so early failures tell us where the floor is.
Calibration anchor (LABBench2): realistic multi-step agentic work typically
drops 26–46% from single-query accuracy. Qwen 2.5 Coder 14B is at 99.0% on
V1+V2 single-query, so the floor we're looking at is ~53–73% on this suite.
70% first-try is the Phase 14 success target.

Usage:
    python benchmark_agentic.py                         # run all tasks
    python benchmark_agentic.py --only create_calculator
    python benchmark_agentic.py --config my_config.yaml
    python benchmark_agentic.py --output results.json   # custom output

Results are written to bench_agentic_<model>_<timestamp>.json with per-task
pass/fail, iterations, tool calls, wall time, and failure reason for analysis.
Use graph_gap_analyzer.py-style re-analysis to spot dominant failure modes.

Zero servers, pure in-process, runs the model locally.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# On Windows, stdout defaults to cp1252 which crashes on model output
# containing characters like → (U+2192). Reconfigure to UTF-8 with
# replacement so benchmark event printing never aborts the task.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_CORE_ROOT = PROJECT_ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

logger = logging.getLogger("bench_agentic")


# ── Data types ──────────────────────────────────────────────────


@dataclass
class AgenticTask:
    """A single multi-step task the agent has to complete."""

    name: str
    difficulty: int  # 1 (trivial) to 5 (hard)
    goal: str
    setup: Callable[[Path], None]
    check: Callable[[Path, Any], tuple[bool, str]]
    max_iterations: int = 15
    max_wall_time: float = 600.0


@dataclass
class TaskResult:
    name: str
    difficulty: int
    passed: bool
    reason: str
    iterations: int
    tool_calls: int
    wall_time: float
    stop_reason: str
    failure_types: list[str] = field(default_factory=list)
    compactions: int = 0  # how many context-compaction passes fired during this task
    est_tokens: int = 0   # coarse total generation cost (prompt+completion), for fail-cost ranking
    # Which of the three ceilings blocked this task (COSMO-Agent decomposition).
    # See classify_failure_axes(). Additive/empty by default; one or more of
    # {"schema-format", "toolchain", "feasibility"} on a failed task.
    failure_axes: list[str] = field(default_factory=list)


# ── Task library ────────────────────────────────────────────────


def setup_empty(ws: Path) -> None:
    pass


# ── Ambiguous-anchor stress fixture ──
#
# Designed to reliably produce a 2-streak of STALE_ANCHOR or
# multi-match TOOL_REJECTED failures so the self_heal layer can be
# A/B-measured. The model gets a 100-line file with three functions
# all named `process`, with similar surrounding lines. It must rename
# ONLY the second one. Common failure pattern: the model emits
# `def process(items):` as old_string → matches 3 times → rejection
# with "matches 3 times" error → retries with slightly more context
# from read_file output → either still ambiguous (multi-match again)
# or includes the `    47\t` line-number prefix → STALE_ANCHOR. Either
# path produces same-class consecutive failures.

def _setup_ambiguous_anchor(ws: Path) -> None:
    body = []
    body.append('"""Three processors; only one is the batch-style one."""')
    body.append("")
    body.append("")
    # First `process` — handles streaming input
    body.append("def process(items):")
    body.append("    \"\"\"Stream items one at a time.\"\"\"")
    body.append("    for item in items:")
    body.append("        if item is None:")
    body.append("            continue")
    body.append("        yield item.upper()")
    body.append("")
    body.append("")
    # Pad with unrelated content so the model can't just count line numbers
    for i in range(20):
        body.append(f"def helper_{i}(x):")
        body.append(f"    return x + {i}")
        body.append("")
    body.append("")
    # SECOND `process` — handles BATCH input. THIS is the one to rename.
    body.append("def process(items):")
    body.append("    \"\"\"Group items into batches before yielding.\"\"\"")
    body.append("    batch_size = 16")
    body.append("    buf = []")
    body.append("    for item in items:")
    body.append("        buf.append(item)")
    body.append("        if len(buf) >= batch_size:")
    body.append("            yield buf")
    body.append("            buf = []")
    body.append("    if buf:")
    body.append("        yield buf")
    body.append("")
    body.append("")
    for i in range(20, 40):
        body.append(f"def helper_{i}(x):")
        body.append(f"    return x * {i}")
        body.append("")
    body.append("")
    # THIRD `process` — recursive
    body.append("def process(items):")
    body.append("    \"\"\"Recursive tree walker.\"\"\"")
    body.append("    if not items:")
    body.append("        return")
    body.append("    head, *tail = items")
    body.append("    yield head")
    body.append("    yield from process(tail)")
    body.append("")
    (ws / "processors.py").write_text("\n".join(body) + "\n", encoding="utf-8")


def _check_ambiguous_anchor(ws: Path) -> tuple[bool, str]:
    p = ws / "processors.py"
    if not p.is_file():
        return False, "processors.py is missing"
    text = p.read_text(encoding="utf-8")
    # Must have exactly 2 `process` and 1 `process_batch` at the def level.
    process_defs = sum(
        1 for line in text.splitlines() if line.startswith("def process(")
    )
    batch_defs = sum(
        1 for line in text.splitlines() if line.startswith("def process_batch(")
    )
    if process_defs != 2 or batch_defs != 1:
        return False, (
            f"expected 2 `def process(` + 1 `def process_batch(`, "
            f"got {process_defs} + {batch_defs}"
        )
    # The renamed function must be the BATCH one (contains batch_size).
    # Crude check: locate the `def process_batch(` line, scan the next
    # 12 lines for `batch_size`.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("def process_batch("):
            window = "\n".join(lines[i:i + 12])
            if "batch_size" not in window:
                return False, "process_batch was renamed but its body is not the batch one"
            break
    # Must still parse.
    try:
        compile(text, "processors.py", "exec")
    except SyntaxError as exc:
        return False, f"processors.py SyntaxError: {exc}"
    return True, "second `process` renamed to `process_batch`, file parses"



def setup_calculator(ws: Path) -> None:
    (ws / "calculator.py").write_text(
        '"""Tiny calculator."""\n\n\ndef add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n',
        encoding="utf-8",
    )


def setup_broken_docstring(ws: Path) -> None:
    (ws / "geometry.py").write_text(
        "def area_of_circle(radius):\n    import math\n    return math.pi * radius ** 2\n",
        encoding="utf-8",
    )


def setup_failing_test(ws: Path) -> None:
    (ws / "paginate.py").write_text(
        "def paginate(items, page_size):\n"
        "    pages = []\n"
        "    for i in range(0, len(items), page_size):\n"
        "        pages.append(items[i : i + page_size])\n"
        "    # Bug: extra empty page appended\n"
        "    pages.append([])\n"
        "    return pages\n",
        encoding="utf-8",
    )
    (ws / "test_paginate.py").write_text(
        "from paginate import paginate\n\n\n"
        "def test_paginate_three():\n"
        "    assert paginate([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n\n\n"
        "def test_paginate_empty():\n"
        "    assert paginate([], 2) == []\n",
        encoding="utf-8",
    )


def setup_rename(ws: Path) -> None:
    (ws / "utils.py").write_text(
        "def do_thing(x):\n    return x * 2\n\n\ndef helper():\n    return do_thing(5)\n",
        encoding="utf-8",
    )
    (ws / "main.py").write_text(
        "from utils import do_thing\n\n\nprint(do_thing(10))\n",
        encoding="utf-8",
    )


def setup_add_cli_flag(ws: Path) -> None:
    (ws / "app.py").write_text(
        "import argparse\n"
        "import logging\n\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--name', default='world')\n"
        "    args = parser.parse_args()\n"
        "    logging.basicConfig(level=logging.INFO)\n"
        "    logging.info(f'Hello, {args.name}')\n\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )


def setup_refactor_dataclass(ws: Path) -> None:
    (ws / "person.py").write_text(
        "class Person:\n"
        "    def __init__(self, name, age, email):\n"
        "        self.name = name\n"
        "        self.age = age\n"
        "        self.email = email\n\n"
        "    def greet(self):\n"
        "        return f'Hi, I am {self.name}'\n",
        encoding="utf-8",
    )


def setup_fix_import_cycle(ws: Path) -> None:
    (ws / "a.py").write_text(
        "from b import b_thing\n\n\n"
        "def a_thing():\n"
        "    return 'a:' + b_thing()\n",
        encoding="utf-8",
    )
    (ws / "b.py").write_text(
        "from a import a_thing\n\n\n"
        "def b_thing():\n"
        "    return 'b'\n\n\n"
        "def use_a():\n"
        "    return a_thing()\n",
        encoding="utf-8",
    )


# ── Checks ──────────────────────────────────────────────────────


def check_docstring_added(ws: Path, _result: Any) -> tuple[bool, str]:
    path = ws / "geometry.py"
    if not path.exists():
        return False, "geometry.py missing"
    try:
        import ast
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "area_of_circle":
            doc = ast.get_docstring(node)
            if doc and len(doc.strip()) >= 5:
                return True, f"docstring: {doc.strip()[:60]!r}"
            return False, "area_of_circle exists but has no docstring"
    return False, "area_of_circle not found"


def check_calculator_extended(ws: Path, _result: Any) -> tuple[bool, str]:
    calc = ws / "calculator.py"
    if not calc.exists():
        return False, "calculator.py missing"
    text = calc.read_text(encoding="utf-8")
    if "def multiply" not in text:
        return False, "multiply function missing"
    if "def divide" not in text:
        return False, "divide function missing"
    if "ValueError" not in text:
        return False, "divide does not raise ValueError on /0"
    try:
        compile(text, "calculator.py", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError in calculator.py: {exc}"
    # Actually import and exercise
    import importlib.util
    spec = importlib.util.spec_from_file_location("_bench_calc", calc)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return False, f"import failed: {exc}"
    try:
        assert mod.add(2, 3) == 5
        assert mod.subtract(5, 2) == 3
        assert mod.multiply(4, 3) == 12
        assert mod.divide(10, 2) == 5
        try:
            mod.divide(1, 0)
            return False, "divide(1, 0) did not raise"
        except ValueError:
            pass
        except Exception as exc:
            return False, f"divide(1, 0) raised wrong type: {type(exc).__name__}"
    except AssertionError as exc:
        return False, f"arithmetic assertion failed: {exc}"
    return True, "all 4 functions work + divide-by-zero raises ValueError"


def check_paginate_fixed(ws: Path, _result: Any) -> tuple[bool, str]:
    paginate = ws / "paginate.py"
    if not paginate.exists():
        return False, "paginate.py missing"
    text = paginate.read_text(encoding="utf-8")
    try:
        compile(text, "paginate.py", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    import importlib.util
    spec = importlib.util.spec_from_file_location("_bench_paginate", paginate)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return False, f"import failed: {exc}"
    try:
        result = mod.paginate([1, 2, 3, 4, 5], 2)
        expected = [[1, 2], [3, 4], [5]]
        if result != expected:
            return False, f"paginate([1..5], 2) = {result}, expected {expected}"
        if mod.paginate([], 2) != []:
            return False, f"paginate([], 2) should be []"
    except Exception as exc:
        return False, f"paginate raised: {exc}"
    return True, "off-by-one fixed, both tests pass"


def check_rename(ws: Path, _result: Any) -> tuple[bool, str]:
    utils = ws / "utils.py"
    main = ws / "main.py"
    if not utils.exists() or not main.exists():
        return False, "utils.py or main.py missing"
    utils_text = utils.read_text(encoding="utf-8")
    main_text = main.read_text(encoding="utf-8")
    if "def do_thing" in utils_text:
        return False, "utils.py still has old name `do_thing`"
    if "def compute" not in utils_text:
        return False, "utils.py missing new name `compute`"
    if "do_thing" in main_text:
        return False, "main.py still references `do_thing`"
    if "compute" not in main_text:
        return False, "main.py missing `compute` reference"
    # Verify it runs
    try:
        compile(utils_text, "utils.py", "exec")
        compile(main_text, "main.py", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError after rename: {exc}"
    return True, "renamed in both files, both compile"


def check_add_cli_flag(ws: Path, _result: Any) -> tuple[bool, str]:
    app = ws / "app.py"
    if not app.exists():
        return False, "app.py missing"
    text = app.read_text(encoding="utf-8")
    try:
        compile(text, "app.py", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    if "--verbose" not in text:
        return False, "no --verbose argparse argument"
    # The flag should actually wire into logging — look for DEBUG level set
    if "DEBUG" not in text:
        return False, "--verbose doesn't set logging level to DEBUG"
    # Run it to verify: with --verbose, DEBUG logging should activate
    import subprocess
    try:
        proc = subprocess.run(
            [sys.executable, "app.py", "--verbose"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(ws),
        )
    except Exception as exc:
        return False, f"app.py --verbose failed to run: {exc}"
    if proc.returncode != 0:
        return False, f"app.py --verbose exited {proc.returncode}: {proc.stderr[-200:]}"
    return True, "--verbose flag added, wires to logging, runs cleanly"


def check_refactor_dataclass(ws: Path, _result: Any) -> tuple[bool, str]:
    p = ws / "person.py"
    if not p.exists():
        return False, "person.py missing"
    text = p.read_text(encoding="utf-8")
    if "@dataclass" not in text:
        return False, "no @dataclass decorator"
    if "from dataclasses import" not in text:
        return False, "missing `from dataclasses import dataclass`"
    # def __init__ should be gone
    if "def __init__" in text:
        return False, "old __init__ still present — should be replaced by @dataclass"
    try:
        compile(text, "person.py", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    # Exercise it
    import importlib.util
    spec = importlib.util.spec_from_file_location("_bench_person", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        person = mod.Person(name="Alice", age=30, email="alice@example.com")
        assert person.name == "Alice"
        assert person.age == 30
        assert person.email == "alice@example.com"
        assert "Alice" in person.greet()
    except Exception as exc:
        return False, f"Person dataclass broken: {exc}"
    return True, "converted to @dataclass, all fields work, greet() preserved"


def check_fix_import_cycle(ws: Path, _result: Any) -> tuple[bool, str]:
    a_path = ws / "a.py"
    b_path = ws / "b.py"
    if not (a_path.exists() and b_path.exists()):
        return False, "a.py or b.py missing"
    # The real test: can we import a.py and b.py without ImportError?
    import subprocess
    try:
        proc = subprocess.run(
            [
                sys.executable, "-c",
                "import a; import b; print(a.a_thing(), b.b_thing(), b.use_a())",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(ws),
        )
    except Exception as exc:
        return False, f"import check failed to run: {exc}"
    if proc.returncode != 0:
        err = proc.stderr[-300:].strip()
        return False, f"import still broken: {err}"
    # Functions must still produce the same output
    out = proc.stdout.strip()
    if "a:b" not in out:
        return False, f"a.a_thing() output wrong: {out}"
    if "b" not in out:
        return False, f"b.b_thing() output wrong: {out}"
    return True, f"import cycle fixed, output: {out}"


def check_todo_cli(ws: Path, _result: Any) -> tuple[bool, str]:
    required = ["todo.py", "storage.py", "cli.py", "test_todo.py"]
    missing = [f for f in required if not (ws / f).exists()]
    if missing:
        return False, f"missing files: {missing}"
    for f in required:
        try:
            compile((ws / f).read_text(encoding="utf-8"), f, "exec")
        except SyntaxError as exc:
            return False, f"SyntaxError in {f}: {exc}"

    # Try running pytest
    import subprocess
    try:
        proc = subprocess.run(
            ["pytest", "-q", str(ws)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(ws),
        )
    except Exception as exc:
        return False, f"pytest invocation failed: {exc}"
    if proc.returncode != 0:
        return False, f"pytest failed (rc={proc.returncode}): {proc.stdout[-300:]}"

    # Try running the CLI
    try:
        add_proc = subprocess.run(
            [sys.executable, "cli.py", "add", "buy-milk"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(ws),
        )
    except Exception as exc:
        return False, f"cli.py add failed to run: {exc}"
    if add_proc.returncode != 0:
        return False, f"cli.py add returned {add_proc.returncode}: stderr={add_proc.stderr[-200:]}"

    try:
        list_proc = subprocess.run(
            [sys.executable, "cli.py", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(ws),
        )
    except Exception as exc:
        return False, f"cli.py list failed to run: {exc}"
    if list_proc.returncode != 0:
        return False, f"cli.py list returned {list_proc.returncode}: stderr={list_proc.stderr[-200:]}"
    if "buy-milk" not in list_proc.stdout:
        return False, f"cli.py list output missing buy-milk: {list_proc.stdout[-200:]}"
    return True, "all files, tests pass, CLI add+list work"


# ── Multi-language setups & checks ──────────────────────────────


def setup_js_reducer(ws: Path) -> None:
    (ws / "sum.js").write_text(
        "function sumRange(arr, from, to) {\n"
        "  let total = 0;\n"
        "  // Bug: should iterate up to and INCLUDING `to`, not exclusive\n"
        "  for (let i = from; i < to; i++) {\n"
        "    total += arr[i];\n"
        "  }\n"
        "  return total;\n"
        "}\n\n"
        "module.exports = { sumRange };\n",
        encoding="utf-8",
    )
    (ws / "sum.test.js").write_text(
        "const { sumRange } = require('./sum');\n\n"
        "const arr = [10, 20, 30, 40, 50];\n"
        "const got = sumRange(arr, 1, 3);\n"
        "const want = 90;  // 20 + 30 + 40\n"
        "if (got !== want) {\n"
        "  console.error(`FAIL: sumRange(arr, 1, 3) = ${got}, want ${want}`);\n"
        "  process.exit(1);\n"
        "}\n"
        "console.log('PASS sumRange inclusive');\n",
        encoding="utf-8",
    )


def check_js_reducer_fixed(ws: Path, _result: Any) -> tuple[bool, str]:
    if shutil.which("node") is None:
        return False, "node not installed — cannot verify"
    import subprocess
    try:
        proc = subprocess.run(
            ["node", "sum.test.js"],
            capture_output=True, text=True, timeout=15, cwd=str(ws),
        )
    except Exception as exc:
        return False, f"test runner failed: {exc}"
    if proc.returncode != 0:
        return False, f"test failed: {proc.stderr.strip() or proc.stdout.strip()}"
    return True, "JS test passes after reducer fix"


def setup_add_json_field(ws: Path) -> None:
    (ws / "package.json").write_text(
        '{\n'
        '  "name": "demo",\n'
        '  "version": "0.1.0",\n'
        '  "scripts": {\n'
        '    "start": "node index.js"\n'
        '  }\n'
        '}\n',
        encoding="utf-8",
    )


def check_add_json_field(ws: Path, _result: Any) -> tuple[bool, str]:
    p = ws / "package.json"
    if not p.exists():
        return False, "package.json missing"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"
    if data.get("license") != "MIT":
        return False, f"license field missing or wrong: got {data.get('license')!r}"
    if data.get("name") != "demo":
        return False, "existing name field was modified"
    if "scripts" not in data or data["scripts"].get("start") != "node index.js":
        return False, "existing scripts.start was damaged"
    return True, "license=MIT added, existing fields preserved"


def setup_fix_yaml_indent(ws: Path) -> None:
    # Broken: `host` is one space short of lining up with `port` under `database`
    (ws / "config.yml").write_text(
        "service: webapp\n"
        "database:\n"
        "  port: 5432\n"
        " host: localhost\n"  # <-- only 1 space of indent, should be 2
        "  name: appdb\n"
        "features:\n"
        "  - auth\n"
        "  - billing\n",
        encoding="utf-8",
    )


def check_fix_yaml_indent(ws: Path, _result: Any) -> tuple[bool, str]:
    p = ws / "config.yml"
    if not p.exists():
        return False, "config.yml missing"
    try:
        import yaml  # type: ignore
    except ImportError:
        return False, "pyyaml not installed — cannot verify"
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return False, f"YAML still invalid: {exc}"
    if not isinstance(data, dict):
        return False, f"expected a mapping, got {type(data).__name__}"
    db = data.get("database")
    if not isinstance(db, dict):
        return False, f"database is not a mapping: {db!r}"
    if db.get("port") != 5432 or db.get("host") != "localhost" or db.get("name") != "appdb":
        return False, f"database mapping missing keys: {db}"
    if data.get("features") != ["auth", "billing"]:
        return False, f"features list damaged: {data.get('features')}"
    return True, "YAML parses, database mapping preserved"


def setup_write_bash_lister(ws: Path) -> None:
    (ws / "a.py").write_text("x = 1\n", encoding="utf-8")
    (ws / "b.py").write_text("y = 2\n", encoding="utf-8")
    (ws / "notes.txt").write_text("hi\n", encoding="utf-8")
    sub = ws / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("z = 3\n", encoding="utf-8")
    (sub / "readme.md").write_text("# sub\n", encoding="utf-8")


def check_write_bash_lister(ws: Path, _result: Any) -> tuple[bool, str]:
    p = ws / "list_py.sh"
    if not p.exists():
        return False, "list_py.sh missing"
    # Prefer an explicit bash binary path. On Windows, `bash` on PATH can
    # resolve to WSL, which refuses to exec scripts that live outside the
    # WSL filesystem. Look for git-bash's /usr/bin/bash first.
    bash = None
    for candidate in (r"C:\Program Files\Git\usr\bin\bash.exe", r"C:\Program Files\Git\bin\bash.exe"):
        if Path(candidate).exists():
            bash = candidate
            break
    if bash is None:
        bash = shutil.which("bash")
    if bash is None:
        return False, "bash not installed — cannot verify"
    import subprocess
    try:
        proc = subprocess.run(
            [bash, "list_py.sh"],
            capture_output=True, text=True, timeout=10, cwd=str(ws),
        )
    except Exception as exc:
        return False, f"bash failed: {exc}"
    if proc.returncode != 0:
        return False, f"bash script exited {proc.returncode}: stderr={proc.stderr.strip()[:200]}"
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    # Normalize: backslashes -> forward slashes so relative-path and
    # Windows-absolute-path forms compare on the same axis.
    normalized = [ln.replace("\\", "/") for ln in lines]

    # Suffix-based match. The script may emit relative paths ("a.py",
    # "./a.py", "sub/c.py") or absolute paths
    # ("C:/Users/.../bench_write_bash_lister_xxx/a.py"). Any line whose
    # suffix matches `<required>` or `/<required>` counts as a hit.
    # Decorative noise lines (PowerShell `--------` separators,
    # `FullName` column headers, etc.) won't suffix-match a real .py
    # path, so they're naturally ignored.
    def _has_required(req: str) -> bool:
        return any(ln == req or ln.endswith("/" + req) for ln in normalized)

    required = ["a.py", "b.py", "sub/c.py"]
    missing = [r for r in required if not _has_required(r)]
    if missing:
        return False, f"output missing {missing} — got: {sorted(normalized)[:6]}"
    # Negative check: any actual .txt or .md file should NOT appear as
    # a path. Match on .txt or .md as a path suffix, not anywhere in
    # the line, so column headers like 'FullName' don't trip the rule.
    bad = [ln for ln in normalized if ln.endswith(".txt") or ln.endswith(".md")]
    if bad:
        return False, f"output includes non-.py files: {bad}"
    return True, f"bash script lists all required .py files (a.py, b.py, sub/c.py)"


def setup_extend_real_gallery(ws: Path) -> None:
    """Copy gallery.html verbatim from the real salon-studio-template repo.
    Tests whether the agent can extend a real production static site without
    breaking existing structure."""
    src = Path(__file__).parent.parent / "salon-studio-template" / "gallery.html"
    if not src.exists():
        raise FileNotFoundError(f"fixture source missing: {src}")
    (ws / "gallery.html").write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8",
    )


def check_extend_real_gallery(ws: Path, _result: Any) -> tuple[bool, str]:
    p = ws / "gallery.html"
    if not p.exists():
        return False, "gallery.html missing"
    text = p.read_text(encoding="utf-8")
    # Well-formedness check — run through html.parser and fail on unclosed tags.
    from html.parser import HTMLParser
    class _Checker(HTMLParser):
        def __init__(self):
            super().__init__()
            self.error = None
        def error(self, message):  # noqa
            self.error = message
    try:
        _Checker().feed(text)
    except Exception as exc:
        return False, f"html.parser rejected file: {exc}"
    # Structural checks: no existing categories deleted, new category added.
    for cat in ("color", "balayage", "cuts", "bridal", "extensions"):
        if f'data-filter="{cat}"' not in text:
            return False, f"original filter data-filter=\"{cat}\" was removed"
    if 'data-filter="styling"' not in text:
        return False, "missing new filter button with data-filter=\"styling\""
    styling_items = text.count('data-category="styling"')
    if styling_items < 2:
        return False, f"expected >=2 gallery items with data-category=\"styling\", found {styling_items}"
    # Original item count floor: 12 was the starting fixture. We don't want a
    # rewrite that silently drops them.
    item_count = text.count("gallery-item")
    # gallery-item appears once on each grid tile's div AND potentially in
    # a CSS-like comment. The fixture uses it on tile divs and grid CSS;
    # a realistic lower bound is 10 tile divs + new additions.
    if item_count < 10:
        return False, f"gallery lost items: only {item_count} 'gallery-item' occurrences"
    # Sanity: nav + header intact
    if "site-header" not in text or "main-nav" not in text:
        return False, "site header/nav was damaged"
    return True, f"gallery extended ({styling_items} styling items, {item_count} gallery-item mentions)"


def setup_add_ts_interface(ws: Path) -> None:
    (ws / "user.ts").write_text(
        "// TODO: add a User interface with { id: number, name: string, email?: string }\n"
        "// TODO: add a greet(u: User) function that returns `Hello, ${u.name}`\n",
        encoding="utf-8",
    )
    # A test harness that uses the exports — fails TS compile if the types are wrong
    (ws / "user.test.ts").write_text(
        "import { greet, User } from './user';\n\n"
        "const u: User = { id: 1, name: 'Ada' };\n"
        "const msg: string = greet(u);\n"
        "if (msg !== 'Hello, Ada') {\n"
        "  console.error(`FAIL: got ${msg}`);\n"
        "  process.exit(1);\n"
        "}\n"
        "console.log('PASS');\n",
        encoding="utf-8",
    )


def check_add_ts_interface(ws: Path, _result: Any) -> tuple[bool, str]:
    # We don't require tsc to be installed. A structural check is enough:
    # (a) user.ts exists and parses as a Node-compatible JS file after type
    # erasure (we emit `node --check` on a stripped copy); (b) the content
    # mentions the required shapes.
    p = ws / "user.ts"
    if not p.exists():
        return False, "user.ts missing"
    text = p.read_text(encoding="utf-8")
    if "interface User" not in text and "type User" not in text:
        return False, "no `interface User` or `type User` declaration"
    if "id" not in text or "name" not in text:
        return False, "User type must include id and name fields"
    if "function greet" not in text and "const greet" not in text and "greet =" not in text:
        return False, "greet function not found"
    if "export" not in text:
        return False, "user.ts must export User and greet"
    # Soft syntax check via node --check on a de-typed copy: strip `: TYPE`
    # annotations, `interface ... { ... }` blocks, and `as TYPE` casts, then
    # run node --check. This catches gross parse errors without needing tsc.
    if shutil.which("node") is None:
        return True, "user.ts has required declarations (node not installed, skipped syntax check)"
    import re
    stripped = re.sub(r"interface\s+\w+\s*\{[^}]*\}", "", text, flags=re.DOTALL)
    stripped = re.sub(r":\s*[\w\[\]<>|&?., ]+(?=[,)=;\n])", "", stripped)
    stripped = re.sub(r"\bas\s+\w+", "", stripped)
    tmp = ws / "_user_typecheck.js"
    tmp.write_text(stripped, encoding="utf-8")
    import subprocess
    try:
        proc = subprocess.run(
            ["node", "--check", "_user_typecheck.js"],
            capture_output=True, text=True, timeout=10, cwd=str(ws),
        )
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    if proc.returncode != 0:
        return False, f"user.ts has JS-level syntax errors after type strip: {proc.stderr.strip()[:200]}"
    return True, "user.ts has User type + greet export, parses as JS"


# ── Task suite ──────────────────────────────────────────────────

TASKS: list[AgenticTask] = [
    AgenticTask(
        name="add_docstring",
        difficulty=1,
        goal=(
            "Read geometry.py and add a short docstring to the area_of_circle "
            "function explaining what it computes. Do not change any other behavior."
        ),
        setup=setup_broken_docstring,
        check=check_docstring_added,
        max_iterations=6,
    ),
    AgenticTask(
        name="extend_calculator",
        difficulty=2,
        goal=(
            "Read calculator.py. Add a multiply(a, b) function and a divide(a, b) "
            "function that raises ValueError on division by zero. Do not change "
            "the existing add or subtract functions."
        ),
        setup=setup_calculator,
        check=check_calculator_extended,
        max_iterations=8,
    ),
    AgenticTask(
        name="fix_paginate",
        difficulty=3,
        goal=(
            "The tests in test_paginate.py are failing. Read both files, figure out "
            "the bug in paginate.py, and fix it. Do not change the tests. Then run "
            "the tests with run_tests and confirm they pass."
        ),
        setup=setup_failing_test,
        check=check_paginate_fixed,
        max_iterations=10,
    ),
    AgenticTask(
        name="rename_function",
        difficulty=3,
        goal=(
            "Rename the function `do_thing` to `compute` in utils.py, and update "
            "every reference to it in main.py. Both files should still run after "
            "the rename. Use grep to find all references before editing."
        ),
        setup=setup_rename,
        check=check_rename,
        max_iterations=10,
    ),
    AgenticTask(
        name="add_cli_flag",
        difficulty=3,
        goal=(
            "Read app.py. Add a new `--verbose` argparse flag (store_true). When "
            "--verbose is passed, set the logging level to DEBUG instead of INFO. "
            "Preserve all existing functionality. Then run `python app.py --verbose` "
            "via run_bash and confirm it runs without error."
        ),
        setup=setup_add_cli_flag,
        check=check_add_cli_flag,
        max_iterations=10,
    ),
    AgenticTask(
        name="refactor_dataclass",
        difficulty=4,
        goal=(
            "Read person.py. Convert the Person class to a @dataclass. "
            "Import dataclass from dataclasses, remove the __init__ method, "
            "and keep the greet() method unchanged. The class should still "
            "work with the same constructor arguments (name, age, email)."
        ),
        setup=setup_refactor_dataclass,
        check=check_refactor_dataclass,
        max_iterations=10,
    ),
    AgenticTask(
        name="fix_import_cycle",
        difficulty=4,
        goal=(
            "a.py and b.py have a circular import that will fail at import time. "
            "Read both files to understand the cycle. Break it by moving the "
            "function that uses the cross-file dependency INTO the file that "
            "needs it, or by making the imports lazy (inside the function that "
            "needs them). After your fix, importing both a and b must succeed "
            "and calling a.a_thing(), b.b_thing(), and b.use_a() must all work."
        ),
        setup=setup_fix_import_cycle,
        check=check_fix_import_cycle,
        max_iterations=12,
    ),
    AgenticTask(
        name="build_todo_cli",
        difficulty=5,
        goal=(
            "Build a minimal todo list CLI from scratch in the current directory. "
            "Requirements:\n"
            "1. todo.py — Todo dataclass (id: int, title: str, done: bool default False).\n"
            "2. storage.py — load_todos(path) and save_todos(path, todos) using JSON.\n"
            "3. cli.py — argparse subcommands: add TITLE, list, done ID, clear. Uses todos.json.\n"
            "4. test_todo.py — pytest tests for Todo creation, save/load round-trip.\n"
            "5. Run the tests with run_tests (pytest) and confirm they pass.\n"
            "6. Run 'python cli.py add buy-milk' and 'python cli.py list' via run_bash. "
            "If stderr shows a SyntaxError or Traceback, fix and retry before declaring done."
        ),
        setup=setup_empty,
        check=check_todo_cli,
        max_iterations=20,
        max_wall_time=900.0,
    ),
    # ── Multi-language tasks (Python-only harness would miss real daily-driver work) ──
    AgenticTask(
        name="add_json_field",
        difficulty=1,
        goal=(
            "Read package.json. Add a top-level `\"license\": \"MIT\"` field so the "
            "final JSON has name, version, scripts, AND license. Do not change any "
            "existing fields. Use edit_file or write_file, then verify the file "
            "is still valid JSON."
        ),
        setup=setup_add_json_field,
        check=check_add_json_field,
        max_iterations=6,
    ),
    AgenticTask(
        name="fix_yaml_indent",
        difficulty=2,
        goal=(
            "Read config.yml. It has a YAML indentation error that makes it fail "
            "to parse. Fix the indentation so the file parses as a valid YAML "
            "mapping with `database` containing port (5432), host (localhost), "
            "and name (appdb). Do not change any values. Do not add or remove "
            "keys. Only fix whitespace."
        ),
        setup=setup_fix_yaml_indent,
        check=check_fix_yaml_indent,
        max_iterations=6,
    ),
    AgenticTask(
        name="fix_js_reducer",
        difficulty=2,
        goal=(
            "Read sum.js and sum.test.js. The sumRange(arr, from, to) function is "
            "supposed to sum array elements from index `from` to index `to` "
            "INCLUSIVE on both ends. It has an off-by-one bug. Fix sum.js (not "
            "the test), then run `node sum.test.js` via run_bash and confirm it "
            "prints PASS."
        ),
        setup=setup_js_reducer,
        check=check_js_reducer_fixed,
        max_iterations=8,
    ),
    AgenticTask(
        name="write_bash_lister",
        difficulty=2,
        goal=(
            "Create list_py.sh — a bash script that prints the relative path of "
            "every .py file under the current directory (recursively), one per "
            "line, and nothing else. It must NOT print .txt, .md, or any other "
            "extension. After creating it, run `bash list_py.sh` via run_bash "
            "and confirm the output includes a.py, b.py, and sub/c.py."
        ),
        setup=setup_write_bash_lister,
        check=check_write_bash_lister,
        max_iterations=8,
    ),
    AgenticTask(
        name="add_ts_interface",
        difficulty=3,
        goal=(
            "Read user.ts. Replace the two TODO comments with:\n"
            "1. `export interface User { id: number; name: string; email?: string; }`\n"
            "2. `export function greet(u: User): string { return `Hello, ${u.name}`; }`\n"
            "The file must export BOTH `User` and `greet`. Use a backtick template "
            "literal for the return string. Do not add any other exports."
        ),
        setup=setup_add_ts_interface,
        check=check_add_ts_interface,
        max_iterations=8,
    ),
    # ── Stress fixture: ambiguous-anchor edits (designed to trigger
    # STALE_ANCHOR/multi-match streaks so self_heal can be measured). ──
    AgenticTask(
        name="ambiguous_anchor_rename",
        difficulty=4,
        goal=(
            "Read processors.py. The file defines THREE functions all named "
            "`process` with identical signatures. Rename ONLY the SECOND "
            "function (the one that handles BATCH input — has `batch_size` in "
            "its body) to `process_batch`. The other two `process` functions "
            "must keep their name unchanged. After renaming, the file must "
            "still parse cleanly (auto_verify will check)."
        ),
        setup=lambda ws: _setup_ambiguous_anchor(ws),
        check=lambda ws, _r: _check_ambiguous_anchor(ws),
        max_iterations=15,
        max_wall_time=600.0,
    ),
    # ── Real-codebase fixture: extend a real production HTML file ──
    AgenticTask(
        name="extend_real_gallery",
        difficulty=4,
        goal=(
            "Read gallery.html — it is a real production HTML page for a salon "
            "website. Add a new category \"styling\" to the photo gallery:\n"
            "1. Add a new filter button with `data-filter=\"styling\"` inside the "
            "existing .gallery-filters row, matching the format of the other "
            "filter buttons (e.g. Color, Balayage).\n"
            "2. Add at least 2 new `.gallery-item` divs with "
            "`data-category=\"styling\"` inside .gallery-grid, matching the "
            "placeholder-img + overlay pattern of the existing items.\n"
            "DO NOT delete or rename any existing filter or gallery item. DO NOT "
            "rewrite unrelated sections (header, nav, hero). The page must remain "
            "well-formed HTML."
        ),
        setup=setup_extend_real_gallery,
        check=check_extend_real_gallery,
        max_iterations=12,
        max_wall_time=900.0,
    ),
]


# ── Runner ──────────────────────────────────────────────────────


def classify_failure(task: AgenticTask, result: Any, reason: str) -> list[str]:
    """Tag each failure with one or more failure-mode categories."""
    tags: list[str] = []
    low = reason.lower()
    if "syntax" in low:
        tags.append("syntax_error")
    if "missing" in low and ".py" in low:
        tags.append("missing_file")
    if "import" in low:
        tags.append("import_error")
    if "assertion" in low or "paginate([" in low:
        tags.append("wrong_behavior")
    if "pytest failed" in low:
        tags.append("test_failure")
    if "raise" in low and "not" in low:
        tags.append("missing_raise")
    if result is not None:
        stop = getattr(result, "stop_reason", "")
        if stop == "wall_time":
            tags.append("timeout")
        if stop == "max_iterations":
            tags.append("loop_limit")
        if stop == "model_error":
            tags.append("model_error")
        if getattr(result, "iterations", 0) == 1 and getattr(result, "tool_calls", None) == []:
            tags.append("no_tool_calls")
    if not tags:
        tags.append("unknown")
    return tags


# Harness-injected synthetic ToolResults — these are NOT real tool dispatches,
# so a failed one must never be counted as a "toolchain" failure. `parse_error`
# is handled separately (it's the schema-format signal). Everything else here
# is a nudge/gate the agent loop threads back to steer the model.
_SYNTHETIC_TOOL_NAMES = frozenset({
    "parse_error",
    "auto_verify",
    "auto_apply",
    "task_incomplete",
    "stuck_repeat",
    "mutation_gate",
    "truncated_reasoning",
    "self_heal_diagnose",
})


def classify_failure_axes(result: Any, reason: str) -> list[str]:
    """Decompose a task failure into the three COSMO-Agent ceilings.

    PURE INSTRUMENTATION over signals already present on the AgentResult — does
    not change agent behavior. A failed task gets one or more of:

    - "schema-format": the model emitted a tool call but the JSON/format was
      malformed and could not be parsed/dispatched (synthetic `parse_error`
      ToolResults, i.e. the parser produced errors with zero calls that turn).
    - "toolchain": a tool call parsed and dispatched but the tool itself errored
      (a failed ToolResult whose name is a REAL tool, not a harness nudge).
    - "feasibility": the agent ran to completion (answered) but the produced
      code doesn't actually work — task verification failed despite a clean run.

    Returns [] when `result` is None (e.g. setup/runner crash before any run).
    Axes are additive; a task can hit more than one (a run can have malformed
    calls AND end with broken code).
    """
    if result is None:
        return []
    axes: list[str] = []

    tool_results = getattr(result, "tool_results", None) or []

    # schema-format: any synthetic parse_error observation means the model
    # emitted a call the parser/validator could not turn into a dispatchable
    # ToolCall (multi-line JSON, broken quotes, etc).
    if any(getattr(r, "name", "") == "parse_error" for r in tool_results):
        axes.append("schema-format")

    # toolchain: a real tool dispatched but returned failure. Exclude the
    # harness-injected synthetic results (parse_error/auto_verify/gates/nudges).
    if any(
        not getattr(r, "success", True)
        and getattr(r, "name", "") not in _SYNTHETIC_TOOL_NAMES
        for r in tool_results
    ):
        axes.append("toolchain")

    # feasibility: the run completed normally (model gave a final answer) yet
    # the task's verification still failed — the code itself doesn't work. We
    # only assert this on a clean stop; timeouts/loop-limits/model-errors are
    # process failures, not code-feasibility failures, and are captured by the
    # other axes / stop_reason.
    if getattr(result, "stop_reason", "") == "answered":
        axes.append("feasibility")

    # Fallback: a failed task that produced no parse errors, no failed real
    # tool, and did not stop on "answered" (e.g. max_iterations / wall_time /
    # model_error). Attribute to schema-format only if the model never managed
    # a clean dispatch at all; otherwise leave as "process" so it's visible
    # without being miscredited to a code ceiling.
    if not axes:
        axes.append("process")
    return axes


def run_one_task(
    task: AgenticTask,
    config_path: str,
    auto_approve_risky: bool = True,
    architect_mode: bool = False,
    context_char_budget: Optional[int] = None,
    shared_model: Any = None,
    auto_flag: bool = False,
    enable_self_heal: bool = True,
    json_recovery: bool = False,
    toolset: Optional[str] = None,
) -> TaskResult:
    """Run one task against the agent. Uses a fresh tmp workspace each time.

    If *shared_model* is provided (a pre-loaded BaseModel), it is reused
    instead of loading a fresh model per task. This is critical for models
    with long load times (e.g. Gemma 4 at ~4 min per load). The caller is
    responsible for loading before and unloading after the full bench run.
    """
    from main import UltraliteCodeAssistant
    from engine.agent_memory import AgentMemory, _project_id

    print(f"\n-- [{task.name}] difficulty {task.difficulty} --")
    print(f"   goal: {task.goal[:120]}{'...' if len(task.goal) > 120 else ''}")

    with tempfile.TemporaryDirectory(prefix=f"bench_{task.name}_") as tmp:
        ws = Path(tmp).resolve()
        try:
            task.setup(ws)
        except Exception as exc:
            return TaskResult(
                name=task.name, difficulty=task.difficulty, passed=False,
                reason=f"setup failed: {exc}", iterations=0, tool_calls=0,
                wall_time=0.0, stop_reason="setup_error", failure_types=["setup_error"],
                failure_axes=["process"],
            )

        # Build agent via run_agent_fast's shell approach but use an Agent
        # directly so we can capture the AgentResult object.
        from engine.agent import Agent, AgentEvent
        from engine.agent_builtins import build_default_registry
        from engine.agent_tools import ToolCall
        from densanon.core.config import Config

        config = Config(config_path)
        config.setup_logging()

        # Isolate memory so tasks don't share state
        with tempfile.TemporaryDirectory(prefix=f"mem_{task.name}_") as mem_root:
            memory = AgentMemory(workspace=ws, root=Path(mem_root))
            registry = build_default_registry(ws, memory=memory, toolset=toolset)

            owns_model = shared_model is None
            if shared_model is not None:
                bm = shared_model
            else:
                try:
                    from engine.base_model import BaseModel as LocalBaseModel
                    spec = _load_speculative_config_local(config_path)
                    bm = LocalBaseModel(config.base_model, speculative_config=spec)
                except ImportError:
                    from densanon.core.model_loader import BaseModel as CoreBaseModel
                    bm = CoreBaseModel(config.base_model)
                bm.load()

            # Minimal event renderer — also shows model prose (truncated)
            # so we can diagnose tasks that fail with no tool calls
            def on_event(e: AgentEvent) -> None:
                if e.type == "iteration":
                    print(f"   [iter {e.iteration}]")
                elif e.type == "model_text":
                    text = (e.payload or "").strip()
                    if text and "<tool_call>" not in text:
                        preview = text.replace("\n", " ")
                        if len(preview) > 220:
                            preview = preview[:220] + " ..."
                        print(f"     model: {preview}")
                elif e.type == "tool_call":
                    args_s = ", ".join(f"{k}={str(v)[:40]}" for k, v in e.payload.arguments.items())
                    print(f"     -> {e.payload.name}({args_s[:80]})")
                elif e.type == "tool_result":
                    r = e.payload
                    if r.success:
                        c = str(r.content)[:80]
                        print(f"        ok: {c}")
                    else:
                        print(f"        err: {r.error}")
                elif e.type == "pre_finish_retry":
                    p = e.payload or {}
                    print(
                        f"   [auto-retry #{p.get('retry', '?')}] "
                        f"{p.get('feedback', '')[:120]}"
                    )
                elif e.type == "compacted":
                    p = e.payload or {}
                    print(
                        f"   [compacted #{e.iteration}] {p.get('total_before', 0)} → "
                        f"{p.get('total_after', 0)} chars (budget {p.get('budget', 0)}, "
                        f"elided {p.get('elided_chars', 0)})"
                    )
                elif e.type == "final":
                    print(f"   [done]")

            # Workspace hint
            hint = f"Workspace: {ws}\nProject type: Python (ephemeral benchmark task)"

            # Per-task strategy hints. Injected into the system prompt when the
            # goal matches a known-hard pattern. Keeps the model from picking a
            # wrong-but-plausible approach on the first turn.
            goal_lower = task.goal.lower()
            if "circular import" in goal_lower or "import cycle" in goal_lower:
                hint += (
                    "\n\nCircular-import fix — use exactly these two edits, in this order, "
                    "then stop editing:\n"
                    "1. edit_file path='b.py' old_string='from a import a_thing\\n\\n\\n' "
                    "new_string='' — removes the top-level cross-module import line along "
                    "with the blank lines that follow it.\n"
                    "2. edit_file path='b.py' "
                    "old_string='def use_a():\\n    return a_thing()' "
                    "new_string='def use_a():\\n    from a import a_thing\\n    "
                    "return a_thing()' — adds the lazy import as the first line inside "
                    "use_a.\n"
                    "Do NOT touch a.py. Do NOT move or rename any function. After the two "
                    "edits, verify with run_bash "
                    "`python -c \"import a, b; print(a.a_thing(), b.b_thing(), b.use_a())\"` "
                    "and only then give the final answer."
                )
            # HTML gallery insertion: hint to use insert_at_line instead of
            # edit_file for large HTML files with many ambiguous </div> tags.
            if "gallery" in goal_lower and "gallery-item" in goal_lower:
                hint += (
                    "\n\nHTML insertion tip: this file has many repeated </div> tags, so "
                    "edit_file will fail with 'matches N times'. Instead:\n"
                    "1. read_file the HTML to find the exact line numbers.\n"
                    "2. Use insert_at_line(path, line, text) to insert new elements "
                    "BEFORE a specific line number. This avoids the anchor-matching "
                    "problem entirely.\n"
                    "3. For the filter button: find the line with the last "
                    "filter-btn and insert_at_line on the next line.\n"
                    "4. For gallery items: find the closing </div> of .gallery-grid "
                    "and insert_at_line before it."
                )

            # Pull temperature + max_tokens from the loaded config so that
            # R1-distill's temp=0.6 / max_tokens=4096 profile works without
            # a code change. Fall back to Agent defaults if the config
            # object doesn't expose them (older densanon-core builds).
            try:
                bm_cfg = config.base_model
                cfg_temp = getattr(bm_cfg, "temperature", None)
                cfg_max = getattr(bm_cfg, "max_tokens", None)
                cfg_path = getattr(bm_cfg, "path", "") or ""
            except Exception:
                cfg_temp = None
                cfg_max = None
                cfg_path = ""
            max_tokens_per_turn = int(cfg_max) if cfg_max else 1024
            # Clamp to a sane upper bound so a misconfigured config can't
            # blow up the per-turn inference budget.
            max_tokens_per_turn = max(512, min(max_tokens_per_turn, 8192))
            # Auto-detect reasoning models via model path or config name. These
            # ship with `<think>` reasoning enabled by default and burn 2-4k
            # tokens per turn before emitting any tool call. The suppress_think
            # assistant-header prefix pre-closes an empty <think> so the model
            # jumps straight to action.
            #   - R1-distill family
            #   - Anything with "reasoning" in the name
            #   - Qwen3 family (32B, 30B-A3B, etc — all reasoning-mode by default)
            path_lower = str(cfg_path).lower()
            suppress_think = (
                ("r1" in path_lower and "distill" in path_lower)
                or "reasoning" in path_lower
                or "qwen3" in path_lower
            )
            if architect_mode:
                from engine.architect_agent import ArchitectAgent
                agent = ArchitectAgent(
                    model=bm,
                    registry=registry,
                    system_prompt_extra=hint,
                    workspace_root=ws,
                    memory=memory,
                    auto_verify_python=True,
                    max_iterations_per_step=6,
                    max_wall_time=task.max_wall_time,
                    max_tokens_per_turn=max_tokens_per_turn,
                    temperature=cfg_temp if cfg_temp is not None else 0.1,
                    confirm_risky=(lambda _c: True) if auto_approve_risky else None,
                    # Bench is a trusted, controlled environment — auto-approve
                    # the supply-chain gate the same way as the risky gate so a
                    # benign fixture (e.g. a package.json with a lifecycle
                    # script) can never cause a false-refusal that fails a task.
                    confirm_supply_chain=(lambda _c, _r: True) if auto_approve_risky else None,
                    on_event=on_event,
                )
            else:
                agent_kwargs = dict(
                    model=bm,
                    registry=registry,
                    system_prompt_extra=hint,
                    workspace_root=ws,
                    memory=memory,
                    auto_verify_python=True,
                    suppress_think=suppress_think,
                    max_iterations=task.max_iterations,
                    max_wall_time=task.max_wall_time,
                    max_tokens_per_turn=max_tokens_per_turn,
                    temperature=cfg_temp if cfg_temp is not None else 0.1,
                    confirm_risky=(lambda _c: True) if auto_approve_risky else None,
                    confirm_supply_chain=(lambda _c, _r: True) if auto_approve_risky else None,
                    on_event=on_event,
                    enable_self_heal=enable_self_heal,
                    json_recovery=json_recovery,
                )
                if context_char_budget is not None:
                    agent_kwargs["context_char_budget"] = context_char_budget
                # Wire task.check as pre_finish_check so the model gets
                # feedback on partial work before the run ends.
                def _pre_finish():
                    ok, reason = task.check(ws, None)
                    return None if ok else reason
                agent_kwargs["pre_finish_check"] = _pre_finish
                agent_kwargs["pre_finish_max_retries"] = 2
                agent = Agent(**agent_kwargs)

            start = time.monotonic()
            try:
                result = agent.run(task.goal)
            except Exception as exc:
                if owns_model:
                    bm.unload()
                return TaskResult(
                    name=task.name, difficulty=task.difficulty, passed=False,
                    reason=f"agent.run raised: {exc}",
                    iterations=0, tool_calls=0,
                    wall_time=time.monotonic() - start,
                    stop_reason="exception",
                    failure_types=["exception"],
                    failure_axes=["process"],
                )
            if owns_model:
                bm.unload()

            # Surface json_recovery activity so a 0-firing A/B result is
            # distinguishable from a 0-help one (mirrors the self_heal finding).
            jr_fired = getattr(agent, "_json_recovery_fired", 0)
            if jr_fired:
                jr_ok = getattr(agent, "_json_recovery_recovered", 0)
                print(f"   json_recovery: fired {jr_fired}x, recovered {jr_ok}x")

            passed, reason = task.check(ws, result)
            tags = [] if passed else classify_failure(task, result, reason)
            axes = [] if passed else classify_failure_axes(result, reason)

            # Auto-flag-on-fail: silently write YAML augmentor entries for
            # any known failure pattern detected in this run. Only fires when
            # auto_flag is enabled. No-op for clean runs.
            if auto_flag:
                try:
                    from engine.failure_flagger import flag, summarize
                    from engine.yaml_augmentor_builder import write_all
                    repo_root = Path(__file__).resolve().parent
                    records = flag(result, task.goal)
                    if records:
                        paths = write_all(records, task.goal, repo_root)
                        if paths:
                            counts = summarize(records)
                            summary = ", ".join(f"{k}x{v}" for k, v in counts.items())
                            print(f"   auto-flagged: {summary} -> {len(paths)} YAML(s)")
                except Exception as exc:
                    # Never let the flagger crash the bench
                    print(f"   auto-flag failed: {exc}")

            print(f"   result: {'PASS' if passed else 'FAIL'} — {reason}")
            return TaskResult(
                name=task.name,
                difficulty=task.difficulty,
                passed=passed,
                reason=reason,
                iterations=result.iterations,
                tool_calls=len(result.tool_calls),
                wall_time=result.wall_time,
                stop_reason=result.stop_reason,
                failure_types=tags,
                compactions=getattr(result, "compactions", 0),
                est_tokens=getattr(result, "est_tokens", 0),
                failure_axes=axes,
            )


def _load_speculative_config_local(config_path: str):
    """Local copy of main._load_speculative_config to avoid re-importing main."""
    try:
        import yaml
        from engine.native_speculative import NativeSpeculativeConfig
    except ImportError:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except OSError:
        return None
    spec = raw.get("speculative") or {}
    if not spec.get("enabled", False):
        return None
    return NativeSpeculativeConfig(
        enabled=bool(spec.get("enabled", False)),
        mode=str(spec.get("mode", "prompt_lookup")),
        num_pred_tokens=int(spec.get("num_pred_tokens", 10)),
        max_ngram_size=int(spec.get("max_ngram_size", 2)),
        draft_model_path=str(spec.get("draft_model_path", "")),
        draft_gpu_layers=int(spec.get("draft_gpu_layers", 99)),
        draft_context_length=int(spec.get("draft_context_length", 4096)),
    )


# ── Main ────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 14 agentic benchmark")
    parser.add_argument(
        "--config", default="config_agent14b.yaml",
        help="Agent config YAML (default: config_agent14b.yaml)",
    )
    parser.add_argument(
        "--only", default=None, metavar="NAME",
        help="Run only the task with this name",
    )
    parser.add_argument(
        "--min-difficulty", type=int, default=1,
        help="Only run tasks with difficulty >= this (default 1)",
    )
    parser.add_argument(
        "--max-difficulty", type=int, default=5,
        help="Only run tasks with difficulty <= this (default 5)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: bench_agentic_<timestamp>.json)",
    )
    parser.add_argument(
        "--no-yes", action="store_true",
        help="Disable auto-approve of risky tools (default: approve, since bench is unattended)",
    )
    parser.add_argument(
        "--architect", action="store_true",
        help="Run tasks via ArchitectAgent (plan-then-execute) instead of flat Agent",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, metavar="N",
        help="Run the whole selected suite N times and report pass rate + wall-time "
             "mean/stddev per task. Default 1 (single run). Use this to distinguish "
             "deterministic passes from flaps.",
    )
    parser.add_argument(
        "--share-model", action="store_true",
        help="Load the model once and share across all tasks. Essential for models "
             "with long load times (Gemma 4 ~4min). Default: reload per task.",
    )
    parser.add_argument(
        "--context-budget", type=int, default=None, metavar="CHARS",
        help="Override Agent.context_char_budget. Use a small value (e.g. 4000) to "
             "force transcript compaction during a bench run — useful for testing "
             "the auto-compact code path. Default: Agent's built-in budget (~52000 chars ≈ 16k tokens).",
    )
    parser.add_argument(
        "--auto-flag", action="store_true",
        help="After every task, run the failure flagger and write YAML augmentor "
             "entries to data/auto_generated_review/. Silent on clean tasks. "
             "Use this on bench runs to harvest failure patterns into the augmentor library.",
    )
    parser.add_argument(
        "--auto-promote", action="store_true",
        help="After the bench finishes, run the promoter on the review queue: "
             "validate (schema + replay) + dedup + copy passing YAMLs into "
             "data/augmentor_examples/agentic/. Implies --auto-flag. Use this for "
             "fully-autonomous overnight runs where you want the harvest to land "
             "directly in the retrieval index. Default: harvest only, manual /promote later.",
    )
    parser.add_argument(
        "--no-self-heal", action="store_true",
        help="Suppress the self_heal layer entirely. As of 2026-05-05 the layer "
             "defaults to OFF on the Agent (it never fired across 31 bench runs in "
             "the 2026-05-05 A/B, so it doesn't earn its keep on the standard stack); "
             "this flag is redundant for default runs but kept for symmetry with "
             "--self-heal-on so future toggling is explicit. Use --self-heal-on to "
             "engage the layer for stress scenarios.",
    )
    parser.add_argument(
        "--self-heal-on", action="store_true",
        help="Engage the self_heal diagnose-and-repair layer (default OFF as of "
             "2026-05-05). Useful when running stress benches or smaller models "
             "with weaker tool-call discipline that may form same-class failure streaks.",
    )
    parser.add_argument(
        "--toolset", default=None,
        help="Curated tool profile to bench (coding/refactor/git/web/full). "
             "Default None = lean 10-tool core. Use to A/B that a themed subset "
             "holds the 97.6%% baseline where the 24-tool 'full' set regresses.",
    )
    parser.add_argument(
        "--json-recovery", action="store_true",
        help="Engage the json_recovery layer (default OFF): on an unparseable "
             "tool call (parser returns errors, zero calls) do ONE ChatML "
             "regeneration pass asking the model for valid JSON before falling "
             "through to the normal parse-error feedback. Targets the "
             "JSON-quote-recovery ceiling. Use for an A/B vs the baseline; "
             "per-task firing counts are printed so 0-firing is visible.",
    )
    args = parser.parse_args()

    # --auto-promote implies --auto-flag (otherwise nothing lands in the
    # review queue for the promoter to walk).
    if getattr(args, "auto_promote", False):
        args.auto_flag = True

    # Logging: info for UCA, warning for llama-cpp noise
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    tasks = [
        t for t in TASKS
        if args.min_difficulty <= t.difficulty <= args.max_difficulty
    ]
    if args.only:
        tasks = [t for t in tasks if t.name == args.only]
        if not tasks:
            print(f"No task matching --only {args.only!r}. Available: {[t.name for t in TASKS]}")
            return 2

    print(f"\nPhase 14 agentic benchmark — {len(tasks)} tasks")
    print(f"Config: {args.config}")
    print()

    # Optional model preloading — avoids per-task reload overhead for slow-
    # loading models like Gemma 4 (~4 min per load vs ~4s for Qwen 14B).
    shared_model = None
    if args.share_model:
        from densanon.core.config import Config
        config = Config(args.config)
        config.setup_logging()
        print("Pre-loading model (--share-model)...")
        try:
            from engine.base_model import BaseModel as LocalBaseModel
            spec = _load_speculative_config_local(args.config)
            shared_model = LocalBaseModel(config.base_model, speculative_config=spec)
        except ImportError:
            from densanon.core.model_loader import BaseModel as CoreBaseModel
            shared_model = CoreBaseModel(config.base_model)
        shared_model.load()
        print("Model loaded — sharing across all tasks.\n")

    results: list[TaskResult] = []
    repeats = max(1, args.repeat)
    aborted = False
    t0 = time.monotonic()
    for run_idx in range(repeats):
        if repeats > 1:
            print(f"\n--- Run {run_idx + 1} of {repeats} ---")
        for task in tasks:
            try:
                r = run_one_task(
                    task, args.config,
                    auto_approve_risky=not args.no_yes,
                    architect_mode=args.architect,
                    context_char_budget=args.context_budget,
                    shared_model=shared_model,
                    auto_flag=args.auto_flag,
                    # default OFF; --self-heal-on engages, --no-self-heal
                    # is redundant but kept for explicit-suppression callers.
                    enable_self_heal=(args.self_heal_on and not args.no_self_heal),
                    json_recovery=args.json_recovery,
                    toolset=args.toolset,
                )
            except KeyboardInterrupt:
                print("\n[aborted by user]")
                aborted = True
                break
            except Exception:
                traceback.print_exc()
                r = TaskResult(
                    name=task.name, difficulty=task.difficulty, passed=False,
                    reason="runner crashed (see traceback above)",
                    iterations=0, tool_calls=0, wall_time=0.0,
                    stop_reason="runner_crash",
                    failure_types=["runner_crash"],
                    failure_axes=["process"],
                )
            results.append(r)
        if aborted:
            break

    total_wall = time.monotonic() - t0
    if shared_model is not None:
        shared_model.unload()
        print("Shared model unloaded.")

    # Flat per-run report (unchanged format)
    passed_count = sum(1 for r in results if r.passed)
    total_runs = len(results)
    print(f"\n=== Results ({passed_count}/{total_runs} passed, {total_wall:.0f}s total) ===\n")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(
            f"  [{mark}] [diff {r.difficulty}] {r.name:<20} "
            f"iter={r.iterations:<2} calls={r.tool_calls:<2} "
            f"wall={r.wall_time:5.1f}s stop={r.stop_reason}"
        )
        if not r.passed:
            print(f"       -> {r.reason}")
            if r.failure_types:
                print(f"       tags: {', '.join(r.failure_types)}")

    # Per-task aggregate (only meaningful when repeats > 1)
    by_task: dict[str, list[TaskResult]] = {}
    for r in results:
        by_task.setdefault(r.name, []).append(r)

    if repeats > 1:
        print(f"\n=== Per-task aggregate over {repeats} repeats ===\n")
        import statistics
        for name, runs in by_task.items():
            passes = sum(1 for r in runs if r.passed)
            walls = [r.wall_time for r in runs]
            mean_w = statistics.mean(walls) if walls else 0.0
            std_w = statistics.stdev(walls) if len(walls) > 1 else 0.0
            print(
                f"  {name:<24} {passes}/{len(runs)} pass "
                f"({passes / len(runs):.0%}) "
                f"wall mean={mean_w:5.1f}s stddev={std_w:4.1f}s"
            )

    # Failure mode histogram
    if any(not r.passed for r in results):
        print("\nFailure modes:")
        counts: dict[str, int] = {}
        for r in results:
            if r.passed:
                continue
            for tag in r.failure_types:
                counts[tag] = counts.get(tag, 0) + 1
        for tag, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {tag:<20} {n}")

        # Fail-cost ranking (Open Agent Leaderboard: failed runs cost 20-54%
        # more tokens than successful ones). Surface the most wasteful failures
        # first so fixes are prioritized by token-leverage, not just frequency.
        failed = [r for r in results if not r.passed and r.est_tokens]
        if failed:
            print("\nFailures by token cost (most wasteful first):")
            for r in sorted(failed, key=lambda r: -r.est_tokens):
                tags = ",".join(r.failure_types) or "?"
                print(f"  {r.est_tokens:>8} tok  {r.name:<24} [{tags}] ({r.stop_reason})")

        # Failures by axis (COSMO-Agent decomposition): tell us WHICH ceiling
        # blocked each task — a malformed-JSON ceiling (schema-format), a tool
        # that errored (toolchain), or code that ran but doesn't work
        # (feasibility). A task can land on more than one axis, so axis counts
        # may sum to more than the failed-task count. Pure instrumentation —
        # prioritizes engineering effort by ceiling, not just task name.
        axis_failed = [r for r in results if not r.passed]
        if axis_failed:
            axis_counts: dict[str, int] = {}
            for r in axis_failed:
                for ax in (r.failure_axes or ["process"]):
                    axis_counts[ax] = axis_counts.get(ax, 0) + 1
            print("\nFailures by axis (which ceiling blocked the task):")
            for ax in ("schema-format", "toolchain", "feasibility", "process"):
                if ax in axis_counts:
                    print(f"  {ax:<16} {axis_counts[ax]}")
            # Any axes outside the canonical set (future-proofing) sort last.
            for ax, n in sorted(axis_counts.items(), key=lambda kv: -kv[1]):
                if ax not in ("schema-format", "toolchain", "feasibility", "process"):
                    print(f"  {ax:<16} {n}")

    # JSON output
    output_path = args.output or f"bench_agentic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import statistics as _stats
    by_task_payload = []
    for name, runs in by_task.items():
        walls = [r.wall_time for r in runs]
        passes = sum(1 for r in runs if r.passed)
        by_task_payload.append({
            "name": name,
            "difficulty": runs[0].difficulty,
            "total_runs": len(runs),
            "passed": passes,
            "pass_rate": passes / len(runs),
            "wall_mean": _stats.mean(walls) if walls else 0.0,
            "wall_stddev": _stats.stdev(walls) if len(walls) > 1 else 0.0,
            "wall_min": min(walls) if walls else 0.0,
            "wall_max": max(walls) if walls else 0.0,
        })
    payload = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "repeats": repeats,
        "total_tasks": total_runs,
        "passed": passed_count,
        "pass_rate": passed_count / max(total_runs, 1),
        "total_wall_time": total_wall,
        "tasks": [
            {
                "name": r.name,
                "difficulty": r.difficulty,
                "passed": r.passed,
                "reason": r.reason,
                "iterations": r.iterations,
                "tool_calls": r.tool_calls,
                "wall_time": r.wall_time,
                "stop_reason": r.stop_reason,
                "failure_types": r.failure_types,
                "failure_axes": r.failure_axes,
                "compactions": r.compactions,
            }
            for r in results
        ],
        "by_task": by_task_payload,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults written to {output_path}")

    # Auto-promote: end-of-run pass over the review queue, validating +
    # deduping + copying passing YAMLs into the live retrieval index.
    # Implies --auto-flag (harvest before promote). Off by default to
    # avoid accidentally polluting retrieval; opt-in for unattended runs.
    if getattr(args, "auto_promote", False):
        try:
            from engine.augmentor_promoter import promote_all, summarize as p_summarize
            repo_root = Path(__file__).resolve().parent
            promo_results = promote_all(repo_root)
            counts = p_summarize(promo_results)
            print(
                f"Auto-promote: {counts['promoted']} promoted, "
                f"{counts['skipped']} skipped (dedup/exists), "
                f"{counts['rejected']} rejected (validation failed)"
            )
            if counts["rejected"]:
                # Surface the first few rejections so the user knows what to investigate
                rejected = [r for r in promo_results if not r.promoted
                            and "validation failed" in r.reason]
                for r in rejected[:5]:
                    print(f"  REJECT {r.source.name}: {r.reason}")
        except Exception as exc:
            print(f"Auto-promote failed: {exc}")

    return 0 if passed_count == total_runs else 1


if __name__ == "__main__":
    sys.exit(main())
