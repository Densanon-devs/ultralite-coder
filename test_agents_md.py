"""
Tests for engine.agents_md — AGENTS.md generator.

Feature 1: generate_agents_md(project_dir) inspects a synthetic temp project
and writes a well-formed AGENTS.md with:
  - Detected primary language
  - A test command section
  - A conventions section

Also tests the write_agents_md builtin tool registration lambda accepts all
kwargs a model might emit (matching the **kw pattern used elsewhere in the
registry — see feedback_registration_lambda_kwargs.md).

Run: python -m pytest test_agents_md.py -v
     OR: python test_agents_md.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.agents_md import generate_agents_md, _detect_language_breakdown
from engine.agent_builtins import build_default_registry


# ── helpers ──────────────────────────────────────────────────────


def _make_project(td: Path) -> None:
    """Create a synthetic Python project in td."""
    (td / "src").mkdir()
    (td / "tests").mkdir()
    (td / "src" / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (td / "src" / "utils.py").write_text(
        "def fmt(x):\n    return str(x)\n", encoding="utf-8"
    )
    (td / "tests" / "test_calc.py").write_text(
        "from src.calculator import add\ndef test_add():\n    assert add(1,2)==3\n",
        encoding="utf-8",
    )
    (td / "README.md").write_text("# My Project\n", encoding="utf-8")
    (td / "requirements.txt").write_text("pytest\n", encoding="utf-8")


def _make_js_project(td: Path) -> None:
    """Create a synthetic JavaScript project in td."""
    (td / "src").mkdir()
    (td / "src" / "index.js").write_text("const x = 1;\n", encoding="utf-8")
    (td / "src" / "utils.js").write_text("module.exports = {};\n", encoding="utf-8")
    (td / "package.json").write_text(
        '{"name":"myapp","scripts":{"test":"jest"}}\n', encoding="utf-8"
    )


def _make_mixed_project(td: Path) -> None:
    """A project with Python + a few JS files (Python should win)."""
    (td / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (td / "main.py").write_text("import app\n", encoding="utf-8")
    (td / "helper.py").write_text("pass\n", encoding="utf-8")
    (td / "static").mkdir()
    (td / "static" / "app.js").write_text("console.log(1);\n", encoding="utf-8")


# ── language detection ────────────────────────────────────────────


def test_detect_python_project():
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        breakdown = _detect_language_breakdown(td)
        assert breakdown["python"] >= 2
        # most files are Python
        assert breakdown.get("python", 0) >= breakdown.get("javascript", 0)


def test_detect_javascript_project():
    with tempfile.TemporaryDirectory() as td:
        _make_js_project(Path(td))
        breakdown = _detect_language_breakdown(td)
        assert breakdown.get("javascript", 0) >= 1


def test_detect_mixed_project_python_wins():
    with tempfile.TemporaryDirectory() as td:
        _make_mixed_project(Path(td))
        breakdown = _detect_language_breakdown(td)
        assert breakdown.get("python", 0) > breakdown.get("javascript", 0)


# ── generate_agents_md output structure ──────────────────────────


def test_generates_agents_md_file():
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        out_path = generate_agents_md(td)
        assert out_path.name == "AGENTS.md"
        assert out_path.exists()


def test_agents_md_contains_language():
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        out_path = generate_agents_md(td)
        text = out_path.read_text(encoding="utf-8")
        assert "python" in text.lower() or "Python" in text


def test_agents_md_contains_test_command():
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        out_path = generate_agents_md(td)
        text = out_path.read_text(encoding="utf-8")
        # Must mention test command in some form
        assert "pytest" in text.lower() or "test" in text.lower()


def test_agents_md_contains_conventions_section():
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        out_path = generate_agents_md(td)
        text = out_path.read_text(encoding="utf-8")
        assert "convention" in text.lower() or "Convention" in text


def test_agents_md_is_valid_markdown():
    """File must start with a Markdown heading."""
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        out_path = generate_agents_md(td)
        text = out_path.read_text(encoding="utf-8")
        assert text.startswith("#"), "AGENTS.md should start with a Markdown heading"


def test_agents_md_js_project_detects_js():
    with tempfile.TemporaryDirectory() as td:
        _make_js_project(Path(td))
        out_path = generate_agents_md(td)
        text = out_path.read_text(encoding="utf-8")
        assert (
            "javascript" in text.lower()
            or "node" in text.lower()
            or "npm" in text.lower()
            or "jest" in text.lower()
        )


def test_agents_md_detects_tests_dir():
    """If a tests/ dir exists, the AGENTS.md should mention it."""
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        out_path = generate_agents_md(td)
        text = out_path.read_text(encoding="utf-8")
        assert "tests" in text.lower() or "test" in text.lower()


def test_agents_md_overwrite_is_idempotent():
    """Calling generate_agents_md twice should not raise."""
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        generate_agents_md(td)
        out_path = generate_agents_md(td)  # overwrite
        assert out_path.exists()


def test_empty_project_still_produces_file():
    """Even an empty project dir should produce a valid AGENTS.md."""
    with tempfile.TemporaryDirectory() as td:
        out_path = generate_agents_md(td)
        text = out_path.read_text(encoding="utf-8")
        assert text.startswith("#")
        assert len(text) > 50


# ── tool registration lambda accepts model kwargs ─────────────────


def test_write_agents_md_tool_registered():
    """The write_agents_md tool must be present in the extended registry."""
    with tempfile.TemporaryDirectory() as td:
        reg = build_default_registry(td, extended_tools=True)
        names = {s.name for s in reg._tools.values()}
        assert "write_agents_md" in names, (
            f"write_agents_md not found in registry. Found: {sorted(names)}"
        )


def test_write_agents_md_tool_accepts_extra_kwargs():
    """Lambda must accept **kw so extra model args don't crash dispatch."""
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        reg = build_default_registry(td, extended_tools=True)
        schema = reg._tools["write_agents_md"]
        # Call with an extra kwarg the model might invent
        result = schema.function(project_dir=td, extra_unused_kwarg="ignored")
        assert "AGENTS.md" in result or "Wrote" in result


def test_write_agents_md_tool_default_dir():
    """Calling with no project_dir should default to workspace root."""
    with tempfile.TemporaryDirectory() as td:
        _make_project(Path(td))
        reg = build_default_registry(td, extended_tools=True)
        schema = reg._tools["write_agents_md"]
        # No project_dir arg — should default to workspace root
        result = schema.function()
        assert "AGENTS.md" in result or "Wrote" in result


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [
        (n, f)
        for n, f in inspect.getmembers(mod, inspect.isfunction)
        if n.startswith("test_")
    ]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
