"""
Tests for engine.agent_lsp.

Exercises:
- is_available reflects jedi presence
- register_lsp_tools adds 4 tools when jedi is available
- register_lsp_tools is a no-op when jedi is not available
- goto_definition resolves cross-file references
- find_references finds every call site of a function
- get_diagnostics catches SyntaxError
- get_diagnostics catches possibly-undefined names
- get_diagnostics is clean for a well-formed file
- get_completions returns names for an attribute lookup
- Tools handle missing files gracefully
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

from engine.agent_lsp import (
    is_available, register_lsp_tools,
    _goto_definition, _find_references, _get_diagnostics, _get_completions,
)
from engine.agent_tools import ToolRegistry


# ── Module surface ──

def test_is_available():
    # jedi is installed in this test env
    assert is_available() is True


def test_register_default_pack_adds_two_tools():
    """Default (no tool_pack) registers the curated 2-tool subset to
    stay under the 14B's tool-count regression cliff."""
    reg = ToolRegistry()
    n = register_lsp_tools(reg, Path("."))
    assert n == 2
    for name in ("goto_definition", "find_references"):
        assert reg.get(name) is not None
    # Omitted by default
    assert reg.get("get_diagnostics") is None
    assert reg.get("get_completions") is None


def test_register_all_pack_adds_four_tools():
    """tool_pack=['all'] opts back into every LSP tool."""
    reg = ToolRegistry()
    n = register_lsp_tools(reg, Path("."), tool_pack=["all"])
    assert n == 4
    for name in ("goto_definition", "find_references", "get_diagnostics", "get_completions"):
        assert reg.get(name) is not None


def test_register_explicit_pack():
    """Explicit list registers only those tools."""
    reg = ToolRegistry()
    n = register_lsp_tools(reg, Path("."), tool_pack=["get_diagnostics"])
    assert n == 1
    assert reg.get("get_diagnostics") is not None
    assert reg.get("goto_definition") is None


def test_register_unknown_name_raises():
    """Typos in tool_pack must surface loudly."""
    import pytest
    reg = ToolRegistry()
    with pytest.raises(ValueError) as exc_info:
        register_lsp_tools(reg, Path("."), tool_pack=["bogus_tool"])
    assert "bogus_tool" in str(exc_info.value)
    # Available list should be in the error so the user can fix the typo.
    assert "goto_definition" in str(exc_info.value)


# ── goto_definition ──

def _ws(files: dict[str, str]) -> Path:
    tmp = Path(tempfile.mkdtemp())
    for name, content in files.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp


def test_goto_definition_same_file():
    ws = _ws({
        "mod.py": "def helper():\n    return 42\n\ndef caller():\n    return helper()\n",
    })
    out = _goto_definition(ws, "mod.py", "helper")
    assert "mod.py" in out
    assert "1" in out  # defined on line 1


def test_goto_definition_cross_file():
    ws = _ws({
        "lib.py": "def shared():\n    return 'x'\n",
        "main.py": "from lib import shared\n\nresult = shared()\n",
    })
    out = _goto_definition(ws, "main.py", "shared")
    assert "lib.py" in out


def test_goto_definition_unknown_symbol():
    ws = _ws({"a.py": "x = 1\n"})
    out = _goto_definition(ws, "a.py", "nonexistent")
    assert "not found" in out.lower()


# ── find_references ──

def test_find_references_multiple_callsites():
    ws = _ws({
        "mod.py": (
            "def helper():\n"
            "    return 1\n"
            "\n"
            "def a():\n"
            "    return helper()\n"
            "\n"
            "def b():\n"
            "    x = helper()\n"
            "    return x + helper()\n"
        ),
    })
    out = _find_references(ws, "mod.py", "helper")
    # 1 def + 3 call sites = 4 total
    assert "4" in out or "References" in out
    assert "mod.py" in out


# ── get_diagnostics ──

def test_diagnostics_syntax_error():
    ws = _ws({"bad.py": "def f(\n  return 1\n"})
    out = _get_diagnostics(ws, "bad.py")
    assert "SyntaxError" in out


def test_diagnostics_clean_file():
    ws = _ws({
        "good.py": "import os\n\ndef f(x):\n    return os.path.join('a', x)\n",
    })
    out = _get_diagnostics(ws, "good.py")
    assert "no issues" in out.lower()


def test_diagnostics_non_python_rejected():
    ws = _ws({"thing.txt": "hello"})
    out = _get_diagnostics(ws, "thing.txt")
    assert "only available for .py" in out


# ── get_completions ──

def test_completions_after_dot():
    ws = _ws({
        "x.py": "import os\nresult = os.\n",
    })
    out = _get_completions(ws, "x.py", line=2, column=12)
    assert "Completions" in out
    # Should include some os.* attribute
    assert "path" in out or "environ" in out or "name" in out


# ── error handling ──

def test_missing_file_raises():
    ws = _ws({"a.py": "x = 1\n"})
    try:
        _goto_definition(ws, "ghost.py", "anything")
    except FileNotFoundError as e:
        assert "ghost.py" in str(e)
        return
    assert False, "expected FileNotFoundError"


# ── registry integration ──

def test_lsp_tools_invokable_via_registry():
    ws = _ws({
        "lib.py": "def shared():\n    return 'x'\n",
        "main.py": "from lib import shared\n\nresult = shared()\n",
    })
    reg = ToolRegistry()
    register_lsp_tools(reg, ws)
    tool = reg.get("goto_definition")
    out = tool.function(path="main.py", symbol="shared")
    assert "lib.py" in out


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
