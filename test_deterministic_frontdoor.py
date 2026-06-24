"""
No-GPU unit tests for the deterministic front-door (engine/deterministic_frontdoor.py).

Every test runs on a tmpdir fixture with ZERO model inference — the whole
point of the front-door is to never call the 14B. Covers, per category:
  - correct execution on a fixture,
  - abstention (returns None) on ambiguous / generative input,
and the default-OFF wiring guard (the REPL flag gate is byte-for-byte legacy
when --frontdoor is absent).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.deterministic_frontdoor import (  # noqa: E402
    DeterministicFrontDoor,
    FrontDoorMatch,
    _looks_generative,
    _has_multiple_clauses,
    _module_importable,
)


@pytest.fixture
def ws():
    d = tempfile.mkdtemp(prefix="fd_test_")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fd(ws: Path) -> DeterministicFrontDoor:
    return DeterministicFrontDoor(ws)


# ── global abstention guards ─────────────────────────────────────


def test_empty_goal_abstains(ws):
    assert _fd(ws).try_handle("") is None
    assert _fd(ws).try_handle("   ") is None


def test_generative_marker_abstains(ws):
    (ws / "a.py").write_text("def old():\n    return 1\n", encoding="utf-8")
    # "rename old to new that returns 2" trips the generative guard.
    assert _fd(ws).try_handle("rename old to new that returns 2") is None


def test_multi_clause_abstains(ws):
    (ws / "a.py").write_text("def old():\n    return 1\n", encoding="utf-8")
    assert _fd(ws).try_handle("rename old to new and add a docstring") is None


def test_helpers():
    assert _looks_generative("fix the bug in parser")
    assert not _looks_generative("rename foo to bar")
    assert _has_multiple_clauses("create a.py and create b.py")
    assert not _has_multiple_clauses("create a.py")


# ── rename ───────────────────────────────────────────────────────


def test_rename_across_files(ws):
    (ws / "mod.py").write_text("def old_name():\n    return 1\n", encoding="utf-8")
    (ws / "use.py").write_text(
        "from mod import old_name\n\nprint(old_name())\n", encoding="utf-8"
    )
    m = _fd(ws).try_handle("rename old_name to new_name")
    assert isinstance(m, FrontDoorMatch)
    assert m.category == "rename"
    assert sorted(m.changed_files) == ["mod.py", "use.py"]
    assert "def new_name()" in (ws / "mod.py").read_text(encoding="utf-8")
    assert "new_name()" in (ws / "use.py").read_text(encoding="utf-8")
    # word-boundary: a substring like old_name_2 must NOT be touched
    assert "old_name" not in (ws / "use.py").read_text(encoding="utf-8")


def test_rename_word_boundary_preserved(ws):
    (ws / "m.py").write_text(
        "old = 1\noldish = 2\nprint(old, oldish)\n", encoding="utf-8"
    )
    m = _fd(ws).try_handle("rename old to fresh")
    assert m is not None
    txt = (ws / "m.py").read_text(encoding="utf-8")
    assert "fresh = 1" in txt
    assert "oldish = 2" in txt  # untouched — boundary respected


def test_rename_phrasing_variants(ws):
    for phrasing in (
        "rename function old to new",
        "rename `old` to `new`",
        "rename old -> new",
        "rename old to new everywhere",
    ):
        d = ws / phrasing.replace(" ", "_").replace("`", "").replace(">", "")
        d.mkdir()
        (d / "x.py").write_text("def old():\n    pass\n", encoding="utf-8")
        m = DeterministicFrontDoor(d).try_handle(phrasing)
        assert m is not None, phrasing
        assert "def new()" in (d / "x.py").read_text(encoding="utf-8")


def test_rename_absent_identifier_abstains(ws):
    (ws / "m.py").write_text("def something():\n    pass\n", encoding="utf-8")
    # old_name doesn't appear anywhere → abstain (model may know better)
    assert _fd(ws).try_handle("rename ghost to specter") is None


def test_rename_breaking_syntax_aborts(ws):
    # Renaming a Python keyword-collision target would break syntax. Use a
    # rename that would produce invalid code: rename `x` to `def`.
    (ws / "m.py").write_text("x = 1\n", encoding="utf-8")
    # "def" isn't a valid identifier per the classifier regex → abstain at parse.
    assert _fd(ws).try_handle("rename x to def") is None
    assert (ws / "m.py").read_text(encoding="utf-8") == "x = 1\n"


def test_rename_non_identifier_abstains(ws):
    (ws / "m.py").write_text("x = 1\n", encoding="utf-8")
    assert _fd(ws).try_handle("rename x to my-thing") is None


def test_rename_same_name_abstains(ws):
    (ws / "m.py").write_text("x = 1\n", encoding="utf-8")
    assert _fd(ws).try_handle("rename x to x") is None


# ── add import ───────────────────────────────────────────────────


def test_add_import_simple(ws):
    (ws / "a.py").write_text("print(os.getcwd())\n", encoding="utf-8")
    m = _fd(ws).try_handle("add import os to a.py")
    assert m is not None and m.category == "add_import"
    assert (ws / "a.py").read_text(encoding="utf-8").startswith("import os\n")


def test_add_from_import(ws):
    (ws / "a.py").write_text("p = Path('.')\n", encoding="utf-8")
    m = _fd(ws).try_handle("add `from pathlib import Path` to a.py")
    assert m is not None
    assert (ws / "a.py").read_text(encoding="utf-8").startswith(
        "from pathlib import Path\n"
    )


def test_add_import_idempotent_noop(ws):
    (ws / "a.py").write_text("import os\nprint(os)\n", encoding="utf-8")
    m = _fd(ws).try_handle("add import os to a.py")
    assert m is not None and m.no_op is True
    # file unchanged (no duplicate import prepended)
    assert (ws / "a.py").read_text(encoding="utf-8") == "import os\nprint(os)\n"


def test_add_import_missing_file_abstains(ws):
    assert _fd(ws).try_handle("add import os to nope.py") is None


def test_add_import_cycle_abstains(ws):
    # a.py and b.py mutually import → adding `from b import thing` to a.py is
    # the ambiguous lazy-import case; defer to the model.
    (ws / "a.py").write_text("from b import b_thing\n\ndef a_thing():\n    pass\n", encoding="utf-8")
    (ws / "b.py").write_text("from a import a_thing\n\ndef b_thing():\n    return a_thing()\n", encoding="utf-8")
    # Try adding another from-b import to a.py — cycle guard should abstain.
    m = _fd(ws).try_handle("add `from b import other` to a.py")
    assert m is None


def test_add_import_generative_abstains(ws):
    (ws / "a.py").write_text("pass\n", encoding="utf-8")
    # not an import-shaped statement
    assert _fd(ws).try_handle("add a logging setup to a.py") is None


# ── trailing newline ─────────────────────────────────────────────


def test_add_trailing_newline(ws):
    (ws / "a.txt").write_text("hello", encoding="utf-8")
    m = _fd(ws).try_handle("add a trailing newline to a.txt")
    assert m is not None and m.category == "trailing_newline"
    assert (ws / "a.txt").read_text(encoding="utf-8") == "hello\n"


def test_add_trailing_newline_noop(ws):
    (ws / "a.txt").write_text("hello\n", encoding="utf-8")
    m = _fd(ws).try_handle("ensure a.txt ends with a newline")
    assert m is not None and m.no_op is True


def test_strip_trailing_newline(ws):
    (ws / "a.txt").write_text("hello\n\n\n", encoding="utf-8")
    m = _fd(ws).try_handle("strip trailing newlines from a.txt")
    assert m is not None
    assert (ws / "a.txt").read_text(encoding="utf-8") == "hello\n"


def test_trailing_newline_missing_file_abstains(ws):
    assert _fd(ws).try_handle("add a trailing newline to ghost.txt") is None


# ── create empty file ────────────────────────────────────────────


def test_create_empty_file(ws):
    m = _fd(ws).try_handle("create empty foo.py")
    assert m is not None and m.category == "create_empty"
    assert (ws / "foo.py").exists()
    assert (ws / "foo.py").read_text(encoding="utf-8") == ""


def test_create_file_variants(ws):
    for i, phrasing in enumerate((
        "create file notes.md",
        "create a new file data.json",
        "create an empty config.yaml",
    )):
        m = _fd(ws).try_handle(phrasing)
        assert m is not None, phrasing


def test_create_with_behavior_abstains(ws):
    # described content/behavior is generative → abstain
    assert _fd(ws).try_handle("create a parser.py that reads CSV files") is None
    assert _fd(ws).try_handle("create a function in helper.py") is None


def test_create_existing_file_abstains(ws):
    (ws / "exists.py").write_text("x = 1\n", encoding="utf-8")
    # don't clobber an existing file
    assert _fd(ws).try_handle("create exists.py") is None
    assert (ws / "exists.py").read_text(encoding="utf-8") == "x = 1\n"


# ── format (conditional on a formatter being installed) ──────────


def test_format_abstains_without_formatter(ws):
    (ws / "a.py").write_text("x=1\n", encoding="utf-8")
    if shutil.which("black") or _module_importable("black") or \
       shutil.which("autopep8") or _module_importable("autopep8"):
        pytest.skip("a formatter is installed; the no-formatter path can't be tested")
    # No formatter → format must abstain (not a safe deterministic op here).
    assert _fd(ws).try_handle("format a.py") is None
    assert _fd(ws).try_handle("format a.py with black") is None


@pytest.mark.skipif(
    not (shutil.which("black") or _module_importable("black")),
    reason="black not installed",
)
def test_format_with_black(ws):
    (ws / "a.py").write_text("x=1\ndef  f( a ,b ):return a+b\n", encoding="utf-8")
    m = _fd(ws).try_handle("format a.py with black")
    assert m is not None and m.category == "format"
    out = (ws / "a.py").read_text(encoding="utf-8")
    assert "x = 1" in out  # black normalized spacing


# ── default-OFF wiring guard ─────────────────────────────────────


def test_repl_flag_gate_default_off(ws, monkeypatch):
    """_maybe_frontdoor must return None when --frontdoor is absent from argv,
    so the default REPL/one-shot path is byte-for-byte the legacy path."""
    import ulcagent
    (ws / "a.py").write_text("def old():\n    pass\n", encoding="utf-8")

    # Default argv (no --frontdoor): even a perfectly mechanical goal defers.
    monkeypatch.setattr(sys, "argv", ["ulcagent"])
    assert ulcagent._maybe_frontdoor("rename old to new", ws) is None
    # file untouched because the model path (not run here) would handle it
    assert "def old()" in (ws / "a.py").read_text(encoding="utf-8")

    # With --frontdoor the same goal is handled deterministically.
    monkeypatch.setattr(sys, "argv", ["ulcagent", "--frontdoor"])
    m = ulcagent._maybe_frontdoor("rename old to new", ws)
    assert m is not None
    assert "def new()" in (ws / "a.py").read_text(encoding="utf-8")


def test_parse_is_side_effect_free(ws):
    """parse() classifies without touching the filesystem."""
    (ws / "a.py").write_text("def old():\n    pass\n", encoding="utf-8")
    plan = _fd(ws).parse("rename old to new")
    assert plan is not None
    # parse alone must not have changed the file
    assert "def old()" in (ws / "a.py").read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
