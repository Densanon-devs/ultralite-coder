"""
Tests for engine.file_index + the `locate` tool wiring.

Covers the failure this feature exists for: ulcagent was asked "where is the
densanon llc folder?" from D:\\LLCWork. Every discovery tool (glob/grep/
list_dir) resolves against the Workspace root, so it searched the wrong drive
and answered "not present" — the folder was on G:. `locate` is the only
non-workspace-anchored discovery primitive, so it must stay that way.

Run: python -m pytest test_file_index.py -v
     OR just: python test_file_index.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine import file_index as fi
from engine.agent_builtins import (
    ASSISTANT_TOOL_NAMES,
    CORE_TOOL_NAMES,
    TOOLSETS,
    build_default_registry,
)


def _isolate(tmp: Path) -> None:
    """Point the module's DB + roots file at a temp dir (no global clobber)."""
    fi._INDEX_DIR = tmp / "index"
    fi._INDEX_DB = tmp / "index" / "file_index.db"
    fi._ROOTS_FILE = tmp / "index_roots.txt"


def _tree(root: Path) -> None:
    (root / "My Drive" / "Densanon LLC").mkdir(parents=True)
    (root / "My Drive" / "Densanon LLC" / "pricing.pdf").write_text("x")
    (root / "Projects" / "densanon-core").mkdir(parents=True)
    (root / "Projects" / "densanon-core" / "setup.py").write_text("x")
    (root / "deep" / "a" / "b" / "c").mkdir(parents=True)
    (root / "deep" / "a" / "b" / "c" / "densanon llc notes.txt").write_text("x")
    # must be pruned
    (root / "Projects" / "node_modules" / "densanon-llc-pkg").mkdir(parents=True)
    (root / "Projects" / "__pycache__").mkdir(parents=True)
    (root / "Projects" / "__pycache__" / "densanon.pyc").write_bytes(b"\x00")


def test_build_and_locate_finds_the_folder():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _isolate(tmp)
        src = tmp / "src"
        src.mkdir()
        _tree(src)

        stats = fi.build(roots=[src])
        assert stats["entries"] > 0

        hits = fi.locate("densanon llc")
        assert hits, "expected a match for 'densanon llc'"
        # Top hit must be the actual folder, not a nested note file or a
        # partial-name sibling like densanon-core.
        top = hits[0]
        assert top.is_dir, f"top hit should be a dir, got {top.path}"
        assert top.path.endswith("Densanon LLC"), top.path


def test_kind_filter():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _isolate(tmp)
        src = tmp / "src"
        src.mkdir()
        _tree(src)
        fi.build(roots=[src])

        assert all(h.is_dir for h in fi.locate("densanon", kind="dir"))
        assert all(not h.is_dir for h in fi.locate("densanon", kind="file"))


def test_noise_dirs_are_pruned():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _isolate(tmp)
        src = tmp / "src"
        src.mkdir()
        _tree(src)
        fi.build(roots=[src])

        paths = [h.path for h in fi.locate("densanon", limit=100)]
        assert not any("node_modules" in p for p in paths), paths
        assert not any("__pycache__" in p for p in paths), paths


def test_multiword_query_is_and_not_or():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _isolate(tmp)
        src = tmp / "src"
        src.mkdir()
        (src / "alpha only").mkdir()
        (src / "beta only").mkdir()
        (src / "alpha beta both").mkdir()
        fi.build(roots=[src])

        paths = [h.path for h in fi.locate("alpha beta", limit=50)]
        assert any("alpha beta both" in p for p in paths)
        assert not any(p.endswith("beta only") for p in paths), paths


def test_nested_roots_are_deduped():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _isolate(tmp)
        src = tmp / "src"
        (src / "inner").mkdir(parents=True)
        (src / "inner" / "thing.txt").write_text("x")

        stats = fi.build(roots=[src, src / "inner"])
        # Only the outer root should have been walked.
        assert len(stats["roots"]) == 1, stats["roots"]


def test_surrogate_names_do_not_kill_a_batch():
    """Windows hands back surrogate-escaped names sqlite cannot encode."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _isolate(tmp)
        con = fi._connect()
        bad = ("D:/x/\udcffbad", "\udcffbad", 0, 1, 1.0, 1)
        good = ("D:/x/good", "good", 0, 1, 1.0, 1)
        written = fi._insert_batch(con, [bad, good])
        con.commit()
        n = con.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        con.close()
        assert written >= 1, "the good row must survive the bad one"
        assert n >= 1


def test_locate_on_missing_index_is_empty_not_error():
    with tempfile.TemporaryDirectory() as td:
        _isolate(Path(td))
        assert fi.locate("anything") == []


def test_stale_and_status():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _isolate(tmp)
        src = tmp / "src"
        src.mkdir()
        (src / "f.txt").write_text("x")
        fi.build(roots=[src])

        st = fi.status()
        assert st["exists"] and st["entries"] >= 1
        assert st["stale"] is False, "a fresh index must not read as stale"

        con = fi._connect()
        fi._set_meta(con, "built_at", str(time.time() - fi.STALE_AFTER_SEC - 60))
        con.commit()
        con.close()
        assert fi.is_stale() is True


def test_roots_file_overrides_defaults():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _isolate(tmp)
        custom = tmp / "custom"
        custom.mkdir()
        fi.save_roots([custom])
        assert fi.configured_roots() == [custom]


def test_default_roots_excludes_c_drive_root():
    """C:\\ wholesale would bury the index in Windows/Program Files."""
    roots = [str(r).replace("\\", "/").rstrip("/").lower() for r in fi.default_roots()]
    assert "c:" not in roots, roots


# ── tool wiring ─────────────────────────────────────────────────

def test_locate_is_opt_in_not_in_core():
    """The lean 10-tool registry benchmarks at 100%; don't silently make it 11."""
    assert "locate" not in CORE_TOOL_NAMES
    assert "locate" in ASSISTANT_TOOL_NAMES

    default = build_default_registry(ROOT)
    names = {t.name for t in default.enabled_tools()}
    assert "locate" not in names, "locate must not appear in the default registry"
    assert len(names) == 10, sorted(names)


def test_assistant_toolset_is_small_and_read_only():
    reg = build_default_registry(ROOT, toolset="assistant")
    names = {t.name for t in reg.enabled_tools()}
    assert "locate" in names
    # Under the ~10-tool accuracy cliff.
    assert len(names) <= 8, sorted(names)
    # No write path in an assistant session pointed at the whole machine.
    for w in ("write_file", "edit_file", "insert_at_line", "apply_patch"):
        assert w not in names, f"{w} should not be in the assistant toolset"


def test_enable_locate_flag_works_without_toolset():
    reg = build_default_registry(ROOT, enable_locate=True)
    assert "locate" in {t.name for t in reg.enabled_tools()}


def test_assistant_toolset_registered():
    assert "assistant" in TOOLSETS


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL  {name}: {e}")
    print()
    if failures:
        print(f"{len(failures)} failed")
        sys.exit(1)
    print("all passed")
