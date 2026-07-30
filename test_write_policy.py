"""
Tests for engine.write_policy + the hybrid mode wiring.

Hybrid mode lets the agent create/move/edit OUTSIDE the current project, which
is a real escalation: `Workspace.is_inside()` was never enforced, and this model
has documented failure modes on path/anchor handling. So the allowlist is the
load-bearing safety control and these tests are the spec for it.

The important cases are the escapes: `..` traversal out of an allowed root, a
symlink/junction planted inside an allowed root pointing at a system directory,
and credential filenames. Each must be REFUSED, not merely prompted.

Run: python -m pytest test_write_policy.py -v
     OR just: python test_write_policy.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine import write_policy as wp
from engine.agent_builtins import TOOLSETS, build_default_registry


def _policy(*roots: Path) -> wp.WritePolicy:
    return wp.WritePolicy(roots=list(roots), extra_roots=[])


# ── allow / refuse ──────────────────────────────────────────────

def test_inside_an_allowed_root_is_allowed():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pol = _policy(root)
        assert pol.check(root / "a.txt")
        assert pol.check(root / "deep" / "nested" / "b.txt")


def test_outside_every_root_is_refused():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pol = _policy(Path(a))
        v = pol.check(Path(b) / "x.txt")
        assert not v.allowed
        assert "outside every allowed write root" in v.reason


def test_dotdot_traversal_cannot_escape():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "allowed"
        root.mkdir()
        pol = _policy(root)
        assert not pol.check(root / ".." / "escaped.txt").allowed


def test_system_directories_are_refused_even_if_a_root_contains_them():
    """A root of C:/ must not make C:/Windows writable."""
    pol = _policy(Path("C:/"))
    for bad in ("C:/Windows/system32/drivers/etc/hosts",
                "C:/Program Files/app/x.dll",
                "C:/ProgramData/thing.dat",
                "C:/$Recycle.Bin/x"):
        v = pol.check(bad)
        assert not v.allowed, f"{bad} should be refused"
        assert "protected system location" in v.reason


def test_credential_filenames_are_refused():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pol = _policy(root)
        for name in ("db.key", "pairing.json", ".env", "id_rsa", "license.key"):
            v = pol.check(root / name)
            assert not v.allowed, f"{name} should be refused"
            assert "credentials or key material" in v.reason


def test_a_root_itself_is_writable():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        assert _policy(root).check(root)


def test_sibling_prefix_is_not_treated_as_inside():
    """/allowed must not authorise /allowed-evil."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        good = base / "allowed"
        evil = base / "allowed-evil"
        good.mkdir()
        evil.mkdir()
        pol = _policy(good)
        assert not pol.check(evil / "x.txt").allowed


def test_symlink_out_of_an_allowed_root_is_refused():
    """resolve() follows the link, so the real target is what gets judged."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        root, outside = Path(a), Path(b)
        link = root / "escape"
        try:
            os.symlink(str(outside), str(link), target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):
            return          # needs privilege on Windows; skip rather than fake it
        pol = _policy(root)
        assert not pol.check(link / "x.txt").allowed


def test_extra_root_covers_the_current_workspace():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pol = wp.WritePolicy(roots=[Path(a)], extra_roots=[Path(b)])
        assert pol.check(Path(b) / "in_workspace.txt")


def test_verdict_is_falsy_when_refused():
    with tempfile.TemporaryDirectory() as td:
        pol = _policy(Path(td))
        assert not pol.check("C:/Windows/x.txt")
        assert pol.check(Path(td) / "ok.txt")


def test_refusal_lists_the_allowed_roots():
    """The model has to be able to self-correct from the message."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pol = _policy(Path(a))
        reason = pol.check(Path(b) / "x.txt").reason
        assert str(Path(a)) in reason


# ── journal + revert ────────────────────────────────────────────

def test_journal_and_revert_a_move(monkeypatch=None):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wp._STATE_DIR = tmp / "state"
        wp._JOURNAL = tmp / "state" / "mutations.jsonl"
        wp._BACKUP_DIR = tmp / "state" / "backups"

        src = tmp / "a.txt"
        dst = tmp / "b.txt"
        src.write_text("hello")
        import shutil
        shutil.move(str(src), str(dst))
        wp.record("move", src, dest=dst)

        entries = wp.journal_entries(limit=5)
        assert entries and entries[-1]["op"] == "move"

        ok, msg = wp.revert(entries[-1])
        assert ok, msg
        assert src.exists() and not dst.exists()


def test_revert_of_an_edit_restores_the_backup():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wp._STATE_DIR = tmp / "state"
        wp._JOURNAL = tmp / "state" / "mutations.jsonl"
        wp._BACKUP_DIR = tmp / "state" / "backups"

        f = tmp / "code.py"
        f.write_text("original\n")
        backup = wp._backup(f)
        assert backup
        wp.record("edit", f, backup=backup)
        f.write_text("BROKEN\n")

        ok, msg = wp.revert(wp.journal_entries(limit=1)[-1])
        assert ok, msg
        assert f.read_text() == "original\n"


def test_revert_refuses_to_clobber_a_recreated_source():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wp._STATE_DIR = tmp / "state"
        wp._JOURNAL = tmp / "state" / "mutations.jsonl"
        wp._BACKUP_DIR = tmp / "state" / "backups"
        src, dst = tmp / "a.txt", tmp / "b.txt"
        dst.write_text("moved")
        src.write_text("something new lives here now")
        ok, msg = wp.revert({"op": "move", "path": str(src), "dest": str(dst)})
        assert not ok and "refusing to overwrite" in msg.lower()


def test_revert_last_on_empty_journal_is_graceful():
    with tempfile.TemporaryDirectory() as td:
        wp._JOURNAL = Path(td) / "none.jsonl"
        msgs = wp.revert_last(1)
        assert msgs and "nothing" in msgs[0].lower()


# ── roots config ────────────────────────────────────────────────

def test_saved_roots_override_defaults():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wp._STATE_DIR = tmp
        wp._ROOTS_FILE = tmp / "write_roots.txt"
        custom = tmp / "custom"
        custom.mkdir()
        wp.save_write_roots([custom])
        assert wp.configured_write_roots() == [custom]


def test_defaults_never_include_a_system_root():
    for r in wp.default_write_roots():
        low = str(r).lower()
        assert "windows" not in low
        assert "program files" not in low
        # AppData is governed by the allowlist (it is simply not a default),
        # not by the deny-list — see the scope note in write_policy.
        assert not low.endswith("appdata")


# ── hybrid profile wiring ───────────────────────────────────────

def test_hybrid_profile_is_exactly_ten_tools():
    """Ten is the benchmarked-good registry size; don't drift off it."""
    assert len(TOOLSETS["hybrid"]) == 10, sorted(TOOLSETS["hybrid"])


def test_hybrid_has_the_four_asked_for_powers():
    names = set(TOOLSETS["hybrid"])
    assert "locate" in names,      "search"
    assert "move_path" in names,   "move"
    assert "write_file" in names,  "create"
    assert "edit_file" in names,   "edit"
    assert "recall" in names,      "DensAssistant"


def test_hybrid_excludes_run_bash():
    """Dropped deliberately to hold the profile at 10; --toolset full has it."""
    assert "run_bash" not in TOOLSETS["hybrid"]
    assert "run_bash" in TOOLSETS["full"]


def test_move_path_requires_confirmation():
    reg = build_default_registry(ROOT, toolset="hybrid",
                                 write_policy=_policy(ROOT))
    tool = reg.get("move_path")
    assert tool is not None and tool.risky


def test_coding_profile_has_no_policy_and_no_new_tools():
    names = {t.name for t in build_default_registry(ROOT).enabled_tools()}
    assert len(names) == 10
    for n in ("move_path", "recall", "locate", "run_capability"):
        assert n not in names, f"{n} leaked into the default registry"


def test_write_is_refused_outside_policy_through_the_tool():
    """End-to-end: the registered write_file honours the allowlist."""
    with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as denied:
        reg = build_default_registry(Path(allowed), toolset="hybrid",
                                     write_policy=_policy(Path(allowed)))
        write = reg.get("write_file")
        target = str(Path(denied) / "sneaky.txt")
        try:
            write.function(path=target, content="nope")
        except PermissionError as e:
            assert "refused" in str(e)
        else:
            raise AssertionError("write outside the allowlist should have raised")
        assert not (Path(denied) / "sneaky.txt").exists()


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
            except Exception as e:
                failures.append((name, f"{type(e).__name__}: {e}"))
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} failed")
        sys.exit(1)
    print("all passed")
