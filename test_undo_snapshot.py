"""
Tests for ulcagent's /undo snapshot (_snapshot_workspace / _do_undo).

Model-free and deterministic. Guards two real defects:

  1. MemoryError crash — the snapshot held every file's bytes in RAM and caught
     only OSError. Pointed at a tree containing multi-GB .gguf models,
     `p.read_bytes()` raised MemoryError (not an OSError), which escaped and
     killed the REPL on the very first goal, after the model had already loaded.

  2. Data loss on /undo — the delete-orphans phase removed any file missing
     from the content snapshot. Once the snapshot is bounded (large + binary
     files intentionally skipped), that would have deleted model weights, APKs
     and build artifacts. Membership must be tested against _undo_known (every
     file SEEN) rather than _undo_snapshot (only files whose bytes were kept).

Run: python -m pytest test_undo_snapshot.py -v
     OR just: python test_undo_snapshot.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import ulcagent as U


def _make_workspace(tmp: Path) -> Path:
    """A workspace mixing source, oversized, binary and noise files."""
    (tmp / "app.py").write_text("print('v1')\n")
    (tmp / "pkg").mkdir()
    (tmp / "pkg" / "mod.py").write_text("X = 1\n")

    # Seen but deliberately not captured.
    (tmp / "model.gguf").write_bytes(b"\x00" * 64)               # binary suffix
    (tmp / "big.txt").write_bytes(b"a" * (3 * 1024 * 1024))      # over per-file cap
    (tmp / "app.exe").write_bytes(b"MZ" + b"\x00" * 32)          # binary suffix

    # Pruned entirely.
    (tmp / "node_modules").mkdir()
    (tmp / "node_modules" / "dep.js").write_text("junk\n")
    (tmp / "__pycache__").mkdir()
    (tmp / "__pycache__" / "app.cpython-310.pyc").write_bytes(b"\x00" * 8)
    return tmp


def test_snapshot_is_bounded_and_does_not_crash():
    with tempfile.TemporaryDirectory() as td:
        ws = _make_workspace(Path(td))
        U._snapshot_warned.clear()
        U._snapshot_workspace(ws)

        assert "app.py" in U._undo_snapshot
        assert str(Path("pkg/mod.py")) in U._undo_snapshot

        # Oversized / binary files are seen but their bytes are never held.
        for name in ("model.gguf", "big.txt", "app.exe"):
            assert name not in U._undo_snapshot, f"{name} should not be captured"
            assert name in U._undo_known, f"{name} should still be known"

        # Noise dirs are pruned, not merely filtered after the fact.
        assert not any("node_modules" in k for k in U._undo_known)
        assert not any("__pycache__" in k for k in U._undo_known)

        assert U._undo_complete is True
        held = sum(len(v) for v in U._undo_snapshot.values())
        assert held < 1024 * 1024, f"snapshot held {held} bytes"


def test_undo_restores_edits_and_removes_created_files():
    with tempfile.TemporaryDirectory() as td:
        ws = _make_workspace(Path(td))
        U._snapshot_warned.clear()
        U._snapshot_workspace(ws)

        (ws / "app.py").write_text("print('v2 BROKEN')\n")
        (ws / "scratch_new.py").write_text("# created by the goal\n")
        U._do_undo(ws)

        assert (ws / "app.py").read_text() == "print('v1')\n"
        assert not (ws / "scratch_new.py").exists()


def test_undo_never_deletes_files_the_snapshot_skipped():
    """The data-loss regression: skipped != newly created."""
    with tempfile.TemporaryDirectory() as td:
        ws = _make_workspace(Path(td))
        U._snapshot_warned.clear()
        U._snapshot_workspace(ws)
        U._do_undo(ws)

        assert (ws / "model.gguf").exists(), "undo deleted a model file"
        assert (ws / "app.exe").exists(), "undo deleted a binary"
        assert (ws / "big.txt").exists(), "undo deleted an oversized file"
        assert (ws / "big.txt").stat().st_size == 3 * 1024 * 1024
        assert (ws / "node_modules" / "dep.js").exists(), "undo deleted a pruned file"


def test_oversized_workspace_declines_instead_of_stalling():
    """Over _SNAPSHOT_MAX_FILES the snapshot opts out — and stays inert."""
    with tempfile.TemporaryDirectory() as td:
        ws = _make_workspace(Path(td))
        orig = U._SNAPSHOT_MAX_FILES
        U._SNAPSHOT_MAX_FILES = 2
        U._snapshot_warned.clear()
        try:
            U._snapshot_workspace(ws)
            assert not U._undo_snapshot
            assert not U._undo_known
            assert U._undo_complete is False

            (ws / "made_after.py").write_text("# new\n")
            U._do_undo(ws)
            # A declined snapshot must neither delete nor restore anything.
            assert (ws / "made_after.py").exists()
            assert (ws / "app.py").read_text() == "print('v1')\n"
        finally:
            U._SNAPSHOT_MAX_FILES = orig


def test_scan_bails_early_on_huge_tree():
    """The early bail is what keeps a 45k-file holding zone off the hot path."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i in range(60):
            (root / f"f{i}.py").write_text("x = 1\n")
        files, too_many = U._scan_workspace_files(root, 10)
        assert too_many is True
        assert len(files) == 11, "scan should stop the moment it exceeds the limit"

        files, too_many = U._scan_workspace_files(root, 500)
        assert too_many is False
        assert len(files) == 60


def test_snapshot_of_a_real_project_tree_is_fast():
    """ulcagent's own repo: a normal project must snapshot without a stall."""
    U._snapshot_warned.clear()
    t0 = time.time()
    U._snapshot_workspace(ROOT)
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"snapshot of {ROOT} took {elapsed:.1f}s"
    assert U._undo_snapshot, "expected to capture this repo's source files"


def test_do_undo_on_empty_snapshot_is_a_noop():
    with tempfile.TemporaryDirectory() as td:
        U._undo_snapshot.clear()
        U._undo_known.clear()
        U._undo_skipped.clear()
        U._do_undo(Path(td))   # must not raise


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
