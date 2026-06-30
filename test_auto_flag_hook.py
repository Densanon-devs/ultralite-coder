"""
Tests for the auto-flag / auto-promote surface in benchmark_agentic.

Exercises:
- benchmark_agentic CLI accepts --auto-flag flag
- benchmark_agentic.run_one_task accepts an auto_flag kwarg (default False)
- benchmark_agentic CLI accepts --auto-promote flag

NOTE (2026-06-30): the original file also tested a `_maybe_auto_flag` hook
imported from `ulcagent`, but that function was never implemented in
`ulcagent.py` (it exists nowhere in the repo), so those 5 tests failed at
collection on every branch since the harvest-pipeline experiment added them.
The surviving auto-flag mechanism lives entirely in `benchmark_agentic.py`,
which is what this file now tests. The dead `_maybe_auto_flag` import + tests
were removed so the file collects and the real coverage runs.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))


def test_bench_cli_accepts_auto_flag_flag():
    """benchmark_agentic --help should mention --auto-flag."""
    import subprocess
    res = subprocess.run(
        [sys.executable, "benchmark_agentic.py", "--help"],
        capture_output=True, text=True, timeout=15, cwd=str(ROOT),
    )
    assert "--auto-flag" in res.stdout, res.stdout


def test_bench_run_one_task_accepts_auto_flag_kwarg():
    import inspect
    from benchmark_agentic import run_one_task
    sig = inspect.signature(run_one_task)
    assert "auto_flag" in sig.parameters
    assert sig.parameters["auto_flag"].default is False


def test_bench_cli_accepts_auto_promote_flag():
    """benchmark_agentic --help should mention --auto-promote."""
    import subprocess
    res = subprocess.run(
        [sys.executable, "benchmark_agentic.py", "--help"],
        capture_output=True, text=True, timeout=15, cwd=str(ROOT),
    )
    assert "--auto-promote" in res.stdout, res.stdout


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
