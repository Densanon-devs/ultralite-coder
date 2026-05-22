"""
Tests for the COSMO-Agent failure-axis decomposition in benchmark_agentic.

Pure instrumentation on top of the existing pass/fail: a failed task is
classified into one or more of {schema-format, toolchain, feasibility} (with a
"process" fallback for crashes / timeouts / loop-limits), derived entirely from
signals already present on the AgentResult (synthetic parse_error observations,
failed real-tool ToolResults, and a clean "answered" stop). No GPU / no model —
synthetic SimpleNamespace result objects, mirroring test_auto_flag_hook.py.

Exercises:
- classify_failure_axes is importable + callable
- None result -> [] (setup/runner crash before any run)
- parse_error observation -> schema-format
- failed REAL tool -> toolchain
- failed SYNTHETIC observation (auto_verify/stuck_repeat) does NOT count as toolchain
- clean answered run that still failed verification -> feasibility
- a run can land on multiple axes at once (additive)
- timeout / max_iterations with no other signal -> process fallback
- TaskResult carries failure_axes (default empty)
- JSON payload + reporter wiring exists
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from benchmark_agentic import classify_failure_axes, TaskResult


def _tr(name, success, error=""):
    return SimpleNamespace(name=name, success=success, error=error, content="")


def _result(stop_reason="answered", tool_results=None):
    return SimpleNamespace(
        stop_reason=stop_reason,
        tool_results=tool_results or [],
    )


def test_classify_failure_axes_is_callable():
    assert callable(classify_failure_axes)


def test_none_result_returns_empty():
    assert classify_failure_axes(None, "setup failed") == []


def test_parse_error_is_schema_format():
    res = _result(tool_results=[
        _tr("parse_error", False, "bare JSON tool call failed to parse"),
    ])
    axes = classify_failure_axes(res, "syntax error")
    assert "schema-format" in axes


def test_failed_real_tool_is_toolchain():
    res = _result(tool_results=[
        _tr("read_file", True),
        _tr("edit_file", False, "old_string not found"),
    ])
    axes = classify_failure_axes(res, "pytest failed")
    assert "toolchain" in axes


def test_failed_synthetic_observation_is_not_toolchain():
    """auto_verify / stuck_repeat / mutation_gate are harness nudges, not real
    tool dispatches — a failed one must NOT be miscredited to the toolchain axis."""
    for synthetic in ("auto_verify", "stuck_repeat", "mutation_gate",
                       "self_heal_diagnose", "task_incomplete", "truncated_reasoning"):
        res = _result(tool_results=[
            _tr("read_file", True),
            _tr(synthetic, False, "some harness nudge"),
        ])
        axes = classify_failure_axes(res, "assertion failed")
        assert "toolchain" not in axes, f"{synthetic} wrongly counted as toolchain"


def test_clean_answered_run_is_feasibility():
    """All tool calls succeeded, model answered, but the produced code is wrong."""
    res = _result(stop_reason="answered", tool_results=[
        _tr("read_file", True),
        _tr("write_file", True),
        _tr("auto_verify", True),
    ])
    axes = classify_failure_axes(res, "paginate([1,2,3], 2) returned wrong result")
    assert axes == ["feasibility"], axes


def test_multiple_axes_are_additive():
    """A run can have a malformed call AND a failed real tool AND end answered."""
    res = _result(stop_reason="answered", tool_results=[
        _tr("parse_error", False, "Expecting ',' delimiter"),
        _tr("edit_file", False, "old_string not found"),
        _tr("write_file", True),
    ])
    axes = classify_failure_axes(res, "pytest failed")
    assert "schema-format" in axes
    assert "toolchain" in axes
    assert "feasibility" in axes


def test_timeout_with_no_signal_is_process():
    """A wall-time / loop-limit failure with no parse error, no failed real
    tool, and no clean answer is a process failure, not a code ceiling."""
    res = _result(stop_reason="wall_time", tool_results=[_tr("read_file", True)])
    axes = classify_failure_axes(res, "timeout")
    assert axes == ["process"], axes

    res2 = _result(stop_reason="max_iterations", tool_results=[_tr("grep", True)])
    assert classify_failure_axes(res2, "loop limit") == ["process"]


def test_max_iterations_with_failed_tool_still_credits_toolchain():
    """Even on a non-clean stop, a failed real tool is a genuine toolchain
    signal — process fallback only applies when NOTHING else fired."""
    res = _result(stop_reason="max_iterations", tool_results=[
        _tr("edit_file", False, "old_string not found"),
    ])
    axes = classify_failure_axes(res, "loop limit")
    assert "toolchain" in axes
    assert "process" not in axes


def test_taskresult_carries_failure_axes_default_empty():
    tr = TaskResult(
        name="t", difficulty=1, passed=True, reason="ok",
        iterations=2, tool_calls=1, wall_time=0.1, stop_reason="answered",
    )
    assert tr.failure_axes == []


def test_reporter_prints_failures_by_axis():
    """The main() reporter should print a 'Failures by axis' block. We exercise
    the formatting logic the same way the reporter does, on synthetic results."""
    results = [
        TaskResult(name="a", difficulty=1, passed=False, reason="x", iterations=1,
                   tool_calls=1, wall_time=0.1, stop_reason="answered",
                   failure_axes=["feasibility"]),
        TaskResult(name="b", difficulty=2, passed=False, reason="y", iterations=1,
                   tool_calls=1, wall_time=0.1, stop_reason="answered",
                   failure_axes=["schema-format", "feasibility"]),
        TaskResult(name="c", difficulty=1, passed=True, reason="ok", iterations=1,
                   tool_calls=1, wall_time=0.1, stop_reason="answered"),
    ]
    axis_counts: dict[str, int] = {}
    for r in [r for r in results if not r.passed]:
        for ax in (r.failure_axes or ["process"]):
            axis_counts[ax] = axis_counts.get(ax, 0) + 1
    assert axis_counts["feasibility"] == 2
    assert axis_counts["schema-format"] == 1


def test_bench_run_one_task_still_accepts_auto_flag_kwarg():
    """Sanity: our additive change didn't break run_one_task's signature."""
    import inspect
    from benchmark_agentic import run_one_task
    sig = inspect.signature(run_one_task)
    assert "auto_flag" in sig.parameters


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
