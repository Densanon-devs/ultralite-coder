"""
Tests for engine.mcp_workhorse — the Claude-Code-drives-ulcagent direction.

Driven in-process against the Server class, so the JSON-RPC contract is
asserted directly without spawning a subprocess or loading a 9 GB GGUF. The
model is stubbed; what is under test is the *delegation harness*, not the
model's coding ability (benchmark_agentic.py already covers that).

The properties that matter here, none of them happy paths:
  * `delegate` must return a job id IMMEDIATELY. If it ever blocks, the whole
    point (driver keeps working while the local model grinds) is gone.
  * stdout is the protocol. Any diagnostic printed to stdout corrupts the
    stream, so logging must go to stderr.
  * The caller is a PROGRAM. There is no human at a y/N prompt, so a refusal
    must surface as a tool error, never as a silent success.
  * A running job cannot be interrupted — cancel must SAY so rather than
    reporting a cancellation that did not happen.

Run: python -m pytest test_mcp_workhorse.py -v
     OR just: python test_mcp_workhorse.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.mcp_workhorse import Job, Server, Workhorse  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────


def _server(tmp: Path, out=None) -> Server:
    logs: list[str] = []
    srv = Server(tmp, out=out or io.StringIO(), err_log=logs.append)
    srv._test_logs = logs           # type: ignore[attr-defined]
    return srv


def _call(srv: Server, name: str, args: dict) -> tuple[bool, str]:
    reply = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args}})
    result = reply["result"]
    return result["isError"], result["content"][0]["text"]


class _StubWorkhorse(Workhorse):
    """Replaces _execute so no model is loaded."""

    def __init__(self, *a, delay: float = 0.0, boom: bool = False, **kw):
        super().__init__(*a, **kw)
        self.delay = delay
        self.boom = boom

    def _execute(self, job: Job):
        time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("simulated model failure")
        job.answer = f"did: {job.goal}"
        job.stop_reason = "answered"
        job.iterations = 2
        job.wall_time = self.delay
        job.files_changed = ["stub.py"]
        job.tool_calls = ["read_file", "edit_file"]


# ── protocol ─────────────────────────────────────────────────────


def test_initialize_reports_server_info():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert r["result"]["serverInfo"]["name"] == "ulcagent-workhorse"
        assert r["result"]["protocolVersion"] == "2024-11-05"


def test_notifications_get_no_reply():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list_is_exactly_the_four_delegation_tools():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        r = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = [t["name"] for t in r["result"]["tools"]]
        assert names == ["delegate", "delegate_result", "delegate_list", "delegate_cancel"], names
        # Every tool must carry a schema, or the driver cannot call it.
        for t in r["result"]["tools"]:
            assert t["inputSchema"]["type"] == "object"
            assert t["description"].strip()


def test_no_file_primitives_are_exposed():
    """Claude Code has better primitives; re-exposing them is pure context tax."""
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        names = set(srv.tools)
        assert not (names & {"read_file", "write_file", "edit_file", "locate", "grep"})


def test_unknown_method_returns_jsonrpc_error():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        r = srv.handle({"jsonrpc": "2.0", "id": 9, "method": "resources/list"})
        assert r["error"]["code"] == -32601


def test_unknown_tool_is_a_tool_error_not_a_crash():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        err, text = _call(srv, "nope", {})
        assert err and "Unknown tool" in text


# ── delegation semantics ─────────────────────────────────────────


def test_delegate_returns_immediately_with_a_job_id():
    """The whole design rests on this call not blocking."""
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        srv.wh = _StubWorkhorse(Path(tmp), delay=2.0)
        t0 = time.monotonic()
        err, text = _call(srv, "delegate", {"goal": "something slow"})
        elapsed = time.monotonic() - t0
        assert not err, text
        assert elapsed < 0.5, f"delegate blocked for {elapsed:.2f}s"
        assert json.loads(text)["job_id"]


def test_empty_goal_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        err, text = _call(srv, "delegate", {"goal": "   "})
        assert err and "non-empty" in text


def test_result_reports_answer_and_files_changed():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        srv.wh = _StubWorkhorse(Path(tmp), delay=0.1)
        _, text = _call(srv, "delegate", {"goal": "edit a file"})
        job_id = json.loads(text)["job_id"]
        _, text = _call(srv, "delegate_result", {"job_id": job_id, "wait_seconds": 30})
        data = json.loads(text)
        assert data["status"] == "done", data
        assert data["answer"] == "did: edit a file"
        assert data["files_changed"] == ["stub.py"]
        assert data["stop_reason"] == "answered"


def test_verbose_adds_tool_calls_and_default_omits_them():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        srv.wh = _StubWorkhorse(Path(tmp), delay=0.1)
        _, text = _call(srv, "delegate", {"goal": "x"})
        job_id = json.loads(text)["job_id"]
        _, text = _call(srv, "delegate_result", {"job_id": job_id, "wait_seconds": 30})
        assert "tool_calls" not in json.loads(text)
        _, text = _call(srv, "delegate_result",
                        {"job_id": job_id, "verbose": True})
        assert json.loads(text)["tool_calls"] == ["read_file", "edit_file"]


def test_model_failure_surfaces_as_error_status_not_silent_success():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        srv.wh = _StubWorkhorse(Path(tmp), boom=True)
        _, text = _call(srv, "delegate", {"goal": "will blow up"})
        job_id = json.loads(text)["job_id"]
        _, text = _call(srv, "delegate_result", {"job_id": job_id, "wait_seconds": 30})
        data = json.loads(text)
        assert data["status"] == "error", data
        assert "simulated model failure" in data["error"]


def test_unknown_job_id_is_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        err, text = _call(srv, "delegate_result", {"job_id": "deadbeef"})
        assert err and "no such job" in text


def test_jobs_run_one_at_a_time():
    """One GPU, one model: the second job must queue, not race the first."""
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        srv.wh = _StubWorkhorse(Path(tmp), delay=1.5)
        _, a = _call(srv, "delegate", {"goal": "first"})
        _, b = _call(srv, "delegate", {"goal": "second"})
        assert json.loads(b)["jobs_ahead"] >= 1, b
        # Both still complete.
        for text in (a, b):
            jid = json.loads(text)["job_id"]
            _, r = _call(srv, "delegate_result", {"job_id": jid, "wait_seconds": 60})
            assert json.loads(r)["status"] == "done"


def test_delegate_list_shows_every_job():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        srv.wh = _StubWorkhorse(Path(tmp), delay=0.05)
        for goal in ("one", "two", "three"):
            _call(srv, "delegate", {"goal": goal})
        _, text = _call(srv, "delegate_list", {})
        assert len(json.loads(text)["jobs"]) == 3


def test_cancel_admits_it_cannot_stop_a_running_job():
    """Reporting a cancellation that did not happen would be worse than refusing."""
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        srv.wh = _StubWorkhorse(Path(tmp), delay=3.0)
        _, text = _call(srv, "delegate", {"goal": "long one"})
        job_id = json.loads(text)["job_id"]
        time.sleep(0.6)                       # let the worker pick it up
        err, msg = _call(srv, "delegate_cancel", {"job_id": job_id})
        assert err, "cancel must not claim success on a running job"
        assert "cannot be interrupted" in msg


def test_max_iterations_is_clamped():
    with tempfile.TemporaryDirectory() as tmp:
        srv = _server(Path(tmp))
        srv.wh = _StubWorkhorse(Path(tmp), delay=0.05)
        _, text = _call(srv, "delegate", {"goal": "x", "max_iterations": 9999})
        job = srv.wh.get(json.loads(text)["job_id"])
        assert job.max_iterations == 50, job.max_iterations


# ── stream hygiene ───────────────────────────────────────────────


def test_logging_never_touches_stdout():
    with tempfile.TemporaryDirectory() as tmp:
        out = io.StringIO()
        srv = _server(Path(tmp), out=out)
        srv.wh = _StubWorkhorse(Path(tmp), delay=0.05)
        _call(srv, "delegate", {"goal": "x"})
        srv._send({"jsonrpc": "2.0", "id": 1, "result": {}})
        # Only the one framed message we sent may appear on stdout.
        lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 1, lines
        json.loads(lines[0])


def test_serve_survives_a_malformed_line():
    with tempfile.TemporaryDirectory() as tmp:
        out = io.StringIO()
        srv = _server(Path(tmp), out=out)
        stdin = io.StringIO(
            "not json\n"
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
        )
        srv.serve(stdin=stdin)
        lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["result"]["serverInfo"]["name"] == "ulcagent-workhorse"


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
