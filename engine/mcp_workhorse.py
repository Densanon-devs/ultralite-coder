"""MCP server exposing ulcagent as a delegated workhorse.

This is deliberately NOT engine/mcp_server.py. That server exposes file
primitives (locate/read/create/move/edit) for DensAssistant, whose client has
no file tools of its own. Claude Code already has better primitives than those,
so re-exposing them would just cost context for duplicate schemas.

What is worth exposing to a smarter driver is the *agent loop*: one call that
hands ulcagent a concrete goal, lets the local 14B burn its own tokens running
read/grep/write/edit until the goal is met, and returns the outcome. The
expensive model plans and reviews; the cheap local model does the mechanical
work.

Protocol: newline-delimited JSON-RPC 2.0 on stdio (same contract as
engine/mcp_server.py). stdout is the protocol — all logging goes to stderr.

Concurrency: one GGUF on one GPU means one generation at a time, so jobs run on
a single worker thread and queue behind each other. `delegate` returns
immediately with a job id regardless; the win is that the driver isn't blocked,
not that stages run in parallel.

Usage:
    python -m engine.mcp_workhorse --workspace D:/LLCWork/some-project

Claude Code registration (.mcp.json):
    {"mcpServers": {"ulcagent": {
        "command": "python",
        "args": ["-m", "engine.mcp_workhorse", "--workspace", "."],
        "cwd": "D:/LLCWork/ultralight-coder"}}}
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_SELF = Path(__file__).resolve().parent.parent

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ulcagent-workhorse"
SERVER_VERSION = "0.1.0"

# Profile -> config file, mirroring ulcagent.PROFILES without importing the CLI
# module (which parses sys.argv at import time).
PROFILE_CONFIGS = {
    "code": _SELF / "config_agent14b.yaml",
    "general": _SELF / "config_agent14b_general.yaml",
}

# Tool names that mutate the filesystem — used to report files_changed.
_MUTATING = {"write_file", "edit_file", "move_path", "create_file",
             "apply_patch", "insert_at_line"}


def _log(msg: str) -> None:
    sys.stderr.write(f"[mcp_workhorse] {msg}\n")
    sys.stderr.flush()


# ── Job bookkeeping ──────────────────────────────────────────────


@dataclass
class Job:
    job_id: str
    goal: str
    workspace: str
    toolset: str
    max_iterations: int
    status: str = "queued"          # queued|running|done|error|cancelled
    answer: str = ""
    stop_reason: str = ""
    iterations: int = 0
    wall_time: float = 0.0
    est_tokens: int = 0
    files_changed: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    error: str = ""
    queued_at: float = field(default_factory=time.monotonic)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def public(self, verbose: bool = False) -> dict:
        d = {
            "job_id": self.job_id,
            "status": self.status,
            "goal": self.goal[:200],
        }
        if self.status in ("done", "error"):
            d.update({
                "answer": self.answer,
                "stop_reason": self.stop_reason,
                "iterations": self.iterations,
                "wall_time_s": round(self.wall_time, 1),
                "est_tokens": self.est_tokens,
                "files_changed": self.files_changed,
            })
            if self.error:
                d["error"] = self.error
            if verbose:
                d["tool_calls"] = self.tool_calls
        elif self.status == "running":
            d["running_for_s"] = round(time.monotonic() - (self.started_at or 0), 1)
        elif self.status == "queued":
            d["queued_for_s"] = round(time.monotonic() - self.queued_at, 1)
        return d


class Workhorse:
    """Owns the warm model and a single-threaded job queue."""

    def __init__(self, workspace: Path, profile: str = "code",
                 default_toolset: str = "coding", model_path: Optional[str] = None,
                 log: Callable[[str], None] = _log):
        self.workspace = workspace
        self.profile = profile
        self.default_toolset = default_toolset
        # Optional override for the GGUF path. The shared configs point at
        # ./models/, which lives on the USB HDD (~4 MB/s) — fine for a
        # long-lived CLI session, painful for a server that respawns whenever
        # the driver reconnects. Point this at a copy on the NVMe.
        self.model_path = model_path
        self._log = log
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._q: "queue.Queue[str]" = queue.Queue()
        self._bm = None                     # warm BaseModel
        self._worker: Optional[threading.Thread] = None

    # ── model ────────────────────────────────────────────────────

    def _ensure_model(self):
        """Load the GGUF once and keep it warm for the session."""
        if self._bm is not None and getattr(self._bm, "model", None) is not None:
            return self._bm
        cfg_path = PROFILE_CONFIGS.get(self.profile)
        if cfg_path is None or not cfg_path.exists():
            raise RuntimeError(f"no config for profile {self.profile!r} at {cfg_path}")
        try:
            from densanon.core.config import Config
            cfg = Config(str(cfg_path))
        except ImportError:
            from engine._config_shim import load_config
            cfg = load_config(str(cfg_path))
        from engine.base_model import BaseModel
        if self.model_path:
            cfg.base_model.path = self.model_path
        t0 = time.monotonic()
        self._log(f"loading model for profile {self.profile!r} "
                  f"from {getattr(cfg.base_model, 'path', '?')} ...")
        bm = BaseModel(cfg.base_model)
        bm.load()
        self._bm = bm
        self._log(f"model ready in {time.monotonic() - t0:.1f}s")
        return bm

    # ── worker ───────────────────────────────────────────────────

    def _ensure_worker(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run_loop, daemon=True,
                                        name="ulcagent-workhorse")
        self._worker.start()

    def _run_loop(self):
        while True:
            job_id = self._q.get()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status == "cancelled":
                    continue
                job.status = "running"
                job.started_at = time.monotonic()
            try:
                self._execute(job)
                with self._lock:
                    if job.status != "cancelled":
                        job.status = "done"
            except Exception as exc:                     # noqa: BLE001
                self._log("job failed:\n" + traceback.format_exc())
                with self._lock:
                    job.status = "error"
                    job.error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    job.finished_at = time.monotonic()

    def _execute(self, job: Job):
        from engine.agent import Agent
        from engine.agent_builtins import build_default_registry
        from engine.agent_memory import AgentMemory
        from engine.write_policy import WritePolicy

        ws = Path(job.workspace).expanduser().resolve()
        model = self._ensure_model()
        memory = AgentMemory(workspace=ws)

        # Writes are allowlist-gated. There is no confirm hook on an MCP path —
        # nobody is at a terminal to answer — so the allowlist is the only
        # control, exactly as in engine/mcp_server.py. Every mutation is still
        # journaled by the tools themselves, so `ulcagent --revert-last N`
        # remains the escape hatch.
        policy = WritePolicy.load(ws)

        registry = build_default_registry(
            ws,
            memory=memory,
            toolset=job.toolset,
            write_policy=policy,
            # No ask_user_fn / confirm_* hooks: an unattended path must never
            # block on a prompt that no human will ever see.
        )

        agent = Agent(
            model=model,
            registry=registry,
            workspace_root=ws,
            memory=memory,
            auto_verify_python=True,
            max_iterations=job.max_iterations,
            max_wall_time=900.0,
            temperature=0.1,
            enable_self_heal=True,
        )

        result = agent.run(job.goal)

        changed: list[str] = []
        calls: list[str] = []
        for call in result.tool_calls:
            name = getattr(call, "name", None) or getattr(call, "tool", "?")
            args = getattr(call, "arguments", None) or getattr(call, "args", {}) or {}
            calls.append(name)
            if name in _MUTATING:
                for key in ("path", "file_path", "destination", "source"):
                    val = args.get(key) if isinstance(args, dict) else None
                    if isinstance(val, str) and val and val not in changed:
                        changed.append(val)

        with self._lock:
            job.answer = result.final_answer
            job.stop_reason = result.stop_reason
            job.iterations = result.iterations
            job.wall_time = result.wall_time
            job.est_tokens = getattr(result, "est_tokens", 0)
            job.files_changed = changed
            job.tool_calls = calls

    # ── public API ───────────────────────────────────────────────

    def submit(self, goal: str, workspace: Optional[str], toolset: Optional[str],
               max_iterations: int) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:8],
            goal=goal,
            workspace=str(workspace or self.workspace),
            toolset=toolset or self.default_toolset,
            max_iterations=max_iterations,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._ensure_worker()
        self._q.put(job.job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> tuple[bool, str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False, f"no such job {job_id!r}"
            if job.status == "queued":
                job.status = "cancelled"
                return True, f"job {job_id} cancelled before it started"
            if job.status == "running":
                # The agent loop has no cooperative cancellation point, so a
                # running job cannot be interrupted without killing the shared
                # model. Say so plainly rather than pretending.
                return False, (f"job {job_id} is already running and cannot be "
                               f"interrupted; it is bounded by max_iterations "
                               f"and a 900s wall clock")
            return False, f"job {job_id} is already {job.status}"


# ── MCP tool surface ─────────────────────────────────────────────
#
# Deliberately four small tools. Claude Code already has file primitives; the
# only thing worth adding to its context is the ability to hand work off. See
# feedback_tool_count_regression — schema bloat is a measurable accuracy tax on
# the model reading them, and that logic applies to the driver too.


def _tools(wh: "Workhorse") -> dict[str, dict[str, Any]]:
    return {
        "delegate": {
            "description": (
                "Hand one concrete, self-contained task to the local ulcagent "
                "(Qwen 14B) to execute with its own file tools. Returns a job_id "
                "immediately; poll delegate_result. Use for mechanical work you "
                "would otherwise spend your own tokens on: writing tests, applying "
                "a rename across files, scaffolding, mechanical refactors, "
                "summarising a directory. Give ONE goal with enough context to act "
                "on alone — the local model does not see this conversation. "
                "It CAN write files (allowlist-gated, journaled)."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": ("The task, stated concretely and standalone. "
                                        "Name exact paths where you know them."),
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Project root to operate in. Defaults to the server's workspace.",
                    },
                    "toolset": {
                        "type": "string",
                        "enum": ["coding", "refactor", "git", "web", "hybrid", "full"],
                        "description": ("Tool profile. 'coding' (10 tools) is the "
                                        "benchmarked default and best for accuracy; "
                                        "widen only when the task needs it."),
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Tool-call rounds before giving up (default 20).",
                    },
                },
                "required": ["goal"],
            },
        },
        "delegate_result": {
            "description": (
                "Fetch a delegated job's status/result by job_id. Optionally block "
                "up to wait_seconds for it to finish. Jobs run one at a time (single "
                "local GPU), so a job may sit 'queued' behind another."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "wait_seconds": {
                        "type": "integer",
                        "description": "Block up to this long waiting for completion (default 0, max 600).",
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Include the tool-call sequence the agent used.",
                    },
                },
                "required": ["job_id"],
            },
        },
        "delegate_list": {
            "description": "List all delegated jobs this session with their status.",
            "schema": {"type": "object", "properties": {}},
        },
        "delegate_cancel": {
            "description": ("Cancel a QUEUED job. A job already running cannot be "
                            "interrupted; it is bounded by max_iterations and a 900s wall clock."),
            "schema": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    }


def _ok(payload: Any) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


class Server:
    def __init__(self, workspace: Path, profile: str = "code",
                 toolset: str = "coding", model_path: Optional[str] = None,
                 out=None, err_log: Callable[[str], None] = _log):
        self.workspace = workspace
        self.wh = Workhorse(workspace, profile=profile, default_toolset=toolset,
                            model_path=model_path, log=err_log)
        self.tools = _tools(self.wh)
        self._out = out or sys.stdout
        self._log = err_log

    def _send(self, obj: dict) -> None:
        self._out.write(json.dumps(obj) + "\n")
        self._out.flush()

    def _call(self, name: str, args: dict) -> dict:
        if name == "delegate":
            goal = (args.get("goal") or "").strip()
            if not goal:
                return _err("delegate requires a non-empty 'goal'.")
            try:
                mi = int(args.get("max_iterations") or 20)
            except (TypeError, ValueError):
                mi = 20
            job = self.wh.submit(goal, args.get("workspace"),
                                 args.get("toolset"), max(1, min(mi, 50)))
            self._log(f"queued job {job.job_id}: {goal[:80]}")
            ahead = sum(1 for j in self.wh.all_jobs()
                        if j.status in ("queued", "running") and j.job_id != job.job_id)
            return _ok({"job_id": job.job_id, "status": job.status,
                        "jobs_ahead": ahead,
                        "note": "poll delegate_result for the outcome"})

        if name == "delegate_result":
            job_id = args.get("job_id") or ""
            job = self.wh.get(job_id)
            if job is None:
                return _err(f"no such job {job_id!r}")
            try:
                wait = int(args.get("wait_seconds") or 0)
            except (TypeError, ValueError):
                wait = 0
            wait = max(0, min(wait, 600))
            deadline = time.monotonic() + wait
            while job.status in ("queued", "running") and time.monotonic() < deadline:
                time.sleep(1.0)
            return _ok(job.public(verbose=bool(args.get("verbose"))))

        if name == "delegate_list":
            jobs = sorted(self.wh.all_jobs(), key=lambda j: j.queued_at)
            return _ok({"jobs": [j.public() for j in jobs]})

        if name == "delegate_cancel":
            ok, msg = self.wh.cancel(args.get("job_id") or "")
            return _ok(msg) if ok else _err(msg)

        return _err(f"Unknown tool {name!r}. Available: {', '.join(self.tools)}")

    def handle(self, msg: dict) -> dict | None:
        method = msg.get("method", "")
        mid = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }}

        if method.startswith("notifications/"):
            return None

        if method == "tools/list":
            listed = [{"name": n, "description": s["description"], "inputSchema": s["schema"]}
                      for n, s in self.tools.items()]
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": listed}}

        if method == "tools/call":
            name = params.get("name", "")
            try:
                result = self._call(name, params.get("arguments") or {})
            except Exception as exc:                      # noqa: BLE001
                self._log("tools/call failed:\n" + traceback.format_exc())
                result = _err(f"{type(exc).__name__}: {exc}")
            return {"jsonrpc": "2.0", "id": mid, "result": result}

        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}

    def serve(self, stdin=None) -> None:
        stream = stdin or sys.stdin
        self._log(f"serving on stdio; workspace={self.workspace}")
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._log(f"bad JSON line: {line[:120]}")
                continue
            reply = self.handle(msg)
            if reply is not None:
                self._send(reply)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    def _opt(flag: str, default: str) -> str:
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    workspace = Path(_opt("--workspace", ".")).expanduser().resolve()
    profile = _opt("--profile", "code")
    toolset = _opt("--toolset", "coding")
    model_path = _opt("--model", "") or None
    Server(workspace, profile=profile, toolset=toolset,
           model_path=model_path).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
