"""REAL agentic A/B: GBNF-constrained tool calls vs free-form, on the 14B.

Single-shot was already proven (test_gbnf_toolcall_ab.py: grammar -> 6/6 valid
<tool_call> JSON by construction). This harness answers the multi-TURN question
the single-shot test cannot: across a full ReAct loop on write-heavy tasks, does
the GBNF grammar REDUCE the JSON-repair burden / retries / failures, tie, or HURT
(over-constrain a valid call)?

It runs each selected task from benchmark_agentic.TASKS twice — grammar OFF then
grammar ON — against the same loaded 14B, instrumenting per condition:

  - pass/fail (task.check)
  - iterations, tool_calls, wall_time, est_tokens
  - parse_errors_total       (Agent: tool-call parse errors across all turns)
  - json_recovery_fired      (Agent: opt-in regen pass; 0 here, recovery OFF)
  - grammar_turns            (Agent: tool-call turns that ran WITH the grammar)
  - repair tiers 2-5 + decode_failed  (agent_tools._REPAIR_TIER_COUNTS, delta'd
    around each task — the precise "repair burden" signal: post-Tier-1 salvage)

The grammar wiring under test is Agent(tool_call_grammar=...). OFF passes None
(legacy path, byte-for-byte unchanged); ON passes the parsed LlamaGrammar.

Run (global Python310 has llama-cpp-python 0.3.20):
  python gbnf_agentic_ab.py
  python gbnf_agentic_ab.py --tasks build_todo_cli,extend_calculator --repeat 2
"""
import sys, os, time, argparse, tempfile, shutil
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

# Reuse the GBNF string the single-shot A/B proved out.
from test_gbnf_toolcall_ab import GBNF
from llama_cpp import LlamaGrammar

import benchmark_agentic as ba
from densanon.core.config import Config
from engine.base_model import BaseModel
from engine.agent import Agent
from engine.agent_builtins import build_default_registry
from engine.agent_memory import AgentMemory, _project_id
import engine.agent_tools as agent_tools


MODEL_PATH = os.environ.get(
    "GBNF_AB_MODEL", "models/qwen2.5-coder-14b-instruct-q4_k_m.gguf"
)

# Write-heavy tasks: each exercises multi-line file writes / multi-file projects,
# i.e. exactly where the multi-line-JSON tool-call ceiling lives.
DEFAULT_TASKS = ["build_todo_cli", "extend_calculator", "write_bash_lister", "add_cli_flag"]


def _task_by_name(name):
    for t in ba.TASKS:
        if t.name == name:
            return t
    raise SystemExit(f"unknown task: {name} (have: {[t.name for t in ba.TASKS]})")


def _build_hint(task, ws):
    """Replicate benchmark_agentic's workspace + per-task strategy hint."""
    hint = f"Workspace: {ws}\nProject type: Python (ephemeral benchmark task)"
    goal_lower = task.goal.lower()
    if "circular import" in goal_lower or "import cycle" in goal_lower:
        hint += (
            "\n\nCircular-import fix — use exactly these two edits, in this order, "
            "then stop editing:\n"
            "1. edit_file path='b.py' old_string='from a import a_thing\\n\\n\\n' "
            "new_string='' — removes the top-level cross-module import line along "
            "with the blank lines that follow it.\n"
            "2. edit_file path='b.py' "
            "old_string='def use_a():\\n    return a_thing()' "
            "new_string='def use_a():\\n    from a import a_thing\\n    "
            "return a_thing()' — adds the lazy import as the first line inside use_a.\n"
            "Do NOT touch a.py."
        )
    if "gallery" in goal_lower and "gallery-item" in goal_lower:
        hint += (
            "\n\nHTML insertion tip: this file has many repeated </div> tags, so "
            "edit_file will fail. Use insert_at_line(path, line, text) instead."
        )
    return hint


def run_one(task, model, use_grammar, grammar, config):
    """Run a single task once. Returns a metrics dict. Mirrors benchmark_agentic
    run_single_task's Agent construction, toggling ONLY the grammar."""
    ws = Path(tempfile.mkdtemp(prefix=f"gbnf_ab_{task.name}_"))
    try:
        task.setup(ws)
        memory = AgentMemory(_project_id(ws))
        registry = build_default_registry(ws, memory=memory)

        bm_cfg = config.base_model
        cfg_temp = getattr(bm_cfg, "temperature", None)
        cfg_max = getattr(bm_cfg, "max_tokens", None)
        max_tokens_per_turn = int(cfg_max) if cfg_max else 1024
        max_tokens_per_turn = max(512, min(max_tokens_per_turn, 8192))

        hint = _build_hint(task, ws)

        def _pre_finish():
            ok, reason = task.check(ws, None)
            return None if ok else reason

        agent = Agent(
            model=model,
            registry=registry,
            system_prompt_extra=hint,
            workspace_root=ws,
            memory=memory,
            auto_verify_python=True,
            max_iterations=task.max_iterations,
            max_wall_time=task.max_wall_time,
            max_tokens_per_turn=max_tokens_per_turn,
            temperature=cfg_temp if cfg_temp is not None else 0.1,
            confirm_risky=lambda _c: True,   # auto-approve risky (matches bench)
            pre_finish_check=_pre_finish,
            pre_finish_max_retries=2,
            # The variable under test:
            tool_call_grammar=grammar if use_grammar else None,
        )

        # Delta the global repair-tier counters around just this run.
        before = agent_tools.get_repair_tier_counts()
        start = time.monotonic()
        result = agent.run(task.goal)
        wall = time.monotonic() - start
        after = agent_tools.get_repair_tier_counts()
        repair_delta = {k: after[k] - before[k] for k in after}

        passed, reason = task.check(ws, result)
        return {
            "passed": passed,
            "reason": reason,
            "iterations": result.iterations,
            "tool_calls": len(result.tool_calls),
            "wall": wall,
            "est_tokens": result.est_tokens,
            "parse_errors": result.parse_errors_total,
            "json_recovery": result.json_recovery_fired,
            "grammar_turns": result.grammar_turns,
            "repair_total": sum(v for k, v in repair_delta.items() if k != "decode_failed"),
            "decode_failed": repair_delta["decode_failed"],
            "repair_breakdown": {k: v for k, v in repair_delta.items() if v},
            "stop_reason": result.stop_reason,
        }
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tasks = [_task_by_name(n) for n in task_names]

    config = Config("config.yaml")
    config.setup_logging()
    config.base_model.path = args.model
    # The default config ships context_length=4096 (tuned for the 0.5B). The
    # 14B agent prompt (system block + 10-tool schemas + transcript) is ~5k
    # tokens on turn 1 alone, so 4096 makes every generate() fail with
    # "Requested tokens exceed context window" at iter 0. ulcagent/the 14B
    # benchmarks run a wider window; 16384 matches the agent's default
    # context_char_budget (~16k tokens) and fits in the 3060's 12GB with the
    # Q4_K_M 14B. Identical for OFF and ON, so it doesn't bias the A/B.
    config.base_model.context_length = 16384
    print(f"loading {args.model} (GPU, n_ctx={config.base_model.context_length}) ...")
    model = BaseModel(config.base_model)
    model.load()
    grammar = LlamaGrammar.from_string(GBNF)
    print("model + grammar ready\n")

    # rows: (task, repeat_idx) -> {OFF: metrics, ON: metrics}
    rows = []
    for task in tasks:
        for r in range(args.repeat):
            label = f"{task.name}" + (f"#{r+1}" if args.repeat > 1 else "")
            print(f"=== {label} : grammar OFF ===")
            off = run_one(task, model, False, grammar, config)
            print(f"    pass={off['passed']} iters={off['iterations']} "
                  f"calls={off['tool_calls']} parse_err={off['parse_errors']} "
                  f"repair={off['repair_total']} fail_decode={off['decode_failed']} "
                  f"recov={off['json_recovery']} t={off['wall']:.0f}s "
                  f"gturns={off['grammar_turns']}")
            if off["repair_breakdown"]:
                print(f"      repair tiers: {off['repair_breakdown']}")
            if not off["passed"]:
                print(f"      FAIL reason: {off['reason'][:160]}")

            print(f"=== {label} : grammar ON  ===")
            on = run_one(task, model, True, grammar, config)
            print(f"    pass={on['passed']} iters={on['iterations']} "
                  f"calls={on['tool_calls']} parse_err={on['parse_errors']} "
                  f"repair={on['repair_total']} fail_decode={on['decode_failed']} "
                  f"recov={on['json_recovery']} t={on['wall']:.0f}s "
                  f"gturns={on['grammar_turns']}")
            if on["repair_breakdown"]:
                print(f"      repair tiers: {on['repair_breakdown']}")
            if not on["passed"]:
                print(f"      FAIL reason: {on['reason'][:160]}")
            print()
            rows.append((label, off, on))

    # ── Results table ──
    print("\n" + "=" * 100)
    print("GBNF tool-call A/B — OFF vs ON (14B, write-heavy agentic tasks)")
    print("=" * 100)
    hdr = (f"{'task':<22} | {'cond':<4} | {'pass':<4} | {'iter':<4} | "
           f"{'calls':<5} | {'p_err':<5} | {'repair':<6} | {'fail':<4} | "
           f"{'time':<6} | {'gturns':<6}")
    print(hdr)
    print("-" * len(hdr))
    agg = {"OFF": dict(p=0, n=0, perr=0, rep=0, fail=0, t=0.0),
           "ON":  dict(p=0, n=0, perr=0, rep=0, fail=0, t=0.0)}
    for label, off, on in rows:
        for cond, m in (("OFF", off), ("ON", on)):
            print(f"{label:<22} | {cond:<4} | "
                  f"{('Y' if m['passed'] else 'N'):<4} | {m['iterations']:<4} | "
                  f"{m['tool_calls']:<5} | {m['parse_errors']:<5} | "
                  f"{m['repair_total']:<6} | {m['decode_failed']:<4} | "
                  f"{m['wall']:<6.0f} | {m['grammar_turns']:<6}")
            a = agg[cond]
            a["p"] += int(m["passed"]); a["n"] += 1
            a["perr"] += m["parse_errors"]; a["rep"] += m["repair_total"]
            a["fail"] += m["decode_failed"]; a["t"] += m["wall"]
    print("-" * len(hdr))
    for cond in ("OFF", "ON"):
        a = agg[cond]
        print(f"{'TOTAL':<22} | {cond:<4} | {a['p']}/{a['n']:<2} | "
              f"{'':<4} | {'':<5} | {a['perr']:<5} | {a['rep']:<6} | "
              f"{a['fail']:<4} | {a['t']:<6.0f} | ")
    print("=" * 100)
    print("p_err = tool-call parse errors (pre-recovery) | repair = post-Tier-1 "
          "salvage firings (tiers 2-5) | fail = bodies that exhausted all repair tiers")
    print("gturns ON should equal tool-call turns; OFF must be 0 (proves the flag).")


if __name__ == "__main__":
    main()
