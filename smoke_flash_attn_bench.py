"""
Throughput micro-bench: Qwen 2.5 Coder 14B Q4_K_M, fixed-length
completion, repeated N times, reports tokens/second.

Used to A/B flash_attn=True vs flash_attn=False on the same hardware
and the same model. Doesn't measure end-to-end agent task quality —
just per-token decode throughput, which is what flash_attn affects.

Usage:
    python smoke_flash_attn_bench.py [--flash] [--max-tokens 200] [--repeat 3]

Note: this script loads the model directly via llama_cpp.Llama,
NOT through engine.base_model, so we can A/B the kwarg in one
script without modifying the production loader.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DEFAULT_MODEL = ROOT / "models" / "qwen2.5-coder-14b-instruct-q4_k_m.gguf"

SHORT_PROMPT = """<|im_start|>system
You are a Python expert.<|im_end|>
<|im_start|>user
Write a Python function that takes a list of integers and returns the
median value. Handle empty lists by returning None. Add a docstring
explaining the algorithm.<|im_end|>
<|im_start|>assistant
"""


# Realistic agent-loop prompt: large system block with tool schemas
# (mirrors what ulcagent constructs) + a multi-turn transcript that
# simulates a mid-session agent state. Approximately 4K input tokens.
def _make_long_prompt(target_tokens: int = 4000) -> str:
    """Build a realistic agent-shaped prompt of roughly target_tokens
    input length. Padding is repeated tool schemas + a fake transcript,
    which is structurally similar to what ulcagent sees mid-session."""
    fake_tool_schema = """  - read_file(path: str, offset?: int, limit?: int) -> str:
    Read a file. Returns numbered lines. Use offset/limit on large files.
  - write_file(path: str, content: str) -> str:
    Write content to a file. Creates parent directories. Returns "Wrote N chars to path".
  - edit_file(path: str, old_string: str, new_string: str, replace_all?: bool) -> str:
    Exact-string replacement. Fails on multi-match unless replace_all.
  - list_dir(path: str, depth?: int) -> str:
    Tree-listing of a directory. Depth 1-5. Skips __pycache__/node_modules/.git.
  - glob(pattern: str) -> list[str]:
    Files matching a glob pattern (e.g. "src/**/*.py"). Max 200 results.
  - grep(pattern: str, path?: str, glob?: str) -> list[str]:
    Ripgrep-style content search. Returns matching lines with file:line.
  - run_bash(cmd: str) -> str:
    Execute a shell command. RISKY — prompts y/N before running.
  - run_tests(framework?: str) -> str:
    Run project tests. Auto-detects pytest/unittest/npm/go/cargo.
"""
    fake_turn = """<|im_start|>user
Continue working on the previous task. The user has reviewed the changes.
<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "read_file", "arguments": {"path": "engine/agent.py", "offset": 100, "limit": 50}}</tool_call><|im_end|>
<|im_start|>tool
<tool_response>
   100  def some_function(arg):
   101      x = compute(arg)
   102      return x
   ...
</tool_response>
<|im_end|>
"""
    # Build until we hit target token count (~4 chars/token)
    body = fake_tool_schema
    while len(body) // 4 < target_tokens - 500:
        body += "\n" + fake_tool_schema + "\n" + fake_turn
    return f"""<|im_start|>system
You are a coding agent. Available tools:

{body}

Use tools to accomplish the user's goal. Respond with tool calls in
Hermes format: <tool_call>{{"name": ..., "arguments": ...}}</tool_call>
<|im_end|>
<|im_start|>user
Now refactor the median function I wrote to also handle empty lists
gracefully and add proper type hints.<|im_end|>
<|im_start|>assistant
"""


DEFAULT_PROMPT = SHORT_PROMPT  # overridden by --long


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash", action="store_true",
                        help="Enable flash_attn=True")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--long", action="store_true",
                        help="Use realistic ~4K-token agent-loop prompt "
                              "(flash_attn benefits scale with seq length)")
    parser.add_argument("--target-tokens", type=int, default=4000,
                        help="--long target input token count")
    args = parser.parse_args()

    prompt = _make_long_prompt(args.target_tokens) if args.long else SHORT_PROMPT
    print(f"Prompt length: ~{len(prompt) // 4} input tokens")

    if not args.model.exists():
        # Fallback: try any qwen 14b in the models dir
        candidates = list(ROOT.joinpath("models").glob("qwen2.5*14b*.gguf"))
        if not candidates:
            print(f"ERROR: model not found at {args.model}, no fallback")
            return
        args.model = candidates[0]
        print(f"Using fallback model: {args.model.name}")

    from llama_cpp import Llama

    llama_kwargs = dict(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=8,
        verbose=False,
    )
    if args.flash:
        llama_kwargs["flash_attn"] = True

    print(f"Loading model with flash_attn={'ON' if args.flash else 'OFF'}...")
    t0 = time.monotonic()
    llm = Llama(**llama_kwargs)
    load_s = time.monotonic() - t0
    print(f"Load time: {load_s:.2f}s")
    print()

    results = []
    for i in range(args.repeat):
        t0 = time.monotonic()
        out = llm(
            prompt,
            max_tokens=args.max_tokens,
            temperature=0.0,
            stop=["<|im_end|>"],
        )
        elapsed = time.monotonic() - t0
        completion = out["choices"][0]["text"]
        usage = out.get("usage", {})
        n_completion = usage.get("completion_tokens",
                                  len(completion.split()) * 2)
        toks_per_s = n_completion / elapsed if elapsed > 0 else 0.0
        results.append({
            "elapsed": elapsed,
            "completion_tokens": n_completion,
            "toks_per_s": toks_per_s,
        })
        print(f"  Run {i+1}: {n_completion} tokens in {elapsed:.2f}s "
              f"= {toks_per_s:.1f} tok/s")

    avg_toks = sum(r["toks_per_s"] for r in results) / len(results)
    avg_time = sum(r["elapsed"] for r in results) / len(results)
    print()
    print(f"Average over {len(results)} runs: {avg_toks:.1f} tok/s "
          f"({avg_time:.2f}s per run)")
    print(f"flash_attn: {'ON' if args.flash else 'OFF'}")


if __name__ == "__main__":
    main()
