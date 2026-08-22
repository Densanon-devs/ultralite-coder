# MCP (Model Context Protocol) — scaffold + activation plan

## Status (2026-04-26)

The MCP-client surface is **scaffolded but not wired**. The interface,
CLI plumbing, and registry hook are all in place; the JSON-RPC stdio
client itself is the one piece left to write before users can run

```bash
ulcagent --mcp densa-deck "build me a Selesnya tokens deck under $100"
```

and have the agent drive the Densa Deck engine through MCP tool calls.

Today, passing `--mcp <anything>` raises `NotImplementedError` so the gap
is loud, not silent. Without `--mcp`, ulcagent's behavior is identical
to before this scaffold — same builtin-only registry, same benchmarks,
same default tool count.

## Why MCP

[MCP](https://modelcontextprotocol.io/) is Anthropic's open standard for
AI clients to call tools on local servers. One protocol, many servers
— Densa Deck, GitHub, Postgres, Filesystem, your own custom ones.
Once ulcagent speaks MCP-client, every MCP server you have on disk
becomes an extension of the agent's tool set, with no per-server
adapter code.

It's also how the densanon-devs portfolio cross-pollinates: each
product (Densa Deck, D-Brief, DensaBooks) ships its own MCP server,
and ulcagent mounts whichever ones the user opts in to via `--mcp`.

## What's already in place

- **`engine/mcp_adapter.py`** — interface stubs for `register_mcp_tools`,
  `resolve_servers`, `is_active`, plus a `_BUILTIN_SERVERS` table for
  shortcut names like `densa-deck`. The activation TODO is documented
  inline.
- **`engine/agent_builtins.py::build_default_registry`** — accepts
  `mcp_servers` and `mcp_tool_pack` parameters. Empty/None = no-op
  (default and the only code path tests exercise today).
- **`ulcagent.py::_parse_mcp_arg`** — parses `--mcp foo,bar` from argv
  and threads it through `_build_agent` → `build_default_registry`.
- **`main.py`** — `--mcp` flag added to argparse; `run_agent_fast`
  accepts `mcp_servers`; both `--agent` and `/agent` REPL paths
  forward the value.

## What activation looks like

Three concrete steps, no surprises:

1. **Add the dep.** `pip install mcp` (or pin in `pyproject.toml` under
   an optional extras group).
2. **Implement `register_mcp_tools` in `engine/mcp_adapter.py`.** The
   docstring already lists the six steps — spawn subprocess, init
   session, list tools, filter by `tool_pack`, translate each MCP
   `Tool` into a `ToolSchema`, track lifecycles. Estimate: half a day
   if the SDK doesn't surprise.
3. **Flip `is_active()` to actually probe `import mcp`.** Done.

## Tool-count regression caveat

The [`feedback_tool_count_regression.md`](../../) memory found that the
14B models in this repo regress hard when the registry crosses ~10 tools
(97.6% → 85.7% on the agent benchmark). Mounting an MCP server like
Densa Deck adds ~17 free-tier tools, which would push the lean registry
well past that line.

The `mcp_tool_pack` parameter on `build_default_registry` is the
mitigation: pass an explicit whitelist and only those tools land in the
registry. E.g. for an MTG deckbuilding session:

```python
build_default_registry(
    workspace,
    mcp_servers=["densa-deck"],
    mcp_tool_pack=["search_cards", "analyze_deck", "run_goldfish"],
)
```

When activation lands, `--mcp` from the CLI should default to a curated
3-5 tool pack per server, with a `--mcp-all` opt-in for users who want
the full surface.

## Built-in shortcuts

Defined in `engine/mcp_adapter._BUILTIN_SERVERS`. Today:

| Shortcut | Command | Source repo |
|---|---|---|
| `densa-deck` | `densa-deck mcp serve` | [densa-deck](https://github.com/Densanon-devs/densa-deck) |

Add new entries here as sister projects ship MCP servers.

## Testing

`tests/test_mcp_adapter_scaffold.py` verifies the no-MCP-servers branch
is a clean no-op and that passing a non-empty list raises
`NotImplementedError` with the configured server names included in
the message. When activation lands, those tests will be extended to
exercise the actual JSON-RPC layer with a stub server.

---

# The other direction: ulcagent AS an MCP server

Everything above is ulcagent as an MCP **client** (calling out to other
servers). There are also two servers that let something else drive
**us**. They expose deliberately different surfaces, and the difference
is the whole design point.

| module | driver | exposes | why |
|---|---|---|---|
| `engine/mcp_server.py` | DensAssistant | file primitives — locate / read / create / move / edit | its client has no file tools of its own |
| `engine/mcp_workhorse.py` | Claude Code (or any strong model) | the **agent loop** — delegate / delegate_result / delegate_list / delegate_cancel | Claude Code already has better primitives; what it lacks is a way to hand work off |

## Why the workhorse exposes no file tools

Re-exposing `read_file`/`write_file` to Claude Code would be worse than
useless: it already has stronger versions, and every extra schema is
context it pays for on every turn. See "Tool-count regression caveat"
above — that tax applies to the *driver* too, not just to us.

The one thing a strong model cannot do for itself is spend someone
else's tokens. So the workhorse exposes exactly one capability —
"run this goal to completion locally" — and returns the outcome.

```
Claude Code (plans, decomposes, reviews)
   └── delegate(goal="add pytest cases for engine/foo.py")  ──► job id, instantly
          └── ulcagent 14B loops read/grep/write/edit locally
   └── delegate_result(job_id) ──► answer, files_changed, stop_reason
```

## Running it

```bash
python -m engine.mcp_workhorse --workspace <project> \
       [--model C:/models/qwen2.5-coder-14b-instruct-q4_k_m.gguf] \
       [--profile code|general] [--toolset coding|refactor|git|web|hybrid|full]
```

Register in a project's `.mcp.json` (see this repo's for a working
example). Point `--workspace` at the project you want worked on; an
individual call can override it with the `workspace` argument.

**Use `--model` to point at a copy on the NVMe.** The shared configs
resolve to `./models/`, which is on the USB HDD (~4 MB/s). The server
respawns whenever the driver reconnects, so a cold load off that drive
is minutes; from `C:` it is ~8s.

## Things that are true and will bite you otherwise

- **`delegate` never blocks.** It returns a job id immediately. This is
  load-bearing — a blocking call would stall the driver for the whole
  run and defeat the purpose.
- **Jobs run one at a time.** One GGUF on one GPU. A second `delegate`
  queues behind the first and reports `jobs_ahead`. Firing several
  stages is still useful (the driver isn't blocked), but they are not
  parallel.
- **A running job cannot be cancelled.** There is no cooperative
  cancellation point in the agent loop, so `delegate_cancel` only stops
  jobs still queued and *says so* for running ones rather than
  reporting a cancellation that did not happen. Runs are bounded by
  `max_iterations` (clamped to 50) and a 900s wall clock.
- **Writes are allowlist-gated and nothing else.** As with
  `engine/mcp_server.py`, no human is at a y/N prompt, so
  `engine/write_policy.py` is the only control. No `ask_user_fn` or
  `confirm_*` hook is installed — an unattended path must never block
  on a prompt no one will see. Every mutation is journaled with a
  backup, so `ulcagent --revert-last N` is the undo.
- **The local model cannot see the driver's conversation.** A goal has
  to stand alone. "Fix the thing we discussed" fails; name the files.
- **stdout is the protocol.** All logging goes to stderr.

## `no_op`: the failure mode that matters most

The 14B intermittently returns an **empty generation** on turn 1, which
the agent loop classifies as `stop_reason="answered"`. Left alone, that
surfaces as a job that did nothing while reporting success — and a
driver building its next stage on it corrupts the whole pipeline.

Measured 2026-08-22, same task, four phrasings:

| phrasing | result |
|---|---|
| "…strips leading and trailing hyphens. **Return the slug.**" | empty generation |
| same, trailing sentence removed | ✔ `write_file` |
| "**Use the write_file tool to** create utils.py…" | empty generation |
| "Create a file hello.txt containing…" | ✔ `write_file` |

It is prompt-*shape* sensitive, not a capability limit — naming the
tool explicitly made it worse, not better, and the failures reproduce
deterministically for a given phrasing. This is the same wall as
`feedback_14b_tool_call_ceilings`: not prompt-fixable in general.

So the server detects it (no tool calls **and** empty answer), retries
once, and if it happens again reports a distinct terminal status
`no_op` carrying a hint telling the driver to rephrase. It never
reports it as `done`.

**Drivers must treat any status that is not `queued`/`running` as
terminal.** Our own test harness span forever when `no_op` was added
because it checked `status in ("done","error","cancelled")`. Check
negatively, not positively.

Rephrasing genuinely recovers — the failing goal above succeeded on the
plainer wording (`status: done`, `files_changed: ["utils.py"]`, output
verified correct).

## Testing

`test_mcp_workhorse.py` (18 tests) drives the `Server` class in-process
with a stubbed model, so it asserts the delegation contract without
loading a 9 GB GGUF. It covers the non-happy paths that matter:
`delegate` returning inside 0.5s even when the job takes seconds, model
failures surfacing as `status: "error"` instead of silent success,
cancel refusing to lie about running jobs, jobs queueing rather than
racing, and stdout carrying nothing but framed JSON-RPC.

Model *quality* is not tested here — `benchmark_agentic.py` already
owns that.
