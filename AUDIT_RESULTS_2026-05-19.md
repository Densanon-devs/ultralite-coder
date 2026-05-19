# Autonomous Audit Pass — 2026-05-19

Branch: `experiment/autonomous-audit-2026-05-19` (from master)
Operator: Claude (autonomous, user away)
Scope: 11 tasks from `project_ultralight_coder_phase14_next.md` open backlog

User's pre-existing WIP on `experiment/mission-state` (`mission_pre_finish_check` + `benchmark_granite_phase1.py`) preserved via `git stash` and restored at end.

Format per task: outcome (INCORPORATED / RECORDED-AND-DUMPED / BLOCKED), evidence, notes.

---

## Task 1: KV-Q8 cache config gate — **INCORPORATED** (commit `e6db303`)

- Added `BaseModelConfig.cache_type_k` / `cache_type_v` (`Optional[str]`, defaults `None`).
- Wired through `Config._load()` so config YAML round-trips.
- `BaseModel._kv_cache_kwargs()` maps human-readable names (`f16`/`q8_0`/`q4_0`/`q4_1`/`q5_0`/`q5_1`/`f32`/`q8_0`) → ggml `GGML_TYPE_*` int constants for llama-cpp-python's `type_k`/`type_v` kwargs. Unknown names warned and ignored.
- TypeError fallback in `BaseModel.load()` if the runtime's llama-cpp-python build doesn't accept the kwargs — loads with stock F16 KV instead of crashing.
- Sample config (commented out) added to `config_agent14b.yaml`.
- 11 smoke tests in `test_kv_cache_config.py` (kwargs mapping, case-insensitive, unknown-type warn, only-K, only-V, mixed, TypeError fallback, YAML round-trip, defaults). 40-test regression sweep on `test_compaction.py` + `test_auto_verify.py` + `test_parallel_tools.py` + `test_diff_surfacing.py` clean.
- **Operator action required to harvest the actual win:** uncomment the two lines in `config_agent14b.yaml`, re-run `benchmark_agentic.py --repeat 5` against `build_todo_cli` + `extend_real_gallery`, and compare against the documented 39-42/42 variance band. If pass rate moves above 42/42 or the build_todo_cli flap closes, ceiling #1 (JSON quote recovery) was KV-pressure, not a model semantic limit. Not run autonomously (requires GPU time + user oversight).

## Task 2: IDK phrasing audit (CLEAR finding) — **RECORDED-AND-DUMPED** (no code change)

Grepped `*.py`, `*.yaml`, and config files for IDK-style hedges that could prime "I can't" / "I don't know" / "unsure" / "unable to" model outputs (per arXiv 2605.01011 CLEAR finding: open-ended uncertainty hedges *paradoxically increase* incorrect selections).

Findings:
- `engine/agent.py` (system prompt) — already CLEAR-compliant. Explicit rule at line 133: *"NEVER respond to `stuck_repeat` with 'I can't complete the task'"*. The prompt anti-hedges rather than offering an IDK escape.
- `engine/failure_flagger.py` — uses IDK-style phrasing as a **detector** (the `give_up_signals` tuple at L156-165), which is correct. Also offers actionable redirects in retry hints, not blank IDKs.
- `engine/yaml_augmentor_builder.py` — embeds a negative example ("Do NOT respond with 'I cannot complete this task'") in YAML templates. Correct.
- `data/augmentor_examples/**.yaml` — `cannot` / `unable` only appear inside code examples (e.g., `if not file.exists(): raise FileNotFoundError(...)`), not in system-prompt hedges. Clean.
- `profiles/*.yaml` — no IDK priming.
- `ulcagent.py` — no IDK priming.

The single "When in doubt" string in `failure_flagger.py:529` is followed by a concrete action recommendation (`use run_tests instead of run_bash`) — CLEAR-compliant per the paper's "enumerated null with action" pattern.

**Net:** the ULC prompt stack already follows the CLEAR principle. No code change. Future work: when adding new system prompts, augmentor YAMLs, or `.ulcagent` profiles, run this grep pattern as a hygiene check.

## Task 3: HTML output mode for `/export` — **INCORPORATED** (commit `4258b3f`)

- `/export` now accepts `--html` / `--format html` / `--md` / `.html`-suffix-infers. Markdown remains the default.
- HTML output is a self-contained document: embedded CSS (dark-mode aware via `prefers-color-scheme: dark`), severity chips (critical/high/medium/low) auto-rendered at line starts (incl. after bullets/numbers), fenced code blocks → `<pre><code>`, inline backticks → `<code>`, HTML injection escaped via `_html_escape`.
- Pure stdlib (no external links to fonts/CSS) — preserves the air-gap invariant.
- Help text updated; `{dim}` placeholder added to `_print_help` color/no-color paths.
- 24 unit tests in `test_html_export.py`: flag parsing (8), `_render_message_html` (7), `_build_session_html` (5), `_export_session` integration with tmpdir + `_session_log` (4). 75-test regression sweep clean.
- `/review` integration deferred: `/review` currently produces a prompt that goes to the agent, not a direct output. To get HTML output for a review session, run `/review` then `/export --html`. A dedicated `/review --html` flag would need a separate plumbing arc through the agent loop; the `/export --html` path is the cheaper win that covers the same use case.

## Task 4: `mcp` package version audit — **RECORDED-AND-DUMPED** (no action)

- `pip show mcp` → **Version: 1.27.0** (well above the ≥1.19.0 floor flagged in memory; also above 1.23.0 which patched CVE-2025-66416 DNS rebinding per [[experiment_backlog]] entry 92).
- `requirements.txt` does NOT declare an `mcp` dependency. ULC's `engine/mcp_adapter.py` is a scaffold-only adapter (per CLAUDE.md MCP section, interface-only). The installed `mcp` package comes from `claude-agent-sdk-python` / Anthropic CLI, not from ULC.
- **Net:** no upgrade needed; ULC isn't a runtime MCP consumer right now. If `engine/mcp_adapter.py` is ever wired into the agent path, add `mcp>=1.23.0` to `requirements.txt` to lock the CVE patch floor.

## Task 5: PostToolUse JSON repair hook — **INCORPORATED** (commit `c92764e`)

- New module `engine/post_tool_sanitize.py` (~140 LOC). Sits at `ToolRegistry.execute`'s post-execution seam — runs *after* the tool function returns and *before* the result gets serialized for the next model turn.
- Sanitizers (default-on, narrow per [[feedback_dont_change_tool_return_formats]]):
  - Strip lone UTF-16 surrogates (U+D800-U+DFFF) → `�`. These appear in mis-decoded `read_file` returns of binary content and crash `json.dumps` on some Python builds.
  - Escape embedded NUL (`\x00`) + non-whitespace control chars as `\\xHH`. Keeps `\t \n \r` untouched (meaningful whitespace for code).
  - Soft-truncate at 30,000 chars (matches the [[project_ulcagent_web_tools]] 2026-05-10 cap).
- Structure-preserving: recursive on `list`/`tuple`/`dict`; non-str values untouched. Idempotent.
- Per-registry `SanitizerConfig`; opt-out by tool name (default opt-out set: `{"auto_verify"}`). Wholesale disable via `reg._sanitizer_config.enabled = False`.
- 27 unit tests in `test_post_tool_sanitize.py` covering surrogate strip, NUL escape, control-char escape, whitespace preservation, truncation, idempotency, JSON round-trip safety, registry integration with real tool execution, opt-out semantics.
- **525-test full repo sweep clean** (excluding 2 pre-existing broken files unrelated to this change: `test_android.py` calls `sys.exit` at module load; `test_auto_flag_hook.py` imports a missing symbol — both predate the audit branch).

## Task 6: k=5 + Bayesian calibration framework — **INCORPORATED** (commit `8b76bf1`)

- New module `engine/bench_calibration.py` (~280 LOC):
  - `BetaBinomial` posterior on a Bernoulli rate (default uninformative `Beta(1, 1)` prior); mean + equal-tailed credible interval at any level.
  - `regularized_beta` + `beta_inv_cdf` — Numerical Recipes BETACF continued fraction + bisection. Zero scipy dependency, 1e-6 accuracy.
  - `load_repeat_results()` accepts both `benchmark_agentic.py` JSON shapes (`by_task` aggregate from `--repeat N`, or flat `results` list).
  - `calibrate_results()` returns per-task + suite-level posteriors (Beta-Binomial pooling).
  - `apply_human_labels()` reconciles auto-grader passes against a small (~20-30 problem) hand-labeled subset; multiplicative calibration, wider CI honestly surfaces label scarcity.
  - CLI: `python -m engine.bench_calibration <results.json> [--human-labels labels.json] [--out cal.json] [--prior-alpha N --prior-beta M]`.
- Smoke-check on a realistic 41/42 input: mean 0.954, 95% CrI **[0.877, 0.994]**. The "97.6%" point estimate is really a Bayesian interval from 88% to 99%. The `fix_yaml_indent` 2/3 flap: mean 0.6, CrI [0.19, 0.93] — honestly wide at n=3.
- 18 unit tests in `test_bench_calibration.py` (Beta math: 4, BetaBinomial: 5, calibrate_results: 2, load_repeat_results: 2, human-label calibration: 4, CLI: 1). 543-test full sweep clean.
- **Operator action required to harvest the actual win:** run `benchmark_agentic.py --repeat 5` against the 13-task suite (~1 GPU-hour), then `python -m engine.bench_calibration <output.json>` to see calibrated numbers. Optionally hand-label 20-30 problems and pass them via `--human-labels`. Not run autonomously.

## Task 7: G-STEP gate live measurement — **INCORPORATED** (commit `d28085e`)

- Added `engine.tool_gate.GateStats` dataclass with counters split by outcome (`allowed_disabled`, `allowed_action_tool`, `allowed_anchor_ok`, `allowed_no_key_arg`, `rejected_anchor`, `rejected_exploration_cap`, `rejected_by_tool: dict[str, int]`).
- `GateState.stats` field (`default_factory=GateStats`) — backwards-compatible: every existing `GateState()` construction call works.
- `check()` bumps the appropriate counter on every code path (exhaustive coverage of allow/reject branches).
- `fire_rate` property — rejections / (checks - disabled - action_tool). 0.0 means the gate is dead code on this stack — the same observability that surfaced the [[session_2026-05-05_self_heal_ab_finding]] result.
- `AgentResult.gate_stats: dict` — Agent extracts `GateStats.summary()` post-run for inclusion in bench JSON. Empty dict when registry has no gate state.
- 16 new unit tests in `test_tool_gate_stats.py` (counter init, every counter path, fire_rate edge cases, summary keys + json-serializability, backwards-compat). 21 existing G-STEP tests still pass. 559-test full sweep clean.
- **Operator action required:** run `benchmark_agentic.py` with `registry.configure_gate(...)` engaged; inspect each task's `gate_stats` in the JSON output. If `fire_rate` is 0.0 across the run, the gate is the next self_heal-style A/B candidate — flip the default off and keep as opt-in.

## Task 8: Failure-class detection in retry path — **INCORPORATED** (commit `d30c5c5`)

- Extended `engine.self_heal.classify_failure()` with 5 new classes lifted out of the previous `TRACEBACK` catch-all: `IMPORT_ERROR`, `ASSERTION_FAILURE`, `TYPE_ERROR`, `ATTRIBUTE_ERROR`, `KEY_INDEX_ERROR`.
- Each class gets a tailored repair hint in `_PER_CLASS_HINT` so `diagnose_message()` produces a class-targeted prompt instead of generic "two consecutive tool errors" advice. Specifically:
  - `IMPORT_ERROR`: distinguishes "wrong import path" from "package not installed, rewrite in stdlib"
  - `ASSERTION_FAILURE`: directs model to read failing test FIRST, identify divergence, minimal edit
  - `TYPE_ERROR`: enumerates the three common shapes (None / list vs scalar / arity) so the model commits to one
  - `ATTRIBUTE_ERROR`: distinguishes wrong-object (find real attr) from None-at-call-site (fix producer)
  - `KEY_INDEX_ERROR`: enumerates typo / missing-path / off-by-one
- Classifier ordering carefully chosen: `IMPORT_ERROR` checked before `NAME_NOT_DEFINED` (`ModuleNotFoundError` contains "not" + a name but the repair is "fix the import"); `ATTRIBUTE_ERROR` before `TYPE_ERROR`; `ASSERTION_FAILURE` after the typed errors (`assert` text inside an otherwise-typed traceback picks up the right class).
- 28 new unit tests in `test_self_heal_failure_classes.py` (5 new classes × ~3 cases each + ordering conflicts + 8 regression tests confirming existing classes still classify + per-class hint validation). All 52 `self_heal` tests pass. 583-test full sweep clean.

## Tasks 9-11: Model-swap bench scoping — **BLOCKED-AND-DOCUMENTED** (commit `c4fbe27`)

The three benches require GGUF downloads (4.5-18.6 GB) and ~1 GPU-hour each. None of them were run autonomously per the safety policy. What landed instead is a ready-to-run config template per candidate so each bench is a one-command operation when the operator returns.

### Task 9: Qwen3-30B-A3B (front-runner) — template at `config_bench_qwen3_30b_a3b.yaml.template`
- 30.5B-total / 3.3B-active MoE. Q4_K_M GGUF = 18.6 GB.
- **Does NOT fit RTX 3060 12 GB.** Will spill to CPU. Expect tok/s well below the ~20 tok/s Qwen 2.5 Coder 14B baseline.
- Operator floor for daily-driver UX: ≥ 10 tok/s sustained. Below that, NO swap regardless of quality.
- Per [[experiment_backlog]] #79: HF cofounder demo shows Opus-Claude-Code-class quality offline. Credibility gain is the load-bearing argument.
- KV-Q8 strongly recommended at this size — the JSON-quote ceiling case compounds when CPU spill is already slow.
- Bench protocol: `benchmark_agentic.py --config ... --repeat 3 --share-model` then pipe output through `python -m engine.bench_calibration` (Task 6 framework).

### Task 10: Granite 4.1-8B — template at `config_bench_granite_8b.yaml.template`
- 8B dense FP8/Q4_K_M. Fits 12 GB at Q4_K_M (~4.5 GB) with comfortable KV headroom.
- Apache 2.0, BFCL V3 = 68.27 (strong tool-calling number on IBM's bench).
- **Load-bearing question:** does it emit Hermes-format `<tool_call>` JSON natively? Non-Hermes models score ~30% per [[feedback_gemma4_rejected]]. **Run a single-task smoke first before the full 13-task suite.**
- Simon Willison's informal SVG test on the 3B was "all pretty terrible" — code-gen quality at small quants is not yet independently verified beyond IBM's own numbers.
- Bench gates: EvalEval k=5 + Bayesian calibration (Task 6 framework) before declaring a winner against Qwen 2.5 Coder 14B baseline.

### Task 11: AI2 EMO 14B-total/1B-active MoE — template at `config_bench_emo_14b.yaml.template`
- **GGUF NOT YET AVAILABLE as of 2026-05-19.** Template ships but bench is BLOCKED on upstream artifact.
- Check `huggingface.co/collections/allenai/emo` periodically; rerun this task when a Q4_K_M GGUF is published.
- Differentiator: document-level expert routing (experts cluster around domains: health/news/code). 1B active = much faster inference at 14B-total quality. Should be more stable on multi-file agentic tasks where the same code expert stays active.

### Each template includes:
- Exact CLI to run + calibration follow-up
- Model size + VRAM fit notes
- Known compatibility caveats from memory
- Pinned tokenizer-template warnings where relevant
- A "FILL IN" line for the actual GGUF filename

---

## Summary

| # | Task | Outcome | Commit |
|---|---|---|---|
| 1 | KV-Q8 cache config gate | INCORPORATED | `e6db303` |
| 2 | IDK phrasing audit (CLEAR) | RECORDED-AND-DUMPED (no code change — already CLEAR-compliant) | — |
| 3 | HTML output mode for /export | INCORPORATED | `4258b3f` |
| 4 | mcp package version audit | RECORDED-AND-DUMPED (no action — at v1.27.0, well above floor) | — |
| 5 | PostToolUse JSON repair hook | INCORPORATED | `c92764e` |
| 6 | k=5 + Bayesian calibration framework | INCORPORATED | `8b76bf1` |
| 7 | G-STEP gate live measurement | INCORPORATED | `d28085e` |
| 8 | Failure-class detection in retry path | INCORPORATED | `d30c5c5` |
| 9 | Qwen3-30B-A3B bench scoping | BLOCKED-AND-DOCUMENTED (template ready) | `c4fbe27` |
| 10 | Granite 4.1-8B bench scoping | BLOCKED-AND-DOCUMENTED (template ready) | `c4fbe27` |
| 11 | AI2 EMO bench scoping | BLOCKED-AND-DOCUMENTED (template + GGUF watch) | `c4fbe27` |

**Net change: 8 commits on branch `experiment/autonomous-audit-2026-05-19`.**

### Test totals
- New tests: 11 (KV-Q8) + 24 (HTML) + 27 (sanitize) + 18 (calibration) + 16 (gate stats) + 28 (failure classes) = **124 new unit tests**
- Full sweep at end of audit: **583 tests pass** (excluding 2 pre-existing broken test files: `test_android.py` does module-load `sys.exit`; `test_auto_flag_hook.py` imports a missing symbol — both predate this branch).

### Operator action queue (after returning)

1. **Run KV-Q8 ablation** — uncomment the two lines in `config_agent14b.yaml`, re-run `benchmark_agentic.py --repeat 5` against `build_todo_cli` + `extend_real_gallery`, compare against 39-42/42 variance band. Confirms or rejects "JSON quote recovery ceiling is KV pressure" hypothesis.
2. **Run full k=5 + calibration** — `benchmark_agentic.py --repeat 5` on the 13-task suite (~1 GPU-hour), then `python -m engine.bench_calibration <output.json>`. Surfaces real variance instead of point estimates.
3. **Measure G-STEP fire_rate live** — enable the gate via `registry.configure_gate(...)` for one bench run. If `fire_rate == 0.0`, the gate is the next self_heal-style A/B candidate (default off, opt-in).
4. **Validate failure-class targeting in the wild** — once self_heal fires on a real run, inspect which of the 5 new classes are picking up the failures the old generic `TRACEBACK` was eating. If the new classes never fire, leave as-is; if they fire and the model's repair is class-appropriate, this was a real win.
5. **Bench Qwen3-30B-A3B / Granite 8B** when GGUFs are available. Templates already shipped.
6. **Decide on `experiment/autonomous-audit-2026-05-19` branch:** review the 8 commits + merge to master, or cherry-pick individual items. None of them touch the in-flight `experiment/mission-state` work.

### User's WIP preserved
- `experiment/mission-state` branch untouched.
- Uncommitted `mission_pre_finish_check` work (in `engine/agent_builtins.py`, `ulcagent.py`, `test_mission.py`) + untracked `benchmark_granite_phase1.py` are restored via `git stash pop` at the end of this run.
