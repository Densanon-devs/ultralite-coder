# Ultralite Code Assistant — Roadmap

## Status: v0.1.0 Release

**What's done:**
- 359 YAML augmentor examples across 12 languages (Python, JS/TS, Go, Rust, SQL, C, Java, C#, Ruby, Bash, Kotlin, Swift)
- 99.2% on 130-query multi-language benchmark, 99.8% on 400-query Python benchmark
- Auto mode with language-aware routing (rerank1 for <1.5GB, rerank for >=1.5GB)
- 50+ failure routing categories with language detection boosting
- Web UI with streaming, multi-turn conversation, code paste-and-fix, project indexing
- FastAPI server with /generate, /generate/stream, /session/reset, /project/index, /project/status
- Project context: index a codebase for context-aware code generation
- One-click launcher (launch.py) with dep check, GPU detection, model selection
- Noisy startup logs suppressed, offline mode for sentence-transformers
- Model download helper (download_model.py with 8 model options)
- README, LICENSE (MIT), pyproject.toml, clean .gitignore

---

## Phase 14: Agentic Harness — MVP DONE 2026-04-14

**Goal:** Transform benchmark-winning Ultralite Coder into a daily-driver autonomous agent that can do real project work end-to-end. Hard constraints: zero servers, 100% privacy. Targeting "Claude Code on your device" UX.

**Build plan steps (8 total):**

| Step | Description | Status |
|---|---|---|
| 1 | Tool parser/registry in Qwen 2.5 native Hermes format | ✅ `engine/agent_tools.py` |
| 2 | 8 builtin tools (read/write/edit/list/glob/grep/bash/tests) | ✅ `engine/agent_builtins.py` |
| 3 | ReAct loop with budgets + risky confirm + event callbacks | ✅ `engine/agent.py` |
| 4 | Project awareness — workspace hint in system prompt | ✅ (lite, stdlib only — FAISS top-k deferred) |
| 5 | Cross-session memory + `remember` tool | ✅ `engine/agent_memory.py` |
| 6 | Auto-verification loop (Python `compile()` after writes) | ✅ in `engine/agent.py` |
| 7 | `main.py --agent` CLI flag + `/agent` REPL command | ✅ |
| 8 | Agentic benchmark suite (multi-step tasks) | TODO |

**Test coverage:** 4 standalone smoke suites totaling 19 tests:
- `engine/agent_tools.py` — parser/registry, multi-line JSON tolerance
- `engine/agent_builtins.py` — all 9 tools including memory-gated `remember`
- `engine/agent_memory.py` — per-project isolation, cap, clear
- `engine/agent.py` — 9 ReAct loop tests (single-shot, happy path, max iterations, risky deny, unknown tool, model error, auto-verify catch+fix, memory inject+remember, auto-verify opt-out)

Plus end-to-end UCA integration verified with stub model: read → edit → auto_verify → remember chain works against a real workspace.

**Code shipped:**
- `engine/agent_tools.py` (~340 lines) — `ToolSchema`, `ToolCall`, `ToolResult`, `ToolRegistry` with Hermes system block + lenient JSON parser
- `engine/agent_builtins.py` (~660 lines) — 9 builtin tools, `Workspace` path anchor, optional memory tool registration
- `engine/agent_memory.py` (~135 lines) — `AgentMemory` per-project notes
- `engine/agent.py` (~430 lines) — `Agent` class, `AgentEvent`, `AgentResult`, ReAct loop, auto-verify
- `main.py` — `+_build_workspace_hint`, `+_get_agent`, `+_confirm_risky_tool`, `+_render_agent_event`, `+run_agent`, `--agent GOAL` flag, `/agent` REPL command, `AgentMemory` wired in

**Daily-driver entry points (working as of 2026-04-14):**
- One-shot: `python main.py --agent "fix the failing test in tests/test_foo.py"`
- Interactive: `python main.py` then `/agent <goal>` at the REPL prompt
- Both reuse the GGUF loaded by `config.yaml` (Qwen 2.5 Coder 14B is the recommended choice — Phase 13 winner)

**Polish remaining (not blocking daily-driver use):**
- Step 4 full — FAISS-backed top-k file injection from `engine/project_context.py` into the agent's system prompt at run start
- Step 8 — agentic benchmark suite of multi-step real-project tasks; target 70% first-try / 90% with one nudge
- Multi-language auto-verify (currently Python-only via `compile()`; could add `node --check`, `cargo check`, etc.)
- Verbose mode flag for the REPL (currently event renderer is medium-verbose, no quiet/loud toggle)

### Research look-intos for Step 8 (agentic benchmark design)

Three external references worth reading before designing the benchmark suite — all surfaced during the 2026-04-14 research review, gated by "does this give leverage on a shipped product this week?"

- **LABBench2 (arXiv 2604.09554)** — biology benchmark, zero domain overlap, but the headline result is a **26–46% accuracy drop** when tasks move from "knowledge tests" to "realistic research work." Same shape Step 8 is about to hit: Qwen 2.5 Coder 14B is at 99.0% on V1+V2 single-query codegen, and the moment we measure it on "read 3 files, run pytest, parse the failure, edit one line, re-run" the number will crater. Use the 26–46% figure as a calibration anchor for threshold-setting. The stated Phase 14 target (70% first-try, 90% with one nudge) is probably the correct ballpark; LABBench2 is external confirmation that's not pessimism. Do NOT read the paper body — the 1,900-task dataset is biology-specific and not useful.

- **Seven simple steps for log analysis in AI systems (arXiv 2604.09563)** + the **Inspect Scout** library. A methodology paper from the UK AISI / METR evaluation-governance cluster. Proposes a standardized seven-step pipeline for analyzing logs from AI tool-use runs, with code examples in Inspect Scout. Directly relevant to Step 8 because Step 8's real problem isn't "did the task pass" — it's "across N multi-step tool-calling runs, which failure modes dominate, and does a fix for mode A cause regressions in mode B?" Skim the Inspect Scout code samples before writing a bespoke log schema from scratch. The one actionable idea worth stealing verbatim: store *multiple* weak signals per run (tool-call counts, turn counts, auto_verify triggers, risky-tool denials, etc.), not just pass/fail — that gives `graph_gap_analyzer.py`-style re-analysis something structured to work against.

- **Simon Willison — Servo crate exploration (2026-04-13)** — Simon pointed Claude Code at the newly released `servo` v0.1.0 crate and asked it to figure out what the crate could do. Claude Code autonomously built `servo-shot`, a Rust CLI rendering HTML → PNG against stable Rust, successfully screenshotted news.ycombinator.com, tried compiling Servo to WASM, determined infeasibility (threads + SpiderMonkey), and pivoted to a WASM build of `html5ever`/`markup5ever_rcdom` instead. **This is NOT an architectural idea** — don't let it become one. The only use here is as a **reference task shape for the Step 8 benchmark**: "given unfamiliar crate/library X, build a small working CLI that uses it." It's a good real-world agentic test because it exercises tool use (read, write, run_bash), error recovery (the WASM compile failure → pivot), and end-to-end closure (tool actually runs and produces output). Copy the shape, not the paradigm. If Qwen 2.5 Coder 14B running through ultralight-coder's agent loop can do even a watered-down version of this on a familiar Python library, Phase 14 is validated as a daily driver.

**Skipped — not worth the time:**
- **arXiv 2604.09624 (SECL / test-time discriminative distillation)** — belongs in tax-engine's human-review queue, not here. ultralight-coder already wins via augmentor verifiers and auto+language-scoping gate; test-time training would muddy the zero-servers guarantee with no clear benefit.
- **Reddit `1sl6931` "24/7 Headless AI Server on Xiaomi 12 Pro"** — URL verified and content read. Author flashed LineageOS (strips Android UI for ~9 GB free RAM), froze the Android framework, hand-compiled `wpa_supplicant` for headless networking, built a thermal daemon that toggles external active cooling via a Wi-Fi smart plug at 45 °C, caps charging at 80 %, and serves Gemma "4" via Ollama as a **LAN-accessible API**. The post has NO tok/s, NO context length, NO quant level, NO workload numbers — it's an operational showcase, not a benchmark. Still skipped for ultralight-coder: a LAN HTTP endpoint is exactly what the daily-driver zero-servers rule excludes (see `feedback_local_ai_privacy.md`). Parked in `experiment_backlog.md` entry #7 as edge-deployment prior art in case a mobile-game NPC hosting story for Anima ever becomes a real ask.

### Design reference: skill-persistence format (Phase 14, future)

- **Syll (arXiv 2606.07594, GitHub: https://github.com/THU-SAGE/syll)** — MIT-licensed personal-automation agent (2026-06). Its skill artifact format is worth noting as a precedent for Phase 14's eventual skill-persistence layer: skills are stored as individual Markdown files at `~/.syll/workspace/skills/{name}/SKILL.md`, one directory per skill, editable directly by the user via a web UI. LiteLLM-backed so Ollama/vLLM local inference works out of the box. What's relevant to ULC: the per-skill directory layout, the plain-Markdown file format, and the user-editable design — those are all feasible analogs for how Phase 14 might persist learned augmentor patterns beyond the current per-project `notes.md`. What is NOT relevant: Syll's screenshot+OCR GUI control layer (it's a desktop automation agent; ULC is CLI/IDE-only). This is a reference, not a prescription — Jordan decides at Phase 14 design time whether to adopt the SKILL.md pattern or stick with YAML augmentors.

---

## Phase 13: Large Model Integration — DONE 2026-04-13 (followup 2026-04-14)

**Final result:** **Both 14B models hit 198/200 (99.0%) on V1+V2 and 127–128/130 (97.7–98.5%) on multi-language**, closing the gap to the 0.5B–3B baseline to 1–2 queries. The winning config is `auto` mode with language scoping enabled (default behavior of `use_auto_augmentors` for models ≥3 GB).

The followup also revealed a pre-existing bug: graph mode was never actually doing graph retrieval (`_build_graph_augmentors` didn't set `_retrieval_mode="graph"` on clones). Once fixed, graph+scoping ties auto+scoping on coder-14b and LOSES on qwen-14b — the "graph helps large models" hypothesis was never true; the real win was language scoping.

**Journey:** aug 189/181 → raw 193/189 → largemode-1024 196/195 → V1+V2 hardening 198/200 (qwen) → **auto+scoping 198/198 V1+V2** + **127-128/130 multi-lang (+17 qwen, +16 coder)**.

See `BENCHMARKS.md` Phase 13 + Phase 13 Followup sections.

**Code shipped:**
- `engine/augmentors.py` — large-mode gating + language detection + retrieval scoping + `_set_language_scoping` dispatcher; `_build_graph_augmentors` fixed to actually set `_retrieval_mode="graph"` on clones (was silently broken pre-followup)
- `benchmark_phase13.py` — `--augmentor-mode {auto,graph,adaptive,hybrid,none}`, `--query-set {v1v2,v3,v4,all}`, `--max-tokens` default 1024
- `benchmark_realworld.py` — tuple OR-groups in `must_contain`, `run_benchmark` auto-bumps `max_tokens` to 1024 for models ≥3 GB
- `benchmark_realworld_v3.py` / `benchmark_realworld_v4.py` / `benchmark_multilang.py` — idiom hardening (contextmanager, impl<T>, AsyncMock, ChainMap, etc.)
- `rerun_phase13_truncated.py` / `rerun_phase13_hardened.py` / `rerun_phase13_v3v4_hardened.py` — surgical re-run helpers
- `graph_gap_analyzer.py` — read-only offline failure analyzer, surfaced the cross-language bleeding

**Not shipped (deferred):**
- Speculative decoding — `LlamaPromptLookupDecoding` is broken in llama-cpp-python 0.3.4/0.3.9 and community reports say it gives no speedup even when it works. The C++ CLI `llama-speculative-simple` + Qwen 0.5B draft-model path is the only plausible route to the 2.5× community numbers; see experiment backlog.
- The final 2–3 residual failures per model (both V1+V2 and multi-lang) are either check brittleness too marginal to fix without losing discriminative power, or genuine model quirks (truncation, prompt comprehension). Not worth chasing further.

---

## Phase 13 (archived — original plan)

**Goal:** Add Qwen 2.5 Coder 14B and Qwen 2.5 14B (non-coder) to the model roster and find the right scaffolding posture for large models. Hypothesis: augmentor injection (built for sub-3B models) hurts 14B-class models the same way it hurt Qwen Coder 1.5B in Phase 1 (100% → 90% structural with experts).

**Hardware:** RTX 3060 12 GB, Ryzen 5800X, 32 GB DDR4. Pre-phase BIOS tuning: enable XMP/DOCP (RAM is at 2133 MT/s, rated 3600), Resizable BAR, Above 4G Decoding. Both models are ~9 GB Q4_K_M — cannot coexist in 12 GB VRAM, all runs must be serial (process-per-model so VRAM releases between runs).

**Models:**
- `qwen2.5-coder-14b-instruct-q4_k_m.gguf` — HF: `Qwen/Qwen2.5-Coder-14B-Instruct-GGUF`
- `qwen2.5-14b-instruct-q4_k_m.gguf` — HF: `Qwen/Qwen2.5-14B-Instruct-GGUF`
- Drop into `models/` alongside existing GGUFs. Add entries to `download_model.py` (`qwen-coder-14b`, `qwen-14b`).

**Execution plan:**

1. **Download** both GGUFs (~18 GB) to `models/`. Run `python download_model.py --model coder-14b` then `python download_model.py --model qwen-14b`.
2. **Sanity load** — instantiate each via `llama_cpp.Llama(n_gpu_layers=99, n_ctx=4096)`, run a 20-token completion. Back off to `n_ctx=2048` or IQ4_XS quant (~7.4 GB) if OOM.
3. **Raw baseline** — `benchmark_realworld.py --both`, `benchmark_realworld_v3.py`, `benchmark_realworld_v4.py` for each model, `augmentors.enabled=false`. Six runs total, serial, one subprocess per model.
4. **Speculative decoding matrix** — `benchmark_phase13.py --target coder-14b` (and `--target qwen-14b`). Runs **3 configs** per target model: raw baseline, + prompt_lookup (num_pred_tokens=10, default), + prompt_lookup (num_pred_tokens=15, aggressive). Measures tok/s and pass rate per config. Uses `engine/native_speculative.py` with `LlamaPromptLookupDecoding`. Note: the second-Llama draft-model path (Qwen Coder 0.5B → 14B) that community reports 2.5× for is not supported by llama-cpp-python 0.3.9's Python API — that speedup comes from the `llama-speculative-simple` C++ CLI binary and requires a separate subprocess-based benchmark (`benchmark_phase13_cli.py`, future work).
5. **Augmented pass** — six runs from step 3 with augmentors enabled. Compute delta per model, per domain.
6. **Large-model module** — add `modules/code_gen_large/manifest.yaml` (minimal system prompt, no few-shot wrapper, trust the model). Auto-select when `base_model.size_mb > 3000` via a threshold check in `engine/augmentors.py` or `engine/module_manager.py`. Re-run benchmark through ultralight-coder's full stack.
7. **Analysis** — compare 14B raw / 14B augmented / 14B large-module vs the Phase 7 0.5B–3B + augmentor baselines (399/400). Which wins per domain? Is the augmentor system adding or subtracting value at 14B? Does `prompt_lookup` give a meaningful tok/s win on code-heavy prompts with their repetitive structure? Update `BENCHMARKS.md` with results.

**Fine-tuning target (no weight training):** adjust ultralight-coder's output-shaping — prompt templates, augmentor gating, fusion token budgets, stop tokens, temperature — based on what the raw-vs-augmented deltas reveal. "Fine-tune where it helps" means tuning the scaffolding, not the model weights.

**Blockers:**
- BIOS updates (user, before execution starts)
- ~18 GB disk on D: (no issue — 679 GB free)

---

## Priority 3: Expand (after initial release)

### 3.1 Multi-Language Support
Augmentor examples are Python-only. Add packs for:
- JavaScript/TypeScript (most-requested after Python)
- Go (good for small model code gen — simple syntax)
- Rust (harder but high demand)
- SQL (standalone queries)

### 3.2 Project Context
Let users point at a directory and have the system understand their codebase:
- Index files with embeddings
- Include relevant snippets in the prompt
- Understand imports and framework usage

### 3.3 Multi-Agent Pipeline (branch: multi-agent-architect)
3B architect decomposes, 0.5B workers build pieces, 3B assembles.
**Status:** Tested, decomposition works (8-9/10), assembly needs work (3-5/10).
**TODO on that branch:**
1. Pass interface specs to workers so pieces connect
2. Add execution check after assembly (run code, feed errors back)
3. Speed up assembly (concatenate + fix imports, don't rewrite)
4. Add auth/hashing keywords to failure routing

### 3.4 VS Code Extension
Wrap the FastAPI server as a VS Code extension backend. Users select code, right-click, "Generate/Review/Explain with Ultralite."

### 3.5 Benchmark Hardening
- Add execution tests to real-world benchmark (actually run the generated code)
- Track non-deterministic variance (run each query 3x, report consistency)
- Test with deliberately adversarial prompts

---

## Completed Phases

| Phase | Date | Result |
|-------|------|--------|
| 1: Structural benchmark | 2026-03-29 | Proved structural tests are useless |
| 2: Execution benchmark | 2026-03-29 | Qwen 1.5B at 98.1%, direct > augmented |
| 3: Stress benchmark | 2026-03-30 | 0.5B+experts (49%) > 3B direct (42%) |
| 4: Programmer pack | 2026-03-30 | +26-29% boost, 25 examples |
| 5: YAML modularization | 2026-03-30 | 0.5B: 94%, 1.5B: 100% |
| 6: Retrieval strategy | 2026-04-01 | Single injection best for sub-3B |
| 6b: Auto mode | 2026-04-01 | Model-size-adaptive matches hand-tuned |
| 7: Real-world 400 queries | 2026-04-01 | 399/400 (99.8%) |
| 8: Multi-agent (branch) | 2026-04-01 | Decomposition 9/10, assembly 4/10 |
| 9: Ship-ready polish | 2026-04-02 | Multi-turn, paste-fix, streaming, log suppression, offline mode |
| 10: Multi-language | 2026-04-02 | 12 languages, 437 examples, 99.2% benchmark, language-aware routing |
| 11: Project context | 2026-04-02 | Codebase indexing + retrieval, FAISS-backed, UI integration |
| 12: v0.1.0 ship prep | 2026-04-02 | README, LICENSE, pyproject.toml, requirements audit, .gitignore |
