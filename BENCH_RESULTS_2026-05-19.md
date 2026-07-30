# Real bench — master with A3/A4 default-on promotions (2026-05-19 PM)

Model: Qwen 2.5 Coder 14B Q4_K_M, RTX 3060 12GB, config_agent14b.yaml.
Harness: `benchmark_agentic.py --repeat 5 --share-model` (15-task suite).
Raw data: `bench_master_a3on_2026-05-19.json` + `.log`,
`bench_btc_recheck_2026-05-19.json` (build_todo_cli A/B).

## Headline

**66/75 (88%)** at repeat 5. Bayesian calibration (uniform Beta(1,1) prior):
suite mean **0.870**, 95% credible interval **[0.787, 0.935]**.

## Per-task

| Task | Pass | Note |
|---|---|---|
| add_docstring | 5/5 | |
| extend_calculator | 5/5 | |
| fix_paginate | 5/5 | |
| add_cli_flag | 5/5 | |
| fix_import_cycle | 5/5 | |
| add_json_field | 5/5 | |
| fix_js_reducer | 5/5 | |
| write_bash_lister | 5/5 | |
| add_ts_interface | 5/5 | |
| ambiguous_anchor_rename | 5/5 | |
| rename_function | 4/5 | documented flapper |
| refactor_dataclass | 4/5 | documented flapper |
| fix_yaml_indent | 4/5 | documented flapper (structural YAML) |
| extend_real_gallery | 3/5 | documented HTML-ceiling task (192-line real file) |
| build_todo_cli | 1/5 | documented multi-file-JSON ceiling; recheck gave 2/5 |

## Verdict: NO regression from A3/A4 default-on

Excluding the two genuine model-ceiling tasks (`build_todo_cli`,
`extend_real_gallery`), the other 13 tasks scored **62/65 = 95.4%** —
inside the documented 39-42/42 (93-100%) variance band.

### A3 (test-edit hint, default-on) is EXONERATED
Every `build_todo_cli` failure is `missing_file`, and the missing file set
includes `test_todo.py` itself. A3's hint only fires AFTER a successful
test-file write, so on the failing runs it provably never fired. The
failures are the multi-file-JSON ceiling, not the hint.

Caveat: the bench did NOT demonstrate a benefit from A3 either — none of
the passing test-writing tasks needed the hint to pass. A3's status is
**neutral, no harm observed**. Kept default-on (cheap, addresses a real
failure mode that didn't manifest in this task set). Defensible to revert
if zero-unproven-behavior-change is preferred.

### A4 (extended-tool pruning, default-on for extended only)
No-op on this bench — the lean 10-tool path is what `benchmark_agentic.py`
exercises, and pruning only engages on extended (21-tool) registries.
NOT exercised by this run. To validate A4, a separate
`--extended`-mode bench is needed (the extended path is documented to
regress to ~85.7% without pruning; A4 should recover some of that).

### build_todo_cli variance (the A/B)
| Batch | Result |
|---|---|
| Main run (repeat 5) | 1/5 (20%) |
| Recheck (repeat 5) | 2/5 (40%) |
| Combined | 3/10 (30%) |

The 20pp swing between two repeat-5 batches confirms this task is
genuinely high-variance — consistent with the documented "flaps 0-100%"
characterization. Not a deterministic regression.

## Recommended next steps for the operator

1. **A4 validation** — run `benchmark_agentic.py --repeat 3` with an
   extended-tool registry (the bench would need an `--extended` plumb, or
   wire `auto_prune_tools_k` in a one-off). Compare pruned vs unpruned on
   the extended path to confirm A4 recovers the documented 85.7% regression.
2. **build_todo_cli is the standing ceiling.** It's the multi-file-JSON
   emission limit. The opt-in levers shipped today target it: GBNF grammar
   (A1) + grammar-retry (C4) for the JSON discipline, hashline edit (C2)
   for quote-heavy lines. A focused A/B of `build_todo_cli` with those
   flags ON is the highest-value follow-up.
3. **KV-Q8 + IQ4_XS stack swap** (from the AM audit) remains the model-side
   lever if the harness levers plateau.
