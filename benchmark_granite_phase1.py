"""Phase 1 of the Granite Embedding cross-project bench.

Plan: see C:/Users/Jordan/.claude/projects/D--LLCWork/memory/project_granite_embedding_bench.md

This file executes Phase 1: head-to-head comparison of:

  - baseline:  nomic-ai/nomic-embed-text-v2-moe (ULC's current production embedder
               for engine/project_context.py, used via NomicGGUFEmbedder in real
               deployment. Here we run the sentence-transformers PyTorch variant
               so we can drive both candidates through one harness without a
               second loader plumbing.)
  - candidate: ibm-granite/granite-embedding-97m-multilingual-r2 (the Granite R2
               from the 2026-05-17 digest insight)

Quality is the comparison we care about for the Phase-1 stop gate.
Production-faithful GGUF latency / VRAM coexistence is measured separately;
the plan's stop gate is "Granite loses badly on ULC" (no path to any recall
improvement) regardless of latency.

Corpus: ULC's own ``engine/`` directory chunked by ~30-line windows.
Queries: 20 hand-crafted natural-language questions about ULC features, each
with a ground-truth target file. A query passes top-k iff the top-ranked
chunk lives in (or near) the ground-truth file.

Metrics reported per embedder:
  - recall@1 / @3 / @5 / @10  (did the right file show up?)
  - MRR                       (how high was it?)
  - idx_latency_ms_per_doc    (corpus embed speed)
  - query_latency_ms          (single-query embed speed)
  - peak_alloc_mb             (best-effort, PyTorch's process tracker)

Usage:
  cd D:\\LLCWork\\ultralight-coder
  python benchmark_granite_phase1.py

The harness saves results to ``bench_granite_phase1_<timestamp>.json`` for
follow-up analysis and rolls up to a printed comparison table.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ── Models under test ────────────────────────────────────────────────

BASELINE_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
CANDIDATE_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"

# Models in scope per the plan. Granite-311M may be added if 97M is mid.
MODELS = [BASELINE_MODEL, CANDIDATE_MODEL]

# Nomic v2-moe is a contrastive model — different prefix for queries vs passages.
NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_PASSAGE_PREFIX = "search_document: "
GRANITE_QUERY_PREFIX = ""
GRANITE_PASSAGE_PREFIX = ""


@dataclass
class CodeChunk:
    file_path: str  # relative to ULC root
    start_line: int
    end_line: int
    content: str


@dataclass
class GroundTruthQuery:
    query: str
    target_files: list[str]  # match if any chunk in any of these files is in top-k
    description: str = ""


@dataclass
class EmbedderResult:
    name: str
    n_corpus: int
    n_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    idx_total_s: float
    idx_ms_per_doc: float
    query_ms_avg: float
    peak_alloc_mb: float
    per_query: list[dict] = field(default_factory=list)


# ── Corpus loader ────────────────────────────────────────────────────

ULC_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ULC_ROOT / "engine"

CHUNK_LINES = 30
CHUNK_STRIDE = 20  # overlap so a target spanning a boundary still gets seen


def _iter_chunks(file_path: Path) -> Iterable[CodeChunk]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return
    lines = text.splitlines()
    rel = file_path.relative_to(ULC_ROOT).as_posix()
    if not lines:
        return
    i = 0
    while i < len(lines):
        end = min(i + CHUNK_LINES, len(lines))
        chunk = "\n".join(lines[i:end]).strip()
        if chunk:
            yield CodeChunk(
                file_path=rel,
                start_line=i + 1,
                end_line=end,
                content=chunk,
            )
        if end >= len(lines):
            break
        i += CHUNK_STRIDE


def load_corpus() -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for p in sorted(CORPUS_DIR.rglob("*.py")):
        if "__pycache__" in p.parts or p.name.startswith("test_"):
            continue
        for c in _iter_chunks(p):
            chunks.append(c)
    return chunks


# ── Queries with ground-truth target files ───────────────────────────

QUERIES: list[GroundTruthQuery] = [
    GroundTruthQuery(
        "How does the agent run its ReAct tool-call loop?",
        ["engine/agent.py"],
        "Phase 14 ReAct loop in agent.py",
    ),
    GroundTruthQuery(
        "Where is the agent loop's max iterations and wall-time budget set?",
        ["engine/agent.py"],
        "loop budgets",
    ),
    GroundTruthQuery(
        "What format does the agent's tool-call parser expect?",
        ["engine/agent_tools.py"],
        "Hermes tool-call format",
    ),
    GroundTruthQuery(
        "Where are the built-in agent tools like read_file and grep defined?",
        ["engine/agent_builtins.py"],
        "9 builtin agent tools",
    ),
    GroundTruthQuery(
        "How does the agent prompt to confirm risky tools like run_bash?",
        ["engine/agent_builtins.py", "engine/agent.py"],
        "risky-tool confirmation",
    ),
    GroundTruthQuery(
        "Where are cross-session project notes persisted to disk?",
        ["engine/agent_memory.py"],
        "agent_memory notes",
    ),
    GroundTruthQuery(
        "What categories of failures does the harvest pipeline detect?",
        ["engine/failure_flagger.py"],
        "12 failure categories",
    ),
    GroundTruthQuery(
        "How is the project codebase indexed for retrieval at query time?",
        ["engine/project_context.py"],
        "FAISS project indexer",
    ),
    GroundTruthQuery(
        "How is the nomic-embed-text GGUF model wrapped for embedding calls?",
        ["engine/nomic_embedder.py"],
        "NomicGGUFEmbedder",
    ),
    GroundTruthQuery(
        "Where is the prompt assembly with token budgets and XML tags?",
        ["engine/fusion.py"],
        "fusion modes + token budgets",
    ),
    GroundTruthQuery(
        "How does the YAML augmentor system pick examples by query similarity?",
        ["engine/augmentors.py"],
        "augmentor retrieval",
    ),
    GroundTruthQuery(
        "How are auto-generated YAML augmentors built from agent failures?",
        ["engine/yaml_augmentor_builder.py"],
        "harvest pipeline builder",
    ),
    GroundTruthQuery(
        "Where is the architect that decomposes tasks for worker models?",
        ["engine/architect.py"],
        "multi-agent architect",
    ),
    GroundTruthQuery(
        "How are user queries routed to specific modules like code_gen?",
        ["engine/router.py"],
        "module routing",
    ),
    GroundTruthQuery(
        "Where is the GGUF model loaded with llama-cpp-python?",
        ["engine/base_model.py"],
        "GGUF loading",
    ),
    GroundTruthQuery(
        "How does web_search call DuckDuckGo and parse the results?",
        ["engine/web_tools.py"],
        "web_search via DDG",
    ),
    GroundTruthQuery(
        "Where is the dependency graph between augmentor patterns stored?",
        ["engine/pattern_graph.py"],
        "pattern graph",
    ),
    GroundTruthQuery(
        "How are agent failure-success recovery pairs detected for harvesting?",
        ["engine/recovery_detector.py"],
        "recovery detector",
    ),
    GroundTruthQuery(
        "How are auto-built augmentors statically validated before promotion?",
        ["engine/replay_validator.py"],
        "replay validator",
    ),
    GroundTruthQuery(
        "Where are user-issued /learn corrections persisted and retrieved?",
        ["engine/correction_memory.py"],
        "correction memory",
    ),
]


# ── Embedding helpers ────────────────────────────────────────────────


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def _prefixes_for(model_name: str) -> tuple[str, str]:
    """Return (passage_prefix, query_prefix). Contrastive models need different
    prefixes for indexing vs querying."""
    if "nomic" in model_name.lower():
        return NOMIC_PASSAGE_PREFIX, NOMIC_QUERY_PREFIX
    return GRANITE_PASSAGE_PREFIX, GRANITE_QUERY_PREFIX


def _trust_remote_code_for(model_name: str) -> bool:
    """nomic-embed-text-v2-moe ships custom modeling code requiring trust_remote_code=True."""
    return "nomic" in model_name.lower()


def _peak_alloc_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    except Exception:
        pass
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        return 0.0


def _reset_peak_alloc() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ── Evaluation ───────────────────────────────────────────────────────


def evaluate_embedder(
    embedder_name: str,
    corpus: list[CodeChunk],
    queries: list[GroundTruthQuery],
) -> EmbedderResult:
    print(f"\n=== {embedder_name} ===")
    from sentence_transformers import SentenceTransformer

    _reset_peak_alloc()
    print("  loading model...")
    load_kwargs = {}
    if _trust_remote_code_for(embedder_name):
        load_kwargs["trust_remote_code"] = True
    t0 = time.time()
    model = SentenceTransformer(embedder_name, **load_kwargs)
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.1f}s; embedding dim = {model.get_sentence_embedding_dimension()}")

    pass_prefix, query_prefix = _prefixes_for(embedder_name)
    payload = [pass_prefix + c.content for c in corpus]
    print(f"  embedding {len(payload)} chunks (passage prefix={pass_prefix!r})...")
    t0 = time.time()
    corpus_mat = model.encode(
        payload, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False, batch_size=16,
    )
    idx_total = time.time() - t0
    corpus_mat = _l2_normalize(np.asarray(corpus_mat, dtype=np.float32))
    print(f"  indexed in {idx_total:.1f}s ({idx_total/len(payload)*1000:.1f} ms/chunk)")

    print(f"  scoring {len(queries)} queries (query prefix={query_prefix!r})...")
    per_query_records: list[dict] = []
    r1 = r3 = r5 = r10 = 0
    mrr_sum = 0.0
    query_total = 0.0

    for q in queries:
        text = query_prefix + q.query
        t0 = time.time()
        qv = model.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                          show_progress_bar=False)[0]
        query_total += (time.time() - t0)
        qv = _l2_normalize(np.asarray(qv, dtype=np.float32))
        sims = corpus_mat @ qv
        order = np.argsort(-sims)
        target_set = set(q.target_files)

        first_hit_rank = -1
        top_results = []
        for rank_pos, idx in enumerate(order[:10], start=1):
            chunk = corpus[int(idx)]
            top_results.append({
                "rank": rank_pos,
                "file": chunk.file_path,
                "lines": f"{chunk.start_line}-{chunk.end_line}",
                "score": float(sims[int(idx)]),
            })
            if chunk.file_path in target_set and first_hit_rank == -1:
                first_hit_rank = rank_pos

        if first_hit_rank == 1:
            r1 += 1
        if 1 <= first_hit_rank <= 3:
            r3 += 1
        if 1 <= first_hit_rank <= 5:
            r5 += 1
        if 1 <= first_hit_rank <= 10:
            r10 += 1
        if first_hit_rank > 0:
            mrr_sum += 1.0 / first_hit_rank

        per_query_records.append({
            "query": q.query,
            "target_files": q.target_files,
            "first_hit_rank": first_hit_rank,
            "top_3": top_results[:3],
        })

    n = len(queries)
    result = EmbedderResult(
        name=embedder_name,
        n_corpus=len(corpus),
        n_queries=n,
        recall_at_1=r1 / n,
        recall_at_3=r3 / n,
        recall_at_5=r5 / n,
        recall_at_10=r10 / n,
        mrr=mrr_sum / n,
        idx_total_s=idx_total,
        idx_ms_per_doc=idx_total / len(payload) * 1000,
        query_ms_avg=query_total / n * 1000,
        peak_alloc_mb=_peak_alloc_mb(),
        per_query=per_query_records,
    )

    print(f"  R@1={result.recall_at_1:.3f}  R@3={result.recall_at_3:.3f}  R@5={result.recall_at_5:.3f}  R@10={result.recall_at_10:.3f}  MRR={result.mrr:.3f}")
    print(f"  idx {result.idx_ms_per_doc:.1f} ms/doc, query {result.query_ms_avg:.1f} ms/q, peak alloc {result.peak_alloc_mb:.0f} MB")
    return result


# ── Reporting ────────────────────────────────────────────────────────


def _stop_gate(baseline: EmbedderResult, candidate: EmbedderResult) -> tuple[bool, str]:
    """Per the plan: 'if Granite loses badly on ulcagent (no path to recall
    improvement at any k), STOP the whole arc — ALM wiring won't change the
    verdict.' Returns (should_continue, reason)."""
    if candidate.recall_at_10 + 1e-6 < baseline.recall_at_10:
        return False, (
            f"Candidate R@10 {candidate.recall_at_10:.3f} < baseline R@10 {baseline.recall_at_10:.3f}. "
            "Even with k=10, Granite trails — no path to recall improvement."
        )
    if (
        candidate.recall_at_1 < baseline.recall_at_1
        and candidate.recall_at_5 < baseline.recall_at_5
        and candidate.recall_at_10 <= baseline.recall_at_10
    ):
        return False, (
            "Candidate loses at every k. No retrieval lift at any rank — gate fails."
        )
    if candidate.mrr + 0.05 < baseline.mrr and candidate.recall_at_5 + 0.02 < baseline.recall_at_5:
        return False, (
            f"Candidate MRR drops {baseline.mrr - candidate.mrr:.3f} AND R@5 drops "
            f"{baseline.recall_at_5 - candidate.recall_at_5:.3f}. Substantive regression."
        )
    return True, "Candidate at least competitive; Phase 1 gate passes."


def print_comparison(results: list[EmbedderResult]) -> None:
    print("\n" + "=" * 92)
    print(f"{'embedder':<55} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7}")
    print("-" * 92)
    for r in results:
        short = r.name.split("/")[-1]
        print(f"{short:<55} {r.recall_at_1:>7.3f} {r.recall_at_3:>7.3f} {r.recall_at_5:>7.3f} {r.recall_at_10:>7.3f} {r.mrr:>7.3f}")
    print("=" * 92)

    if len(results) >= 2:
        b, c = results[0], results[1]
        print(f"\nDELTA candidate vs baseline ({c.name.split('/')[-1]} - {b.name.split('/')[-1]}):")
        print(f"  R@1  {c.recall_at_1 - b.recall_at_1:+.3f}   "
              f"R@3  {c.recall_at_3 - b.recall_at_3:+.3f}   "
              f"R@5  {c.recall_at_5 - b.recall_at_5:+.3f}   "
              f"R@10 {c.recall_at_10 - b.recall_at_10:+.3f}   "
              f"MRR  {c.mrr - b.mrr:+.3f}")
        cont, reason = _stop_gate(b, c)
        verdict = "PROCEED to Phase 2" if cont else "STOP — gate failed"
        print(f"\nGATE DECISION: {verdict}")
        print(f"  reason: {reason}")


def save_json(results: list[EmbedderResult], out: Path) -> None:
    payload = {
        "phase": 1,
        "corpus_root": str(CORPUS_DIR),
        "chunk_lines": CHUNK_LINES,
        "chunk_stride": CHUNK_STRIDE,
        "results": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


def main() -> None:
    print("Loading corpus...")
    corpus = load_corpus()
    print(f"  {len(corpus)} chunks from {CORPUS_DIR}")
    print(f"  {len(QUERIES)} queries with ground-truth target files")

    results: list[EmbedderResult] = []
    for model_name in MODELS:
        try:
            results.append(evaluate_embedder(model_name, corpus, QUERIES))
        except Exception as exc:
            import traceback
            print(f"\n  FAILED for {model_name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print_comparison(results)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ULC_ROOT / f"bench_granite_phase1_{ts}.json"
    save_json(results, out)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
