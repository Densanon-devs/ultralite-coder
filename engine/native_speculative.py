"""
Native Speculative Decoding — llama-cpp-python draft-model integration.

Sibling to engine/speculative.py, which implements application-level tricks
(parallel generation, n-gram cache, early exit). This module taps llama-cpp's
built-in speculative decoding support via the `draft_model` kwarg on Llama.

### What actually works in llama-cpp-python (verified 2026-04-12, v0.3.9)

Only ONE mode is actually supported by llama-cpp-python's high-level API:

    prompt_lookup:
        Uses LlamaPromptLookupDecoding — no second model required. Scans
        repeated n-grams in the current context and predicts multiple tokens
        at once. Works with any base model. Real speedup on code generation
        whenever the prompt contains repetition the target is likely to
        reproduce (variable names reused in tests, boilerplate, signatures).
        Low risk, bit-identical output, ships today.

### What does NOT work (and why)

A second-Llama draft-model path (e.g., Qwen Coder 0.5B drafting 14B) is
BROKEN from this module in llama-cpp-python 0.3.9. The `draft_model` kwarg
expects an object implementing the `LlamaDraftModel` protocol, and the
library ships exactly one implementation: `LlamaPromptLookupDecoding`. A
raw `Llama` instance does NOT satisfy the protocol.

A custom adapter wrapping a second `Llama` via reset/eval/sample is
theoretically possible but would need KV-cache state tracking and rollback
to be faster than baseline. Without state tracking, every call to the
adapter triggers a full forward pass over the entire context on the draft
model, which kills any speedup. We do not ship this path.

The community-reported 2.5× speedups (ggml-org/llama.cpp discussion #10466)
come from the llama.cpp C++ CLI tool `llama-speculative-simple`, not from
llama-cpp-python's Python API. A subprocess-based benchmark that shells out
to the CLI is a legitimate future direction — see experiment_backlog.md.

### Why draft_model_path config still exists

Kept as a forward-compatible config knob. When/if llama-cpp-python adds a
native second-Llama draft adapter (or we build a subprocess path), the
config key is already wired up. Attempting to enable it today raises a
clear NotImplementedError instead of crashing at generation time.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class NativeSpeculativeConfig:
    """Configuration for native (llama-cpp-python) speculative decoding."""

    enabled: bool = False
    mode: str = "prompt_lookup"  # "prompt_lookup" or "draft_model"

    # prompt_lookup mode
    num_pred_tokens: int = 10
    max_ngram_size: int = 2

    # draft_model mode
    draft_model_path: str = ""
    draft_gpu_layers: int = 99
    draft_context_length: int = 4096


def _probe_prompt_lookup() -> Optional[Any]:
    """Return the LlamaPromptLookupDecoding class if importable, else None."""
    try:
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
        return LlamaPromptLookupDecoding
    except ImportError:
        logger.debug("LlamaPromptLookupDecoding not available in this llama-cpp-python build")
        return None


def _probe_llama() -> Optional[Any]:
    """Return the Llama class if importable, else None."""
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError:
        logger.debug("llama-cpp-python not installed")
        return None


def build_draft_model(cfg: NativeSpeculativeConfig) -> Optional[Any]:
    """
    Construct a draft model object suitable for passing as Llama(draft_model=...).

    Returns None if speculative decoding is disabled, unavailable, or
    misconfigured. Callers should treat None as "run without spec decoding".

    Does NOT raise on failure — this is a perf feature, not a correctness one,
    and we always want the base pipeline to keep working.
    """
    if not cfg.enabled:
        return None

    if cfg.mode == "prompt_lookup":
        return _build_prompt_lookup(cfg)

    if cfg.mode == "draft_model":
        return _build_draft_llama(cfg)

    logger.warning(f"Unknown speculative mode '{cfg.mode}' — disabling")
    return None


def _build_prompt_lookup(cfg: NativeSpeculativeConfig) -> Optional[Any]:
    Cls = _probe_prompt_lookup()
    if Cls is None:
        logger.warning(
            "prompt_lookup speculative decoding requested but "
            "llama_cpp.llama_speculative.LlamaPromptLookupDecoding is unavailable. "
            "Upgrade llama-cpp-python or disable speculative.enabled."
        )
        return None

    try:
        draft = Cls(
            num_pred_tokens=cfg.num_pred_tokens,
            max_ngram_size=cfg.max_ngram_size,
        )
        logger.info(
            f"Native speculative decoding: prompt_lookup "
            f"(num_pred_tokens={cfg.num_pred_tokens}, max_ngram_size={cfg.max_ngram_size})"
        )
        return draft
    except TypeError:
        try:
            draft = Cls(num_pred_tokens=cfg.num_pred_tokens)
            logger.info(
                f"Native speculative decoding: prompt_lookup "
                f"(num_pred_tokens={cfg.num_pred_tokens}, max_ngram_size=<default>)"
            )
            return draft
        except Exception as e:
            logger.warning(f"Failed to build LlamaPromptLookupDecoding: {e}")
            return None
    except Exception as e:
        logger.warning(f"Failed to build LlamaPromptLookupDecoding: {e}")
        return None


def _build_draft_llama(cfg: NativeSpeculativeConfig) -> Optional[Any]:
    """
    Second-Llama draft-model path is intentionally not yet implemented.

    UPDATE 2026-05-19 (Task B3 research): the original 0.3.9-era assessment
    that this path is structurally blocked is partially incorrect. The
    `LlamaDraftModel` ABC requires only:

        class LlamaDraftModel(abc.ABC):
            @abc.abstractmethod
            def __call__(
                self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
            ) -> npt.NDArray[np.intc]: ...

    A custom subclass wrapping a second `Llama` instance IS implementable.
    The hard part is the KV-cache state machine:

      1. On first call: `draft.reset()`, `draft.eval(input_ids)`, sample N.
      2. On subsequent calls, the agent loop's `input_ids` shares a long
         prefix with the previous call (system + tools + earlier transcript).
         To get the published ~2x speedup we MUST detect the shared prefix
         and only `eval()` the suffix, leaving the draft's KV cache intact.
      3. Wrong prefix-detection logic = silent KV corruption = bad drafts.
         Llama-cpp's verification step rejects bad drafts (correctness
         preserved) but wall-clock speedup collapses to zero or negative.
      4. There is no Python API to introspect the draft's internal KV-cache
         token-id sequence; we must track it ourselves.

    The implementation isn't shipped because:
      - Unit tests can verify the state-machine logic against a mock Llama,
        but cannot verify the real KV state stays consistent under llama.cpp's
        C++ internals.
      - Without GPU-time validation, we'd ship code that's unit-test-correct
        but might be wall-clock-neutral or worse.

    DESIGN SKETCH for the operator to implement when they can bench:

        class PairedGGUFDraftModel(LlamaDraftModel):
            def __init__(self, draft_path, n_ctx, gpu_layers):
                self.llm = Llama(model_path=draft_path, n_ctx=n_ctx,
                                 n_gpu_layers=gpu_layers, ...)
                self._cached_ids: list[int] = []

            def __call__(self, input_ids, /, **kw) -> NDArray[intc]:
                ids = input_ids.tolist()
                # Find longest matching prefix with the cached ids.
                shared = self._matching_prefix_len(ids)
                if shared < len(self._cached_ids):
                    # Divergence — we evaluated past where the host's
                    # context now is. Must reset.
                    self.llm.reset()
                    self.llm.eval(ids)
                    self._cached_ids = list(ids)
                elif shared < len(ids):
                    # Append-only suffix path — cheap.
                    self.llm.eval(ids[shared:])
                    self._cached_ids = list(ids)
                # else: no new tokens — sample anyway
                drafted: list[int] = []
                NUM_DRAFT = 10
                for _ in range(NUM_DRAFT):
                    tok = self.llm.sample()
                    drafted.append(tok)
                    self._cached_ids.append(tok)
                    self.llm.eval([tok])
                return np.asarray(drafted, dtype=np.intc)

    Bench protocol when implementing:
      1. Implement against a small smoke task (Qwen 0.5B drafting itself).
      2. Verify token sequences match a no-draft baseline exactly (bad
         drafts must produce identical output, just slower).
      3. Switch to 0.5B drafting 14B Q4_K_M. Run benchmark_agentic.py
         --repeat 5 --share-model and compare to baseline tok/s.
      4. Target: >= 1.6x speedup. Below that, the prefix-detection logic
         likely has a bug — debug before shipping.

    The community-reported 2.5x speedups (ggml-org/llama.cpp discussion
    #10466) come from `llama-speculative-simple`, which has the C++
    batched-verify path. The Python path will be somewhat slower at
    identical draft acceptance rates because of the per-token round-trip
    through the Python interpreter.
    """
    logger.warning(
        "speculative.mode=draft_model is not yet implemented. See the "
        "design sketch in engine/native_speculative.py:_build_draft_llama "
        "for the implementation path. Falling back to no speculative decoding."
    )
    return None


def describe(cfg: NativeSpeculativeConfig) -> str:
    """Short human-readable description for startup logging."""
    if not cfg.enabled:
        return "native speculative: disabled"
    if cfg.mode == "prompt_lookup":
        return f"native speculative: prompt_lookup (n={cfg.num_pred_tokens})"
    if cfg.mode == "draft_model":
        name = Path(cfg.draft_model_path).name if cfg.draft_model_path else "<unset>"
        return f"native speculative: draft_model ({name})"
    return f"native speculative: unknown mode '{cfg.mode}'"
