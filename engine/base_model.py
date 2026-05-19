"""
Base Model — The "Tiny Brain"

Loads and manages the core GGUF model via llama-cpp-python.
Supports dynamic LoRA adapter injection, generation with
configurable parameters, and graceful resource management.

This is the workhorse: it handles raw token generation.
Everything else (routing, memory, fusion) builds on top.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BaseModel:
    """
    Wraps llama-cpp-python to provide:
    - Model loading with quantization/GPU layer control
    - LoRA adapter hot-loading and unloading
    - Text generation with streaming support
    - Token counting for prompt budget management
    - Optional native speculative decoding (prompt_lookup or draft_model)
    """

    def __init__(self, config, speculative_config=None):
        """
        Args:
            config: BaseModelConfig dataclass from engine.config
            speculative_config: Optional NativeSpeculativeConfig.
                When enabled, passes a draft_model to llama-cpp-python's
                Llama constructor for 2-3x throughput. Safe to leave None.
        """
        self.config = config
        self.speculative_config = speculative_config
        self.model = None
        self._loaded_lora: Optional[str] = None
        self._speculative_active: bool = False

    def load(self):
        """Load the base GGUF model into memory."""
        model_path = Path(self.config.path)

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Download a GGUF model and update config.yaml.\n"
                f"Recommended starters:\n"
                f"  - TinyLlama 1.1B: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF\n"
                f"  - Qwen2 1.5B: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF\n"
                f"  - Phi-2 2.7B: https://huggingface.co/TheBloke/phi-2-GGUF"
            )

        logger.info(f"Loading model: {model_path}")
        logger.info(f"  Context: {self.config.context_length} tokens")
        logger.info(f"  GPU layers: {self.config.gpu_layers}")
        logger.info(f"  Threads: {self.config.threads}")

        try:
            from llama_cpp import Llama

            llama_kwargs = dict(
                model_path=str(model_path),
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.gpu_layers,
                n_threads=self.config.threads,
                n_batch=self.config.batch_size,
                verbose=logger.isEnabledFor(logging.DEBUG),
            )

            kv_kwargs = self._kv_cache_kwargs()
            if kv_kwargs:
                llama_kwargs.update(kv_kwargs)
                logger.info(
                    f"  KV cache: k={getattr(self.config, 'cache_type_k', None) or 'f16'}, "
                    f"v={getattr(self.config, 'cache_type_v', None) or 'f16'}"
                )

            draft = self._maybe_build_draft_model()
            if draft is not None:
                llama_kwargs["draft_model"] = draft

            try:
                self.model = Llama(**llama_kwargs)
                self._speculative_active = draft is not None
            except TypeError as e:
                msg = str(e)
                if draft is not None and "draft_model" in msg:
                    logger.warning(
                        "llama-cpp-python in this environment does not accept "
                        "draft_model kwarg. Loading without native speculative decoding."
                    )
                    llama_kwargs.pop("draft_model", None)
                    self.model = Llama(**llama_kwargs)
                    self._speculative_active = False
                elif kv_kwargs and ("type_k" in msg or "type_v" in msg):
                    logger.warning(
                        "llama-cpp-python in this environment does not accept "
                        "type_k/type_v kwargs. Loading with default F16 KV cache."
                    )
                    for k in ("type_k", "type_v"):
                        llama_kwargs.pop(k, None)
                    self.model = Llama(**llama_kwargs)
                    self._speculative_active = draft is not None
                else:
                    raise

            # KV cache reuse: NOT auto-enabled. On short prompts (<2000 tokens)
            # the cache save/load overhead EXCEEDS eval savings — measured 15%
            # slower on rename_function (100s vs 87s baseline). Only beneficial
            # on very long multi-turn sessions. Callers can enable manually via:
            #   from llama_cpp import LlamaRAMCache
            #   bm.model.set_cache(LlamaRAMCache(capacity_bytes=2*1024**3))

            if self._speculative_active:
                logger.info("Model loaded successfully (native speculative decoding active)")
            else:
                logger.info("Model loaded successfully")

        except ImportError:
            logger.error(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_lora(self, lora_path: str):
        """
        Load a LoRA adapter on top of the base model.

        Args:
            lora_path: Path to the LoRA adapter file (.gguf)
        """
        if self.model is None:
            raise RuntimeError("Base model not loaded. Call load() first.")

        lora_file = Path(lora_path)
        if not lora_file.exists():
            logger.warning(f"LoRA file not found: {lora_path}")
            return False

        try:
            # llama-cpp-python supports LoRA via model recreation
            # or via the lora_path parameter. For hot-swapping,
            # we use the adapter loading API.
            from llama_cpp import Llama

            logger.info(f"Loading LoRA adapter: {lora_file.name}")

            # For v1, we reload the model with the LoRA applied.
            # Future: use llama_lora_adapter_set for true hot-swap.
            self.model = Llama(
                model_path=self.config.path,
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.gpu_layers,
                n_threads=self.config.threads,
                n_batch=self.config.batch_size,
                lora_path=str(lora_file),
                verbose=logger.isEnabledFor(logging.DEBUG),
            )
            self._loaded_lora = str(lora_file)
            logger.info(f"LoRA adapter loaded: {lora_file.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            return False

    def unload_lora(self):
        """Remove the current LoRA adapter, reverting to base model."""
        if self._loaded_lora is None:
            return

        logger.info("Unloading LoRA adapter, reverting to base model")
        try:
            from llama_cpp import Llama

            self.model = Llama(
                model_path=self.config.path,
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.gpu_layers,
                n_threads=self.config.threads,
                n_batch=self.config.batch_size,
                verbose=logger.isEnabledFor(logging.DEBUG),
            )
            self._loaded_lora = None
        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        stream: bool = False,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        grammar=None,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The full assembled prompt
            max_tokens: Override max generation length
            temperature: Override sampling temperature
            stop: Stop sequences
            stream: If True, returns a generator that yields tokens
            top_p: Nucleus sampling threshold (0.0-1.0)
            top_k: Top-k sampling (0 = disabled)
            repeat_penalty: Repetition penalty (1.0 = none, >1.0 = penalize)
            grammar: GBNF grammar object for constrained generation

        Returns:
            Generated text string (or generator if stream=True)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_tok = max_tokens or self.config.max_tokens
        temp = temperature or self.config.temperature
        stop_seqs = stop or ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]

        logger.debug(f"Generating (max_tokens={max_tok}, temp={temp}, stream={stream})")

        if stream:
            return self._generate_stream(prompt, max_tok, temp, stop_seqs)

        # Build kwargs for llama-cpp
        gen_kwargs = {
            "max_tokens": max_tok,
            "temperature": temp,
            "stop": stop_seqs,
            "echo": False,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if repeat_penalty is not None:
            gen_kwargs["repeat_penalty"] = repeat_penalty
        if grammar is not None:
            gen_kwargs["grammar"] = grammar

        try:
            output = self.model(prompt, **gen_kwargs)

            response = output["choices"][0]["text"].strip()
            usage = output.get("usage", {})
            logger.debug(
                f"Generated {usage.get('completion_tokens', '?')} tokens "
                f"(prompt: {usage.get('prompt_tokens', '?')})"
            )
            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Generation Error: {e}]"

    def _generate_stream(self, prompt: str, max_tokens: int,
                         temperature: float, stop: list[str]):
        """
        Phase 5: Streaming generation — yields tokens one at a time.
        Returns a generator that yields (token_text, is_final) tuples.
        """
        try:
            stream = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                echo=False,
                stream=True,
            )

            for chunk in stream:
                token = chunk["choices"][0]["text"]
                finish = chunk["choices"][0].get("finish_reason")
                yield token, finish is not None

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"[Generation Error: {e}]", True

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a string.
        Useful for prompt budget management.
        """
        if self.model is None:
            # Rough estimate: ~4 chars per token
            return len(text) // 4

        try:
            tokens = self.model.tokenize(text.encode("utf-8"))
            return len(tokens)
        except Exception:
            return len(text) // 4

    # Map llama.cpp GGML quant type names to the integer constants
    # llama-cpp-python's Llama(type_k=..., type_v=...) accepts. Values pulled
    # from ggml.h GGML_TYPE_* enum (stable since 2024).
    _KV_TYPE_CODES = {
        "f32": 0,
        "f16": 1,
        "q4_0": 2,
        "q4_1": 3,
        "q5_0": 6,
        "q5_1": 7,
        "q8_0": 8,
    }

    def _kv_cache_kwargs(self) -> dict:
        """Return llama-cpp-python kwargs for KV cache quant, or {} if defaults."""
        k = getattr(self.config, "cache_type_k", None)
        v = getattr(self.config, "cache_type_v", None)
        out = {}
        for name, val in (("type_k", k), ("type_v", v)):
            if val is None:
                continue
            key = str(val).lower().strip()
            if key not in self._KV_TYPE_CODES:
                logger.warning(
                    f"Unknown KV cache type {val!r}; ignoring (allowed: "
                    f"{sorted(self._KV_TYPE_CODES)})"
                )
                continue
            out[name] = self._KV_TYPE_CODES[key]
        return out

    def _maybe_build_draft_model(self):
        """Build a native speculative draft model if configured. Never raises."""
        cfg = self.speculative_config
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        try:
            from engine.native_speculative import build_draft_model, describe
            draft = build_draft_model(cfg)
            if draft is not None:
                logger.info(describe(cfg))
            return draft
        except Exception as e:
            logger.warning(f"Native speculative draft build failed: {e}")
            return None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def speculative_active(self) -> bool:
        return self._speculative_active

    @property
    def active_lora(self) -> Optional[str]:
        return self._loaded_lora

    def unload(self):
        """Free the model from memory."""
        if self.model is not None:
            logger.info("Unloading base model from memory")
            del self.model
            self.model = None
            self._loaded_lora = None

    def __repr__(self):
        status = "loaded" if self.is_loaded else "not loaded"
        lora = f", lora={Path(self._loaded_lora).name}" if self._loaded_lora else ""
        return f"BaseModel({status}{lora})"
