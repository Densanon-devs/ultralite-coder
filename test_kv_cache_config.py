"""Smoke tests for KV cache quantization config wiring (Task 1, 2026-05-19).

These do NOT load a real model — they exercise the BaseModelConfig parsing,
the _kv_cache_kwargs mapping, and the TypeError fallback so we can ship the
config gate without needing GPU time. The "did it actually move the JSON
ceiling" measurement is a separate user-supervised benchmark run.
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from engine.base_model import BaseModel
from engine.config import BaseModelConfig


class TestKVCacheKwargs(unittest.TestCase):
    def _bm(self, **kw):
        cfg = BaseModelConfig(**kw)
        return BaseModel(cfg)

    def test_defaults_no_kv_kwargs(self):
        bm = self._bm()
        self.assertEqual(bm._kv_cache_kwargs(), {})

    def test_q8_both(self):
        bm = self._bm(cache_type_k="q8_0", cache_type_v="q8_0")
        self.assertEqual(bm._kv_cache_kwargs(), {"type_k": 8, "type_v": 8})

    def test_mixed_q8k_f16v(self):
        bm = self._bm(cache_type_k="q8_0", cache_type_v="f16")
        self.assertEqual(bm._kv_cache_kwargs(), {"type_k": 8, "type_v": 1})

    def test_case_insensitive(self):
        bm = self._bm(cache_type_k="Q8_0", cache_type_v="  Q4_0 ")
        self.assertEqual(bm._kv_cache_kwargs(), {"type_k": 8, "type_v": 2})

    def test_unknown_type_warned_and_ignored(self):
        bm = self._bm(cache_type_k="q3_xx", cache_type_v="q8_0")
        with self.assertLogs("engine.base_model", level=logging.WARNING) as ctx:
            kw = bm._kv_cache_kwargs()
        self.assertEqual(kw, {"type_v": 8})
        self.assertTrue(any("Unknown KV cache type" in m for m in ctx.output))

    def test_only_k_set(self):
        bm = self._bm(cache_type_k="q8_0")
        self.assertEqual(bm._kv_cache_kwargs(), {"type_k": 8})

    def test_only_v_set(self):
        bm = self._bm(cache_type_v="q5_1")
        self.assertEqual(bm._kv_cache_kwargs(), {"type_v": 7})


class TestKVCacheLoaderFallback(unittest.TestCase):
    """Verify graceful TypeError fallback when llama-cpp-python doesn't
    accept type_k/type_v (older builds)."""

    def test_fallback_when_kwargs_rejected(self):
        cfg = BaseModelConfig(
            path="dummy.gguf", cache_type_k="q8_0", cache_type_v="q8_0"
        )
        bm = BaseModel(cfg)

        # Pretend the GGUF exists.
        with patch("engine.base_model.Path") as mock_path, \
             patch("llama_cpp.Llama") as mock_llama:
            mock_path.return_value.exists.return_value = True
            # First call: simulate llama-cpp-python rejecting type_k.
            # Second call (fallback): succeed with a dummy model.
            sentinel_model = object()
            mock_llama.side_effect = [
                TypeError("unexpected keyword argument 'type_k'"),
                sentinel_model,
            ]
            bm.load()

        self.assertIs(bm.model, sentinel_model)
        # Second call must omit type_k/type_v.
        second_kwargs = mock_llama.call_args_list[1].kwargs
        self.assertNotIn("type_k", second_kwargs)
        self.assertNotIn("type_v", second_kwargs)

    def test_first_load_passes_kwargs_when_supported(self):
        cfg = BaseModelConfig(
            path="dummy.gguf", cache_type_k="q8_0", cache_type_v="q5_0"
        )
        bm = BaseModel(cfg)

        with patch("engine.base_model.Path") as mock_path, \
             patch("llama_cpp.Llama") as mock_llama:
            mock_path.return_value.exists.return_value = True
            mock_llama.return_value = object()
            bm.load()

        kwargs = mock_llama.call_args.kwargs
        self.assertEqual(kwargs.get("type_k"), 8)
        self.assertEqual(kwargs.get("type_v"), 6)


class TestConfigYAMLParsing(unittest.TestCase):
    def test_yaml_with_kv_fields_round_trips(self):
        import tempfile
        from pathlib import Path
        from engine.config import Config

        yaml_text = """
base_model:
  path: models/foo.gguf
  context_length: 8192
  gpu_layers: 99
  threads: 4
  cache_type_k: q8_0
  cache_type_v: q8_0
"""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.yaml"
            p.write_text(yaml_text)
            cfg = Config(str(p))

        self.assertEqual(cfg.base_model.cache_type_k, "q8_0")
        self.assertEqual(cfg.base_model.cache_type_v, "q8_0")

    def test_yaml_without_kv_fields_keeps_defaults_none(self):
        import tempfile
        from pathlib import Path
        from engine.config import Config

        yaml_text = """
base_model:
  path: models/foo.gguf
"""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.yaml"
            p.write_text(yaml_text)
            cfg = Config(str(p))

        self.assertIsNone(cfg.base_model.cache_type_k)
        self.assertIsNone(cfg.base_model.cache_type_v)


if __name__ == "__main__":
    unittest.main()
