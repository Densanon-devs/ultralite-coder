#!/usr/bin/env python3
"""
Ultralite Code Assistant — CLI Entry Point

A local coding assistant powered by tiny code-specialized models.
Branched from Plug-in Intelligence Engine.

Usage:
    python main.py                    # Interactive mode
    python main.py --config my.yaml   # Custom config
    python main.py --dry-run          # Test routing without model
    python main.py --list-modules     # Show available modules
    python main.py --explain "prompt" # Explain routing decision
    python main.py --no-pipeline      # Disable async pipeline
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

_CORE_ROOT = PROJECT_ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from densanon.core.config import Config
from densanon.core.model_loader import BaseModel
from densanon.core.router import Router
from densanon.core.modules import ModuleManager
from densanon.core.memory import MemorySystem
from densanon.core.fusion import FusionLayer
from densanon.core.pipeline import Pipeline
from densanon.core.cache import KVCacheManager
from densanon.core.adapters import MicroAdapterEngine
from densanon.core.tools import ToolRegistry
from engine.augmentors import AugmentorRouter  # KEEP - unique to ultralight-coder
from engine.agent import Agent, AgentEvent
from engine.agent_builtins import build_default_registry
from engine.agent_memory import AgentMemory
from engine.agent_tools import ToolCall
from densanon.core.cache import SpeculativeEngine
from densanon.core.project_context import ProjectIndex

logger = logging.getLogger("UCA")


def _load_speculative_config(config_path: str):
    """
    Parse the `speculative:` section of config_path directly into a
    NativeSpeculativeConfig, bypassing densanon-core's Config class (which
    doesn't know about this section). Returns None if the section is
    missing, disabled, or the engine helper is unavailable.
    """
    try:
        import yaml
        from engine.native_speculative import NativeSpeculativeConfig
    except ImportError as exc:
        logger.debug(f"Speculative config unavailable: {exc}")
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except OSError as exc:
        logger.debug(f"Could not read {config_path} for speculative config: {exc}")
        return None

    spec = raw.get("speculative") or {}
    if not spec.get("enabled", False):
        return None

    return NativeSpeculativeConfig(
        enabled=bool(spec.get("enabled", False)),
        mode=str(spec.get("mode", "prompt_lookup")),
        num_pred_tokens=int(spec.get("num_pred_tokens", 10)),
        max_ngram_size=int(spec.get("max_ngram_size", 2)),
        draft_model_path=str(spec.get("draft_model_path", "")),
        draft_gpu_layers=int(spec.get("draft_gpu_layers", 99)),
        draft_context_length=int(spec.get("draft_context_length", 4096)),
    )


class UltraliteCodeAssistant:
    """
    Main orchestrator. Wires all components together
    and runs the inference loop for code assistance.
    """

    def __init__(self, config_path: str = "config.yaml", dry_run: bool = False, use_pipeline: bool = True):
        self.config = Config(config_path)
        self._apply_tuned_config()
        self.config.setup_logging()
        self.dry_run = dry_run

        logger.info(f"=== {self.config.system.name} v{self.config.system.version} ===")

        # Core components — speculative_config is only passed if the installed
        # BaseModel accepts it (the local engine/base_model.py does, the
        # densanon-core one does not). Phase 13 added it opt-in via config.
        spec_cfg = getattr(self.config, "speculative", None)
        try:
            self.base_model = BaseModel(self.config.base_model, speculative_config=spec_cfg)
        except TypeError:
            self.base_model = BaseModel(self.config.base_model)
        self.router = Router(self.config.router)
        self.modules = ModuleManager(self.config.modules)
        self.memory = MemorySystem(self.config.memory)
        self.fusion = FusionLayer(
            self.config.fusion,
            token_counter=self.base_model.count_tokens,
        )

        # Pipeline
        pipeline_enabled = use_pipeline and self.config.pipeline.enabled
        self.pipeline = Pipeline(
            parallel_workers=self.config.pipeline.parallel_workers,
            enable_queue=self.config.pipeline.enable_generation_queue and not dry_run,
        )

        # KV Cache
        self.kv_cache = KVCacheManager(
            context_length=self.config.base_model.context_length,
            max_generation_tokens=self.config.base_model.max_tokens,
        )
        self.kv_cache.set_token_counter(self.base_model.count_tokens)

        # Micro-adapters
        self.micro_adapters = None
        if self.config.micro_adapters.enabled:
            self.micro_adapters = MicroAdapterEngine(
                storage_dir=self.config.micro_adapters.storage_dir,
                embedding_model=self.config.micro_adapters.embedding_model,
                min_cluster_size=self.config.micro_adapters.min_cluster_size,
                max_adapters=self.config.micro_adapters.max_adapters,
                regenerate_interval=self.config.micro_adapters.regenerate_interval,
            )

        # Tools
        self.tools = ToolRegistry()

        # Augmentor system (code-focused, YAML examples + auto mode)
        self.augmentor_router = AugmentorRouter(yaml_dir="data/augmentor_examples")
        self._augmentors_enabled = self._should_enable_augmentors()

        # Response cache
        self.speculative = SpeculativeEngine(storage_dir="data/cache")

        # Project context indexer — densanon-core's Config does not always
        # expose project_context, so guard the lookup and degrade gracefully.
        project_ctx_cfg = getattr(self.config, "project_context", None)
        if project_ctx_cfg is not None:
            try:
                self.project_index = ProjectIndex(project_ctx_cfg)
            except Exception as exc:
                logger.warning(f"ProjectIndex init failed, disabling: {exc}")
                self.project_index = None
        else:
            self.project_index = None

        # Phase 14 agentic harness — lazy-built on first /agent or --agent use
        self._agent: Agent | None = None
        self._auto_approve_risky: bool = False

        # Performance tracking
        self._perf_history: list[dict] = []

    def _should_enable_augmentors(self) -> bool:
        """Enable augmentors for all supported models.

        Phase 6 proved auto mode works for all model sizes:
        - Sub-1.5GB: rerank1 (single example injection)
        - 1.5GB+: rerank (two example injection)
        """
        model_path = Path(self.config.base_model.path)
        if not model_path.exists():
            return True
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Augmentor system: ON (model={size_mb:.0f}MB, 230 YAML examples)")
        return True

    def _apply_tuned_config(self):
        """Load and apply tuned_config.json if it exists."""
        tuned_path = Path(self.config.config_path).parent / "tuned_config.json"
        if not tuned_path.exists():
            return

        try:
            import json
            with open(tuned_path) as f:
                tuned = json.load(f)

            opt = tuned.get("optimal", {})
            applied = []

            if "threads" in opt:
                self.config.base_model.threads = opt["threads"]
                applied.append(f"threads={opt['threads']}")
            if "temperature" in opt:
                self.config.base_model.temperature = opt["temperature"]
                applied.append(f"temperature={opt['temperature']}")
            if "batch_size" in opt:
                self.config.base_model.batch_size = opt["batch_size"]
                applied.append(f"batch_size={opt['batch_size']}")
            if "gpu_layers" in opt:
                self.config.base_model.gpu_layers = opt["gpu_layers"]
                applied.append(f"gpu_layers={opt['gpu_layers']}")
            if "chat_format" in opt:
                self.config.fusion.chat_format = opt["chat_format"]
                applied.append(f"chat_format={opt['chat_format']}")

            if applied:
                logger.info(f"Applied tuned config: {', '.join(applied)}")

        except Exception as e:
            logger.warning(f"Failed to load tuned config: {e}")

    def initialize(self):
        """Load model, discover modules, start pipeline."""
        available = self.modules.discover()
        logger.info(f"Available modules: {available}")

        if self.router.classifier.needs_training:
            logger.info("Training classifier...")
            stats = self.router.train_classifier()
            logger.info(f"Classifier training result: {stats}")

        if not self.dry_run:
            try:
                self.base_model.load()
            except FileNotFoundError as e:
                print(f"\n{e}\n")
                print("Download a model first: python download_model.py")
                print("Or use --dry-run to test routing.\n")
                sys.exit(1)
            except ImportError:
                print(
                    "\nllama-cpp-python not installed.\n"
                    "Install with: pip install llama-cpp-python\n"
                    "Or use --dry-run to test without a model.\n"
                )
                sys.exit(1)

            def generate_fn(prompt, max_tokens=None, temperature=None, **kwargs):
                return self.base_model.generate(
                    prompt, max_tokens=max_tokens, temperature=temperature, **kwargs,
                )
            self.pipeline.set_generate_fn(generate_fn)

            def stream_fn(prompt, max_tokens=None, temperature=None):
                return self.base_model.generate(
                    prompt, max_tokens=max_tokens, temperature=temperature, stream=True,
                )
            self.pipeline.set_stream_fn(stream_fn)
        else:
            logger.info("DRY RUN mode — no model loaded")

        # Initialize augmentor embeddings + auto mode
        try:
            from densanon.core.embeddings import get_embedder
            embedder = get_embedder()
            if embedder:
                self.augmentor_router.init_embeddings(embedder)
                self.project_index.init_embedder(embedder)
        except Exception as e:
            logger.debug(f"Augmentor embeddings not initialized: {e}")

        if self._augmentors_enabled:
            model_path = Path(self.config.base_model.path)
            model_size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 1000
            self.augmentor_router.use_auto_augmentors(model_size_mb)

        self.pipeline.start()
        logger.info("Ready")

    def process(self, user_input: str) -> str:
        """Full pipeline: route -> augmentor/fusion -> generate -> post-process."""
        result = self._handle_commands(user_input)
        if result is not None:
            return result

        perf = {"start": time.monotonic()}

        # Check response cache
        cached = self.speculative.try_cache(user_input)
        if cached is not None:
            perf["cache_hit"] = True
            perf["total"] = time.monotonic() - perf["start"]
            self.memory.add_interaction("user", user_input)
            self.memory.add_interaction("assistant", cached)
            self._perf_history.append(perf)
            return cached

        # Route
        routing = self.router.route(
            user_prompt=user_input,
            conversation_history=self.memory.short_term.get_turns(),
            available_modules=self.modules.available_modules,
        )
        perf["route"] = time.monotonic() - perf["start"]

        # Augmentor system
        if not self.dry_run and self._augmentors_enabled and routing.selected_modules:
            module_hint = routing.selected_modules[0]

            # Build extra context for augmentor: conversation history + project context
            extra_parts = []
            conv_history = self.memory.short_term.get_context()
            if conv_history:
                extra_parts.append(conv_history)
            if self.project_index.is_indexed:
                project_ctx = self.project_index.get_context(user_input, top_k=3)
                if project_ctx:
                    extra_parts.append(project_ctx)
            extra_ctx = "\n\n".join(extra_parts)

            augmentor_result = self.augmentor_router.process(
                query=user_input,
                model=self.base_model,
                chat_format=self.config.fusion.chat_format,
                module_hint=module_hint,
                extra_context=extra_ctx,
                gen_kwargs={"max_tokens": self.config.base_model.max_tokens,
                            "temperature": self.config.base_model.temperature},
            )
            if augmentor_result is not None:
                perf["augmentor"] = augmentor_result.augmentor_name
                perf["augmentor_attempts"] = augmentor_result.attempts
                response = augmentor_result.response
                perf["generation"] = time.monotonic() - perf["start"] - perf["route"]
                self._post_process(user_input, response, routing, perf)
                return response

        # Micro-adapter selection
        active_adapter = None
        adapter_applied = None
        if self.micro_adapters:
            active_adapter = self.micro_adapters.select_adapter(user_input)
            if active_adapter:
                adapter_applied = self.micro_adapters.apply(
                    active_adapter,
                    base_temperature=self.config.base_model.temperature,
                    base_max_tokens=self.config.base_model.max_tokens,
                )
                perf["adapter"] = active_adapter.name

        # Parallel I/O
        def _load_modules_and_lora():
            active = self.modules.get_multiple(routing.selected_modules)
            lora_modules = [m for m in active if m.manifest.lora_path]
            if lora_modules:
                top_lora = max(lora_modules, key=lambda m: m.manifest.priority)
                if self.base_model.active_lora != top_lora.manifest.lora_path:
                    self.base_model.load_lora(top_lora.manifest.lora_path)
            elif self.base_model.active_lora:
                self.base_model.unload_lora()
            return active

        def _recall_memory():
            return self.memory.recall(user_input)

        def _predictive_preload():
            if self.config.modules.predictive_preload:
                predicted = self.modules.predict_next_modules(
                    routing.selected_modules,
                    top_k=self.config.modules.preload_top_k,
                )
                if predicted:
                    self.modules.preload(predicted)
                    return predicted
            return []

        io_result = self.pipeline.run_parallel_io(
            module_loader=_load_modules_and_lora,
            memory_retriever=_recall_memory,
            preloader=_predictive_preload,
            knowledge_loader=lambda: None,
        )
        active_modules = io_result.active_modules
        memory_context = io_result.memory_context
        perf["parallel_io"] = time.monotonic() - perf["start"] - perf.get("route", 0)

        # Fusion
        if adapter_applied and adapter_applied.get("context_prompt"):
            memory_context["adapter"] = adapter_applied["context_prompt"]

        tool_prompt = self.tools.get_tool_prompt()
        if tool_prompt:
            memory_context["tools"] = tool_prompt

        # Project context — retrieve relevant code from indexed project
        if self.project_index.is_indexed:
            project_ctx = self.project_index.get_context(user_input)
            if project_ctx:
                memory_context["project"] = project_ctx

        prompt = self.fusion.assemble(
            user_input=user_input,
            active_modules=active_modules,
            memory_context=memory_context,
            blend_weights=routing.blend_weights,
        )

        # Generate
        gen_start = time.monotonic()
        if self.dry_run:
            response = self._dry_run_response(routing, active_modules, prompt, perf)
        else:
            gen_kwargs = {}
            if adapter_applied:
                gen_kwargs["temperature"] = adapter_applied["temperature"]
                gen_kwargs["max_tokens"] = adapter_applied["max_tokens"]
            response = self.pipeline.generate(prompt, **gen_kwargs)

        perf["generation"] = time.monotonic() - gen_start
        self._post_process(user_input, response, routing, perf)
        return response

    def process_stream(self, user_input: str):
        """Streaming version of process(). Yields tokens."""
        result = self._handle_commands(user_input)
        if result is not None:
            yield result
            return

        if self.dry_run:
            yield self.process(user_input)
            return

        routing = self.router.route(
            user_prompt=user_input,
            conversation_history=self.memory.short_term.get_turns(),
            available_modules=self.modules.available_modules,
        )

        def _load_modules_and_lora():
            active = self.modules.get_multiple(routing.selected_modules)
            lora_modules = [m for m in active if m.manifest.lora_path]
            if lora_modules:
                top_lora = max(lora_modules, key=lambda m: m.manifest.priority)
                if self.base_model.active_lora != top_lora.manifest.lora_path:
                    self.base_model.load_lora(top_lora.manifest.lora_path)
            elif self.base_model.active_lora:
                self.base_model.unload_lora()
            return active

        io_result = self.pipeline.run_parallel_io(
            module_loader=_load_modules_and_lora,
            memory_retriever=lambda: self.memory.recall(user_input),
            preloader=lambda: None,
            knowledge_loader=lambda: None,
        )

        prompt = self.fusion.assemble(
            user_input=user_input,
            active_modules=io_result.active_modules,
            memory_context=io_result.memory_context,
            blend_weights=routing.blend_weights,
        )

        full_response = []
        for token, is_final in self.pipeline.generate_stream(prompt):
            full_response.append(token)
            yield token

        response = "".join(full_response).strip()
        self.memory.add_interaction("user", user_input)
        self.memory.add_interaction("assistant", response)
        if routing.selected_modules:
            self.router.record_interaction(user_input, routing.selected_modules)
            self.modules.record_usage(routing.selected_modules)

    def _post_process(self, user_input: str, response: str, routing, perf: dict):
        """Post-processing: memory, classifier training, cache."""
        self.memory.add_interaction("user", user_input)
        self.memory.add_interaction("assistant", response)

        module = routing.selected_modules[0] if routing.selected_modules else None
        self.speculative.record(
            user_input, response, module=module, quality_verified=True,
        )

        if routing.selected_modules:
            self.router.record_interaction(user_input, routing.selected_modules)
            self.modules.record_usage(routing.selected_modules)

        if self.micro_adapters and routing.selected_modules:
            self.micro_adapters.record_interaction(
                prompt=user_input, response=response, modules=routing.selected_modules,
            )

        self.modules.cleanup_stale()

        perf["total"] = time.monotonic() - perf["start"]
        self._perf_history.append(perf)
        if len(self._perf_history) > 50:
            self._perf_history = self._perf_history[-50:]

    def _handle_commands(self, user_input: str) -> str | None:
        """Handle /commands."""
        cmd = user_input.strip().lower()

        if cmd == "/modules":
            modules = self.modules.list_all()
            if not modules:
                return "No modules found."
            lines = ["Available modules:"]
            for m in modules:
                cached = f" [{m['tier']}]" if m["cached"] else ""
                lines.append(f"  {m['name']}{cached} — {m['description']}")
            return "\n".join(lines)

        if cmd == "/memory":
            status = self.memory.status()
            return (
                f"Memory: {status['short_term_turns']} turns, "
                f"{status['long_term_count']} long-term entries"
            )

        if cmd.startswith("/remember "):
            fact = user_input[10:].strip()
            self.memory.remember(fact, source="user", importance=0.8)
            return f"Stored: {fact}"

        if cmd.startswith("/explain "):
            query = user_input[9:].strip()
            return self.router.explain(query)

        if cmd == "/status":
            return (
                f"Model: {'loaded' if self.base_model.is_loaded else 'not loaded'}\n"
                f"Augmentors: {'ON' if self._augmentors_enabled else 'OFF'}\n"
                f"Modules: {len(self.modules.available_modules)} available\n"
                f"Memory: {self.memory.status()['short_term_turns']} turns"
            )

        if cmd == "/perf":
            if not self._perf_history:
                return "No performance data yet."
            last = self._perf_history[-1]
            lines = ["Last turn:"]
            for k, v in last.items():
                if k == "start":
                    continue
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.3f}s")
                else:
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        if cmd in ("/help", "/h"):
            return (
                "Commands:\n"
                "  /modules    — List available modules\n"
                "  /memory     — Memory status\n"
                "  /remember X — Store a fact\n"
                "  /explain X  — Explain routing for a prompt\n"
                "  /status     — System status\n"
                "  /perf       — Last turn performance\n"
                "  /help       — This help\n"
                "  /quit       — Exit"
            )

        if cmd in ("/quit", "/exit", "/q"):
            return "__QUIT__"

        return None

    def _dry_run_response(self, routing, active_modules, prompt, perf):
        """Generate a mock response for dry-run mode."""
        lines = [
            "[DRY RUN]",
            f"Routed to: {routing.selected_modules}",
            f"Mode: {routing.routing_mode}",
            f"Modules loaded: {[m.manifest.name for m in active_modules]}",
            f"Prompt length: ~{len(prompt)//4} tokens",
        ]
        return "\n".join(lines)

    # ── Phase 14 agentic harness ────────────────────────────────

    def _build_workspace_hint(self, workspace: Path) -> str:
        """
        Cheap project-awareness snippet for the agent's system prompt:
        cwd, detected project type, top-level layout. Pure stdlib, no indexing.
        """
        lines = [f"Workspace: {workspace}"]

        markers = {
            "pyproject.toml": "Python (pyproject.toml)",
            "requirements.txt": "Python (requirements.txt)",
            "package.json": "Node.js / TypeScript",
            "Cargo.toml": "Rust",
            "go.mod": "Go",
            "build.gradle": "Java/Kotlin (Gradle)",
            "pom.xml": "Java (Maven)",
            "Gemfile": "Ruby",
            "composer.json": "PHP",
            "CMakeLists.txt": "C/C++ (CMake)",
        }
        detected = [label for marker, label in markers.items() if (workspace / marker).exists()]
        if detected:
            lines.append("Project type: " + ", ".join(detected))

        noise = {
            "__pycache__", "node_modules", ".git", ".venv", "venv",
            ".mypy_cache", ".pytest_cache", "dist", "build", ".idea", ".vscode",
        }
        try:
            entries: list[str] = []
            for child in sorted(workspace.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if child.name.startswith(".") or child.name in noise:
                    continue
                entries.append(child.name + ("/" if child.is_dir() else ""))
                if len(entries) >= 40:
                    entries.append("...")
                    break
            if entries:
                lines.append("Top-level: " + ", ".join(entries))
        except OSError:
            pass

        return "\n".join(lines)

    def _get_agent(self, workspace: Path | None = None) -> Agent:
        """Lazy-build the Agent. Reuses the already-loaded base_model."""
        if workspace is None:
            workspace = Path.cwd()
        if self._agent is not None:
            return self._agent

        memory = AgentMemory(workspace=workspace)
        # Reviewable staged edits (opt-in via --review). When on, every
        # mutating file tool routes its final write through a confirm hook
        # that shows the exact diff and prompts before the bytes land. Default
        # off = byte-for-byte legacy behavior. Orthogonal to --yes (which only
        # gates risky tools like run_bash); the two compose.
        confirm_edit = self._confirm_edit if getattr(self, "_review_edits", False) else None
        registry = build_default_registry(
            workspace, memory=memory, ask_user_fn=self._ask_user,
            extended_tools=True,
            mcp_servers=getattr(self, "_mcp_servers", None),
            enable_web=getattr(self, "_enable_web", False),
            confirm_edit=confirm_edit,
        )
        hint = self._build_workspace_hint(workspace)

        self._agent = Agent(
            model=self.base_model,
            registry=registry,
            system_prompt_extra=hint,
            workspace_root=Path(workspace).resolve(),
            memory=memory,
            auto_verify_python=True,
            max_iterations=20,
            max_wall_time=600.0,
            max_tokens_per_turn=1024,
            confirm_risky=self._confirm_risky_tool,
            confirm_destructive=self._confirm_destructive_command,
            confirm_supply_chain=self._confirm_supply_chain_command,
            on_event=self._render_agent_event,
        )
        return self._agent

    def _ask_user(self, question: str) -> str:
        print(f"\n  [agent asks] {question}")
        try:
            answer = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "(no response)"
        return answer or "(no response)"

    def _confirm_risky_tool(self, call: ToolCall) -> bool:
        """
        Interactive y/N prompt for risky tools (run_bash, etc.). Default deny.
        If self._auto_approve_risky is set (e.g. --yes flag), auto-approve and
        log the call instead of prompting.
        """
        if getattr(self, "_auto_approve_risky", False):
            print(f"\n  ! Risky tool (auto-approved): {call.name}")
            for k, v in call.arguments.items():
                shown = str(v)
                if len(shown) > 200:
                    shown = shown[:200] + "..."
                print(f"    {k} = {shown}")
            return True

        print()
        print(f"  ! Risky tool: {call.name}")
        for k, v in call.arguments.items():
            shown = str(v)
            if len(shown) > 200:
                shown = shown[:200] + "..."
            print(f"    {k} = {shown}")
        try:
            answer = input("  Approve? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("  -> denied")
            return False
        approved = answer in ("y", "yes")
        print(f"  -> {'approved' if approved else 'denied'}")
        return approved

    def _confirm_edit(self, path, old_content, new_content, diff_text) -> bool:
        """
        Reviewable staged edit (opt-in via --review). Show the EXACT unified
        diff for a pending file write and prompt y/N before the bytes land on
        disk. Default DENY on Ctrl+C / EOF. Wired inside the tool's write path
        (engine.agent_builtins._apply_file_write), so the previewed diff is
        byte-identical to what gets written.

        Orthogonal to --yes / _auto_approve_risky: that flag auto-approves
        *risky* tools (run_bash, web), but it does NOT bypass edit review — a
        user can run `--yes --review` to auto-run shells while still
        hand-approving every file edit.
        """
        print()
        print(f"  [review edit] {path}")
        if diff_text:
            for ln in diff_text.splitlines():
                print(f"    {ln}")
        else:
            preview = new_content if len(new_content) <= 2000 else (
                new_content[:2000] + f"\n... ({len(new_content) - 2000} more chars)"
            )
            for ln in preview.splitlines():
                print(f"    +{ln}")
        try:
            answer = input("  Apply this edit? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("  -> rejected (interrupt)")
            return False
        approved = answer in ("y", "yes")
        print(f"  -> {'applied' if approved else 'rejected'}")
        return approved

    def _confirm_destructive_command(self, call: "ToolCall", matches: list) -> bool:
        """
        High-friction confirmation for run_bash commands that match the
        destructive-command denylist (rm -rf, ~/ expansion, format, dd to
        /dev, git reset --hard, DROP TABLE, etc.). Mandatory prompt —
        --yes / _auto_approve_risky does NOT bypass this. Requires the
        user to type the exact phrase `yes i am sure` (case-insensitive,
        whitespace-normalized) — a single 'y' is rejected. Default deny
        on Ctrl+C / EOF / any other input.

        See engine/destructive_command_gate.py and the May 2026 r/ClaudeAI
        `rm -rf ~/` (717 GB Windows-wipe) incident for rationale.
        """
        from engine.destructive_command_gate import format_warning

        command = call.arguments.get("command", "")
        print()
        print(format_warning(command, matches))
        print()
        try:
            answer = input(
                "  To execute, type exactly  yes i am sure  (anything else = deny): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("  -> denied (interrupt)")
            return False
        # Normalize whitespace inside the answer so accidental double
        # spaces don't punish the user — but the words must be exact.
        normalized = " ".join(answer.split())
        approved = normalized == "yes i am sure"
        print(f"  -> {'APPROVED (destructive)' if approved else 'denied'}")
        return approved

    def _confirm_supply_chain_command(self, call: "ToolCall", risks: list) -> bool:
        """
        High-friction confirmation for run_bash / run_tests calls flagged by
        the supply-chain gate (fetch-and-execute, repo auto-exec drop points,
        or a command lifted from program output — the June 2026 Miasma worm /
        0din Axiom attack class). Mandatory prompt: --yes / _auto_approve_risky
        does NOT bypass this. Requires typing the exact phrase `yes i trust
        this` (a single 'y' is rejected). Default deny on Ctrl+C / EOF.

        See engine/supply_chain_gate.py.
        """
        from engine.supply_chain_gate import format_warning

        command = call.arguments.get("command", "") if isinstance(call.arguments, dict) else ""
        print()
        print(format_warning(command, risks))
        print()
        try:
            answer = input(
                "  To execute, type exactly  yes i trust this  (anything else = deny): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("  -> denied (interrupt)")
            return False
        normalized = " ".join(answer.split())
        approved = normalized == "yes i trust this"
        print(f"  -> {'APPROVED (supply-chain)' if approved else 'denied'}")
        return approved

    def _render_agent_event(self, event: AgentEvent) -> None:
        """Print agent events to the console as the loop runs."""
        if event.type == "iteration":
            print(f"\n[iter {event.iteration}]")
        elif event.type == "model_text":
            text = (event.payload or "").strip()
            if text and "<tool_call>" not in text:
                # Plain reasoning text between/before tool calls
                preview = text if len(text) < 400 else text[:400] + " ..."
                print(f"  thought: {preview}")
        elif event.type == "tool_call":
            call = event.payload
            args_preview = ", ".join(
                f"{k}={str(v)[:60] + ('...' if len(str(v)) > 60 else '')}"
                for k, v in call.arguments.items()
            )
            print(f"  -> {call.name}({args_preview})")
        elif event.type == "tool_result":
            result = event.payload
            if result.success:
                content_str = str(result.content)
                if len(content_str) > 200:
                    content_str = content_str[:200] + " ..."
                # Strip newlines for one-line display
                content_str = content_str.replace("\n", " | ")
                print(f"     ok: {content_str}")
            else:
                print(f"     err: {result.error}")
        elif event.type == "final":
            print("\n[done]")
        elif event.type == "stopped":
            if event.payload and event.payload != "answered":
                print(f"[stopped: {event.payload}]")

    def run_agent(self, goal: str, workspace: Path | None = None) -> None:
        """Run one agentic task end-to-end and print the final answer."""
        if not self.base_model.is_loaded:
            print("Model not loaded — cannot run agent.")
            return

        agent = self._get_agent(workspace)
        print(f"\n=== Agent goal ===\n{goal}\n")
        try:
            result = agent.run(goal)
        except KeyboardInterrupt:
            print("\n[agent aborted by user]")
            return

        print("\n=== Final answer ===")
        print(result.final_answer or "(no final answer)")
        print(
            f"\n[{result.iterations} iterations, "
            f"{len(result.tool_calls)} tool calls, "
            f"{result.wall_time:.1f}s, stop={result.stop_reason}]"
        )

    # ── Phase 14 standalone agent entry ─────────────────────────

    @staticmethod
    def run_agent_fast(
        config_path: str,
        goal: str,
        workspace: Path | None = None,
        auto_approve_risky: bool = False,
        mcp_servers: list[str] | None = None,
        enable_web: bool = False,
        review_edits: bool = False,
    ) -> int:
        """
        Lightweight agent entry: load Config + BaseModel + Agent only, skip
        the full UCA stack (router, modules, memory, fusion, augmentors,
        sentence-transformer embedder, FAISS, project_index).

        Motivation: the agent loop only needs the GGUF and the tool registry.
        Loading the rest costs ~2 minutes of HuggingFace HEAD requests +
        embedder init AND allocates the sentence-transformer on CUDA, which
        competes for VRAM with the main model on a 12 GB card. Skipping it
        drops startup from ~2 minutes to ~5 seconds and gives the main model
        full VRAM for its KV cache.

        Uses the LOCAL `engine.base_model.BaseModel` (not densanon-core's)
        because the local one accepts `speculative_config` for prompt_lookup
        draft decoding — an agent-loop-relevant speedup that the shared
        densanon-core BaseModel doesn't support. This is the one divergence
        between the agent fast path and the rest of UCA's model loading.

        Reuses UCA's agent instance methods (`run_agent`, `_get_agent`,
        `_render_agent_event`, `_confirm_risky_tool`, `_build_workspace_hint`)
        by constructing a bare UCA shell via `object.__new__` and setting
        only the attributes those methods touch.
        """
        config = Config(config_path)
        config.setup_logging()

        logger.info(
            f"=== {config.system.name} v{config.system.version} (agent mode) ==="
        )
        logger.info(f"Loading model (agent mode): {config.base_model.path}")

        # densanon-core's Config class doesn't expose the `speculative:` YAML
        # section, so parse it ourselves for the agent fast path. Without
        # this, prompt_lookup draft decoding is silently disabled no matter
        # what config_agent14b.yaml says.
        spec_cfg = _load_speculative_config(config_path)
        try:
            from engine.base_model import BaseModel as LocalBaseModel
            base_model = LocalBaseModel(config.base_model, speculative_config=spec_cfg)
            if spec_cfg is not None and getattr(spec_cfg, "enabled", False):
                logger.info(
                    f"Speculative decoding: {spec_cfg.mode} "
                    f"(num_pred_tokens={getattr(spec_cfg, 'num_pred_tokens', '?')})"
                )
        except ImportError as exc:
            logger.warning(
                f"Local engine.base_model unavailable ({exc}); falling back to "
                f"densanon-core BaseModel without speculative decoding."
            )
            base_model = BaseModel(config.base_model)
        base_model.load()

        uca = object.__new__(UltraliteCodeAssistant)
        uca.base_model = base_model
        uca._agent = None
        uca._auto_approve_risky = auto_approve_risky
        # Stash for `_get_agent()` so the registry build picks them up.
        # When the scaffold is empty/None this is a no-op; when populated
        # the agent_builtins MCP hook fires.
        uca._mcp_servers = mcp_servers or []
        uca._enable_web = enable_web
        uca._review_edits = review_edits

        try:
            uca.run_agent(goal, workspace=workspace)
            return 0
        finally:
            base_model.unload()

    def shutdown(self):
        """Clean shutdown."""
        shutdown_fn = getattr(self.pipeline, "shutdown", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception as exc:
                logger.debug(f"Pipeline shutdown raised: {exc}")
        self.base_model.unload()
        logger.info("Shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="Ultralite Code Assistant"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Test without model")
    parser.add_argument("--no-pipeline", action="store_true", help="Disable async pipeline")
    parser.add_argument("--list-modules", action="store_true", help="List available modules")
    parser.add_argument("--explain", type=str, help="Explain routing for a prompt")
    parser.add_argument("--train", action="store_true", help="Train the classifier")
    parser.add_argument(
        "--agent",
        type=str,
        metavar="GOAL",
        help="Run one agentic task in the current directory and exit (Phase 14)",
    )
    parser.add_argument(
        "--agent-full-init",
        action="store_true",
        help="Use the full UCA init path for --agent (slower, loads embedder/router/augmentors). "
             "Default is the fast path that skips everything except the base model.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-approve all risky tool calls (run_bash, etc.) without prompting. "
             "Use for unattended runs; interactive use should leave this off.",
    )
    parser.add_argument(
        "--mcp",
        type=str,
        default="",
        metavar="SERVERS",
        help="Comma-separated MCP servers to mount (e.g. 'densa-deck'). "
             "Scaffolded but NOT WIRED — passing a non-empty value raises "
             "NotImplementedError today; the flag exists so future activation "
             "is one flip in engine/mcp_adapter.py. See docs/MCP.md.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Enable web_search + fetch_url tools. Both flagged risky — the "
             "agent loop prompts y/N before each call (unless --yes). Default "
             "off preserves the zero-server invariant.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Review each file edit before it's written: show the exact diff "
             "and prompt y/N (default deny). Orthogonal to --yes (which only "
             "gates risky tools); the two compose — `--yes --review` auto-runs "
             "shells but still hand-approves every file edit.",
    )
    args = parser.parse_args()

    # Fast agent path: skip the full UCA init entirely. Only Config + BaseModel
    # + Agent are needed for the tool loop, and the rest (embedder on CUDA,
    # FAISS, augmentor embeddings, HF HEAD requests) costs ~2 minutes of
    # startup + contends for VRAM with the main model.
    if args.agent and not args.agent_full_init:
        mcp_servers = [s.strip() for s in args.mcp.split(",") if s.strip()]
        return UltraliteCodeAssistant.run_agent_fast(
            args.config, args.agent,
            auto_approve_risky=args.yes,
            mcp_servers=mcp_servers,
            enable_web=args.web,
            review_edits=args.review,
        )

    engine = UltraliteCodeAssistant(
        config_path=args.config,
        dry_run=args.dry_run,
        use_pipeline=not args.no_pipeline,
    )
    engine.initialize()

    if args.list_modules:
        print(engine.process("/modules"))
        return

    if args.explain:
        print(engine.router.explain(args.explain))
        return

    if args.train:
        stats = engine.router.train_classifier()
        print(f"Training result: {stats}")
        return

    if args.agent:
        engine._auto_approve_risky = args.yes
        engine._enable_web = args.web
        engine._review_edits = args.review
        engine.run_agent(args.agent)
        engine.shutdown()
        return

    # Interactive REPL
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    print(f"\n  Ultralite Code Assistant v{engine.config.system.version}")
    print(f"  Model: {Path(engine.config.base_model.path).name}")
    print(f"  Type /help for commands, /quit to exit\n")

    while True:
        try:
            user_input = input("you> ").strip()
            if not user_input:
                continue

            if user_input.startswith("/agent "):
                goal = user_input[len("/agent "):].strip()
                if goal:
                    engine.run_agent(goal)
                else:
                    print("Usage: /agent <goal>")
                continue

            response = engine.process(user_input)
            if response == "__QUIT__":
                break

            if use_rich and ("```" in response or "**" in response or "#" in response):
                console.print(Markdown(response))
            else:
                print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except EOFError:
            break

    engine.shutdown()


if __name__ == "__main__":
    main()
