"""
Configuration Manager
Loads and validates system configuration from YAML.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from engine.native_speculative import NativeSpeculativeConfig

logger = logging.getLogger(__name__)


@dataclass
class BaseModelConfig:
    path: str = "models/your-model.gguf"
    context_length: int = 2048
    gpu_layers: int = 0
    threads: int = 4
    temperature: float = 0.7
    max_tokens: int = 512
    batch_size: int = 512
    # KV cache quantization. None = default F16. Q8_0 halves KV VRAM at ~lossless
    # quality and can stabilize long tool-call chains where F16 KV gets evicted
    # under context pressure. Accepts "f16" | "q8_0" | "q4_0" | "q5_0" | "q5_1".
    # Requires llama-cpp-python built with KV quant support; falls back gracefully.
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None


@dataclass
class RoutingRule:
    keywords: list[str] = field(default_factory=list)
    module: str = ""
    priority: int = 5


@dataclass
class ClassifierConfig:
    """Phase 2+4: Learned classifier settings."""
    model_path: str = "data/router_model"       # Where to save/load trained classifier
    min_training_samples: int = 10              # Minimum samples before classifier activates
    confidence_threshold: float = 0.4           # Below this, fall back to rule-based
    retrain_interval: int = 50                  # Retrain after N new interactions
    max_features: int = 5000                    # TF-IDF vocabulary size
    # Phase 4: Neural classifier options
    type: str = "tfidf"                         # "tfidf" or "neural"
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence-transformers model for neural classifier


@dataclass
class BlendingConfig:
    """Phase 2: Multi-module blending settings."""
    enabled: bool = True
    strategy: str = "weighted"                  # "weighted", "priority", "equal"
    max_blend_modules: int = 3                  # Max modules to blend in one response
    conflict_resolution: str = "priority"       # "priority", "score", "longest"
    weight_decay: float = 0.7                   # Score decay for lower-ranked modules


@dataclass
class RouterConfig:
    mode: str = "rule_based"                    # "rule_based", "classifier", "hybrid"
    max_active_modules: int = 3
    default_modules: list[str] = field(default_factory=list)
    rules: list[RoutingRule] = field(default_factory=list)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    blending: BlendingConfig = field(default_factory=BlendingConfig)


@dataclass
class ModulesConfig:
    directory: str = "modules"
    cache_ttl: int = 300
    max_cached: int = 5
    # Phase 3: Smart cache settings
    predictive_preload: bool = True     # Pre-load modules based on co-occurrence patterns
    preload_top_k: int = 2             # How many modules to speculatively pre-load


@dataclass
class PipelineConfig:
    """Phase 3: Async pipeline settings."""
    enabled: bool = True
    parallel_workers: int = 3           # Thread pool size for parallel I/O
    enable_generation_queue: bool = True # Generation queue for multi-request
    generation_timeout: int = 120       # Max seconds to wait for generation


@dataclass
class KVCacheConfig:
    """Phase 3: KV cache optimization settings."""
    enabled: bool = True
    compression_threshold: float = 0.80 # Trigger conversation compression at this context usage


@dataclass
class ShortTermMemoryConfig:
    max_turns: int = 20
    max_tokens: int = 1024


@dataclass
class LongTermMemoryConfig:
    enabled: bool = True
    backend: str = "simple"                     # "simple" (keyword) or "faiss" (vector)
    storage_dir: str = "data/memory"
    top_k: int = 5
    similarity_threshold: float = 0.5
    embedding_model: str = "all-MiniLM-L6-v2"  # Phase 4: sentence-transformers model for FAISS


@dataclass
class SystemMemoryConfig:
    enabled: bool = True
    knowledge_dir: str = "data/knowledge"


@dataclass
class CompressorConfig:
    """Phase 4: Conversation compression settings."""
    max_summary_sentences: int = 10            # Max sentences to keep in compressed state
    max_topics: int = 8                        # Max topic keywords to track


@dataclass
class MicroAdapterConfig:
    """Phase 4: Micro-adapter generation settings."""
    enabled: bool = True
    storage_dir: str = "data/adapters"
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence-transformers model
    min_cluster_size: int = 5                   # Minimum interactions per cluster
    max_adapters: int = 8                       # Maximum number of adapters
    regenerate_interval: int = 20               # Regenerate after N new interactions


@dataclass
class ProjectContextConfig:
    enabled: bool = False
    storage_dir: str = "data/project_index"
    top_k: int = 5
    similarity_threshold: float = 0.35
    max_chunk_lines: int = 40                   # Max lines per chunk
    overlap_lines: int = 5                      # Overlap between chunks
    file_extensions: list[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".go", ".rs", ".c", ".h", ".cpp",
        ".java", ".cs", ".rb", ".kt", ".swift",
        ".sql", ".sh", ".bash",
        ".yaml", ".yml", ".json", ".toml",
        ".md", ".txt",
    ])
    ignore_patterns: list[str] = field(default_factory=lambda: [
        "__pycache__", "node_modules", ".git", ".venv", "venv",
        "dist", "build", ".egg-info", ".tox", ".mypy_cache",
        "*.pyc", "*.min.js", "*.min.css",
    ])
    max_file_size_kb: int = 500                 # Skip files larger than this


@dataclass
class MemoryConfig:
    enabled: bool = True
    short_term: ShortTermMemoryConfig = field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)
    system: SystemMemoryConfig = field(default_factory=SystemMemoryConfig)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)


@dataclass
class FusionConfig:
    mode: str = "structured"                    # "simple" (v1) or "structured" (v2 XML-tagged)
    chat_format: str = "chatml"                 # "chatml", "raw", "llama2", "alpaca"
    system_prompt: str = "You are a modular AI assistant."
    max_prompt_tokens: int = 1536
    # Phase 2: Token budget allocation (fractions of max_prompt_tokens)
    budget_system: float = 0.15                 # System prompt budget
    budget_modules: float = 0.25                # Module context budget
    budget_memory: float = 0.20                 # Long-term memory budget
    budget_conversation: float = 0.30           # Conversation history budget
    budget_reserve: float = 0.10                # Reserve for user input + formatting


@dataclass
class NPCConfig:
    enabled: bool = False
    active_profile: Optional[str] = None
    profiles_dir: str = "data/npc_profiles"
    json_output: bool = True
    output_schema: dict = field(default_factory=lambda: {
        "dialogue": "string",
        "emotion": "string",
        "action": "string|null"
    })


@dataclass
class SystemConfig:
    name: str = "Plug-in Intelligence Engine"
    version: str = "0.1.0"
    log_level: str = "INFO"


class Config:
    """
    Central configuration object. Loads from YAML and provides
    typed access to all configuration sections.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.base_dir = self.config_path.parent

        self.system = SystemConfig()
        self.base_model = BaseModelConfig()
        self.router = RouterConfig()
        self.modules = ModulesConfig()
        self.memory = MemoryConfig()
        self.fusion = FusionConfig()
        self.npc = NPCConfig()
        self.pipeline = PipelineConfig()
        self.kv_cache = KVCacheConfig()
        self.micro_adapters = MicroAdapterConfig()
        self.project_context = ProjectContextConfig()
        self.speculative = NativeSpeculativeConfig()

        if self.config_path.exists():
            self._load()
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")

    def _load(self):
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        # System
        if "system" in raw:
            s = raw["system"]
            self.system = SystemConfig(
                name=s.get("name", self.system.name),
                version=s.get("version", self.system.version),
                log_level=s.get("log_level", self.system.log_level),
            )

        # Base Model
        if "base_model" in raw:
            bm = raw["base_model"]
            self.base_model = BaseModelConfig(
                path=self._resolve_path(bm.get("path", self.base_model.path)),
                context_length=bm.get("context_length", self.base_model.context_length),
                gpu_layers=bm.get("gpu_layers", self.base_model.gpu_layers),
                threads=bm.get("threads", self.base_model.threads),
                temperature=bm.get("temperature", self.base_model.temperature),
                max_tokens=bm.get("max_tokens", self.base_model.max_tokens),
                batch_size=bm.get("batch_size", self.base_model.batch_size),
                cache_type_k=bm.get("cache_type_k", self.base_model.cache_type_k),
                cache_type_v=bm.get("cache_type_v", self.base_model.cache_type_v),
            )

        # Router
        if "router" in raw:
            r = raw["router"]
            rules = []
            for rule_data in r.get("rules", []):
                rules.append(RoutingRule(
                    keywords=rule_data.get("keywords", []),
                    module=rule_data.get("module", ""),
                    priority=rule_data.get("priority", 5),
                ))

            # Phase 2+4: Classifier config
            cls_raw = r.get("classifier", {})
            classifier = ClassifierConfig(
                model_path=self._resolve_path(cls_raw.get("model_path", "data/router_model")),
                min_training_samples=cls_raw.get("min_training_samples", 10),
                confidence_threshold=cls_raw.get("confidence_threshold", 0.4),
                retrain_interval=cls_raw.get("retrain_interval", 50),
                max_features=cls_raw.get("max_features", 5000),
                type=cls_raw.get("type", "tfidf"),
                embedding_model=cls_raw.get("embedding_model", "all-MiniLM-L6-v2"),
            )

            # Phase 2: Blending config
            blend_raw = r.get("blending", {})
            blending = BlendingConfig(
                enabled=blend_raw.get("enabled", True),
                strategy=blend_raw.get("strategy", "weighted"),
                max_blend_modules=blend_raw.get("max_blend_modules", 3),
                conflict_resolution=blend_raw.get("conflict_resolution", "priority"),
                weight_decay=blend_raw.get("weight_decay", 0.7),
            )

            self.router = RouterConfig(
                mode=r.get("mode", self.router.mode),
                max_active_modules=r.get("max_active_modules", self.router.max_active_modules),
                default_modules=r.get("default_modules", self.router.default_modules),
                rules=rules,
                classifier=classifier,
                blending=blending,
            )

        # Modules
        if "modules" in raw:
            m = raw["modules"]
            self.modules = ModulesConfig(
                directory=self._resolve_path(m.get("directory", self.modules.directory)),
                cache_ttl=m.get("cache_ttl", self.modules.cache_ttl),
                max_cached=m.get("max_cached", self.modules.max_cached),
                predictive_preload=m.get("predictive_preload", self.modules.predictive_preload),
                preload_top_k=m.get("preload_top_k", self.modules.preload_top_k),
            )

        # Memory
        if "memory" in raw:
            mem = raw["memory"]
            st = mem.get("short_term", {})
            lt = mem.get("long_term", {})
            sys_mem = mem.get("system", {})

            self.memory = MemoryConfig(
                enabled=mem.get("enabled", self.memory.enabled),
                short_term=ShortTermMemoryConfig(
                    max_turns=st.get("max_turns", 20),
                    max_tokens=st.get("max_tokens", 1024),
                ),
                long_term=LongTermMemoryConfig(
                    enabled=lt.get("enabled", True),
                    backend=lt.get("backend", "simple"),
                    storage_dir=self._resolve_path(lt.get("storage_dir", "data/memory")),
                    top_k=lt.get("top_k", 5),
                    similarity_threshold=lt.get("similarity_threshold", 0.5),
                    embedding_model=lt.get("embedding_model", "all-MiniLM-L6-v2"),
                ),
                system=SystemMemoryConfig(
                    enabled=sys_mem.get("enabled", True),
                    knowledge_dir=self._resolve_path(sys_mem.get("knowledge_dir", "data/knowledge")),
                ),
                compressor=CompressorConfig(
                    max_summary_sentences=mem.get("compressor", {}).get("max_summary_sentences", 10),
                    max_topics=mem.get("compressor", {}).get("max_topics", 8),
                ),
            )

        # Fusion
        if "fusion" in raw:
            fu = raw["fusion"]
            self.fusion = FusionConfig(
                mode=fu.get("mode", self.fusion.mode),
                chat_format=fu.get("chat_format", self.fusion.chat_format),
                system_prompt=fu.get("system_prompt", self.fusion.system_prompt),
                max_prompt_tokens=fu.get("max_prompt_tokens", self.fusion.max_prompt_tokens),
                budget_system=fu.get("budget_system", self.fusion.budget_system),
                budget_modules=fu.get("budget_modules", self.fusion.budget_modules),
                budget_memory=fu.get("budget_memory", self.fusion.budget_memory),
                budget_conversation=fu.get("budget_conversation", self.fusion.budget_conversation),
                budget_reserve=fu.get("budget_reserve", self.fusion.budget_reserve),
            )

        # NPC
        if "npc" in raw:
            n = raw["npc"]
            self.npc = NPCConfig(
                enabled=n.get("enabled", self.npc.enabled),
                active_profile=n.get("active_profile", self.npc.active_profile),
                profiles_dir=self._resolve_path(n.get("profiles_dir", self.npc.profiles_dir)),
                json_output=n.get("json_output", self.npc.json_output),
                output_schema=n.get("output_schema", self.npc.output_schema),
            )

        # Phase 3: Pipeline
        if "pipeline" in raw:
            p = raw["pipeline"]
            self.pipeline = PipelineConfig(
                enabled=p.get("enabled", self.pipeline.enabled),
                parallel_workers=p.get("parallel_workers", self.pipeline.parallel_workers),
                enable_generation_queue=p.get("enable_generation_queue", self.pipeline.enable_generation_queue),
                generation_timeout=p.get("generation_timeout", self.pipeline.generation_timeout),
            )

        # Phase 3: KV Cache
        if "kv_cache" in raw:
            kv = raw["kv_cache"]
            self.kv_cache = KVCacheConfig(
                enabled=kv.get("enabled", self.kv_cache.enabled),
                compression_threshold=kv.get("compression_threshold", self.kv_cache.compression_threshold),
            )

        # Phase 4: Micro-adapters
        if "micro_adapters" in raw:
            ma = raw["micro_adapters"]
            self.micro_adapters = MicroAdapterConfig(
                enabled=ma.get("enabled", self.micro_adapters.enabled),
                storage_dir=self._resolve_path(ma.get("storage_dir", "data/adapters")),
                embedding_model=ma.get("embedding_model", self.micro_adapters.embedding_model),
                min_cluster_size=ma.get("min_cluster_size", self.micro_adapters.min_cluster_size),
                max_adapters=ma.get("max_adapters", self.micro_adapters.max_adapters),
                regenerate_interval=ma.get("regenerate_interval", self.micro_adapters.regenerate_interval),
            )

        # Project Context
        if "project_context" in raw:
            pc = raw["project_context"]
            self.project_context = ProjectContextConfig(
                enabled=pc.get("enabled", self.project_context.enabled),
                storage_dir=self._resolve_path(pc.get("storage_dir", self.project_context.storage_dir)),
                top_k=pc.get("top_k", self.project_context.top_k),
                similarity_threshold=pc.get("similarity_threshold", self.project_context.similarity_threshold),
                max_chunk_lines=pc.get("max_chunk_lines", self.project_context.max_chunk_lines),
                overlap_lines=pc.get("overlap_lines", self.project_context.overlap_lines),
                file_extensions=pc.get("file_extensions", self.project_context.file_extensions),
                ignore_patterns=pc.get("ignore_patterns", self.project_context.ignore_patterns),
                max_file_size_kb=pc.get("max_file_size_kb", self.project_context.max_file_size_kb),
            )

        # Native speculative decoding (llama-cpp-python draft model)
        if "speculative" in raw:
            sp = raw["speculative"]
            self.speculative = NativeSpeculativeConfig(
                enabled=sp.get("enabled", self.speculative.enabled),
                mode=sp.get("mode", self.speculative.mode),
                num_pred_tokens=sp.get("num_pred_tokens", self.speculative.num_pred_tokens),
                max_ngram_size=sp.get("max_ngram_size", self.speculative.max_ngram_size),
                draft_model_path=self._resolve_path(
                    sp.get("draft_model_path", self.speculative.draft_model_path)
                ) if sp.get("draft_model_path") else self.speculative.draft_model_path,
                draft_gpu_layers=sp.get("draft_gpu_layers", self.speculative.draft_gpu_layers),
                draft_context_length=sp.get("draft_context_length", self.speculative.draft_context_length),
            )

        logger.info(f"Configuration loaded from {self.config_path}")

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the config file's directory."""
        p = Path(path)
        if not p.is_absolute():
            p = self.base_dir / p
        return str(p)

    def setup_logging(self):
        """Configure logging based on system settings."""
        level = getattr(logging, self.system.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def __repr__(self):
        return (
            f"Config(\n"
            f"  system={self.system}\n"
            f"  base_model={self.base_model}\n"
            f"  router={self.router}\n"
            f"  memory={self.memory}\n"
            f")"
        )
