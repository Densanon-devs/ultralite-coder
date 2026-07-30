"""Lightweight keyword-matched augmentor injection for the agent path.

The full AugmentorRouter (engine/augmentors.py) requires sentence-transformer
embedder + ~2 min of HF init. The agent path was deliberately designed to
skip that (see main.py::run_agent_fast), so reusing the full router would
re-introduce the startup cost we eliminated.

This module is the agent-path equivalent: deterministic keyword matching
over selected augmentor categories that produce useful demos for tool-loop
goals (security/, agentic/web_research_to_file/, etc). No embeddings, no
HF downloads, sub-ms match time.

Selection logic mirrors the spirit of `_LARGE_MODE_AGENTIC_KEYWORDS` in
augmentors.py — keep the Phase 13 baseline preserved by returning empty
on goals that don't match any agentic-domain trigger keyword.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Categories the agent path pulls from. Each entry is (relative path under
# data/augmentor_examples, list of trigger keyword groups). Order matters —
# earlier categories are preferred when multiple match.
_AGENT_AUGMENTOR_CATEGORIES = [
    {
        "path": "agentic/web_research_to_file",
        "keywords": [
            # Web tool keywords (model invokes these)
            "web_search", "fetch_url",
            # Goal phrases that combine research + file write
            "look up", "research the", "find the latest", "find the current",
            # Generic combinations the model handles via this pattern
            "search the web", "look online",
        ],
    },
    {
        "path": "security",
        "keywords": [
            # Tool/library names
            "scapy", "nmap", "nfcpy", "bleak", "proxmark",
            # Pentest verbs/nouns
            "pentest", "penetration test", "security audit",
            "packet capture", "port scan", "arp scan", "dns enum", "axfr",
            "rfid", "mifare", "wiegand",
            "ble gatt", "wifi scan", "rogue ap", "iwlist",
            "sniffer", "scanner",
            # Reporting verbs
            "findings report", "pcap analysis",
            "normalize severity", "security findings", "scan findings",
            "vulnerability report", "pentest report", "cvss",
            # Webapp pentest (added 2026-05-10)
            "intercepting proxy", "http proxy", "burp",
            "jwt", "json web token",
            "wappalyzer", "fingerprint webapp", "webapp recon",
            "ssl audit", "tls audit", "cipher suite",
            "passive scan", "security headers",
        ],
    },
    {
        "path": "agentic",
        "keywords": [
            # Agent-loop patterns the agentic library covers (json quote
            # leaks, fstring traps, etc). These tend to fire on
            # diagnostic/repair goals.
            "json", "fstring", "f-string", "edit_file", "old_string",
            "stuck_repeat", "tool_rejected",
        ],
    },
]


def _load_yaml_dir(category_path: Path) -> list[dict]:
    """Load every YAML in a category dir. Returns flat list of example dicts."""
    examples: list[dict] = []
    if not category_path.exists():
        return examples
    for yaml_file in sorted(category_path.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"agent_augmentor: failed to parse {yaml_file}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        category_name = data.get("category", yaml_file.stem)
        for ex in data.get("examples", []):
            if not isinstance(ex, dict):
                continue
            ex_with_meta = dict(ex)
            ex_with_meta["__category"] = category_name
            ex_with_meta["__source_file"] = str(yaml_file.name)
            examples.append(ex_with_meta)
    return examples


def _score_example(example: dict, goal_lower: str, category_triggered: bool) -> int:
    """Per-example relevance score. Higher = more relevant.

    Args:
        category_triggered: True if the example's category matched a trigger
            keyword in the goal. Adds a base score so any example from a
            triggered category passes min_score even when its specific text
            doesn't overlap the goal much. Without this, a goal like
            "look up the latest pydantic v2 syntax" triggers the
            web_research category but no example scores high enough on
            text overlap to inject.
    """
    score = 4 if category_triggered else 0
    # Tag matches (exact tag tokens vs goal substring)
    for tag in example.get("tags", []) or []:
        tag_s = str(tag).lower()
        if tag_s and tag_s in goal_lower:
            score += 3
    # Category match — treat its tokens as soft signals
    cat = str(example.get("__category", "")).lower()
    for token in re.split(r"[_\W]+", cat):
        if token and len(token) >= 4 and token in goal_lower:
            score += 2
    # Query-text overlap (4+ char content words from the example query)
    query_text = str(example.get("query", "")).lower()
    for word in set(re.findall(r"[a-zA-Z][a-zA-Z_]{3,}", query_text)):
        if word in goal_lower:
            score += 1
    return score


def build_agent_augmentor(
    examples_dir: str | Path = "data/augmentor_examples",
    max_examples: int = 2,
    min_score: int = 4,
):
    """Returns a callable suitable for Agent(augment_for_goal=...).

    Args:
        examples_dir: Repo-relative or absolute path to the augmentor root.
            Defaults to "data/augmentor_examples" — assumes cwd is the
            ultralight-coder repo root.
        max_examples: Cap on examples injected per goal. 2 is a safe default
            given the 14B's tool-count regression at 21+ tools — adding too
            many augmentor blocks dilutes the system prompt the same way.
        min_score: Minimum relevance score for an example to inject. Below
            this, return empty (no injection) — preserves the Phase 13
            baseline for goals that don't strongly match any category.

    The returned callable signature: `(goal: str) -> str`. Returns empty
    string on no match (which Agent treats as no injection).
    """
    base_dir = Path(examples_dir)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir

    # Closure-scoped cache so each builder has its own. Avoids module-level
    # contamination when multiple builders point at different example dirs
    # (matters in tests; harmless in prod where there's one builder).
    _cache: dict[str, list[dict]] = {}
    _cache_lock = threading.Lock()

    def _get_examples(rel_path: str) -> list[dict]:
        with _cache_lock:
            if rel_path in _cache:
                return _cache[rel_path]
            loaded = _load_yaml_dir(base_dir / rel_path)
            _cache[rel_path] = loaded
            return loaded

    def augment(goal: str) -> str:
        if not goal or not goal.strip():
            return ""
        goal_lower = goal.lower()

        # Phase 1: figure out which category(ies) the goal triggers
        triggered_categories: set[str] = set()
        for cat in _AGENT_AUGMENTOR_CATEGORIES:
            for kw in cat["keywords"]:
                if kw in goal_lower:
                    triggered_categories.add(cat["path"])
                    break
        if not triggered_categories:
            return ""

        # Phase 2: collect examples from triggered categories, score, pick best
        candidates: list[tuple[int, dict]] = []
        for rel_path in triggered_categories:
            for ex in _get_examples(rel_path):
                score = _score_example(ex, goal_lower, category_triggered=True)
                if score >= min_score:
                    candidates.append((score, ex))

        if not candidates:
            return ""

        candidates.sort(key=lambda t: -t[0])

        # De-dupe by category — prefer one example per distinct category
        # to avoid 2 nearly-identical examples from the same source.
        seen_cats: set[str] = set()
        picked: list[dict] = []
        for _, ex in candidates:
            cat = ex.get("__category", "")
            if cat in seen_cats:
                continue
            seen_cats.add(cat)
            picked.append(ex)
            if len(picked) >= max_examples:
                break

        if not picked:
            return ""

        # Phase 3: format as a system-prompt block
        blocks = ["## Relevant patterns from your library\n"]
        for ex in picked:
            cat = ex.get("__category", "?")
            query = str(ex.get("query", "")).strip()
            solution = str(ex.get("solution", "")).strip()
            blocks.append(f"### Pattern: {cat}\n")
            blocks.append(f"**Example task:** {query}\n")
            blocks.append(f"**Approach:**\n\n{solution}\n")
        return "\n".join(blocks)

    return augment
