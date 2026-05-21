#!/usr/bin/env python3
"""Augmentor "black-hole" audit.

Scaling Laws of Skills (arXiv 2605.16508) formalizes why a growing skill
library degrades routing: single-step routing accuracy decays logarithmically
with library size, and the worst offenders are "black-hole skills" — overly
general entries whose triggers absorb queries that belong elsewhere. This is
the mechanism behind ULC's measured 21-tool regression (97.6% -> 85.7%).

ULC's force-inject routing layer is `FAILURE_PATTERNS` in engine/augmentors.py
(category -> trigger keywords; a query containing a trigger force-injects that
category regardless of embedding similarity). This script is a static guard
over that table — it does NOT run the model. It flags:

  1. BREADTH  — triggers broad enough to over-fire (single common word, no
                code anchor, or a bare "word " prefix like "type " / "func ").
  2. DRIFT    — triggers that appear in MORE THAN ONE category (cross-family
                contention: the same query force-injects multiple categories,
                exactly the failure mode the paper describes).

Run before promoting new augmentors (composes with the harvest pipeline's
augmentor_promoter). It's an audit, not an edit — output is a flagged list.

Usage:
    python audit_augmentors.py
"""

from __future__ import annotations

import sys

from engine.augmentors import FAILURE_PATTERNS

# Code-anchor characters make a trigger specific (an API name, call, type
# param, dunder) rather than a broad natural-language word.
_CODE_ANCHOR = set("(_.[]<>@:/#")

# Single words that are common enough across coding queries to over-fire if
# used as a bare trigger. Not exhaustive — the breadth score below also
# catches short single words generically; this set is for the obvious verbs.
_GENERIC_WORDS = {
    "write", "create", "build", "add", "fix", "make", "run", "get", "set",
    "use", "code", "function", "method", "class", "object", "value", "data",
    "type", "generic", "interface", "promise", "template", "serialize",
    "deserialize", "validate", "render", "parse", "tokenize", "semaphore",
}


def _has_code_anchor(raw: str) -> bool:
    return any(c in _CODE_ANCHOR for c in raw) or any(c.isupper() for c in raw) \
        or any(c.isdigit() for c in raw)


def _breadth_flags(raw: str) -> list[str]:
    """Return reasons this trigger is broad. Empty list = well-scoped."""
    flags: list[str] = []
    norm = raw.strip().lower()
    words = norm.split()
    anchored = _has_code_anchor(raw)
    if not norm:
        return ["empty"]
    if not anchored and len(words) == 1:
        # A bare single word with no code anchor force-injects on any query
        # mentioning it. "fastapi" is fine (rare/specific); "type"/"generic"
        # are not. Use the generic-word set + length as the discriminator.
        if norm in _GENERIC_WORDS:
            flags.append("single-generic-word")
        elif len(norm) <= 6:
            flags.append("single-short-word")
    if raw.endswith(" ") and len(words) == 1 and not anchored:
        # "type ", "func ", "interface " — a bare word with a trailing space
        # still substring-matches inside many unrelated queries.
        flags.append("bare-word-prefix")
    if not anchored and 0 < len(norm) <= 3:
        flags.append("ultra-short")
    return flags


def main() -> int:
    # trigger -> categories it appears in (lowercased, stripped for grouping)
    trigger_categories: dict[str, list[str]] = {}
    total_triggers = 0
    for category, triggers in FAILURE_PATTERNS.items():
        for raw in triggers:
            total_triggers += 1
            key = raw.strip().lower()
            trigger_categories.setdefault(key, [])
            if category not in trigger_categories[key]:
                trigger_categories[key].append(category)

    n_categories = len(FAILURE_PATTERNS)
    print("=" * 78)
    print("AUGMENTOR BLACK-HOLE AUDIT  (FAILURE_PATTERNS force-inject routing)")
    print("=" * 78)
    print(f"categories: {n_categories}   triggers: {total_triggers}   "
          f"unique triggers: {len(trigger_categories)}")

    # ── DRIFT: triggers in >1 category ──
    drift = {t: cats for t, cats in trigger_categories.items() if len(cats) > 1}
    print(f"\n[DRIFT] triggers force-injecting MORE THAN ONE category: {len(drift)}")
    print("-" * 78)
    if drift:
        for t in sorted(drift, key=lambda k: (-len(drift[k]), k)):
            print(f"  {t!r:42} -> {', '.join(sorted(drift[t]))}")
    else:
        print("  (none — no cross-category contention)")

    # ── BREADTH: broad triggers per category ──
    broad: list[tuple[str, str, list[str]]] = []
    for category, triggers in FAILURE_PATTERNS.items():
        for raw in triggers:
            fl = _breadth_flags(raw)
            if fl:
                broad.append((category, raw, fl))
    print(f"\n[BREADTH] triggers broad enough to over-fire: {len(broad)}")
    print("-" * 78)
    if broad:
        for category, raw, fl in sorted(broad, key=lambda x: (x[0], x[1])):
            print(f"  {category:22} {raw!r:24} {', '.join(fl)}")
    else:
        print("  (none — all triggers are scoped/anchored)")

    # ── Per-category trigger counts (library-growth pressure) ──
    print("\n[SIZE] triggers per category (largest first — growth pressure):")
    print("-" * 78)
    for category, triggers in sorted(
        FAILURE_PATTERNS.items(), key=lambda kv: -len(kv[1])
    )[:8]:
        print(f"  {category:22} {len(triggers)} triggers")

    print("\n" + "=" * 78)
    flagged = len(drift) + len(broad)
    print(f"SUMMARY: {len(drift)} drift + {len(broad)} broad = {flagged} flagged "
          f"of {total_triggers} triggers ({100*flagged/total_triggers:.1f}%).")
    print("Drift triggers are the highest priority (cross-family contention).")
    print("Broad triggers should be scoped (add a code anchor or a 2nd word) "
          "OR confirmed harmless if the word is rare in practice.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
