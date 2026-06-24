"""
bench_frontdoor_precision.py — PRECISION measurement of the deterministic
front-door classifier (engine/deterministic_frontdoor.py).

The front-door intercepts mechanical coding goals before the 14B. Its prime
danger is a FALSE POSITIVE: classifying (and later executing) a deterministic
edit on a goal that actually needed the model. A bad rename/import/format on a
goal it misread = silent code corruption — far worse than over-deferring.

Existing unit tests prove it abstains on a handful of generative phrasings.
This bench measures PRECISION on a realistic, hand-labeled mixed stream of
coding goals: does the classifier ever fire (HANDLE) on a goal that should DEFER?

We measure the CLASSIFY decision only (parse() — side-effect-free), not
execution. parse() returning non-None == HANDLE; None == DEFER. A throwaway
tempdir backs the constructor; no files are written, no model is loaded.

Run:  python bench_frontdoor_precision.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:  # Windows cp1252 stdout would crash on any non-ASCII; force UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.deterministic_frontdoor import DeterministicFrontDoor


# ── Hand-labeled goal set ────────────────────────────────────────
# Each entry: (goal, expected, bucket)
#   expected: "HANDLE" (truly mechanical, safe to execute deterministically)
#             "DEFER"  (needs the model — firing here would corrupt code)

GOALS: list[tuple[str, str, str]] = [
    # ── mechanical_handleable → expect HANDLE ──────────────────────
    # rename
    ("rename foo to bar", "HANDLE", "rename"),
    ("rename calculate_total to compute_total", "HANDLE", "rename"),
    ("rename the function process to handle", "HANDLE", "rename"),
    ("rename old_name -> new_name", "HANDLE", "rename"),
    ("rename `getUser` to `fetchUser`", "HANDLE", "rename"),
    ("rename the variable count to counter", "HANDLE", "rename"),
    ("rename the class Widget to Gadget", "HANDLE", "rename"),
    ("rename foo to bar everywhere", "HANDLE", "rename"),

    # add_import
    ("add import os to utils.py", "HANDLE", "add_import"),
    ("add `from pathlib import Path` to main.py", "HANDLE", "add_import"),
    ("add import sys to main.py", "HANDLE", "add_import"),
    ("add import json in config.py", "HANDLE", "add_import"),
    ("add from collections import defaultdict to graph.py", "HANDLE", "add_import"),
    ("insert import re to parser.py", "HANDLE", "add_import"),

    # format (only HANDLE if a formatter is installed; see note below)
    ("format main.py", "HANDLE", "format"),
    ("reformat utils.py", "HANDLE", "format"),
    ("format config.py with black", "HANDLE", "format"),
    ("format src/app.py", "HANDLE", "format"),

    # trailing_newline
    ("ensure trailing newline in config.py", "HANDLE", "trailing_newline"),
    ("ensure config.py ends with a newline", "HANDLE", "trailing_newline"),
    ("add a trailing newline to main.py", "HANDLE", "trailing_newline"),
    ("add a final newline to utils.py", "HANDLE", "trailing_newline"),
    ("strip trailing newlines from data.txt", "HANDLE", "trailing_newline"),
    ("remove the trailing newline from notes.md", "HANDLE", "trailing_newline"),

    # create_empty
    ("create empty file parser.py", "HANDLE", "create_empty"),
    ("create parser.py", "HANDLE", "create_empty"),
    ("create a new file models.py", "HANDLE", "create_empty"),
    ("create an empty file __init__.py", "HANDLE", "create_empty"),
    ("create file tests/test_api.py", "HANDLE", "create_empty"),

    # ── generative → expect DEFER ──────────────────────────────────
    ("fix the bug in login", "DEFER", "generative"),
    ("write a CSV parser", "DEFER", "generative"),
    ("add error handling to the API", "DEFER", "generative"),
    ("optimize the query", "DEFER", "generative"),
    ("explain what this function does", "DEFER", "generative"),
    ("refactor the auth module", "DEFER", "generative"),
    ("add tests for the parser", "DEFER", "generative"),
    ("implement a binary search", "DEFER", "generative"),
    ("debug the failing test", "DEFER", "generative"),
    ("validate the user input", "DEFER", "generative"),
    ("summarize the changes in this file", "DEFER", "generative"),
    ("review the pull request", "DEFER", "generative"),
    ("why does this crash", "DEFER", "generative"),
    ("how should I structure the database", "DEFER", "generative"),
    ("calculate the total from the rows", "DEFER", "generative"),
    ("compute the checksum of the file", "DEFER", "generative"),
    ("write the endpoint that returns users", "DEFER", "generative"),
    ("make the function faster", "DEFER", "generative"),
    ("add logging throughout the service", "DEFER", "generative"),
    ("clean up this code", "DEFER", "generative"),
    ("update the README", "DEFER", "generative"),
    ("convert this to async", "DEFER", "generative"),
    ("document the public API", "DEFER", "generative"),

    # ── ADVERSARIAL near-misses → expect DEFER (the key bucket) ─────
    # mechanical surface, generative rider / ambiguity
    ("rename foo to bar and make it async", "DEFER", "adversarial"),
    ("rename the function that handles login", "DEFER", "adversarial"),
    ("rename the variable to something clearer", "DEFER", "adversarial"),
    ("rename the helper to a better name", "DEFER", "adversarial"),
    ("rename foo to bar then run the tests", "DEFER", "adversarial"),
    ("rename foo to bar and update the callers", "DEFER", "adversarial"),
    ("add import os and use it to list files", "DEFER", "adversarial"),
    ("add the missing import", "DEFER", "adversarial"),
    ("add an import for the json module", "DEFER", "adversarial"),
    ("add whatever imports are needed to main.py", "DEFER", "adversarial"),
    ("create parser.py that reads CSV", "DEFER", "adversarial"),
    ("create a config file", "DEFER", "adversarial"),
    ("create a new module that handles auth", "DEFER", "adversarial"),
    ("create models.py with a User class", "DEFER", "adversarial"),
    ("create the file and add a main function", "DEFER", "adversarial"),
    ("format the output nicely", "DEFER", "adversarial"),
    ("format the code so it reads better", "DEFER", "adversarial"),
    ("reformat the function to be more readable", "DEFER", "adversarial"),
    ("ensure the file is well formatted", "DEFER", "adversarial"),
    ("add a newline between the functions", "DEFER", "adversarial"),
    ("rename it to bar", "DEFER", "adversarial"),
    ("rename the thing to processor", "DEFER", "adversarial"),
    ("add import os to the file that needs it", "DEFER", "adversarial"),
    ("create a parser", "DEFER", "adversarial"),
    ("format and lint utils.py", "DEFER", "adversarial"),
    ("rename foo to bar, baz to qux", "DEFER", "adversarial"),
    ("add error handling and rename the function", "DEFER", "adversarial"),
    ("make a file called parser.py and fill it in", "DEFER", "adversarial"),
]


def classify(fd: DeterministicFrontDoor, goal: str) -> tuple[str, str | None]:
    """Return (decision, matched_category). decision is HANDLE or DEFER."""
    plan = fd.parse(goal)
    if plan is None:
        return "DEFER", None
    # Plan dataclass name → category label.
    cat = type(plan).__name__.removeprefix("_").removesuffix("Plan")
    mapping = {
        "Rename": "rename",
        "AddImport": "add_import",
        "Format": "format",
        "TrailingNewline": "trailing_newline",
        "CreateEmpty": "create_empty",
    }
    return "HANDLE", mapping.get(cat, cat.lower())


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="frontdoor_precision_")
    fd = DeterministicFrontDoor(tmp)

    # Note on the format bucket: _classify_format abstains when NO formatter
    # (black/autopep8) is installed — so on a machine without black, format
    # goals correctly DEFER. That is NOT a precision failure (deferring is
    # always safe). We detect formatter availability and adjust the expected
    # label for format goals so recall isn't penalized for an env condition.
    formatter_available = (
        DeterministicFrontDoor._pick_formatter(None) is not None
    )

    rows = []
    for goal, expected, bucket in GOALS:
        decision, matched = classify(fd, goal)
        eff_expected = expected
        if bucket == "format" and not formatter_available:
            # Environment can't format → DEFER is the correct, safe outcome.
            eff_expected = "DEFER"
        rows.append({
            "goal": goal,
            "expected": eff_expected,
            "raw_expected": expected,
            "bucket": bucket,
            "decision": decision,
            "matched": matched,
        })

    # ── Precision / recall over the CLASSIFY decision ──
    # HANDLE-truly-mechanical = true positive; HANDLE-should-defer = FALSE POS.
    handled = [r for r in rows if r["decision"] == "HANDLE"]
    true_mech = [r for r in rows if r["expected"] == "HANDLE"]

    tp = [r for r in handled if r["expected"] == "HANDLE"]
    fp = [r for r in handled if r["expected"] == "DEFER"]  # the dangerous ones
    fn = [r for r in true_mech if r["decision"] == "DEFER"]  # missed mechanicals

    precision = (len(tp) / len(handled)) if handled else 1.0
    recall = (len(tp) / len(true_mech)) if true_mech else 1.0

    print("=" * 70)
    print("DETERMINISTIC FRONT-DOOR — CLASSIFIER PRECISION BENCH")
    print("=" * 70)
    print(f"Total goals          : {len(rows)}")
    print(f"black/autopep8 found : {formatter_available}"
          + ("" if formatter_available else "  (format goals expected to DEFER)"))
    print(f"Goals HANDLED        : {len(handled)}")
    print(f"Truly mechanical     : {len(true_mech)}")
    print()
    print(f"PRECISION = {precision:.4f}  "
          f"(of {len(handled)} handled, {len(tp)} truly mechanical, "
          f"{len(fp)} FALSE POSITIVE)")
    print(f"RECALL    = {recall:.4f}  "
          f"(of {len(true_mech)} mechanical, {len(tp)} caught, "
          f"{len(fn)} deferred)")
    print()

    # ── False positives (THE DANGEROUS LIST) ──
    print("-" * 70)
    print("FALSE POSITIVES (goal should DEFER but classifier would HANDLE):")
    print("  these are the code-corruption risks — must be ZERO")
    print("-" * 70)
    if fp:
        for r in fp:
            print(f"  [!!] ({r['matched']:<16}) {r['goal']!r}")
    else:
        print("  (none)  OK")
    print()

    # ── False negatives (safe, informational) ──
    print("-" * 70)
    print("FALSE NEGATIVES (mechanical goal deferred — safe, just lost speedup):")
    print("-" * 70)
    if fn:
        for r in fn:
            tag = "" if r["raw_expected"] == r["expected"] else " [env: no formatter]"
            print(f"  [--] ({r['bucket']:<16}) {r['goal']!r}{tag}")
    else:
        print("  (none)")
    print()

    # ── Per-category / per-bucket breakdown ──
    print("-" * 70)
    print("PER-BUCKET BREAKDOWN:")
    print("-" * 70)
    buckets = {}
    for r in rows:
        b = buckets.setdefault(r["bucket"], {"handle": 0, "defer": 0,
                                             "exp_handle": 0, "exp_defer": 0,
                                             "correct": 0, "n": 0})
        b["n"] += 1
        b["handle" if r["decision"] == "HANDLE" else "defer"] += 1
        b["exp_handle" if r["expected"] == "HANDLE" else "exp_defer"] += 1
        if r["decision"] == r["expected"]:
            b["correct"] += 1
    print(f"  {'bucket':<22} {'n':>3} {'correct':>8} {'handled':>8} {'deferred':>9}")
    for name, b in sorted(buckets.items()):
        print(f"  {name:<22} {b['n']:>3} {b['correct']:>8} "
              f"{b['handle']:>8} {b['defer']:>9}")
    print()

    # ── Per-matched-category breakdown (what fired and on what) ──
    print("-" * 70)
    print("PER-MATCHED-CATEGORY (what the classifier fired as):")
    print("-" * 70)
    by_cat = {}
    for r in handled:
        c = by_cat.setdefault(r["matched"], {"good": 0, "bad": 0})
        if r["expected"] == "HANDLE":
            c["good"] += 1
        else:
            c["bad"] += 1
    if by_cat:
        for cat, c in sorted(by_cat.items()):
            flag = "  <-- HAS FALSE POSITIVES" if c["bad"] else ""
            print(f"  {cat:<18} fired {c['good'] + c['bad']:>2} "
                  f"(good {c['good']}, FALSE POS {c['bad']}){flag}")
    else:
        print("  (classifier fired on nothing)")
    print()

    # ── Verdict ──
    print("=" * 70)
    if precision == 1.0:
        print(f"VERDICT: PASS — precision 1.0, zero false positives. "
              f"Recall {recall:.3f}.")
    else:
        print(f"VERDICT: FAIL — precision {precision:.4f} < 1.0. "
              f"{len(fp)} false positive(s) above can corrupt code.")
    print("=" * 70)

    # Bar: precision must be perfect. A false positive = silent corruption.
    assert precision == 1.0, (
        f"PRECISION {precision:.4f} != 1.0 — {len(fp)} false positive(s): "
        + "; ".join(f"{r['goal']!r}->{r['matched']}" for r in fp)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
