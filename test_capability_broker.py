"""
Tests for engine.capability_broker + the flash-toolkit catalog.

The broker exists because registering flash-toolkit's 46 scripts as 46 tool
schemas would push the 14B far past the ~10-tool accuracy cliff (22 tools
already costs ~14 points). Two brokered tools keep the registry small.

The safety-invariant tests here matter more than the plumbing ones. These
scripts were written for a human at a keyboard, and several bury a destructive
y/n prompt mid-flow — duplicate-finder offers to DELETE every duplicate, temp
-cleaner offers to empty the Recycle Bin. The catalog answers those prompts by
position, so a drifted prompt sequence or a stray "y" would destroy files with
no human in the loop. Hence: no literal "y" anywhere, ever.

Run: python -m pytest test_capability_broker.py -v
     OR just: python test_capability_broker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine import capability_broker as cb
from engine.agent_builtins import build_default_registry

CATALOG_DIR = ROOT / "data" / "capabilities"


# ── catalog integrity ───────────────────────────────────────────

def test_catalog_loads():
    caps = cb.load_catalogs(CATALOG_DIR)
    assert caps, "expected capabilities to load"
    assert len({c.name for c in caps}) == len(caps), "duplicate capability names"


def test_every_catalogued_script_exists():
    missing = [f"{c.name} -> {c.script_path}"
               for c in cb.load_catalogs(CATALOG_DIR) if not c.available()]
    assert not missing, f"catalog points at missing scripts: {missing}"


def test_prompts_only_reference_declared_args():
    """Catalog drift guard: a prompt referencing an undeclared arg would silently
    answer "" and desync every later answer."""
    for c in cb.load_catalogs(CATALOG_DIR):
        declared = {a.name for a in c.args}
        for spec in c.prompts:
            if "arg" in spec:
                assert spec["arg"] in declared, \
                    f"{c.name}: prompt references undeclared arg {spec['arg']!r}"


def test_every_capability_has_a_prompt_sequence():
    for c in cb.load_catalogs(CATALOG_DIR):
        assert c.prompts, f"{c.name} has no prompt sequence — it would hang"


# ── safety invariants ───────────────────────────────────────────

def test_no_literal_yes_anywhere_in_the_catalog():
    """The single most important invariant in this file."""
    for c in cb.load_catalogs(CATALOG_DIR):
        for spec in c.prompts:
            lit = str(spec.get("literal", "")).strip().lower()
            assert lit not in ("y", "yes"), \
                f"{c.name} answers a prompt with {lit!r} — could confirm a destructive action"


def test_safety_tail_never_confirms():
    tail = [t.strip().lower() for t in cb._SAFETY_TAIL]
    assert "y" not in tail and "yes" not in tail, cb._SAFETY_TAIL
    assert "0" in cb._SAFETY_TAIL, "tail must be able to exit a menu loop"


def test_find_duplicates_declines_deletion():
    """duplicate-finder line 166 offers to delete all duplicates."""
    cap = cb.get("find_duplicates", CATALOG_DIR)
    assert cap is not None
    rendered = cb.build_stdin(cap, {"path": "D:/tmp"}).split("\n")
    # path, then the delete prompt -> must be "n"
    assert rendered[0] == "D:/tmp"
    assert rendered[1] == "n", f"delete prompt answered {rendered[1]!r}"
    assert cap.safety == "read"


def test_temp_cleanup_declines_recycle_bin_and_is_write_class():
    cap = cb.get("temp_cleanup", CATALOG_DIR)
    assert cap is not None
    assert cap.is_write, "temp_cleanup modifies the machine"
    rendered = cb.build_stdin(cap, {}).split("\n")
    assert rendered[0] == "2", "default mode should be the safest option"
    assert rendered[1] == "n", "Recycle Bin prompt must stay 'n'"


def test_battery_report_does_not_launch_a_browser():
    cap = cb.get("battery_report", CATALOG_DIR)
    rendered = cb.build_stdin(cap, {}).split("\n")
    assert rendered[0] == "n"


def test_destructive_scripts_are_not_catalogued():
    names = {c.script.lower().replace("\\", "/") for c in cb.load_catalogs(CATALOG_DIR)}
    for banned in ("file-shredder", "user-manager", "process-manager", "hosts-editor"):
        assert not any(banned in n for n in names), \
            f"{banned} must not be catalogued without a verified prompt sequence"


def test_only_expected_write_capabilities():
    writes = {c.name for c in cb.load_catalogs(CATALOG_DIR) if c.is_write}
    assert writes == {"temp_cleanup"}, f"unexpected write-class capabilities: {writes}"


# ── stdin rendering ─────────────────────────────────────────────

def test_arg_substitution_and_defaults():
    cap = cb.get("disk_health", CATALOG_DIR)
    assert cb.build_stdin(cap, {"section": "2"}).startswith("2\n")
    assert cb.build_stdin(cap, {}).startswith("1\n"), "should fall back to default"


def test_missing_required_arg_raises():
    cap = cb.get("folder_sizes", CATALOG_DIR)
    try:
        cb.build_stdin(cap, {})
    except cb.CatalogError as e:
        assert "path" in str(e)
    else:
        raise AssertionError("expected CatalogError for missing required arg")


def test_unknown_arg_names_the_typo():
    cap = cb.get("folder_sizes", CATALOG_DIR)
    try:
        cb.build_stdin(cap, {"pth": "D:/"})
    except cb.CatalogError as e:
        assert "pth" in str(e), f"error should name the bad arg: {e}"
    else:
        raise AssertionError("expected CatalogError for unknown arg")


def test_stdin_always_ends_with_newline():
    for c in cb.load_catalogs(CATALOG_DIR):
        args = {a.name: "X" for a in c.args if a.required}
        assert cb.build_stdin(c, args).endswith("\n")


# ── search ──────────────────────────────────────────────────────

def test_search_ranks_the_obvious_answer_first():
    for query, expected in [("disk space", "folder_sizes"),
                            ("duplicate files", "find_duplicates"),
                            ("what is installed", "installed_software"),
                            ("who is on my network", "network_scan")]:
        hits = cb.search(query, limit=3, catalog_dir=CATALOG_DIR)
        assert hits, f"no hit for {query!r}"
        assert hits[0].name == expected, \
            f"{query!r} -> {hits[0].name}, expected {expected}"


def test_search_empty_query_lists_everything():
    hits = cb.search("", limit=50, catalog_dir=CATALOG_DIR)
    assert len(hits) == len([c for c in cb.load_catalogs(CATALOG_DIR) if c.available()])


def test_format_search_flags_write_capabilities():
    text = cb.format_search("temp cleanup", limit=5, catalog_dir=CATALOG_DIR)
    assert "MODIFIES MACHINE" in text


def test_unknown_capability_is_a_message_not_a_crash():
    out = cb.run("does_not_exist", {}, CATALOG_DIR)
    assert "No capability named" in out


# ── registry wiring ─────────────────────────────────────────────

def test_capability_tools_are_opt_in():
    names = {t.name for t in build_default_registry(ROOT).enabled_tools()}
    assert "run_capability" not in names
    assert "list_capabilities" not in names
    assert len(names) == 10, sorted(names)


def test_assistant_toolset_has_broker_and_stays_small():
    reg = build_default_registry(ROOT, toolset="assistant")
    names = {t.name for t in reg.enabled_tools()}
    assert {"locate", "list_capabilities", "run_capability"} <= names
    assert len(names) <= 10, sorted(names)
    for w in ("write_file", "edit_file", "insert_at_line", "apply_patch"):
        assert w not in names


def test_read_capabilities_run_without_a_prompt():
    """Reads are classified read precisely so they don't nag."""
    for name in ("system_info", "folder_sizes", "find_duplicates", "file_hash"):
        cap = cb.get(name, CATALOG_DIR)
        assert not cap.needs_confirm, f"{name} should not require confirmation"


def test_write_and_flagged_capabilities_require_confirmation():
    assert cb.get("temp_cleanup", CATALOG_DIR).needs_confirm
    # network_scan changes nothing locally but probes the whole subnet.
    assert cb.get("network_scan", CATALOG_DIR).needs_confirm


def test_write_capability_refuses_without_a_confirm_hook():
    """Fail-safe: an unattended session must not silently clean temp files."""
    out = cb.run("temp_cleanup", {}, CATALOG_DIR, confirm=None)
    assert "NOT run" in out, out


def test_write_capability_respects_a_declining_hook():
    out = cb.run("temp_cleanup", {}, CATALOG_DIR, confirm=lambda cap, args: False)
    assert "declined" in out, out


def test_enable_capabilities_flag():
    reg = build_default_registry(ROOT, enable_capabilities=True)
    assert "run_capability" in {t.name for t in reg.enabled_tools()}


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL  {name}: {e}")
    print()
    if failures:
        print(f"{len(failures)} failed")
        sys.exit(1)
    print("all passed")
