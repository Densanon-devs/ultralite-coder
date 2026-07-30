"""
Guards `ulcagent --help` against drift.

This test exists because the help text HAD drifted: `--toolset` shipped without
ever being documented, and `--mission` was listed twice. Both are the same
failure — help is edited by hand while flags and profiles are added elsewhere,
so nothing catches the gap.

Rather than assert a fixed list of strings (which rots the same way), these
tests derive the expectations from the code itself: every flag ulcagent actually
parses out of sys.argv, and every profile in TOOLSETS, must appear in the help.

Run: python -m pytest test_help_coverage.py -v
     OR just: python test_help_coverage.py
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import ulcagent
from engine.agent_builtins import TOOLSETS

_SOURCE = (ROOT / "ulcagent.py").read_text(encoding="utf-8")


def _help_text() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ulcagent._print_help()
    return re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


def _flags_the_code_actually_parses() -> set[str]:
    """Every `--flag` ulcagent tests for in sys.argv, taken from the source."""
    found = set()
    for pattern in (r'"(--[a-z][a-z0-9-]+)" in sys\.argv',
                    r'_a == "(--[a-z][a-z0-9-]+)"',
                    r'sys\.argv\.index\("(--[a-z][a-z0-9-]+)"\)'):
        found.update(re.findall(pattern, _SOURCE))
    return found


# Flags that are intentionally undocumented in the user-facing help.
_INTENTIONALLY_UNDOCUMENTED = {
    "--frontdoor",        # experimental opt-in, documented in the module docstring
    "--agent-full-init",  # escape hatch for the fast path
    "--lsp",              # experiment branch
    "--cli",              # launcher-level, not ulcagent's own
}


def test_every_parsed_flag_is_documented():
    help_text = _help_text()
    missing = sorted(
        f for f in _flags_the_code_actually_parses()
        if f not in _INTENTIONALLY_UNDOCUMENTED and f not in help_text
    )
    assert not missing, (
        f"flags parsed by ulcagent but absent from --help: {missing}. "
        f"Add them to _HELP_TEXT, or to _INTENTIONALLY_UNDOCUMENTED with a reason."
    )


def test_every_toolset_profile_is_documented():
    help_text = _help_text()
    missing = [name for name in TOOLSETS if not re.search(rf"^\s+{name}\s+\d+", help_text, re.M)]
    assert not missing, (
        f"toolset profiles missing from --help: {sorted(missing)}. "
        f"The --toolset block should list each profile and its tool count."
    )


def test_documented_tool_counts_match_reality():
    """A stale count in the help is worse than no count."""
    help_text = _help_text()
    for name, tools in TOOLSETS.items():
        m = re.search(rf"^\s+{name}\s+(\d+)", help_text, re.M)
        if m is None:
            continue          # covered by the test above
        assert int(m.group(1)) == len(tools), (
            f"help says {name} has {m.group(1)} tools, actually {len(tools)}"
        )


def test_no_flag_is_documented_twice():
    """--mission was listed twice before this test existed."""
    help_text = _help_text()
    dupes = []
    for flag in _flags_the_code_actually_parses():
        # Count only definition lines (flag at the start of an indented entry),
        # so prose mentions elsewhere don't trip it.
        n = len(re.findall(rf"^\s+{re.escape(flag)}(?:\s|$)", help_text, re.M))
        if n > 1:
            dupes.append((flag, n))
    assert not dupes, f"flags documented more than once: {dupes}"


def test_the_headline_capabilities_are_findable():
    """The four powers hybrid mode was built for should be discoverable."""
    help_text = _help_text().lower()
    for term in ("locate", "move_path", "recall", "--write-root", "--revert-last"):
        assert term.lower() in help_text, f"{term} is not mentioned in --help"


def test_densassistant_integration_is_documented_both_ways():
    help_text = _help_text()
    assert "mcp_server" in help_text, "the DensAssistant->ulcagent direction is undocumented"
    assert "recall" in help_text, "the ulcagent->DensAssistant direction is undocumented"


def test_help_renders_without_color_codes_leaking():
    """The template uses {bold}/{end} placeholders — an unsubstituted one is a bug."""
    raw = _help_text()
    for placeholder in ("{bold}", "{end}", "{dim}", "{cyan}"):
        assert placeholder not in raw, f"unsubstituted {placeholder} in help output"


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
            except Exception as e:
                failures.append((name, f"{type(e).__name__}: {e}"))
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} failed")
        sys.exit(1)
    print("all passed")
