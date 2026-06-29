"""Tests for the --toolset named tool-profile feature (engine.agent_builtins).

The 14B regresses past ~10 unrelated tools (feedback_tool_count_regression), so
a named toolset is core-10 + a small themed add-on. These lock in the profile
membership, the legacy extended_tools path (unchanged), and toolset authority.
"""
import tempfile

import pytest

from engine.agent_builtins import (
    build_default_registry,
    TOOLSETS,
    CORE_TOOL_NAMES,
    EXTENDED_TOOL_NAMES,
    WEB_TOOL_NAMES,
)


def _names(**kw):
    ws = tempfile.mkdtemp()
    return set(build_default_registry(ws, **kw).status()["tools"])


def test_legacy_lean_unchanged():
    # No toolset, no extended -> the proven lean core of 10.
    assert _names() == set(CORE_TOOL_NAMES)
    assert len(CORE_TOOL_NAMES) == 10


def test_legacy_extended_unchanged():
    # extended_tools=True without a toolset still yields core+extended (22),
    # exactly as before this feature landed.
    got = _names(extended_tools=True)
    assert got == set(CORE_TOOL_NAMES) | set(EXTENDED_TOOL_NAMES)
    assert len(got) == 22


@pytest.mark.parametrize("toolset", sorted(TOOLSETS))
def test_toolset_matches_profile_exactly(toolset):
    assert _names(toolset=toolset) == set(TOOLSETS[toolset])


def test_every_profile_contains_core():
    for ts, allowed in TOOLSETS.items():
        assert CORE_TOOL_NAMES <= allowed, f"{ts} dropped a core tool"


def test_profiles_stay_under_count_cliff():
    # Themed profiles must stay well under the 21-tool regression cliff.
    for ts in ("coding", "refactor", "git", "web"):
        assert len(TOOLSETS[ts]) <= 16, f"{ts} too large: {len(TOOLSETS[ts])}"


def test_full_is_whole_universe():
    assert _names(toolset="full") == (
        set(CORE_TOOL_NAMES) | set(EXTENDED_TOOL_NAMES) | set(WEB_TOOL_NAMES)
    )


def test_toolset_overrides_extended_and_web():
    # toolset is authoritative: coding stays lean even with --extended/--web.
    assert _names(toolset="coding", extended_tools=True, enable_web=True) == set(CORE_TOOL_NAMES)


def test_unknown_toolset_raises():
    with pytest.raises(ValueError):
        _names(toolset="nope")
