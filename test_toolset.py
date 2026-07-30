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
    _TOOLSET_UNIVERSE,
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


# Profiles come in two classes now:
#   * CODING profiles are core-10 + a themed add-on (refactor/git/web/full).
#   * SCOPED profiles are deliberately REDUCED — `assistant` answers questions
#     about the machine and must not be able to write to it.
#   * HYBRID profiles are scoped too, but DO write — outside the workspace even —
#     so they are governed by engine.write_policy's allowlist instead of by
#     omitting the tools. See test_write_policy.py for that half.
CODING_PROFILES = ("coding", "refactor", "git", "web", "full")
SCOPED_PROFILES = ("assistant",)
HYBRID_PROFILES = ("hybrid",)


def test_profile_classes_cover_every_toolset():
    """If someone adds a profile, they must classify it here."""
    classified = set(CODING_PROFILES) | set(SCOPED_PROFILES) | set(HYBRID_PROFILES)
    assert classified == set(TOOLSETS), (
        "unclassified toolset — add it to CODING_PROFILES, SCOPED_PROFILES "
        "or HYBRID_PROFILES"
    )


def test_hybrid_profiles_are_small_and_carry_the_asked_for_powers():
    for ts in HYBRID_PROFILES:
        allowed = TOOLSETS[ts]
        assert len(allowed) <= 10, f"{ts} is {len(allowed)} tools — over the cliff"
        for needed in ("locate", "move_path", "write_file", "edit_file", "recall"):
            assert needed in allowed, f"{ts} is missing {needed}"


def test_every_coding_profile_contains_core():
    for ts in CODING_PROFILES:
        assert CORE_TOOL_NAMES <= TOOLSETS[ts], f"{ts} dropped a core tool"


def test_scoped_profiles_have_no_write_tools():
    """The point of a reduced profile: it cannot modify files."""
    for ts in SCOPED_PROFILES:
        allowed = TOOLSETS[ts]
        for w in ("write_file", "edit_file", "insert_at_line", "apply_patch",
                  "rename_symbol", "add_import", "restore"):
            assert w not in allowed, f"{ts} must not expose {w}"


def test_profiles_stay_under_count_cliff():
    # Themed profiles must stay well under the 21-tool regression cliff.
    for ts in ("coding", "refactor", "git", "web"):
        assert len(TOOLSETS[ts]) <= 16, f"{ts} too large: {len(TOOLSETS[ts])}"


def test_full_is_whole_universe():
    # The universe grew an assistant tier (locate + the capability broker), so
    # assert against it directly rather than re-deriving from the coding sets.
    assert _names(toolset="full") == set(_TOOLSET_UNIVERSE)
    assert set(CORE_TOOL_NAMES) | set(EXTENDED_TOOL_NAMES) | set(WEB_TOOL_NAMES)         <= set(_TOOLSET_UNIVERSE)


def test_toolset_overrides_extended_and_web():
    # toolset is authoritative: coding stays lean even with --extended/--web.
    assert _names(toolset="coding", extended_tools=True, enable_web=True) == set(CORE_TOOL_NAMES)


def test_unknown_toolset_raises():
    with pytest.raises(ValueError):
        _names(toolset="nope")
