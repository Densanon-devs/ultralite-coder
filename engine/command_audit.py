"""Post-execution filesystem-delta audit for destructive shell commands.

`engine/destructive_command_gate.py` stops (or forces a mandatory y/N on) a
matched command *before* it runs. This module is the complementary *after*
layer: once an approved destructive command has actually executed, it records
what the command *really* changed on disk.

Why both layers exist:

  - The gate is a **prediction** — "this command string looks destructive."
  - The audit is **ground truth** — "these files were actually deleted /
    created / modified."

A free-form shell command does not declare its blast radius in any structured
way, so a literal "declared-vs-actual" comparison is not computable. Instead we
diff the workspace against a snapshot taken immediately before execution. Two
uses for the result:

  1. **Forensic log** — an append-only record of every consequential
     filesystem action (the kind an EU-AI-Act-style runtime audit layer
     requires), written to ``<workspace>/.ulcagent_audit.log``.
  2. **Model-visible divergence** — the actual delta is threaded back to the
     agent as a synthetic ``command_audit`` observation so it can react to
     "I deleted more than I intended" on the very next turn. This mirrors the
     existing ``auto_verify`` threading.

Snapshots are a *bounded* walk (skips vendor/cache dirs, caps the entry count)
so the cost stays negligible. This only ever fires on the handful of commands
that match the destructive gate and are approved, so the walk happens rarely.

See also: ``engine/destructive_command_gate.py`` (the pre-execution half),
``feedback_destructive_command_hook.md`` (the cross-project rule).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Directories never worth walking for an audit snapshot — vendored deps,
# caches, and VCS internals. Mirrors the list_dir tool's skip set. Keeping
# these out bounds the cost and avoids flagging churn the agent didn't cause
# (e.g. a tool writing into __pycache__).
_IGNORE_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)

# The audit log itself lives in the workspace; never let writing it register
# as a change the next snapshot picks up.
AUDIT_LOG_NAME = ".ulcagent_audit.log"

# Hard cap on snapshot entries. Above this the workspace is too large to
# snapshot cheaply on every destructive command, so we skip the delta (the
# forensic log still records that the command ran, just without a file list).
DEFAULT_MAX_ENTRIES = 20_000

# (mtime_ns, size_bytes) keyed by workspace-relative POSIX path.
Snapshot = Dict[str, Tuple[int, int]]


@dataclass(frozen=True)
class CommandAudit:
    """The actual filesystem delta produced by one destructive command."""

    command: str
    matched_pattern_ids: List[str]
    deleted: List[str] = field(default_factory=list)
    created: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    # True when the snapshot couldn't be taken (workspace too large / not a
    # dir). The command still ran; we just have no file-level delta.
    skipped: bool = False
    skip_reason: str = ""

    def changed_any(self) -> bool:
        return bool(self.deleted or self.created or self.modified)

    def total_changes(self) -> int:
        return len(self.deleted) + len(self.created) + len(self.modified)


def snapshot_workspace(
    root: Optional[Path], max_entries: int = DEFAULT_MAX_ENTRIES
) -> Optional[Snapshot]:
    """Bounded walk of ``root`` → {relpath: (mtime_ns, size)}.

    Returns ``None`` if ``root`` is unset, not a directory, or has more than
    ``max_entries`` files (too large to snapshot cheaply). Skips vendor/cache
    dirs and the audit log itself. Files that vanish mid-walk are ignored.
    """
    if root is None:
        return None
    try:
        base = Path(root).resolve()
    except OSError:
        return None
    if not base.is_dir():
        return None

    snap: Snapshot = {}
    for dirpath, dirnames, filenames in os.walk(base):
        # Prune ignored dirs in place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for name in filenames:
            if name == AUDIT_LOG_NAME:
                continue
            full = Path(dirpath) / name
            try:
                st = full.stat()
            except OSError:
                # Raced/symlink-dangling/permission — skip silently.
                continue
            try:
                rel = full.resolve().relative_to(base).as_posix()
            except (OSError, ValueError):
                continue
            snap[rel] = (st.st_mtime_ns, st.st_size)
            if len(snap) > max_entries:
                return None
    return snap


def diff_snapshots(
    before: Optional[Snapshot],
    after: Optional[Snapshot],
    command: str,
    matched_pattern_ids: List[str],
) -> CommandAudit:
    """Compute the delta between two snapshots into a :class:`CommandAudit`.

    If either snapshot is ``None`` (couldn't be taken), the audit is marked
    ``skipped`` — the command ran but no file-level delta is available.
    """
    if before is None or after is None:
        return CommandAudit(
            command=command,
            matched_pattern_ids=list(matched_pattern_ids),
            skipped=True,
            skip_reason="workspace snapshot unavailable (unset or too large)",
        )

    before_keys = set(before)
    after_keys = set(after)
    deleted = sorted(before_keys - after_keys)
    created = sorted(after_keys - before_keys)
    modified = sorted(
        k for k in (before_keys & after_keys) if before[k] != after[k]
    )
    return CommandAudit(
        command=command,
        matched_pattern_ids=list(matched_pattern_ids),
        deleted=deleted,
        created=created,
        modified=modified,
    )


def _cap_list(items: List[str], cap: int = 50) -> List[str]:
    """Trim a long path list for display, appending an overflow marker."""
    if len(items) <= cap:
        return items
    return items[:cap] + [f"... and {len(items) - cap} more"]


def format_audit_observation(audit: CommandAudit) -> str:
    """Human/model-readable summary of what a destructive command actually did."""
    if audit.skipped:
        return (
            "command_audit: destructive command executed; filesystem delta "
            f"not captured ({audit.skip_reason})."
        )
    if not audit.changed_any():
        return (
            "command_audit: destructive command executed but changed no files "
            "under the workspace."
        )
    lines = [
        "command_audit: post-execution filesystem delta for the destructive "
        "command you just ran:",
    ]
    if audit.deleted:
        lines.append(f"  deleted ({len(audit.deleted)}):")
        lines += [f"    - {p}" for p in _cap_list(audit.deleted)]
    if audit.created:
        lines.append(f"  created ({len(audit.created)}):")
        lines += [f"    + {p}" for p in _cap_list(audit.created)]
    if audit.modified:
        lines.append(f"  modified ({len(audit.modified)}):")
        lines += [f"    ~ {p}" for p in _cap_list(audit.modified)]
    lines.append(
        "  If any of these were NOT intended, stop and tell the user before "
        "continuing."
    )
    return "\n".join(lines)


def append_audit_log(
    log_path: Path, audit: CommandAudit, when: Optional[datetime] = None
) -> None:
    """Append one audit record to the workspace audit log (best-effort).

    Append-only plain text; one timestamped block per destructive command.
    Failures to write are swallowed — the audit log must never break the
    agent loop.
    """
    ts = (when or datetime.now(timezone.utc)).isoformat()
    patterns = ", ".join(audit.matched_pattern_ids) or "(none)"
    block = [
        f"=== {ts} ===",
        f"command: {audit.command}",
        f"matched: {patterns}",
    ]
    if audit.skipped:
        block.append(f"delta: SKIPPED ({audit.skip_reason})")
    elif not audit.changed_any():
        block.append("delta: no files changed under workspace")
    else:
        for p in audit.deleted:
            block.append(f"  deleted: {p}")
        for p in audit.created:
            block.append(f"  created: {p}")
        for p in audit.modified:
            block.append(f"  modified: {p}")
    block.append("")
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(block) + "\n")
    except OSError:
        pass
