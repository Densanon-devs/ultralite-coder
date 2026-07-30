"""
Write policy — the guardrail that makes machine-wide mutation safe enough to allow.

Context for why this is strict. Until now every agent file tool resolved paths
against the Workspace root, and `Workspace.is_inside()` was written but never
called — so the boundary was nominal, not enforced. Hybrid mode deliberately
lets the agent create/move/edit outside the current project, which turns that
nominal boundary into a real risk: this model has documented ceilings on
ambiguous-anchor edits and multi-line JSON, so a mis-parsed path is a realistic
failure, not a hypothetical one.

Two mechanisms:

  1. ALLOWLIST. Writes are permitted only under roots the user opted into.
     Everything else is REFUSED outright rather than prompted — a prompt you see
     fifty times is a prompt you stop reading, and system directories should be
     unreachable by construction, not by vigilance.

  2. JOURNAL. Every mutation is recorded, and overwritten/deleted content is
     copied aside first, so a bad edit is reversible after the fact instead of
     merely regrettable.

Stdlib only. Nothing leaves the machine.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_STATE_DIR = Path.home() / ".ultralight-coder"
_ROOTS_FILE = _STATE_DIR / "write_roots.txt"
_JOURNAL = _STATE_DIR / "mutations.jsonl"
_BACKUP_DIR = _STATE_DIR / "backups"

# Never writable, even if a configured root would otherwise contain them (e.g.
# someone adds C:/ as a root). Matched as path segments, so C:/Windows and
# D:/x/Windows both trip it.
#
# Scope note: this list is the BACKSTOP for locations that must never be
# writable — breaking them breaks the OS. It is deliberately NOT a list of
# "everything sensitive". The allowlist is the primary control, and AppData is
# already absent from the defaults. Denying `appdata` as a segment was tried and
# reverted: it silently made every Windows temp directory unwritable (they live
# under AppData/Local/Temp) and would have quietly voided an AppData root a user
# deliberately added. Specific secret FILENAMES are still refused everywhere via
# _DENY_NAMES, which is the protection that actually matters there.
_DENY_SEGMENTS = frozenset({
    "windows", "winsxs", "system32", "syswow64", "programdata",
    "program files", "program files (x86)", "$recycle.bin",
    "system volume information", "recovery", "boot", "$winreagent",
})

# Files we refuse to touch wherever they live: secrets and key material.
_DENY_NAMES = frozenset({
    "db.key", "pairing.json", "id_rsa", "id_ed25519", ".env",
    "credentials.json", "token.json", "license.key",
})


def _norm(p: Path) -> str:
    return str(p).replace("\\", "/").rstrip("/").lower()


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:      # so `if policy.check(...)` reads naturally
        return self.allowed


def default_write_roots() -> list[Path]:
    """Where writing is plausible on this machine, in priority order."""
    home = Path.home()
    candidates = [
        Path("D:/LLCWork"),
        Path("G:/My Drive"),
        home / "Documents",
        home / "Desktop",
        home / "Downloads",
    ]
    return [c for c in candidates if c.exists()]


def configured_write_roots() -> list[Path]:
    if _ROOTS_FILE.exists():
        try:
            lines = [l.strip() for l in _ROOTS_FILE.read_text(encoding="utf-8").splitlines()]
            roots = [Path(l) for l in lines if l and not l.startswith("#")]
            live = [r for r in roots if r.exists()]
            if live:
                return live
        except OSError:
            pass
    return default_write_roots()


def save_write_roots(roots: Iterable[Path]) -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    body = "\n".join(str(r) for r in roots)
    _ROOTS_FILE.write_text(
        "# Directories ulcagent may CREATE / MOVE / EDIT inside (hybrid mode).\n"
        "# One path per line. Anything outside these is refused, not prompted.\n"
        "# Delete this file to return to the auto-detected defaults.\n" + body + "\n",
        encoding="utf-8",
    )
    return _ROOTS_FILE


@dataclass
class WritePolicy:
    roots: list[Path]
    extra_roots: list[Path] = None      # e.g. the current workspace

    @classmethod
    def load(cls, workspace: Optional[Path] = None) -> "WritePolicy":
        extra = [workspace] if workspace else []
        return cls(roots=configured_write_roots(), extra_roots=extra)

    def all_roots(self) -> list[Path]:
        return [r for r in list(self.roots) + list(self.extra_roots or []) if r]

    def check(self, path: Path | str, op: str = "write") -> Verdict:
        """Whether *op* may touch *path*."""
        try:
            # resolve() collapses .. and follows junctions/symlinks, so a link
            # planted inside an allowed root cannot be used to escape it.
            target = Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            return Verdict(False, f"path could not be resolved: {exc}")

        segments = {s.lower() for s in target.parts}
        for bad in _DENY_SEGMENTS:
            if bad in segments:
                return Verdict(False, f"'{bad}' is a protected system location")

        if target.name.lower() in _DENY_NAMES:
            return Verdict(False, f"{target.name} holds credentials or key material")

        t = _norm(target)
        for root in self.all_roots():
            try:
                r = _norm(Path(root).expanduser().resolve())
            except (OSError, RuntimeError):
                continue
            if t == r or t.startswith(r + "/"):
                return Verdict(True, f"inside allowed root {root}")

        listed = "\n".join(f"    {r}" for r in self.all_roots()) or "    (none configured)"
        return Verdict(
            False,
            f"{target} is outside every allowed write root. Allowed:\n{listed}\n"
            f"  Add one with: ulcagent --write-root \"<path>\"",
        )

    def describe(self) -> str:
        return "\n".join(f"  {r}" for r in self.all_roots()) or "  (none)"


# ── journal + reversal ───────────────────────────────────────────

def _backup(path: Path) -> Optional[str]:
    """Copy a file aside before it's overwritten or removed."""
    if not path.is_file():
        return None
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = _BACKUP_DIR / f"{stamp}-{os.getpid()}-{path.name}"
    n = 0
    while dest.exists():
        n += 1
        dest = _BACKUP_DIR / f"{stamp}-{os.getpid()}-{n}-{path.name}"
    try:
        shutil.copy2(path, dest)
        return str(dest)
    except OSError:
        return None


def record(op: str, path: Path | str, *, dest: Optional[Path | str] = None,
           backup: Optional[str] = None, note: str = "") -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "op": op,
        "path": str(path),
        "dest": str(dest) if dest else "",
        "backup": backup or "",
        "note": note,
    }
    try:
        with _JOURNAL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def journal_entries(limit: int = 20) -> list[dict]:
    if not _JOURNAL.exists():
        return []
    try:
        lines = _JOURNAL.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-max(1, limit):]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def revert(entry: dict) -> tuple[bool, str]:
    """Undo one journal entry. Returns (ok, message)."""
    op = entry.get("op", "")
    path = Path(entry.get("path", ""))
    dest = Path(entry.get("dest")) if entry.get("dest") else None
    backup = entry.get("backup") or ""

    try:
        if op == "move" and dest is not None:
            if not dest.exists():
                return False, f"{dest} no longer exists — cannot move it back"
            if path.exists():
                return False, f"{path} exists again — refusing to overwrite it"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(path))
            return True, f"moved back: {dest} -> {path}"

        if op == "create":
            if not path.exists():
                return False, f"{path} is already gone"
            bak = _backup(path)
            path.unlink()
            return True, f"deleted created file {path}" + (f" (copy at {bak})" if bak else "")

        if op in ("edit", "overwrite", "delete"):
            if not backup or not Path(backup).is_file():
                return False, f"no backup recorded for {path}"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, path)
            return True, f"restored {path} from backup"

        return False, f"don't know how to revert op {op!r}"
    except OSError as exc:
        return False, f"revert failed: {exc}"


def revert_last(count: int = 1) -> list[str]:
    """Revert the most recent *count* mutations, newest first."""
    entries = journal_entries(limit=200)
    msgs: list[str] = []
    for entry in reversed(entries):
        if len(msgs) >= max(1, count):
            break
        ok, msg = revert(entry)
        msgs.append(("ok: " if ok else "skipped: ") + msg)
        if ok:
            record("revert", entry.get("path", ""), note=f"reverted {entry.get('op')}")
    return msgs or ["nothing in the mutation journal to revert"]
