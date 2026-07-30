"""
Machine-wide filename index — the primitive ulcagent was missing.

Why this exists: every discovery tool (`glob`, `grep`, `list_dir`) resolves
paths against the agent's Workspace root, so ulcagent physically could not
answer "where is the Densanon LLC folder?" — the folder lives on G:, the
workspace was D:\LLCWork. It walked the wrong tree and reported "not present".

This module builds a SQLite index of file and directory NAMES across a few
configured roots, so `locate("densanon llc")` answers in milliseconds from
anywhere. Metadata only: it calls os.scandir and never opens a file, so
indexing a Google-Drive mirror does not trigger downloads.

Stdlib only (sqlite3 + os). No server, no daemon, nothing leaves the box —
same privacy invariant as the rest of agent mode.
"""
from __future__ import annotations

import os
import sqlite3
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_INDEX_DIR = Path.home() / ".ultralight-coder" / "index"
_INDEX_DB = _INDEX_DIR / "file_index.db"
_ROOTS_FILE = Path.home() / ".ultralight-coder" / "index_roots.txt"

# Refresh automatically once the index is older than this.
STALE_AFTER_SEC = 24 * 3600

# Directory names we never descend into. Two classes: OS/system noise that
# nobody asks about, and dependency/build trees that would swamp the index with
# hundreds of thousands of vendored files.
SKIP_DIRS = frozenset({
    # OS / system
    "$Recycle.Bin", "System Volume Information", "Windows", "WinSxS",
    "Recovery", "PerfLogs", "$WinREAgent", "OneDriveTemp",
    # caches + package/dependency trees
    "AppData", ".cache", "node_modules", "__pycache__", ".venv", "venv",
    "site-packages", ".git", ".gradle", ".expo", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".next", ".nuxt", "Cargo.lock",
    ".cargo", ".rustup", ".nuget", ".m2", ".npm", ".yarn", ".pnpm-store",
    # our own artifacts
    "models", "dist", "build", "target",
})

# Individual path segments that mark a tree as uninteresting even at depth.
_SKIP_SUFFIXES = (".egg-info", ".dist-info")

_MAX_FILES_PER_ROOT = 2_000_000       # runaway backstop


@dataclass
class Hit:
    path: str
    name: str
    is_dir: bool
    size: int
    mtime: float

    @property
    def kind(self) -> str:
        return "dir" if self.is_dir else "file"


# ── roots ────────────────────────────────────────────────────────

def default_roots() -> list[Path]:
    """Sensible roots for a personal machine, in priority order.

    The user's home (minus AppData, pruned by SKIP_DIRS), every other fixed
    drive, and a Google-Drive / OneDrive mirror if one is mounted. Deliberately
    NOT C:\\ wholesale — Windows + Program Files are ~500k files nobody asks
    "where is" about, and they'd dominate the index.
    """
    roots: list[Path] = []
    home = Path.home()
    if home.exists():
        roots.append(home)

    for letter in string.ascii_uppercase:
        if letter == "C":
            continue                      # covered by home; rest is OS noise
        drive = Path(f"{letter}:/")
        try:
            if drive.exists() and os.path.ismount(str(drive)):
                roots.append(drive)
        except OSError:
            continue

    # Cloud mirrors often sit under a drive already added above; adding the
    # "My Drive" subfolder too is harmless (dedup happens in build()).
    for candidate in (Path("G:/My Drive"), home / "OneDrive"):
        if candidate.exists():
            roots.append(candidate)
    return roots


def configured_roots() -> list[Path]:
    """Roots from ~/.ultralight-coder/index_roots.txt, else default_roots()."""
    if _ROOTS_FILE.exists():
        try:
            lines = [l.strip() for l in _ROOTS_FILE.read_text(encoding="utf-8").splitlines()]
            roots = [Path(l) for l in lines if l and not l.startswith("#")]
            live = [r for r in roots if r.exists()]
            if live:
                return live
        except OSError:
            pass
    return default_roots()


def save_roots(roots: Iterable[Path]) -> Path:
    _ROOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(str(r) for r in roots)
    _ROOTS_FILE.write_text(
        "# ulcagent index roots — one path per line. Delete this file to "
        "return to auto-detected defaults.\n" + body + "\n",
        encoding="utf-8",
    )
    return _ROOTS_FILE


# ── schema ───────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_INDEX_DB))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=OFF")     # rebuildable cache, not precious data
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS entries (
            path    TEXT PRIMARY KEY,
            name_lc TEXT NOT NULL,
            is_dir  INTEGER NOT NULL,
            size    INTEGER NOT NULL DEFAULT 0,
            mtime   REAL    NOT NULL DEFAULT 0,
            depth   INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_name_lc ON entries(name_lc);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    return con


def _set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def get_meta(key: str, default: str = "") -> str:
    if not _INDEX_DB.exists():
        return default
    try:
        con = _connect()
        row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        con.close()
        return row[0] if row else default
    except sqlite3.Error:
        return default


# ── build ────────────────────────────────────────────────────────

def _walk_root(root: Path) -> Iterable[tuple[str, str, int, int, float, int]]:
    """Yield (path, name_lc, is_dir, size, mtime, depth) under *root*.

    os.scandir only — never opens a file, so cloud-mirror placeholders stay
    unhydrated. entry.stat() is served from the directory enumeration on
    Windows, so size/mtime are effectively free.
    """
    root_str = str(root)
    base_depth = root_str.replace("\\", "/").count("/")
    stack = [root_str]
    seen = 0
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        name = entry.name
                        is_dir = entry.is_dir(follow_symlinks=False)
                        if is_dir:
                            if name in SKIP_DIRS or name.endswith(_SKIP_SUFFIXES):
                                continue
                            stack.append(entry.path)
                        try:
                            st = entry.stat(follow_symlinks=False)
                            size, mtime = st.st_size, st.st_mtime
                        except OSError:
                            size, mtime = 0, 0.0
                        p = entry.path.replace("\\", "/")
                        yield (p, name.lower(), 1 if is_dir else 0,
                               size, mtime, p.count("/") - base_depth)
                        seen += 1
                        if seen >= _MAX_FILES_PER_ROOT:
                            return
                    except OSError:
                        continue
        except OSError:
            continue


_INSERT_SQL = ("INSERT OR REPLACE INTO entries"
               "(path,name_lc,is_dir,size,mtime,depth) VALUES(?,?,?,?,?,?)")


def _sanitize_row(row: tuple) -> tuple:
    """Strip unpaired surrogates from a row's text fields.

    Windows hands back surrogate-escaped names for files whose names aren't
    valid UTF-16 (they exist — a few show up under C:\\Users). sqlite3 refuses
    to encode those, and one bad name would otherwise kill a 20k-row batch.
    """
    path, name_lc, *rest = row
    def clean(s: str) -> str:
        return s.encode("utf-8", "replace").decode("utf-8", "replace")
    return (clean(path), clean(name_lc), *rest)


def _insert_batch(con: sqlite3.Connection, batch: list[tuple]) -> int:
    """Insert a batch, falling back to per-row sanitising only if needed.

    The happy path stays a single executemany — sanitising every path up front
    would cost an extra encode on ~500k rows for the handful that need it.
    """
    try:
        con.executemany(_INSERT_SQL, batch)
        return len(batch)
    except (UnicodeEncodeError, sqlite3.Error):
        pass
    written = 0
    for row in batch:
        for candidate in (row, _sanitize_row(row)):
            try:
                con.execute(_INSERT_SQL, candidate)
                written += 1
                break
            except (UnicodeEncodeError, sqlite3.Error):
                continue
    return written


def build(roots: Optional[list[Path]] = None,
          progress: Optional[callable] = None) -> dict:
    """(Re)build the index from scratch. Returns a stats dict."""
    roots = roots or configured_roots()
    # Drop roots nested inside an earlier root so we don't index them twice.
    uniq: list[Path] = []
    for r in roots:
        rs = str(r).replace("\\", "/").rstrip("/").lower()
        if any(rs == u or rs.startswith(u + "/") for u in
               (str(x).replace("\\", "/").rstrip("/").lower() for x in uniq)):
            continue
        uniq.append(r)

    started = time.time()
    con = _connect()
    con.execute("DELETE FROM entries")
    per_root: dict[str, int] = {}
    total = 0
    for root in uniq:
        n = 0
        batch: list[tuple] = []
        for row in _walk_root(root):
            batch.append(row)
            n += 1
            if len(batch) >= 20000:
                _insert_batch(con, batch)
                batch.clear()
                if progress:
                    progress(f"{root}: {n:,} entries")
        if batch:
            _insert_batch(con, batch)
        per_root[str(root)] = n
        total += n
        if progress:
            progress(f"{root}: {n:,} entries (done)")

    elapsed = time.time() - started
    _set_meta(con, "built_at", str(time.time()))
    _set_meta(con, "entries", str(total))
    _set_meta(con, "roots", "\n".join(str(r) for r in uniq))
    _set_meta(con, "build_sec", f"{elapsed:.1f}")
    con.commit()
    con.close()
    return {"entries": total, "roots": per_root, "elapsed_sec": elapsed}


def age_sec() -> Optional[float]:
    """Seconds since the last build, or None if never built."""
    raw = get_meta("built_at")
    if not raw:
        return None
    try:
        return time.time() - float(raw)
    except ValueError:
        return None


def is_stale() -> bool:
    a = age_sec()
    return a is None or a > STALE_AFTER_SEC


def status() -> dict:
    a = age_sec()
    return {
        "exists": _INDEX_DB.exists(),
        "entries": int(get_meta("entries", "0") or 0),
        "age_sec": a,
        "stale": is_stale(),
        "roots": [r for r in get_meta("roots").splitlines() if r],
        "db_path": str(_INDEX_DB),
        "build_sec": get_meta("build_sec", ""),
    }


# ── query ────────────────────────────────────────────────────────

def _tokens(query: str) -> list[str]:
    return [t for t in
            "".join(c if (c.isalnum() or c in "._-") else " " for c in query.lower()).split()
            if t]


def locate(query: str, *, kind: str = "any", limit: int = 20) -> list[Hit]:
    """Find entries whose path contains every token of *query*.

    Ranking favours what a person means by "where is X":
      1. the token sequence appears in the entry's own NAME, not just an ancestor
      2. exact name match
      3. directories before files (people ask for folders)
      4. shallower paths before deeply nested ones
    """
    toks = _tokens(query)
    if not toks or not _INDEX_DB.exists():
        return []

    where = ["(" + " AND ".join(["path LIKE ?"] * len(toks)) + ")"]
    params: list = [f"%{t}%" for t in toks]
    if kind == "dir":
        where.append("is_dir = 1")
    elif kind == "file":
        where.append("is_dir = 0")

    joined = " ".join(toks)
    sql = f"""
        SELECT path, name_lc, is_dir, size, mtime, depth,
               CASE WHEN instr(name_lc, ?) > 0 THEN 0 ELSE 1 END AS name_hit,
               CASE WHEN name_lc = ? THEN 0 ELSE 1 END AS exact
        FROM entries
        WHERE {' AND '.join(where)}
        ORDER BY name_hit, exact, is_dir DESC, depth, length(path)
        LIMIT ?
    """
    con = _connect()
    try:
        rows = con.execute(sql, [joined, joined] + params + [max(1, int(limit))]).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()

    out: list[Hit] = []
    for path, name_lc, is_dir, size, mtime, _depth, _nh, _ex in rows:
        out.append(Hit(path=path, name=Path(path).name, is_dir=bool(is_dir),
                       size=size, mtime=mtime))
    return out


def ensure_index(progress: Optional[callable] = None) -> dict:
    """Build the index if it has never been built. Returns status()."""
    if not _INDEX_DB.exists() or int(get_meta("entries", "0") or 0) == 0:
        build(progress=progress)
    return status()
